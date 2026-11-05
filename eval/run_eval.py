import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

from core import database, retrieval
from core.embeddings import cosine, embed
from core.lifecycle import OllamaLifecycleJudge
from core.llm_adapter import OllamaLLMAdapter
from core.pipeline import process_text_note

EVAL_DIR = Path(__file__).resolve().parent


def actual_outcome(result):
    if result.get("lifecycle"):
        return result["lifecycle"]["label"].lower()
    if result.get("fusion"):
        return result["fusion"]["decision"]
    return None


def similarity_of(result):
    fusion = result.get("fusion") or {}
    if fusion.get("similarity") is not None:
        return fusion["similarity"]
    return result.get("similarity")


def slot_text(slots):
    return " ".join(str(v) for v in slots.values() if v).lower()


def run_scenario(sc, llm, judge, tmp):
    database.DB_PATH = Path(tmp) / f"scenario_{sc['id']}.db"   # never touches your real data
    database.init_db()

    baseline, steps = [], []
    for step in sc["steps"]:
        start = time.time()
        result = process_text_note(step["text"], llm=llm, judge=judge)
        seconds = time.time() - start
        baseline.append((step["text"], embed(step["text"])))

        with database.get_connection() as conn:
            frag = conn.execute("SELECT intent, slots_json FROM fragments WHERE id = ?",
                                (result["fragment_id"],)).fetchone()
        outcome = actual_outcome(result)
        exp_stab, exp_out = step["expect_stability"], step["expect_outcome"]
        steps.append({
            "text": step["text"],
            "expect_stability": exp_stab, "stability": result["stability"],
            "stability_ok": None if exp_stab is None else result["stability"] == exp_stab,
            "expect_outcome": exp_out, "outcome": outcome,
            "outcome_ok": None if exp_out is None else outcome == exp_out,
            "similarity": similarity_of(result),
            "intent": frag["intent"], "slots": json.loads(frag["slots_json"] or "{}"),
            "seconds": round(seconds, 1),
        })

    query = None
    if sc.get("query"):
        cur = sc["current"].lower()
        stale = [s.lower() for s in sc.get("stale", [])]
        sys_blobs = [slot_text(h["slots"]) for h in retrieval.search(sc["query"], limit=3)]
        qv = embed(sc["query"])
        ranked = sorted(baseline, key=lambda b: cosine(qv, b[1]), reverse=True)[:3]
        base_blobs = [b[0].lower() for b in ranked]
        query = {
            "text": sc["query"],
            "system_current": bool(sys_blobs) and cur in sys_blobs[0],
            "system_stale": any(s in b for s in stale for b in sys_blobs),
            "baseline_current": cur in base_blobs[0],
            "baseline_stale": any(s in b for s in stale for b in base_blobs),
        }

    with database.get_connection() as conn:
        active = conn.execute("SELECT COUNT(*) FROM knowledge WHERE status = 'active'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM fragments WHERE stability = 'pending'").fetchone()[0]
        reviews = conn.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'").fetchone()[0]

    return {"id": sc["id"], "title": sc["title"], "hard": sc["hard"],
            "has_stale": bool(sc.get("stale")), "steps": steps, "query": query,
            "system_entries": active + pending, "baseline_entries": len(baseline),
            "ideal_entries": sc["ideal_entries"], "pending_reviews": reviews}


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def rate(results, key, include_hard=True):
    scored = [s[key] for r in results if include_hard or not r["hard"]
              for s in r["steps"] if s[key] is not None]
    return sum(scored), len(scored)


def build_markdown(results):
    L = ["# Evaluation results", ""]
    queries = [(r, r["query"]) for r in results if r["query"]]
    stale_q = [(r, q) for r, q in queries if r["has_stale"]]
    all_steps = [s for r in results for s in r["steps"]]

    L += ["## Summary", "", "| Metric | Result |", "|---|---|"]
    L.append(f"| Notes evaluated | {len(all_steps)} in {len(results)} scenarios |")
    L.append(f"| Stabilization correct (all) | {pct(*rate(results, 'stability_ok'))} |")
    L.append(f"| Stabilization correct (without hard cases) | {pct(*rate(results, 'stability_ok', False))} |")
    L.append(f"| Integration decision correct (all) | {pct(*rate(results, 'outcome_ok'))} |")
    L.append(f"| Integration decision correct (without hard cases) | {pct(*rate(results, 'outcome_ok', False))} |")
    L.append("")
    L += ["| Comparison | Our system | Plain-note baseline |", "|---|---|---|"]
    L.append(f"| Search returns the current value first | "
             f"{pct(sum(q['system_current'] for _, q in queries), len(queries))} | "
             f"{pct(sum(q['baseline_current'] for _, q in queries), len(queries))} |")
    L.append(f"| Search shows an outdated value (queries with a changed fact) | "
             f"{pct(sum(q['system_stale'] for _, q in stale_q), len(stale_q))} | "
             f"{pct(sum(q['baseline_stale'] for _, q in stale_q), len(stale_q))} |")
    L.append(f"| Stored entries (ideal: {sum(r['ideal_entries'] for r in results)}) | "
             f"{sum(r['system_entries'] for r in results)} | {sum(r['baseline_entries'] for r in results)} |")
    L.append(f"| Questions waiting for the user | {sum(r['pending_reviews'] for r in results)} | n/a |")
    L.append(f"| Average seconds per note | {sum(s['seconds'] for s in all_steps) / len(all_steps):.1f} | n/a |")

    L += ["", "## Per scenario", "",
          "| # | Scenario | Hard | Stability ok | Decision ok | Search (ours / baseline) | Entries (ours / baseline / ideal) |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        a, b = rate([r], "stability_ok")
        c, d = rate([r], "outcome_ok")
        q = r["query"]
        search = "n/a" if not q else (f"{'ok' if q['system_current'] else 'miss'}"
                                      f"{', stale' if q['system_stale'] else ''} / "
                                      f"{'ok' if q['baseline_current'] else 'miss'}"
                                      f"{', stale' if q['baseline_stale'] else ''}")
        L.append(f"| {r['id']} | {r['title']} | {'yes' if r['hard'] else ''} | {a}/{b} | {c}/{d} | "
                 f"{search} | {r['system_entries']} / {r['baseline_entries']} / {r['ideal_entries']} |")

    L += ["", "## Failures (what to look at)", ""]
    found = False
    for r in results:
        for s in r["steps"]:
            if s["stability_ok"] is False or s["outcome_ok"] is False:
                found = True
                L.append(f"- Scenario {r['id']}{' (hard)' if r['hard'] else ''}: \"{s['text']}\"  "
                         f"expected {s['expect_stability']}/{s['expect_outcome']}, "
                         f"got {s['stability']}/{s['outcome']}; intent={s['intent']}, slots={s['slots']}")
        q = r["query"]
        if q and (not q["system_current"] or q["system_stale"]):
            found = True
            L.append(f"- Scenario {r['id']} search \"{q['text']}\": "
                     f"current found={q['system_current']}, outdated shown={q['system_stale']}")
    if not found:
        L.append("- None")

    L += ["", "## Similarity values behind the thresholds", "",
          "| Scenario | Note | Expected | Got | Similarity |", "|---|---|---|---|---|"]
    for r in results:
        for s in r["steps"]:
            if s["similarity"] is not None:
                L.append(f"| {r['id']} | {s['text']} | {s['expect_outcome']} | {s['outcome']} | {s['similarity']} |")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="*", help="scenario ids to run")
    parser.add_argument("--file", default="scenarios.json", help="scenario file inside eval/")
    parser.add_argument("--out", default="results", help="output name, without extension")
    args = parser.parse_args()

    scenarios = json.loads((EVAL_DIR / args.file).read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]

    llm, judge = OllamaLLMAdapter(), OllamaLifecycleJudge()
    results = []
    # ignore_cleanup_errors: Windows may keep a database file locked a moment longer.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for sc in scenarios:
            print(f"Scenario {sc['id']}: {sc['title']} ...", flush=True)
            try:
                results.append(run_scenario(sc, llm, judge, tmp))
            except Exception as e:
                print(f"  stopped: {e}\n  Is Ollama running?")
                break
        gc.collect()   # releases open database connections before the folder is deleted

    if not results:
        return
    (EVAL_DIR / f"{args.out}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = build_markdown(results)
    (EVAL_DIR / f"{args.out}.md").write_text(report, encoding="utf-8")
    print("\n" + report)

if __name__ == "__main__":
    main()