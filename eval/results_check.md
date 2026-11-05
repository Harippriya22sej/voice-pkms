# Evaluation results

## Summary

| Metric | Result |
|---|---|
| Notes evaluated | 2 in 1 scenarios |
| Stabilization correct (all) | 2/2 (100%) |
| Stabilization correct (without hard cases) | 2/2 (100%) |
| Integration decision correct (all) | 2/2 (100%) |
| Integration decision correct (without hard cases) | 2/2 (100%) |

| Comparison | Our system | Plain-note baseline |
|---|---|---|
| Search returns the current value first | 1/1 (100%) | 0/1 (0%) |
| Search shows an outdated value (queries with a changed fact) | 0/1 (0%) | 1/1 (100%) |
| Stored entries (ideal: 1) | 1 | 2 |
| Questions waiting for the user | 0 | n/a |
| Average seconds per note | 5.6 | n/a |

## Per scenario

| # | Scenario | Hard | Stability ok | Decision ok | Search (ours / baseline) | Entries (ours / baseline / ideal) |
|---|---|---|---|---|---|---|
| 2 | Meeting moved with explicit change wording |  | 2/2 | 2/2 | ok / miss, stale | 1 / 2 / 1 |

## Failures (what to look at)

- None

## Similarity values behind the thresholds

| Scenario | Note | Expected | Got | Similarity |
|---|---|---|---|---|
| 2 | Actually, move the meeting with Anita to Thursday. | update | update | 0.675 |
