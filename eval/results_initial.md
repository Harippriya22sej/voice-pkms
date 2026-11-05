# Evaluation results

## Summary

| Metric | Result |
|---|---|
| Notes evaluated | 26 in 13 scenarios |
| Stabilization correct (all) | 24/25 (96%) |
| Stabilization correct (without hard cases) | 21/22 (95%) |
| Integration decision correct (all) | 20/22 (91%) |
| Integration decision correct (without hard cases) | 18/19 (95%) |

| Comparison | Our system | Plain-note baseline |
|---|---|---|
| Search returns the current value first | 7/9 (78%) | 6/9 (67%) |
| Search shows an outdated value (queries with a changed fact) | 2/4 (50%) | 4/4 (100%) |
| Stored entries (ideal: 16) | 17 | 26 |
| Questions waiting for the user | 3 | n/a |
| Average seconds per note | 5.9 | n/a |

## Per scenario

| # | Scenario | Hard | Stability ok | Decision ok | Search (ours / baseline) | Entries (ours / baseline / ideal) |
|---|---|---|---|---|---|---|
| 1 | Pending note completed by a later note |  | 2/2 | 1/1 | ok / ok | 1 / 2 / 1 |
| 2 | Meeting moved with explicit change wording |  | 2/2 | 1/2 | miss, stale / miss, stale | 1 / 2 / 1 |
| 3 | Conflicting time without change wording goes to review |  | 2/2 | 2/2 | n/a | 1 / 2 / 1 |
| 4 | More detailed repeat is a refinement |  | 2/2 | 2/2 | ok / ok | 1 / 2 / 1 |
| 5 | Two different purchases stay separate |  | 2/2 | 2/2 | n/a | 2 / 2 / 2 |
| 6 | Vague task completed by a later note |  | 1/2 | 1/1 | ok / miss | 2 / 2 / 1 |
| 7 | Call without a time completed later |  | 2/2 | 1/1 | ok / ok | 1 / 2 / 1 |
| 8 | Same day, different people (hard) | yes | 2/2 | 1/2 | n/a | 1 / 2 / 2 |
| 9 | Two unrelated ideas stay separate |  | 2/2 | 2/2 | n/a | 2 / 2 / 2 |
| 10 | Time changed with the word 'change' |  | 2/2 | 2/2 | ok / ok, stale | 1 / 2 / 1 |
| 11 | Task deadline updated |  | 2/2 | 2/2 | ok / ok, stale | 1 / 2 / 1 |
| 12 | Exact duplicate note |  | 2/2 | 2/2 | ok / ok | 1 / 2 / 1 |
| 13 | Correction that names no subject (hard) | yes | 1/1 | 1/1 | miss, stale / miss, stale | 2 / 2 / 1 |

## Failures (what to look at)

- Scenario 2: "Actually, move the meeting with Anita to Thursday."  expected stable/update, got stable/ask; intent=schedule_meeting, slots={'who': 'Anita', 'what': None, 'when': 'Thursday', 'where': None}
- Scenario 2 search "when is my meeting with Anita?": current found=False, outdated shown=True
- Scenario 6: "I have to complete the chemistry assignment."  expected absorbed/new, got stable/new; intent=do_task, slots={'who': None, 'what': 'chemistry assignment', 'when': None, 'where': None}
- Scenario 8 (hard): "Call Dr. Rao on Friday."  expected stable/unrelated, got stable/ask; intent=call_person, slots={'who': 'Dr. Rao', 'what': None, 'when': 'Friday', 'where': None}
- Scenario 13 search "when do I call Ravi?": current found=False, outdated shown=True

## Similarity values behind the thresholds

| Scenario | Note | Expected | Got | Similarity |
|---|---|---|---|---|
| 1 | The call with Ravi is on Friday at 5. | new | new | 0.543 |
| 2 | Actually, move the meeting with Anita to Thursday. | update | ask | 0.675 |
| 3 | Call Karthik on Wednesday morning. | contradict | contradict | 0.973 |
| 4 | Call Meena on Saturday at 6 in the evening. | fused | fused | 0.946 |
| 5 | Buy a new pen for exams. | new | new | 0.217 |
| 7 | Call Priya on Wednesday. | new | new | 0.768 |
| 8 | Call Dr. Rao on Friday. | unrelated | ask | 0.546 |
| 9 | Maybe I should learn guitar this winter. | new | new | 0.024 |
| 10 | Change the electrician call to 6pm. | update | update | 0.787 |
| 11 | Actually, submit the physics assignment on Tuesday instead. | update | update | 0.841 |
| 12 | Call Divya on Monday at 3. | fused | fused | 1.0 |
