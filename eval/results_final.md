# Evaluation results

## Summary

| Metric | Result |
|---|---|
| Notes evaluated | 20 in 10 scenarios |
| Stabilization correct (all) | 20/20 (100%) |
| Stabilization correct (without hard cases) | 18/18 (100%) |
| Integration decision correct (all) | 16/18 (89%) |
| Integration decision correct (without hard cases) | 15/16 (94%) |

| Comparison | Our system | Plain-note baseline |
|---|---|---|
| Search returns the current value first | 6/7 (86%) | 3/7 (43%) |
| Search shows an outdated value (queries with a changed fact) | 0/3 (0%) | 3/3 (100%) |
| Stored entries (ideal: 12) | 12 | 20 |
| Questions waiting for the user | 2 | n/a |
| Average seconds per note | 5.1 | n/a |

## Per scenario

| # | Scenario | Hard | Stability ok | Decision ok | Search (ours / baseline) | Entries (ours / baseline / ideal) |
|---|---|---|---|---|---|---|
| 101 | Pending call completed by a later note |  | 2/2 | 1/1 | ok / ok | 1 / 2 / 1 |
| 102 | Meeting postponed (change wording: postponed) |  | 2/2 | 2/2 | ok / miss, stale | 1 / 2 / 1 |
| 103 | Different deadline without change wording goes to review |  | 2/2 | 2/2 | n/a | 1 / 2 / 1 |
| 104 | More detailed purchase is a refinement |  | 2/2 | 1/2 | miss / miss | 1 / 2 / 1 |
| 105 | Same day, different people (hard) | yes | 2/2 | 1/2 | n/a | 2 / 2 / 2 |
| 106 | Two unrelated ideas stay separate |  | 2/2 | 2/2 | n/a | 2 / 2 / 2 |
| 107 | Vague task completed by a later note |  | 2/2 | 1/1 | ok / miss | 1 / 2 / 1 |
| 108 | Time changed with 'instead' |  | 2/2 | 2/2 | ok / ok, stale | 1 / 2 / 1 |
| 109 | Same reminder worded differently |  | 2/2 | 2/2 | ok / ok | 1 / 2 / 1 |
| 110 | Dinner moved with 'change' |  | 2/2 | 2/2 | ok / miss, stale | 1 / 2 / 1 |

## Failures (what to look at)

- Scenario 104: "Buy a ruled notebook from the stationery shop."  expected stable/fused, got stable/ask; intent=buy_item, slots={'who': None, 'what': 'ruled notebook', 'when': None, 'where': 'stationery shop'}
- Scenario 104 search "what notebook do I buy?": current found=False, outdated shown=False
- Scenario 105 (hard): "Call the plumber on Friday."  expected stable/unrelated, got stable/new; intent=call_person, slots={'who': 'plumber', 'what': None, 'when': 'Friday', 'where': None}

## Similarity values behind the thresholds

| Scenario | Note | Expected | Got | Similarity |
|---|---|---|---|---|
| 101 | The call with Lakshmi is on Thursday at 10. | new | new | 0.599 |
| 102 | The meeting with Arjun is postponed to Wednesday. | update | update | 0.664 |
| 103 | Submit the lab record on Friday. | contradict | contradict | 0.963 |
| 104 | Buy a ruled notebook from the stationery shop. | fused | ask | 0.697 |
| 105 | Call the plumber on Friday. | unrelated | new | 0.334 |
| 107 | I need to finish the maths worksheet. | new | new | 0.524 |
| 108 | Call the dentist at 4 instead. | update | update | 0.8 |
| 109 | Don't forget to call Hari on Saturday. | fused | fused | 0.959 |
| 110 | Change the dinner with Mom to Saturday. | update | update | 0.787 |
