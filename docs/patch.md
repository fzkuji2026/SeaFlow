# Strategy optimization workflow

The objective is percentile-adjusted payoff on the current field. A green test
suite proves legality and regression coverage; it does not prove rating gain.

## 1. Establish the frozen-batch baseline

Use the completed approximately 200-game deployment-local ID ledger as the
retention unit. Shared agent counters and SDK `game_over` callbacks are not
batch membership because another process can change the former and the latter
misses games completed by the opponent's final move.
Latest 20/50 views are operational diagnostics only and never authorize a
mid-batch code change. Read both positive and negative games.
`logs/mistakes/` alone is selection-biased. The report replays each game through
current code, so stale live actions are not mistaken for current behavior.

## 2. Classify the proposed change

Changes need one of these evidence types:

- **Dominance:** positive final-round payoff versus zero, or non-negative trade
  versus a negative one.
- **Continuation value:** compare discounted payoff now with a probability-
  weighted next action, using observed rejection/concession history.
- **Scenario reference:** compare with `glee_agent/reference.json`, generated
  from the public GLEE corpus by `scripts/build_reference.py`.
- **Repeated current-field behavior:** the same scenario is still negative in
  the 20/50 window and current replay takes the problematic action.

A change may span several functions when the scenario contract spans classify,
calculation, model guidance, and execution. Keep unrelated families separable
so they can be reverted independently.

## 3. Give the model computed context, not authority over numbers

`glee_agent/diagnose.py` computes payoff, surplus, posterior EV, public medians,
discounted continuation, repetition, and signal response rates. `brief.py`
explains how to compare them. `judge.py` exposes only legal alternatives and
calls the model only for bounded bargaining and negotiation ambiguity.
Persuasion, offer generation, and dominance cases remain deterministic.

## 4. Verification

```bash
PYTHONPATH=vendor:. pytest -q
PYTHONPATH=vendor:. python3 scripts/check_patch.py
git diff --check
```

Add a named replay test for every non-trivial branch. Test both the new action
and the nearest case that must remain unchanged.

## 5. Deployment and retention

Live configuration uses explicit recorded concurrency, bounded model judging,
deterministic messages, and frozen policy mutation. Every start appends a
snapshot to `logs/deployments.jsonl`; every newly assigned ID is appended once
under `logs/batch_ids/`. The current deployment value is 16.

Review and modify only after the frozen batch has drained. Keep a dominance
change unless replay contradicts its assumptions. For empirical changes, retain
only exact cells that pass the chronological evidence gate in
`docs/modeling-workflow.md`; do not compensate by changing an unrelated cell.
