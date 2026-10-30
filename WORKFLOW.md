# GLEE method

Live play combines the Python tree, observed opponent behavior, and public GLEE
configuration medians. Numbers stay in code. The model (`GLEE_LLM_JUDGE=1`)
only handles ambiguous scenarios and picks among precomputed legal alternatives;
it never invents a price, split, or EV. Timeout or a bad pick falls back to the
heuristic. Message generation remains off so decision calls receive the latency
budget.

```
game JSON
  → rules.parse
  → family classify
  → canonical scenario_id / state_id
  → one cell / one procedure          # deterministic fallback
  → exact-cell evidence lookup
  → candidate accept / continue / payoff estimates
  → evidence-gated action selector
  → diagnose snapshot (path + calc)
  → legal alternatives (code)
  → scenario ambiguity gate
  → Grok pick among those names       # <=1 call / 8s, 4s timeout
  → deterministic message
  → sanitize
  → POST
```

## Optimization loop

1. Run one frozen batch of about 200 locally assigned games. Do not modify
   strategy while the batch has active games. The deployment-local ID ledger,
   not shared agent totals or SDK `game_over` callbacks, controls the limit.
2. Persist every replay and decision trace, then compare the batch with the
   preceding frozen windows. Negative rating is triaged before it is treated as
   an actionable error.
3. `glee_agent/reference.json` supplies same-scenario public medians generated
   from 77,467 official games by `scripts/build_reference.py`.
4. Build exact scenario/state labels with `glee_agent.segments`; fit only cells
   that pass chronological holdout and causal-direction checks. Parent fallback
   order is defined in `docs/modeling-workflow.md`.
5. Prefer dominance, discounted continuation value, opponent response rates,
   and posterior EV over fixed global thresholds.
6. Run the complete tests, replay validation, and `scripts/check_patch.py`;
   commit separable changes.
7. Each run records the workflow revision, policy and assigned game IDs so a
   batch can be compared with the preceding ones.

## Official scoring invariants

- Optimize only our own payoff percentile within the exact configuration and
  role. Opponent payoff, fairness, and total efficiency do not affect ranking.
- A per-game rating tick is opponent-strength adjusted. Its sign is evidence
  about that game, not proof that one action is generally right; require the
  same path on named games plus 20/50-game deployment windows.
- Generalize across the 960 configurations with roles, ratios, information,
  horizons, discounting, and observed behavior. Do not transfer an absolute
  price or pot size between configurations.
- Keep every action legal and bounded. A timeout or five invalid moves is scored
  at the 5th percentile; a legal `WalkAway` is a game action and must not be
  confused with abandonment.

Implementation and verification details: `docs/patch.md`.
Segmentation, evidence gates, indicator ownership, and trace fields:
`docs/modeling-workflow.md`.

Where the tree lives:

| family | classify | dispatch |
| --- | --- | --- |
| bargaining | `workflows/bargaining/classify.py` | `_plan.compose` → `_plan._run` |
| negotiation | `workflows/negotiation/classify.py` | `routes/<regime>.handle` → `_tactics` (`_hidden_dead` / `_hidden_strong_buyer` / `_hidden_strong_seller` / `_hidden_clock`) |
| persuasion | `workflows/persuasion/classify.py` | `_beliefs.compose_buy` / `compose_sell` → `execute_buy` |

Numbers and paid lessons live in those functions and in `tests/test_strategy.py` (`test_<family>_<game8>_*`). Do not copy them into skills.

Inspect a finished game with the calc tool, not mental arithmetic:

```bash
PYTHONPATH=vendor:. python3 scripts/diagnose.py --file REPLAY.json --round N --action offer
```

`calc` is share / ZOPA / profit / EV / 1.10×. Missing field → add it in `glee_agent/diagnose.py`.
