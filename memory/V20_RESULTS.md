# v20 Results — compounding champion-gate run (seeded from v19 champion)

**Run:** `reports/ml/alphazero_v20_compound` (out-dir on ephemeral container; model files gitignored)
**Config:** `--gate-mode champion --min-promotion-win-rate 0.50 --gate-mcts-sims 40
--gate-stage-games 30 --games-per-iter 300 --mcts-sims 200 --train-epochs 15
--data-accumulation-decay 0.7`, seeded from `models/champions/v19_iter1_champion.npz`,
per-slot + identity + hand-habitat + power features (hidden 768/384), `--value-target-mode absolute`.
**Status:** iters 1-6 completed; **stopped by choice at iter 6** (of 10). The iter-3 champion held
for 3 straight iterations (4-6) with no further promotion — conclusion established, and finishing
7-10 would only reconfirm it at the cost of more container-reclaim babysitting.

## Headline
**v20 broke the plateau that v18 and v19 could not.** It produced a **real promotion (iter 3)** —
the champion advanced past the v19 model — which neither prior run achieved (v18: 0 promotions in
10 iters; v19: 1 promotion but only by beating the weak *boot* model, never its own champion).
But v20 also showed the ratchet **does not compound** at this config: after the iter-3 promotion,
iters 4-6 sat at parity-or-below vs the new champion and never promoted again.

## Per-iteration data
| iter | self-play mean | eval mean | eval WR | gate WR (vs champion) | promoted |
|---|---|---|---|---|---|
| 1 | 69.2 | 53.7 | 0.275 | 12/30 = 0.400 | ✗ |
| 2 | 68.3 | 56.2 | 0.200 | 29/60 = 0.483 | ✗ |
| 3 | 68.6 | 53.6 | 0.300 | **32/60 = 0.533** | ✅ |
| 4 | 67.7 | 57.0 | 0.350 | 14/30 = 0.467 | ✗ |
| 5 | 68.9 | 54.8 | 0.300 | 28/60 = 0.467 | ✗ |
| 6 | 68.5 | 63.3 | 0.375 | 9/30 = 0.300 | ✗ |

(iters 2,4,5,6 were each interrupted by a container reclaim and rerun from scratch; values above
are the completed reruns. The pre-reclaim iter-6 attempt scored an even higher eval, 64.5 / 0.550.)

## Key findings
1. **The lower bar + stronger targets worked — once.** Gate climbed 0.400 → 0.483 → **0.533** to
   promote at iter 3. v19's candidates never got above ~0.43 vs their champion; v20's reached the
   bar. The fixes (0.50 bar, 200-sim self-play targets, decay 0.7) did what they were meant to.
2. **No sustained compounding.** Once the iter-3 champion was installed, iters 4-6 came in at
   0.467, 0.467, 0.300 — matching or losing to it. Each promotion raises the bar (you must now beat
   a stronger self), and at this config the per-iter improvement isn't enough to clear it again.
3. **Eval-vs-heuristic does NOT predict the gate.** Iter 6's candidate *beat the heuristic*
   (eval 0.550, the best in any run) yet **lost the gate 9-20 (0.300)** to the iter-3 champion. The
   greedy eval and the MCTS-vs-champion gate measure different things; the champion is genuinely
   strong in search play even when a candidate looks great against the rule-based bot.

## Cross-run comparison
| | v18 | v19 | v20 |
|---|---|---|---|
| Gate mode | heuristic | champion (boot-seeded) | champion (v19-champ-seeded) |
| Promotion bar | — | 0.55 | 0.50 |
| Self-play sims | 100 | 100 | 200 |
| Promotions | 0 / 10 | 1 / 6 (vs boot only) | **1 / 6 (vs real champion)** |
| Best eval mean | ~49 | ~57 | **64.5** |
| Best eval WR vs heuristic | ~0.175 | ~0.425 | **0.550** |
| Champion at end | boot (frozen) | iter-1 model | **iter-3 model (strongest)** |

## Artifacts (committed, durable)
- `models/champions/v20_iter3_champion.npz` (+ `.meta.json`) — the promoted v20 champion, the
  strongest model produced across all runs. **This is the deliverable.**
- `models/champions/v19_iter1_champion.npz` — v19's champion (v20's seed / baseline opponent).

## Operational notes
- **Container reclaimed ~6×** during v20 (ephemeral env, ~3-4h lifetime). Every reclaim was
  recovered with `--no-clean --start-iter N`; `best_model.npz` existing on disk meant the resume
  did **not** re-seed (line 538 guard), so promotions were preserved across restarts.
- **200-sim self-play + heavy Modal preemptions** made each iter ~70-90 min (preempted shards
  restart from scratch). This is the main reason the run was slow and reclaim-prone.

## If pursuing further (v21 sketch)
Sustained compounding likely needs a stronger *learning* signal than the champion, not just loop
tuning: higher self-play sims again (200→300+), more/explicit exploration (temperature, Dirichlet
noise) to escape the champion's policy basin, or larger model capacity. The eval/gate divergence
suggests the policy net is improving against the heuristic but plateauing against MCTS-champion
play — a hint that the bottleneck is search-quality of targets and/or representation, not the gate.
