# v19 Results — champion-gate AlphaZero run

**Run:** `reports/ml/alphazero_v19_champion` (out-dir on ephemeral container; model files gitignored)
**Config:** `--gate-mode champion --min-promotion-win-rate 0.55 --gate-mcts-sims 60
--gate-stage-games 30 --games-per-iter 200 --mcts-sims 100 --train-epochs 10`,
seeded from `reports/ml/_boot/boot_model.npz`, per-slot + identity + hand-habitat + power features
(hidden 768/384), `--value-target-mode absolute`.
**Status:** iters 1-6 completed; capped at 8 but **stopped after iter 6 by choice** — iters 7-8
would rerun the identical config vs the identical champion and only reconfirm the plateau
(iter 7 was already losing 3-7 in its gate when the container was reclaimed).

## Headline
**The v19 hypothesis is validated.** Switching the promotion gate from `heuristic` to `champion`
fixed v18's structural failure: v18 promoted **0 times in 10 iters** (champion frozen at the boot
model, eval stuck in the 40s), whereas v19 **promoted on iter 1** and produced a clearly stronger
model. But the run then converged to a **single-promotion regime**: later candidates get within
~0.40 of the iter-1 champion but never clear the 0.55 bar.

## Per-iteration data
| iter | self-play mean | eval mean (vs heuristic) | eval WR | gate WR (vs champion) | promoted |
|---|---|---|---|---|---|
| 1 | 44.5 (boot) | 43.5 | 0.150 | **56/60 = 0.933** | ✅ |
| 2 | 67.4 | 50.7 | 0.225 | 9/30 = 0.300 | ✗ |
| 3 | 67.3 | 47.0 | 0.075 | 13/30 = 0.433 | ✗ |
| 4 | 65.1 | 52.4 | 0.225 | 9/30 = 0.300 | ✗ |
| 5 | 65.3 | 56.0–57.0 | 0.225–0.425 | 12/30 = 0.400 | ✗ |
| 6 | 65.7 | 53.1 | 0.225 | 12/30 = 0.400 | ✗ |
| 7 | 65.9 | 48.4 | 0.225 | interrupted (3-7 at 10 games) | — |

After the iter-1 promotion, self-play quality jumped **44.5 → ~66** (and held there, since the
champion didn't advance again). Eval-vs-heuristic peaked at **57.0 / WR 0.425** — vs v18's best of
~49 / 0.175.

## v18 vs v19
| | v18 (heuristic gate) | v19 (champion gate) |
|---|---|---|
| Promotions | 0 / 10 | 1 / 6 |
| Champion at end | boot model (frozen) | iter-1 model (real) |
| Self-play mean | ~44–50 (flat) | ~66 (post-promotion) |
| Eval mean (best) | ~49 | ~57 |
| Eval WR (best) | ~0.175 | ~0.425 |

## Why the ratchet stalled after iter 1 (analysis, verified against code + log)
Note: the replay buffer **was on** (`decay=0.5`, growing 31200→72800 samples across iters — see log
`Combined dataset: ... (new=10400 + replay)`), so this is **not** a data-quantity problem.
1. **Candidate is anchored to the champion.** Each candidate **warm-starts from `best_model.npz`**
   (`applied=32`) **and** trains on a replay buffer dominated by that same champion's self-play
   (decay 0.5 → recent iters, all from the iter-1 champion, dominate). It is effectively fine-tuning
   the champion on the champion's own games → reproduces it and lands slightly *below* from training
   noise → gate ~0.40.
2. **The 0.55 bar is demanding for marginal gains** given NN-vs-NN gate variance at 30-60 games.
3. **Weak improvement signal** — at `mcts_sims=100`, the search target may be only marginally better
   than the champion net, leaving little to climb toward. (v20 addresses this with more sims + a
   0.50 bar + looser anchor.)

## Operational notes
- **Gate is the cost bottleneck:** ~50–66 min of each ~85–90 min iter is the NN-vs-NN MCTS gate
  (`gate-mcts-sims=60`). Self-play (Modal) ~23–32 min, train ~1 min, eval ~2 min.
- **Container reclaimed 3×** (ephemeral env, ~4–8h lifetime). Each time, resumed cleanly with
  `--no-clean --start-iter N` (datasets + `best_model.npz` persist on disk). No learning lost.
- **Modal preemptions** were frequent but self-healing (each shard auto-restarts with same input).
- ⚠️ **The promoted iter-1 champion (`best_model.npz`, 5.5 MB) is gitignored and lives only on the
  ephemeral container** — it will be lost on the next reclaim unless force-committed or copied off.
