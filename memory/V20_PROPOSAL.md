# v20 Proposal — make the champion ratchet compound

## Goal
v19 proved champion-gate works (1 promotion, strong model) but stalled in a **single-promotion
regime**: candidates cluster at ~0.40 gate WR vs the iter-1 champion and never clear 0.55. v20's
goal is **repeated promotions** so the champion compounds across iterations.

## Diagnosis recap (see V19_RESULTS.md)
Candidates warm-start from the champion and train only on that champion's ~10.4k self-play samples
for 10 epochs — they reproduce the champion and land slightly below it. The 0.55 bar plus gate
variance then blocks marginal gains. Root issues: **weak per-iter training signal**, **low data
diversity**, and a **demanding bar**.

## Proposed changes (in priority order)
1. **Lower the promotion bar to 0.50** (`--min-promotion-win-rate 0.50`). Standard AlphaZero. Lets
   genuine-but-marginal improvements promote, which is the whole point of a ratchet. Biggest
   expected impact, zero extra compute.
2. **Multi-iter replay buffer for training.** Train each candidate on the last **K=3** iters of
   self-play (not just the latest), for more and more-diverse data. (Check whether the orchestrator
   already supports a replay-window flag; if not, this is a small code change in
   `train_factorized_bc.py` / the auto_improve loop.)
3. **Stronger self-play targets:** `--mcts-sims 100 → 150` (or 200). Better policy/value targets at
   the cost of ~1.5–2× self-play time (still Modal-sharded, ~30–60 min/iter).
4. **More training signal:** `--train-epochs 10 → 15` and consider `--games-per-iter 200 → 300`.
5. **(Optional) exploration:** raise self-play temperature / Dirichlet noise so candidates explore
   beyond the champion's policy. Only if 1–4 still stall.

## Operational changes (make long runs survivable)
The bottleneck and the thing that makes container reclaims expensive is the **gate**. With the bar
at 0.50 and a staged gate, cut gate cost so each iter fits comfortably inside a container window:
- `--gate-mcts-sims 60 → 40` and keep `--gate-stage-games 30` (early-stop on clear decisions).
- Target **≤60 min/iter** so 8–10 iters fit in fewer reclaim cycles.
- Expect to resume ~2–4× via `--no-clean --start-iter N`; that workflow is proven.

## Suggested launch (subject to replay-buffer flag check)
```bash
nohup env PYTHONPATH=. python -m backend.ml.auto_improve_alphazero \
  --out-dir reports/ml/alphazero_v20_compound \
  --train-init-model-path reports/ml/_boot/boot_model.npz \
  --value-target-mode absolute --use-modal \
  --gate-mode champion \
  --min-promotion-win-rate 0.50 \
  --gate-mcts-sims 40 --gate-stage-games 30 \
  --iterations 10 --games-per-iter 300 --mcts-sims 150 \
  --train-epochs 15 --eval-games 40 --promotion-games 40 --dataset-workers 32 \
  --pilot-early-stop-iter 999 \
  --use-per-slot-encoding --enable-identity-features \
  --use-hand-habitat-features --use-power-features \
  > reports/ml/alphazero_v20_compound_run.log 2>&1 &
```

## Success criterion
**≥3 promotions** across the run, with self-play mean and eval-vs-heuristic both trending up past
v19's plateau (self-play >66, eval >57). If still single-promotion after lowering the bar +
replay buffer, the limiter is the model/representation, not the gate — revisit encoder/capacity.

## Open decision before launch
- Confirm whether a **replay-window flag** already exists; if not, decide whether to make the small
  code change (recommended) or launch without it first.
- Decide whether to **preserve v19's iter-1 champion** (force-commit the 5.5 MB `best_model.npz` or
  copy off-container) before it's reclaimed — useful as a v20 baseline opponent.
