# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Tests
```bash
pytest                                        # all 492 tests
pytest backend/tests/test_engine.py -v       # single file
pytest -k "test_powers"                      # filter by name
pytest backend/tests/test_alphazero.py -x -q # fast fail, quiet
```
All tests live in `backend/tests/`. Config is in `pyproject.toml` (`testpaths = ["backend/tests"]`, `pythonpath = ["."]`).

### Backend server
```bash
uvicorn backend.main:app --reload            # http://localhost:8000
```
`load_all(EXCEL_FILE)` runs at startup, loading all bird/bonus/goal data from `wingspan-20260128.xlsx` into singleton registries. The server must be running for the frontend to work.

### Frontend
```bash
cd frontend && npm run dev                   # http://localhost:5173 (proxies /api → :8000)
cd frontend && npm run build && npm run preview
```

### ML pipeline — AlphaZero training loop
```bash
# Smoke test (local, ~5 min)
python -m backend.ml.auto_improve_alphazero \
  --out-dir reports/ml/alphazero_smoke \
  --iterations 1 --games-per-iter 5 --mcts-sims 20 \
  --train-epochs 2 --eval-games 5 --promotion-games 10 \
  --dataset-workers 1

# Resume a run from iter N
python -m backend.ml.auto_improve_alphazero --out-dir reports/ml/alphazero_vN \
  --no-clean --start-iter N [... same flags as original launch ...]
```

### Solo single-seed optimization (current direction)
Deterministic single-player score maximization. For one fixed seed the *deal*
is fixed (pre-shuffled deck stack, starting hand, bonus cards, round goals,
seeded birdfeeder reroll stream); the opening *draft* keep-decision and every
in-game move are searched to maximize the final score. Because whole games are
scored to the end, the search discovers engines (food gathering, round-end teal
caches like Sri Lanka Blue-Magpie, card tucking) that a one-step greedy
evaluator misses. No opponent and no learned model are required.
```bash
# One fixed seed (draft + play searched; prints the best trajectory)
python -m backend.ml.solo_seed_optimizer single --seed 42 --games 150 --show-trajectory

# Many seeds fanned across Modal containers -> best-line dataset + analytics
python -m backend.ml.solo_seed_optimizer multi --seeds 0-999 --games-per-seed 150 \
  --use-modal --seeds-per-shard 1 --out reports/ml/solo_seed/best_lines.jsonl
```
`modal_solo.py` is the Modal dispatcher (one container per seed shard).
See `memory/SOLO_SEED_FINDINGS.md` for the current bird/bonus tier list.

### Other ML utilities
```bash
python -m backend.ml.alphazero_self_play      # standalone self-play data gen
python -m backend.ml.train_factorized_bc      # standalone training
python -m backend.ml.evaluate_factorized_bc   # standalone eval
```

---

## Architecture

### Data flow (startup → game → solve)

```
wingspan-20260128.xlsx
    └─ backend/data/loader.py        openpyxl → raw rows
    └─ backend/data/registries.py    BirdRegistry / BonusRegistry / GoalRegistry
                                     (loaded once, shared as module singletons)
    └─ backend/models/               Pure dataclasses — no logic
         bird.py, player.py, game_state.py, board.py, enums.py, ...
    └─ backend/engine/               Mutates game state
         rules.py        Legality checks (can_play_bird, food cost, etc.)
         actions.py      execute_action() + activate_row() (brown power loop)
         scoring.py      calculate_score() → ScoreBreakdown
         timed_powers.py Pink power resolution at end of round
    └─ backend/powers/               Bird power resolution
         registry.py     get_power(bird) → PowerEffect instance
         base.py         PowerEffect ABC, PowerContext, PowerResult
         templates/      One file per power category (gain_food, lay_eggs,
                         draw_cards, predator, special, unique, ...)
    └─ backend/solver/               Decision-making
         move_generator.py   generate_all_moves(game, player) → list[Move]
         heuristics.py       Score moves heuristically
         simulation.py       Fast rollout + execute_move_on_sim()
         monte_carlo.py      MC evaluation
    └─ backend/api/                  FastAPI routers
         routes_game.py, routes_solver.py, routes_data.py,
         routes_setup.py, routes_ml.py
         schemas.py, serializers.py  Pydantic models + JSON serialization
    └─ frontend/src/lib/api/
         client.ts      Typed fetch wrappers
         types.ts       TypeScript mirrors of backend schemas
```

### Power system

Every bird has a power resolved through `backend/powers/`. `get_power(bird)` returns a `PowerEffect` instance. Powers are executed via `power.execute(ctx)` returning a `PowerResult`. `activate_row()` in `actions.py` is the brown-power loop that iterates the habitat and fires each bird's power.

Special cases to know:
- `RepeatPower` (Gray Catbird, Northern Mockingbird): re-executes the best same-habitat brown power
- `CopyNeighborBrownPower`: copies adjacent bird's brown power — mutual recursion guard exists between these two
- `MoveBird` (8 birds): moves rightmost bird to best destination habitat
- 4 "play on top" birds (Common Buzzard, Red Kite, etc.): overlap existing slots

`BirdSlot` has flags: `counts_double`, `is_sideways`, `is_sideways_blocked`.

### Game rules — important / non-obvious

This is **Wingspan** with the five sets currently released and present in
`wingspan-20260128.xlsx`: **Core, Oceania, Asia, European, and Promo-UK** (471
birds total: core 180, oceania 95, asia 90, european 81, promo_uk 25). No other
promo sets (promoAsia / promoCA / promoEurope / promoNZ / promoUS) — those release
later and are not in the data or the `GameSet` enum yet. Note that base/Core is
North-American themed, so birds like American Robin, Song Sparrow, and the
hummingbirds (Anna's, Black-Chinned) are legitimately in-scope Core cards.

- **Food cost & nectar** (`rules.py` `can_pay_food_cost` / `find_food_payment_options`):
  every bird costs food, deducted on play. Nectar substitutes **1-for-1** for any
  food. The Oceania **2-for-1** rule (any 2 food tokens pay for 1 required food)
  is a *real* rule but applies **only to bird-play costs** — never to bird-power
  costs (caching, predator discards) or the birdfeeder reset (exactly 1 food),
  which spend exact food via `food_supply.has/spend` and `_pay_bonus_cost`.
- **Power colors** (`PowerColor`): brown = "when activated", white/none = "when
  played", pink = "once between turns", **teal = round end**, yellow = game end.
  Teal/yellow fire in `timed_powers.py`. **Cached food scores 1 VP per token**
  (`score_cached_food`), so round-end cache engines (e.g. Sri Lanka Blue-Magpie)
  compound with board size.
- **Setup / draft** (`solver/setup_advisor.py` `analyze_setup`): dealt 5 birds +
  2 bonus; keep a **total of 5** (birds + food), at most **1 of each** of the 5
  non-nectar food types, keep **1 of 2** bonus cards, plus **1 free nectar**
  (Oceania). Good lines often keep *fewer* birds and more food.
- **Solo scoring** (`num_players < 2`): round goals use **fixed-target** scoring
  (`_compute_round_goal_scores_solo`) — full target → 1st-place pts, half → 2nd,
  else 0, capped at the real per-round values (max 22). Nectar has its own solo
  branch (`_score_nectar_solo`). Kept modest so a specialist VP engine still wins
  while forfeiting goals/nectar.

### ML pipeline

> **Status:** the 2-player AlphaZero self-play loop (v4–v20) produced weak
> champions — the greedy NN loses to the rule-based heuristic ~70% of the time
> and plays a degenerate egg-spam strategy. It is **superseded by the solo
> single-seed optimization above**, which is the current direction (stronger
> strategy + a clean best-line dataset). The loop below is kept for reference and
> as the eventual competitive fine-tuning stage. Old `reports/ml/alphazero_v*`
> run artifacts were deleted.

The AlphaZero loop in `auto_improve_alphazero.py` runs four steps per iteration:

1. **Self-play data gen** — `generate_self_play_dataset()` in `alphazero_self_play.py`. Both players use MCTS (`mcts.py`, UCB-PUCT). Value target = player's absolute final score (normalized by `score_scale=120`). State vector produced by `StateEncoder` (≈2827-dim with all features enabled).

2. **Training** — `train_bc()` in `train_factorized_bc.py`. PyTorch, MPS/CUDA. Factorized multi-head policy (`action_type`, `play_habitat`, `gain_food_primary`, `draw_mode`, `lay_eggs_bin`, `play_cost_bin`, `play_power_color`) + auxiliary value head. Saves `.npz` weights compatible with `FactorizedPolicyModel` (numpy inference, no PyTorch at serve time).

3. **Eval** — `evaluate_factorized_vs_heuristic()` in `evaluate_factorized_bc.py`. Greedy NN vs rule-based heuristic. Note: greedy NN scores ~25–35 pts lower than MCTS NN.

4. **Promotion gate** — candidate vs champion (heuristic or NN vs NN). Promotes on ≥50% win rate. On promotion, `best_model.npz` is updated.

**Cloud dispatch:** `--use-modal` shards self-play across 32 Modal.com containers (see `modal_selfplay.py`). Local `backend/` is mounted fresh at dispatch time so code fixes apply immediately without redeployment.

**State encoder features** (flags required at training and must match inference model):
- `--use-per-slot-encoding` — per-slot bird encoding (adds ≈750 dims, triggers hidden=768/384 auto-scale)
- `--enable-identity-features --identity-hash-dim N` — bird identity hashing
- `--use-hand-habitat-features` — hand×board synergy features
- `--use-tray-per-slot-encoding` — full tray card features
- `--use-opponent-board-encoding` — full opponent board features
- `--use-power-features` — power-effect encoding for own board+hand

**Model files:** `.npz` weights + embedded `metadata_json` array. Load with `FactorizedPolicyModel(path)` or `StateEncoder.resolve_for_model(model.meta)` to reconstruct the correct encoder from saved metadata.

### Key invariants

- All bird/goal/bonus data lives in `wingspan-20260128.xlsx`: the 5 released sets
  (Core, Oceania, Asia, European, Promo-UK = 471 birds). `create_training_game`
  deals from the whole registry, so scope is governed entirely by what's in the
  xlsx + the `GameSet` enum — both currently hold exactly these 5 sets. Future
  promo sets (promoAsia/CA/Europe/NZ/US) are NOT yet present and must not be added
  to training games until released.
- `backend/data/registries.py` must be initialized with `load_all(EXCEL_FILE)` before any game engine code runs. Tests do this in `setup_module()` or `conftest.py`.
- `GameState` is a mutable dataclass; `simulation.py` uses `copy.deepcopy()` for rollouts.
- Combined training JSONLs are ≈2.75 GB and are auto-deleted after training. Disk abort at <5 GB free.
- Current direction is **solo single-seed optimization** (`backend/ml/solo_seed_optimizer.py`); dataset in `reports/ml/solo_seed/best_lines_*.jsonl`, findings in `memory/SOLO_SEED_FINDINGS.md`. The old AlphaZero `alphazero_v*` runs were deleted.
- `reports/ml/**/*.jsonl` and `*.npz` are gitignored; the solo dataset is force-added so it survives container recycling.
