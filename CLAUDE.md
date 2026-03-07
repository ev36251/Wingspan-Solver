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

### ML pipeline

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

- All bird/goal/bonus data lives in `wingspan-20260128.xlsx`. No Americas expansion, no hummingbirds, no promo sets.
- `backend/data/registries.py` must be initialized with `load_all(EXCEL_FILE)` before any game engine code runs. Tests do this in `setup_module()` or `conftest.py`.
- `GameState` is a mutable dataclass; `simulation.py` uses `copy.deepcopy()` for rollouts.
- Combined training JSONLs are ≈2.75 GB and are auto-deleted after training. Disk abort at <5 GB free.
- `reports/ml/alphazero_v12/` is the current training run. Resume command is in `memory/MEMORY.md`.
