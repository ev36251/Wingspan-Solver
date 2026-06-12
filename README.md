# Wingspan Solver

An AlphaZero-style reinforcement learning engine for the [Wingspan](https://stonemaiergames.com/games/wingspan/) board game. The system trains entirely through self-play: two AI agents play thousands of games against each other, and a neural network learns from the resulting data using Monte Carlo Tree Search (MCTS) for both data generation and evaluation. The policy and value network is implemented in PyTorch, trained with behavioral cloning on self-play data, and served via a FastAPI backend with a React frontend. Self-play is parallelized across 32 Modal.com cloud containers for fast iteration.

---

## Architecture

```
wingspan-20260128.xlsx
    └─ backend/data/loader.py + registries.py   Bird/bonus/goal data, loaded once at startup
    └─ backend/models/                           Pure dataclasses: GameState, Player, Board, enums
    └─ backend/engine/                           Game rules
         rules.py      Legality checks (can_play_bird, food costs, egg limits)
         actions.py    execute_action() + activate_row() (brown power loop)
         scoring.py    calculate_score() → ScoreBreakdown
         timed_powers.py  Pink powers (triggered by opponent actions, end-of-round)
    └─ backend/powers/                           446 bird powers
         registry.py     Bird → PowerEffect lookup (explicit_mappings.json)
         templates/      Implementations: gain_food, lay_eggs, draw_cards,
                         tuck_cards, predator, cache_food, play_bird, unique, special
    └─ backend/solver/                           Decision-making
         move_generator.py   All legal moves for a given state
         simulation.py       Fast rollout engine (deepcopy-based)
         monte_carlo.py      MCTS with UCB-PUCT
    └─ backend/ml/                               Training pipeline
         alphazero_self_play.py   Self-play data generation (MCTS both sides)
         train_factorized_bc.py   PyTorch training: factorized policy + value heads
         evaluate_factorized_bc.py  NN vs rule-based heuristic evaluation
         auto_improve_alphazero.py  Outer loop: self-play → train → eval → gate
         modal_selfplay.py          Modal.com cloud dispatch (32 parallel workers)
    └─ backend/api/                              FastAPI routers
    └─ frontend/src/                             React + Vite frontend
```

### Policy network: factorized multi-head design

Rather than outputting a single probability over a flat action space (which would require one logit per legal move and varies hugely in size), the policy is split into **7 factorized heads**:

| Head | Classes | Meaning |
|------|---------|---------|
| `action_type` | 4 | Play bird, gain food, lay eggs, draw cards |
| `play_habitat` | 4 | Forest / Grassland / Wetland / None |
| `play_cost_bin` | 7 | Food cost 0–6+ |
| `play_power_color` | 7 | Brown / White / Pink / Teal / Yellow / None |
| `gain_food_primary` | 6 | Invertebrate / Seed / Fish / Fruit / Rodent / Nectar |
| `draw_mode` | 4 | Deck-only / tray-only / mixed / none |
| `lay_eggs_bin` | 11 | Eggs placed 0–10+ |

Only the heads relevant to the chosen action type are trained for any given move. At inference, head outputs are combined to score and rank candidate moves from the legal move generator.

### Value heads

The network has three value outputs:

- **Score value head**: Predicts the acting player's final absolute score (normalized by ÷80). Used as the leaf evaluation in MCTS instead of random rollouts.
- **Win value head**: Binary sigmoid — did this player win? Used as a secondary signal.
- **Move value head**: Trained on pairs of (good move, bad move) via a ranking loss. Conditionally enabled at inference only when validation pair accuracy ≥ 0.52 and rank margin ≥ 0.01; blended into policy scores at α=0.35 when reliable.

### State encoder (~2827 dimensions, full feature set)

- Per-slot bird encoding for own board (bird identity, habitat, food cost, egg count, power color)
- Identity hashing for hand cards and tray
- Hand × board synergy features
- Opponent board encoding
- Power-effect features for board and hand

### MCTS

Standard UCB-PUCT with the neural network as both prior (policy heads) and leaf evaluator (score value head). Used during self-play data generation and optionally at eval time. Greedy NN (no MCTS) scores ~25–35 pts lower than MCTS NN, which shows the tree search is doing meaningful work beyond what the network alone can express.

### Self-play loop

```
for each iteration:
    1. Self-play      MCTS vs MCTS, both using current best model
                      Modal.com: 32 parallel containers × N games
    2. Train          Behavioral cloning on accumulated self-play data
                      Factorized cross-entropy (policy) + MSE (value)
    3. Eval           Greedy NN vs rule-based heuristic, 40 games
    4. Promotion gate Candidate vs champion, promote if win rate ≥ 50%
                      best_model.npz updated on promotion
```

---

## What it learned / how it improved

The system has run through 17+ training versions, with the current run (`v17_3_long20_improved_fullfeat`) completing 20 iterations. A few genuine learning dynamics observed:

**Early iterations (iter 1–5):** The network starts near-random. Greedy NN scores 33–44 pts. It quickly learns basic action-type selection — that playing high-point birds and laying eggs beats randomly drawing cards every turn.

**Mid-run (iter 6–15):** Scores climb to 55–62 pts. The network begins to learn habitat synergies — that a forest full of brown-power birds compounds on `GAIN_FOOD` actions in ways a flat heuristic misses. The move value head becomes trainable at this stage (pair accuracy crosses 0.52).

**Late iterations (16–20):** Greedy NN stabilizes around 60 pts vs the heuristic's 69 pts. Gate win rate reaches 62.5% (threshold 50%), so the model is promoted. The remaining gap is explained in the Known Limitations section.

**Key architectural inflection:** Early runs used pure AlphaZero (both players use MCTS during self-play, policy learned via policy gradient). This was slow and noisy. Switching to **behavioral cloning on MCTS-generated move choices** (the AlphaZero "distillation" approach) stabilized training significantly and allowed the factorized head design to be introduced cleanly.

---

## Known limitations and where this is going

**Round 3–4 suboptimality.** The model plays well in rounds 1–2 but loses points in late rounds. An encoder audit showed round/goal/bonus features are already well represented (rounds remaining, per-goal progress and rank, bonus next-tier distance), so the gap is attributed to the value target rather than missing inputs: an absolute-score target cannot express urgency relative to the opponent. The differential value target (my_score − opponent_score) is the fix being trained; eval now also logs per-component score breakdowns to verify where points are recovered.

**Greedy NN underperforms vs MCTS NN by ~30 pts.** This means the policy heads alone aren't fully capturing the long-horizon value of moves — the network relies on tree search to look ahead. Improving the value head accuracy (so MCTS evaluations are sharper) and increasing self-play MCTS simulations per move are the main levers here.

**Move value head instability.** The ranking loss that trains the move value head is sensitive to how "negative" examples are sampled from self-play. Poor negative sampling leads to pair accuracy dropping below the 0.52 gate, and the head gets disabled. Better negative mining (e.g., using moves the MCTS explicitly down-ranked vs random moves) would help.

**1v1 focus.** The engine and data pipeline support 2, 3, and 4 player games, but all recent training runs use 2-player games. The policy learned is therefore calibrated for 1v1 — opponent modeling in multiplayer (e.g., blocking bonus cards, adapting to 3-way score gaps) is untrained.

**Power edge cases.** All 446 birds have implemented power logic, but ~15 "unique" birds (RepeatPower, CopyNeighborBrownPower, MoveBird, play-on-top birds) involve complex interactions. These are hand-coded and tested, but rare interaction chains (e.g., Repeat → Copy → another Repeat) have shallow test coverage.

**Next steps:**
- Round-awareness features in state encoder
- Better negative sampling for move value head
- Expand to 3–4 player self-play
- Deeper MCTS at eval time (currently limited by inference speed; exporting to ONNX would help)

---

## How to run

### Prerequisites

```bash
pip install -e ".[dev]"          # Python deps (pytest, uvicorn, torch, etc.)
cd frontend && npm install        # Frontend deps
```

All bird data loads from `wingspan-20260128.xlsx` at startup. This file must be present.

### Backend (game engine + API)

```bash
uvicorn backend.main:app --reload
# http://localhost:8000
```

### Frontend

```bash
cd frontend && npm run dev
# http://localhost:5173  (proxies /api → :8000)
```

### Tests

```bash
pytest                                          # all 492 tests
pytest backend/tests/test_engine.py -v         # single file
pytest -k "test_powers"                        # filter by name
pytest backend/tests/test_alphazero.py -x -q   # fast fail, quiet
```

### ML training (smoke test, ~5 min)

```bash
python -m backend.ml.auto_improve_alphazero \
  --out-dir reports/ml/alphazero_smoke \
  --iterations 1 --games-per-iter 5 --mcts-sims 20 \
  --train-epochs 2 --eval-games 5 --promotion-games 10 \
  --dataset-workers 1
```

### Resume a training run

```bash
python -m backend.ml.auto_improve_alphazero \
  --out-dir reports/ml/alphazero_vN \
  --no-clean --start-iter N [... same flags as original launch ...]
```

### Cloud self-play (Modal.com, 32 workers)

```bash
python -m backend.ml.modal_selfplay [flags]
# Dispatches self-play across 32 Modal containers.
# Local backend/ is mounted fresh — code fixes apply immediately.
```

---

## Project structure

```
backend/
  data/          Bird/bonus/goal loading and registries
  engine/        Rules, actions, scoring, timed powers
  models/        Dataclasses (GameState, Player, Board, BirdSlot, ...)
  powers/        446 bird power implementations
  solver/        Move generation, MCTS, heuristics, simulation
  ml/            AlphaZero training pipeline
  api/           FastAPI routes and Pydantic schemas
frontend/
  src/           React + TypeScript UI
reports/
  ml/            Training run outputs (models, logs, eval results)
backend/tests/   492 pytest tests
```
