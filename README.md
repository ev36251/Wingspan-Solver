# Wingspan Solver

A full rules engine, solver, and strategy-learning pipeline for **Wingspan: Oceania** (base + Oceania expansion). The engine implements every bird power, the full food/nectar economy, and end-of-round/game scoring, served via a FastAPI backend with a React frontend.

Two strategy approaches live in this repo:

1. **Solo single-seed optimization (current direction).** For a fixed, fully deterministic game, the solver searches the opening draft and every move to maximize the final score, replaying the same seed many times. Across many seeds (parallelized on Modal.com) this produces a high-quality dataset of best-scoring lines plus empirical bird/bonus/draft analytics — and it naturally discovers engine-building strategies (food gathering, round-end caching, card tucking) that simpler evaluators miss. See `memory/SOLO_SEED_FINDINGS.md`.
2. **AlphaZero-style self-play (legacy).** Two MCTS agents play each other and a factorized PyTorch policy/value network learns from the data via behavioral cloning. This produced weak champions (see Known Limitations) and is superseded by the solo approach above; it remains for reference and as the eventual competitive fine-tuning stage.

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
    └─ backend/ml/                               Strategy pipeline
         solo_seed_optimizer.py   Deterministic solo single-seed score search (current)
         modal_solo.py            Modal.com dispatch — one container per seed shard
         alphazero_self_play.py   Self-play data generation (MCTS both sides, legacy)
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

## Solo single-seed optimization (current direction)

For one fixed seed, the entire game is made **deterministic**: a pre-shuffled deck stack (the Nth draw is always the same card), fixed starting hand / bonus cards / round goals, and a seeded birdfeeder reroll stream. The optimizer then replays that exact game many times, searching both the opening **draft** (which birds vs. food to keep — the real Oceania rule: keep 5 total, ≤1 of each food type) and every in-game **move**, on a simulated-annealing temperature schedule, keeping the highest-scoring line. Because each replay is scored to the end of the game, the search "sees" engines pay off — something a one-step greedy evaluator cannot.

Each seed is independent, so seeds are fanned out across Modal.com containers (one per seed). Only the single best line from each seed is kept, so per-seed search effort never overfits a model — the number of *distinct* seeds controls generalization.

**Findings over 1000 seeds** (Oceania, 150 games/seed; full detail in `memory/SOLO_SEED_FINDINGS.md`):

- Mean best score ~90 (range 66–140), built from a balanced engine — bird VP, round-end caching, and card tucking — not egg-spam.
- Drafts robustly keep **~2.6 of 5 birds**, taking more food (matches expert intuition).
- The top birds by appearance-and-average-score are the round-end cache engines **Eurasian Nutcracker** and **Sri Lanka Blue-Magpie** — the search rediscovers known strong cards on its own.
- High-scoring lines fill all three habitats evenly and lean on brown ("when activated") and teal ("round end") engines.

This dataset of best-scoring lines is the foundation for the next step: training a policy/value network on *strong* play (rather than the weak self-play data below) and, later, competitive fine-tuning against an opponent.

---

## What it learned / how it improved (legacy AlphaZero self-play)

> The following describes the earlier 2-player self-play runs (v4–v20). These produced weak champions (see Known Limitations) and have been superseded by the solo approach above; the old `reports/ml/alphazero_v*` artifacts were deleted.

A few genuine learning dynamics observed across those runs:

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

### Solo single-seed optimization (current direction)

```bash
# Optimize one fixed seed locally (draft + play searched; prints best line)
python -m backend.ml.solo_seed_optimizer single --seed 42 --games 150 --show-trajectory

# Fan many seeds across Modal containers -> best-line dataset + analytics
python -m backend.ml.solo_seed_optimizer multi --seeds 0-999 --games-per-seed 150 \
  --use-modal --seeds-per-shard 1 --out reports/ml/solo_seed/best_lines.jsonl
```

### ML training — legacy AlphaZero loop (smoke test, ~5 min)

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

### Cloud compute (Modal.com)

```bash
# Solo seed optimization: one container per seed (see command above)
python -m backend.ml.solo_seed_optimizer multi --use-modal [flags]

# Legacy self-play: dispatch across 32 Modal containers
python -m backend.ml.modal_selfplay [flags]
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
  solver/        Move generation, MCTS, heuristics, simulation, setup advisor
  ml/            Solo single-seed optimizer + Modal dispatch, legacy AlphaZero pipeline
  api/           FastAPI routes and Pydantic schemas
frontend/
  src/           React + TypeScript UI
reports/
  ml/            Strategy outputs (solo_seed/ best-line datasets, analytics)
memory/          Findings & proposals (SOLO_SEED_FINDINGS.md, ...)
backend/tests/   492 pytest tests
```
