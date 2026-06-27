# Wingspan Solver

A full rules engine, solver, and **play-assistant web app** for **Wingspan**, covering
the five released sets — **Core, Oceania, Asia, European, and Promo-UK (471 birds total)**.
The engine implements every bird power, the full food/nectar economy, and
end-of-round/game scoring, served via a FastAPI backend with a Svelte frontend.

Three things live in this repo:

1. **The play assistant (the product).** A web companion: enter your real game,
   tap "Recommend," and it returns the strongest move — chosen by a rollout-search
   agent that scores in the low-to-mid **90s** — plus plain-English odds
   ("~70% of the time this finishes 70+ points") and longshot lines ("~3% chance
   you draw this bird → big jump"). See **"In plain English"** above.
2. **Solo single-seed optimization (the dataset/analytics engine).** For a fixed,
   fully deterministic game, the solver searches the opening draft and every move
   to maximize the final score, replaying the same seed many times. Across many
   seeds (parallelized on Modal.com) this produces a high-quality dataset of
   best-scoring lines plus empirical bird/bonus/draft analytics. See
   `memory/SOLO_SEED_FINDINGS.md`.
3. **AlphaZero-style self-play (legacy, superseded).** Two MCTS agents play each
   other and a factorized policy/value network learns by behavioral cloning. This
   produced weak champions and is kept for reference only — see
   **"Legacy / superseded code"** near the bottom.

---

## In plain English (start here)

**What this is.** A Wingspan helper. You type your current game into a web app,
hit "Recommend," and it tells you the strongest move to make — plus your odds of
different final scores.

**How the engine actually "thinks": rollouts.** It doesn't use a formula to
guess whether a move is good. Instead, for each move it's weighing, it *finishes
the game in its head* — it imagines playing all the way to the end and sees what
score it gets. One imagined full game = one **rollout**. It does this many times
and picks the move that leads to the best scores on average. This one idea is the
whole project: judging a move by *playing it out to the end* (instead of guessing)
is worth roughly **+40 points** over "obvious" greedy play, and it's why the
engine discovers real strategies — food engines, end-of-round caching, tucking
cards — because those only pay off later, and rollouts actually *see* later.

**What "R" and "K" mean (the two thinking dials).** It can't imagine *infinite*
games (that'd take forever), so two settings control how hard it thinks:

- **K = top-k = how many candidate moves it seriously considers** each turn.
  `K10` = look at the 10 most promising moves; `K20` = look at 20.
- **R = rollouts = how many full games it imagines per candidate move.**
  `R12` = finish the game 12 times for each move and average; `R36` = 36 times.

So **`K10/R12`** means: *take the 10 best-looking moves, play each out to the end
12 times, average the scores, pick the winner.* More of either dial = more
thinking = usually better moves, but slower. `K20/R36` is the most we tried.

**Where it landed: ~92, and why turning the dials up stops helping.** We cranked
the thinking dials to see if more compute kept buying score (all vs the built-in
opponent, 100 games each):

| thinking budget | average score |
|---|---|
| light (`K8/R6`) | ~88 |
| **default (`K10/R12`)** | **~93** |
| heavy (`K16/R24`) | ~96 |
| heaviest (`K20/R36`) | ~96 *(no better)* |

Two things stand out: more thinking *does* help — but it **flattens out around
96**. Going from R24 to R36 (50% more compute) bought essentially nothing. So
~92–96 isn't a bug; it's a **ceiling**.

**Why ~96 is the ceiling (the whole issue).** We tried hard to break past it with
fancier methods: training a neural net to pick moves, training one to *guess* the
final score without playing it out, different search algorithms (MCTS, deeper
look-ahead), and letting it learn by playing against itself. **Every one of them
failed to beat plain rollout search.** The reason is always the same: the
rollouts are already doing the hard work. When you finish the game 12+ times and
average, that average is such a good judge of a move that a neural net's "hunch"
just gets overruled by the rollouts anyway — the smarter guess gets washed out.
So the ceiling isn't the AI being dumb; it's that **for this game, against this
opponent, ~96 is about the best score reachable**, and the only lever that
reliably helps (more rollouts) has diminishing returns. Past R24 you pay double
the time for noise.

**Bottom line on the engine.** It plays a genuinely strong, *honest* game — it
doesn't cheat by peeking at the deck (it reshuffles the unseen cards before each
imagined game), it beats the rule-based opponent ~90–95% of the time, and it
hits 100+ points in about a third of games. It's about as strong as rollout
search gets here. Use **default** for speed, **heavy (K16/R24)** for the last few
points.

**The app you actually use.** On top of the engine is a web companion: type in
your game, tap "Recommend," and it gives you the strongest move (chosen by that
~92 engine) plus plain-English odds — e.g. *"~70% of the time this finishes 70+
points"* and longshots like *"~3% chance you draw this bird → big jump."* One
honest tradeoff: the *move* comes from the strong engine, but the *score numbers*
come from a faster, rougher stand-in, because a true ~96-level score *distribution*
would take ~an hour per question — useless mid-turn.

---

## Results: from ~60 to the 90s

The agent's score has climbed dramatically over the life of the project. Early
greedy-network and rule-based play scored in the **50s–60s**. The current agent
plays full *honest* games — no peeking at the deck order — and scores a **~92
mean (worst-case floor ~64)** against the rule-based heuristic, which it beats
roughly **90% of the time** (re-measured at n=100 on the latest corrected
engine; best config r12 / top-k 10 / temp 0.3 / determinize).

### Score progression
| stage | typical score |
|-------|---------------|
| rule-based heuristic / early greedy network | ~50–60 |
| net-guided rollout search (solo, held-out seeds) | 72 → 91 |
| **2-player honest rollout search (current)** | **~92 mean, floor ~64** |

### What moved the needle (all measured at n=100 on Modal.com)

- **Rollout search to game-end** is the heart of the agent: judging each move by
  *playing the game to the finish* instead of a one-step score estimate is worth
  ~**+40 points** over greedy play, and it discovers real engines (food loops,
  round-end caches, card tucking) that a shallow evaluator misses.
- **Averaged stochastic rollouts** — scoring a move by the *average* of several
  varied playouts rather than one noisy game — is a real, statistically
  significant gain (63% win rate vs the single-rollout baseline, p = 0.004).
- **Honest play via determinization** — reshuffling the unseen deck before each
  rollout so the agent plans over *plausible* futures instead of peeking at the
  real card order — costs almost nothing (~1 point). The agent wins by building
  *robust* engines, not by knowing the next card.
- **Engine-correctness pass (the floor-raiser).** Auditing the simulator against
  the real Oceania player mat surfaced and fixed several rules bugs: the
  tray/feeder reset now deals **fresh** cards (it used to clear them and never
  refill, so resetting was strictly bad); resets are payable with **nectar**, and
  that nectar **scores** in its habitat; and end-of-round goals no longer award
  points to a player with **zero** of the goal. These fixes raised the bad-deal
  **floor from 52 → 68** — the worst hands now recover, exactly as a strong human
  would by flipping the tray and spending surplus nectar. The real-game replay
  goldens still reproduce perfectly (0 divergences), so the engine is strictly
  *more* faithful after the changes.

### What didn't work (measured and ruled out)

- **Denial / "beat the opponent" objective.** Optimizing *my score − opponent
  score* **loses** to pure score-maximization (40% vs 60%, n = 100). In Wingspan
  player interaction is limited, so a strong selfish engine beats clumsy
  sabotage; even with an opponent-board-aware network, denial only reaches
  break-even. Natural denial (taking strong birds you actually want) is already
  captured by playing well for yourself.
- **Network retraining.** Three separate retrains — a behavioral-cloned
  opponent-aware net, and a fresh solo-flywheel net trained on the corrected
  engine — all failed to beat the existing `solo_net_spread`. At heavy search
  budget the **search dominates the policy prior**, so a new prior barely changes
  the searched output. The lever to push the mean higher is **more search**
  (depth / drafts / rollouts), not a new network.

Full experiment-by-experiment detail (with sample sizes and p-values) lives in
`memory/SOLO_SEED_FINDINGS.md`.

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
    └─ backend/powers/                           471 bird powers
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

## What we've learned & what's next

This phase moved the agent from ~60 to a robust, honest **~92–95 mean** by
*measuring* every lever rather than guessing. The findings (full detail with
sample sizes and p-values in `memory/SOLO_SEED_FINDINGS.md`):

**Search-lever scorecard** (all measured at n=100 on Modal):

| lever | verdict |
|-------|---------|
| Rollout search to game-end | the core engine: ~**+40** over greedy play |
| Averaged stochastic rollouts | ✅ real win (+5, p = 0.004) |
| Determinization (no deck-peeking) | ✅ ~free (~1 pt) — honest play matches peeking |
| Differential / "denial" objective | ❌ hurts (40% vs 60%) — selfish play wins |
| Network retraining (×3 attempts) | ❌ null — at heavy search the *search dominates the prior* |
| Depth-2 lookahead | ❌ hurts at equal budget — the rollout already looks to game end |

**Best config:** depth-1, rollouts 12, top-k 10, temp 0.3, determinize. This is
the practical ceiling for the rollout-search approach.

**Re-tuned on the corrected engine (n=100, honest, same 100 seeds).** Every config
above was originally tuned on an older engine, so a 6-cell sweep was re-run on the
latest corrected engine (rollouts ∈ {8,12} × top-k ∈ {8,10} × temp ∈ {0.3,0.5}).
The optimum did **not** move: r12 / k10 / t0.3 is again the best cell
(mean 91.7, floor 64, 28% ≥100, 91/100 wins). Rollouts is the only directionally
consistent lever (r12 ≥ r8 on every metric at no floor cost), but the r12-vs-r8
lift is **sub-noise** (+2.35 mean, paired sign-test p = 0.47) — all six cells
cluster in 89–92, so the search is saturated across this grid. The corrected-engine
re-measure thus *confirms* the existing best config rather than replacing it.

**Engine correctness is where the real points hid.** Auditing the simulator
against the real Oceania board surfaced and fixed several rules bugs — tray/feeder
reset now deals fresh cards, resets are payable with nectar (which scores), and
round-end goals no longer reward a player with zero of the goal. These raised the
**bad-deal floor from 52 → 68**, and the video-replay goldens still reproduce the
real recorded games perfectly (0 divergences), so the engine is strictly more
faithful.

**Bird coverage:** all **471** birds (Core, European, Oceania, Asia, **+ the 25
Promo UK pack**) have explicit, tested power implementations. A solo + 2-player
tier analysis ranks them; notably, opponent-reactive (pink) birds rank dead-last
solo but jump to the top in 2-player (e.g. Eurasian Skylark 75 → 109 avg).

### Next steps

- **A learned, search-quality value evaluator.** The one path that could raise
  the ceiling rather than nudge it: AlphaZero-style iteration with
  *search-derived* value targets (plain Monte-Carlo targets label every state
  with the one final score and aren't move-discriminative). This is a real
  project, not a knob.
- **Finish the remaining approximate "unique" powers** and audit rare
  interaction chains (Repeat → Copy → Repeat). Low-risk, steady correctness gains
  — the kind that already paid off in the floor result.
- **Multiplayer (3–4 player)** opponent modeling — the pipeline supports it, but
  play is calibrated for 1v1.

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
pytest                                          # full pytest suite
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
  powers/        471 bird power implementations
  solver/        Move generation, MCTS, heuristics, simulation, setup advisor
  ml/            Solo single-seed optimizer + Modal dispatch, legacy AlphaZero pipeline
  api/           FastAPI routes and Pydantic schemas
frontend/
  src/           Svelte + TypeScript UI (the play-assistant app)
reports/
  ml/            Strategy outputs (solo_seed/ best-line datasets, analytics)
memory/          Findings & proposals (SOLO_SEED_FINDINGS.md, ...)
backend/tests/   pytest suite (engine correctness, solver, advisor)
```

---

## Legacy / superseded code

The AlphaZero-style self-play training loop (v4–v20) produced weak champions and
is **superseded** by the rollout-search agent + solo optimizer. It is kept for
reference only and is **not** part of the play assistant. These modules are the
dead loop drivers — safe to ignore:

- `backend/ml/auto_improve_alphazero.py` — the self-play training-loop driver
- `backend/ml/auto_improve.py`, `backend/ml/auto_improve_factorized.py` — older loop drivers
- `backend/ml/self_play_dataset.py` — legacy dataset generator
- `backend/ml/modal_selfplay.py` — Modal dispatch for the legacy loop

> **Note:** a few utilities first written for the loop are still used by current
> code and are *not* legacy — e.g. `alphazero_self_play._encode_policy_visit_targets`
> (reused by the soft-target generator `selfplay_soft.py`) and `mcts.py`. That
> shared usage is why these files stay in place rather than being moved into a
> `legacy/` folder (moving them would break live imports for no real benefit).
