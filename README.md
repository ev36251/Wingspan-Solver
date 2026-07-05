# Wingspan Solver

A complete rules engine, search-based AI, and **play-assistant web app** for the
board game **Wingspan**, covering all five released sets — **Core, Oceania, Asia,
European, and Promo-UK (471 birds)**. The engine implements every bird power, the
full food/nectar economy, and end-of-round/game scoring; a FastAPI backend serves
a Svelte web app that recommends (and now *applies*) moves in a real game.

## Highlights

- **Complete rules engine.** All 471 bird powers have explicit, tested
  implementations, plus nectar, timed (pink/teal/yellow) powers, goal scoring,
  and solo-mode rules. Verified by **golden replays of real recorded games
  (0 divergences)** and a **~1,100-test** pytest suite.
- **A strong, honest AI.** A rollout-search agent that never peeks at hidden
  information (it re-shuffles the unseen deck before every imagined playout)
  scores a **~92 mean vs. the rule-based baseline (~90% win rate)**, rising to
  **~96 at the heavy search preset**. Solo play went **77 → 87** by searching
  the opening draft and deepening mid-game search.
- **Measured, not guessed.** Every candidate improvement was evaluated at
  n = 100 paired games on Modal.com with significance tests. The write-ups keep
  the **negative results**: six independent attempts to beat the search with
  neural networks (retrains, value-net leaves, PUCT-MCTS, self-play soft
  targets) all failed — so the agent's ceiling is *understood*, not assumed.
  Full lab notebook: `memory/SOLO_SEED_FINDINGS.md`.
- **A real product on top.** The web companion recommends the engine's move
  with plain-English odds ("~70% of the time this finishes 70+ points"),
  longshot lines, a draft advisor driven by the same engine, and one-click
  apply that replays the move **through the engine** — brown-power chains,
  pink triggers, nectar bookkeeping and round transitions all happen for you,
  without ever inventing cards the table hasn't revealed.
- **Distributed experimentation.** All data generation and evaluation fans out
  across Modal.com containers (one per seed/config); the local machine only
  dispatches and aggregates.

---

## In plain English (start here)

**What this is.** A Wingspan helper. You type your current game into a web app,
hit "Recommend," and it tells you the strongest move to make — plus your odds of
different final scores — and can apply that move to your tracked game for you.

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

**One place was still under-searched: the opening.** The move-by-move search had
plateaued, but the *draft* (which birds and food to keep before turn one) turned
out to be a second, independent dial. Searching **multiple opening keeps** and
playing each out lifted the solo mean by **+7.4 points across the whole
distribution** — good hands and bad hands alike — and searching deeper in the
mid-game added another ~+2, stacking to **+10 (77 → 87)**. Lesson: when one
search axis saturates, look for the axis nobody is searching.

**Bottom line on the engine.** It plays a genuinely strong, *honest* game — it
doesn't cheat by peeking at the deck, it beats the rule-based opponent ~90% of
the time, and it breaks 100 points in roughly a third of games. Use **default**
for speed, **heavy (K16/R24)** for the last few points.

---

## The play assistant (the product)

A Svelte web app backed by the engine, built for *companion play* — you play a
physical game and the app tracks it, advises, and does the bookkeeping:

- **Recommend.** One tap runs two solvers: a fast lookahead (clickable ranked
  move cards) and the strong rollout-search engine. The engine's pick is
  promoted to the **#1 card with an "engine pick" badge**, alongside
  plain-English "percentage play" sentences ("~70% of the time this finishes
  70+") built from determinized rollouts — the projections re-sample unknown
  opponent hands and deck order per rollout, so the odds are honest.
- **Apply through the engine.** Clicking a recommendation replays it through
  the real rules engine: brown-power chains fire bird by bird, pink powers
  trigger, nectar spend is recorded in the right habitat, the action cube is
  spent, and round transitions (teal powers, goal scoring, nectar discard, cube
  refresh) happen automatically. **Companion mode never invents hidden
  information**: deck draws become face-down card counts and the tray is left
  short for you to enter the card actually revealed at your table.
- **Longshots.** High-ceiling birds still in the deck, with draw odds and a
  conditional score *if you draw them in time to use them* — late-game
  longshots are correctly rare.
- **Draft advisor.** The opening keep-decision is made by the same strong
  engine (not a heuristic): it plays the top openings out and ranks them —
  the +7.4 draft lever, in the product.
- **Screenshot import (Claude vision).** Manual state entry is the slowest part
  of companion play, so the app can read it for you: upload screenshots of the
  game (digital app or photos of the table) and a Claude vision call extracts
  the boards, eggs, cached food, hands, bonus cards, feeder dice, tray, round,
  and nectar into a proposed state. The opening draft screen imports too —
  screenshot the "choose 5 to keep" screen and the dealt birds/bonus cards
  flow straight into the draft advisor. Every card name is fuzzy-matched
  against the 471-bird registry; anything that didn't match cleanly comes back
  as a warning for you to review before applying. Needs `pip install anthropic`
  and an `ANTHROPIC_API_KEY` on the backend (see [How to run](#how-to-run)).
- **After-reset flow, score sheet, max-score bar, multi-player tracking.**

---

## Results: from ~60 to the mid-90s

Early greedy-network and rule-based play scored in the **50s–60s**. The current
agent plays full *honest* games and scores **~92 mean (floor ~64) vs. the
rule-based heuristic, winning ~90%** (n = 100, corrected engine, `K10/R12`,
temp 0.3, determinized), and **~96 at `K16/R24`**.

| stage | typical score |
|-------|---------------|
| rule-based heuristic / early greedy network | ~50–60 |
| net-guided rollout search (solo, held-out seeds) | 72 → 91 |
| 2-player honest rollout search (default budget) | ~92 mean, floor ~64 |
| **+ heavy search preset (`K16/R24`)** | **~96 mean** |
| solo agent + draft search + deeper mid-game | 77 → **87** |

### What moved the needle (all measured at n = 100 on Modal)

- **Rollout search to game-end** — the heart of the agent: judging each move by
  *playing the game to the finish* instead of a one-step score estimate is worth
  ~**+40 points** over greedy play, and it discovers real engines (food loops,
  round-end caches, card tucking) that a shallow evaluator misses.
- **Draft search (+7.4, the biggest late-project win).** Searching 4–6 opening
  keeps instead of committing to one lifts the *whole* score distribution
  (155/200 deals improved), and it **stacks** with deeper mid-game search to
  +10.3 in solo. The opening was the single most under-exploited decision.
- **Search budget (+3).** The one dial that scales: `R6 → R12 → R24` = 87.5 →
  93.2 → 96.2, plateauing at R24 (R36 adds nothing).
- **Averaged stochastic rollouts** — scoring a move by the *average* of several
  varied playouts rather than one noisy game — a statistically significant gain
  (63% win rate vs the single-rollout baseline, p = 0.004).
- **Honest play via determinization** — reshuffling the unseen deck before each
  rollout so the agent plans over *plausible* futures instead of peeking —
  costs ~1 point. The agent wins by building robust engines, not by knowing the
  next card.
- **Engine correctness (the floor-raiser).** Auditing the simulator against the
  real Oceania board fixed several rules bugs: the tray/feeder reset now deals
  **fresh** cards (it used to clear and never refill, so resetting was strictly
  bad); resets are payable with **nectar**, which then **scores** in its
  habitat; round-end goals no longer reward a player with **zero** of the goal.
  These raised the bad-deal **floor from 52 → 68** — the worst hands now
  recover exactly as a strong human would, by flipping the tray and spending
  surplus nectar. The real-game replay goldens still reproduce perfectly.
- **Hot-path engineering (~1.3× throughput, zero behavior change).** Profiling
  the search showed the cost was in Python plumbing, not the model: millions of
  `Enum.__hash__` calls (hashing member name strings) and repeated food-payment
  legality checks. Switching value-keyed enums to identity hashing, memoizing
  the payment solver by `(cost, supply)`, and making bird habitats a canonical
  tuple cut full-game rollouts **70 → 53 ms** — verified bit-identical as an
  identical move *multiset* over 174 real states and identical golden-replay
  scores. Stacks with the distilled student for ~4.5× effective playout
  throughput over the pre-optimization baseline. (It also surfaced a stale
  golden and made move order `PYTHONHASHSEED`-independent.)

### What *didn't* work (measured and ruled out)

This project kept its negative results — they're what make the positive numbers
believable:

| lever | verdict |
|-------|---------|
| Rollout search to game-end | the core engine: ~+40 over greedy |
| Averaged stochastic rollouts | ✅ real win (+5, p = 0.004) |
| Draft search + mid-game depth (solo) | ✅ +10.3, they stack |
| More rollout budget | ✅ +3, plateaus at `K16/R24` |
| Determinization (no deck-peeking) | ✅ ~free — honest play matches peeking |
| Network retraining (×6 attempts) | ❌ null — at heavy search the *search dominates the prior* |
| Learned value function as rollout replacement | ❌ greedy-V 74 / bootstrap 90 / PUCT-MCTS 74, vs full rollouts 93 |
| Distilled fast rollout policy (student) | ⚡ **~3× faster rollouts**; the identity-aware v2 posts the best mean of any tested config (95.5 vs 92.8) at 12% *less* wall-clock — a real speed lever; −6.5 if you pocket the speed unscaled |
| Denial / "beat the opponent" objective | ❌ hurts (40% vs 60%) — selfish play wins in a low-interaction game |
| Depth-2 lookahead (2-player) | ❌ hurts at equal budget — the rollout already looks to game end |

The value-net finding deserves one line: the net *predicts* state value well
(R² ≈ 0.95, one eval worth ~4 playouts of accuracy at 1/300 the cost) but every
way of *using* it inside search loses to real playouts — under argmax pressure
the search exploits the net's residual errors (the optimizer's curse). Its real
use is speed: a ~85-point agent at 1/8 the compute.

The distilled rollout policy is the successful version of that speed idea:
profiling showed ~76% of a playout is the *Python feature encoding*, not the
network, so a tiny student net over ~75 cheap features (counters plus a hashed
bag of the player's own bird identities, distilled from the big net's choices)
plays the rollouts **~3× faster**. Unlike the value net, reinvesting the saved
time in more rollouts doesn't just recover baseline strength — the
identity-aware student posted the best mean of any tested configuration
(95.5 vs 92.8 baseline, n=40 paired, within noise) while running 12% faster.
The big net keeps the root move ranking; the student only plays out the
imagined games. Deployed: the advisor now runs ~3× the rollouts at unchanged
latency. (A pure-engineering variant — caching the encoder's per-bird blocks,
bit-identical output — also ships, speeding every read ~1.35×.)

**Where the project started (and why it changed).** The original approach was an
AlphaZero-style self-play loop: MCTS self-play → behavioral cloning of a
factorized policy/value net → promotion gates. It trained, promoted models, and
plateaued at a degenerate egg-spam policy ~10 points *below* the rule-based
heuristic. It was replaced by the searched-line approach above, and later —
with much better data and infrastructure — five separate attempts to reintroduce
learned components each failed the pre-registered gate. Every experiment, sample
size, and p-value is in `memory/SOLO_SEED_FINDINGS.md`.

### Bird-strategy findings (from ~4,000 optimized games)

- Optimal drafts keep **~2.6 of 5 birds** — food over marginal cards.
- The best birds are round-end cache engines (**Eurasian Nutcracker**,
  **Sri Lanka Blue-Magpie**) — the search rediscovers known strong cards.
- High-scoring lines fill all three habitats and lean on brown ("when
  activated") engines; egg-spam is what *weak* policies converge to.
- Opponent-reactive (pink) birds rank dead-last in solo but jump to the top in
  2-player (Eurasian Skylark: 75 → 109 avg) — the data cleanly separates
  format-dependent card strength.

---

## Architecture

```
wingspan-20260128.xlsx                           All bird/bonus/goal data (5 sets, 471 birds)
    └─ backend/data/loader.py + registries.py    Loaded once at startup into singleton registries
    └─ backend/models/                           Pure dataclasses: GameState, Player, Board, enums
    └─ backend/engine/                           Game rules
         rules.py          Legality checks (can_play_bird, food costs, egg limits)
         actions.py        execute_* actions + activate_row() (brown-power loop)
         scoring.py        calculate_score() → ScoreBreakdown (incl. solo fixed-target goals)
         timed_powers.py   Pink (between-turn), teal (round-end), yellow (game-end) powers
    └─ backend/powers/                           471 bird powers
         registry.py       Bird → PowerEffect lookup (explicit_mappings.json)
         choices.py        Queued explicit power decisions (companion fidelity)
         templates/        gain_food, lay_eggs, draw_cards, tuck_cards, predator,
                           cache_food, flocking, unique, special, ...
    └─ backend/solver/                           Decision-making
         move_generator.py Every legal move for a state
         simulation.py     Fast rollout engine (fast_clone_game: 8.5× cheaper clones)
         setup_advisor.py  Draft advisor (engine-in-draft, multi-opening search)
    └─ backend/engine_search/                    The strong agent's search internals
         belief.py         Determinization: sample unseen deck + opponent hands
    └─ backend/ml/                               Strategy pipeline
         solo_seed_optimizer.py  Deterministic per-seed best-line search
         two_player.py           The deployed rollout-search agent + strength presets
         solo_search.py          Solo net-guided search (draft + k-schedule depth)
         modal_solo.py           Modal.com dispatch — one container per seed shard
         train_factorized_bc.py  Policy-net training (PyTorch; numpy at serve time)
    └─ backend/api/                              FastAPI routers (game, solver, setup, data)
    └─ frontend/src/                             Svelte + TypeScript play-assistant app
```

**The policy network's actual job.** A behavior-cloned, factorized policy net
(trained on searched best-lines) serves as the **move prior and rollout policy**
inside the search — it proposes and plays out candidate moves; the *decision* is
made by rollout returns. Inference is pure numpy (torch is optional at serve
time). Retraining it better was measured to not matter — see the scorecard.

## Solo single-seed optimization (the dataset engine)

For one fixed seed the entire game is deterministic: a pre-shuffled deck stack,
fixed hand/bonus/goals, and a seeded birdfeeder stream. The optimizer replays
that exact game hundreds of times on a cooling temperature schedule — searching
the opening draft *and* every move — and keeps the best line. Seeds are
embarrassingly parallel (one Modal container each), and only the best line per
seed is kept, so per-seed effort never overfits. The resulting dataset
(`reports/ml/solo_seed/`) trains the policy prior and powers the bird-strategy
analytics above.

---

## How to run

### Prerequisites

```bash
pip install -e ".[dev]"          # Python deps (pytest, uvicorn, numpy; torch only for training)
cd frontend && npm install       # Frontend deps
```

All bird data loads from `wingspan-20260128.xlsx` at startup.

### Backend + frontend

```bash
uvicorn backend.main:app --reload    # http://localhost:8000
cd frontend && npm run dev           # http://localhost:5173 (proxies /api → :8000)
```

To enable screenshot import (optional — everything else works without it):

```bash
pip install -e ".[vision]"                                # adds the anthropic SDK
ANTHROPIC_API_KEY=sk-ant-... uvicorn backend.main:app --reload
```

### Tests

```bash
pytest                                   # full suite (~1,100 tests)
pytest backend/tests/test_engine.py -v   # single file
pytest -k "test_powers"                  # filter by name
```

### The 2-player agent (evaluation harness)

```bash
# Agent vs rule-based heuristic, honest play, n=100 on Modal
python -m backend.ml.two_player --mode heuristic --seeds 0-99 --strength strong --use-modal
# --strength: default (K10/R12, ~93) | strong (K16/R24, ~96) | max (K20/R36)
```

### Solo best-line dataset generation

```bash
# One fixed seed locally (draft + play searched; prints the best line)
python -m backend.ml.solo_seed_optimizer single --seed 42 --games 150 --show-trajectory

# Fan seeds across Modal containers → best-line dataset + analytics
python -m backend.ml.solo_seed_optimizer multi --seeds 0-999 --games-per-seed 150 \
  --use-modal --seeds-per-shard 1 --out reports/ml/solo_seed/best_lines.jsonl
```

---

## Project structure

```
backend/
  data/           Bird/bonus/goal loading and registries
  engine/         Rules, actions, scoring, timed powers
  engine_search/  Determinization / hidden-state sampling
  models/         Dataclasses (GameState, Player, Board, BirdSlot, ...)
  powers/         471 bird power implementations + choice queues
  solver/         Move generation, heuristics, simulation, setup advisor
  ml/             Agent, solo optimizer, Modal dispatch, training (+ legacy loop)
  api/            FastAPI routes and Pydantic schemas
  tests/          ~1,100 tests (engine, powers, solver, API, golden replays)
frontend/
  src/            Svelte + TypeScript UI (the play-assistant app)
reports/ml/       Strategy outputs (solo_seed/ best-line datasets, value-gate data)
memory/           The lab notebook: SOLO_SEED_FINDINGS.md (every experiment + p-values)
```

A few `backend/ml/` modules are the retired AlphaZero loop drivers
(`auto_improve*.py`, `self_play_dataset.py`, `modal_selfplay.py`); they're kept
because shared utilities (`mcts.py`, parts of `alphazero_self_play.py`) are
still imported by current code.
