# Solo single-seed optimization — findings

Approach: for each seed, build a fully deterministic solo Wingspan game
(fixed deck stack, fixed hand/bonus/goals, seeded birdfeeder stream) and
replay it ~150 times with a randomized heuristic policy on a cooling
temperature schedule, keeping the best-scoring line. Draft (keep-decision) is
a search dimension. Parallelized across seeds on Modal (one container/seed).

Harness: `backend/ml/solo_seed_optimizer.py` (+ `modal_solo.py`).
Dataset: `reports/ml/solo_seed/best_lines_1000.jsonl` (1000 best lines w/ trajectories).

## Headline results (1000 seeds, 150 games/seed, Oceania)
- Score: mean 90.1, range 66–140.
- Draft: avg **2.62 / 5 birds kept** — the search robustly prefers keeping
  *fewer birds and more food* (stable across 250 and 1000 seeds).
- Score composition (mean): bird_vp 31, round_goals 22 (constant — see caveat),
  eggs 11, tucked_cards 10, cached_food 7, bonus_cards 4, nectar 5.

## Bird tier list (appearance in best lines × avg game score)
1. Eurasian Nutcracker — 73× (avg 99.0)
2. Sri Lanka Blue-Magpie — 71× (avg 98.2)   ← round-end cache engine
3. Brown Shrike — 59× (91.6)
4. Large-Billed Crow — 56× (89.6)
5. Common Chaffinch — 47× (91.6)
The two food-caching round-end (teal) engines lead by a clear margin AND carry
the highest avg scores — strong confirmation that engine birds win when played
to their plan.

## Top bonus cards in best drafts
Small Clutch Specialist (46×), Avian Theriogenologist (35×), Nest Box Builder
(32×), Mechanical Engineer (32×), Oologist (31×), Large Bird Specialist (31×).

## What makes a high-scoring line (top vs bottom quartile)
- More birds on board (7.5 vs 6.3) and **all three habitats filled evenly**
  (~4 each vs ~3.2).
- More **brown** "when activated" engines (5.6 vs 4.5/line) and slightly more
  **teal** round-end caches (0.36 vs 0.25).
- Engine output dominates the gap: cached_food 10.7 vs 4.4, tucked_cards 12.2
  vs 8.3, bird_vp 34.7 vs 27.2, bonus 5.7 vs 3.4. Eggs barely differ.

## Caveats / limitations
- **Bird-identity pair combos are too sparse to mine** at this scale (different
  deal each seed). Synergy is *functional* (power-color / engine archetype),
  not specific named pairs.
- **Round goals score a flat 22 in solo mode** in all 1000 games: with one
  player there is no opponent to rank against, so the player always "wins"
  every goal for a fixed amount. ~1/4 of every score is therefore a dead
  constant the optimizer cannot influence — it learns nothing about goals.
  Fixing this needs threshold-based solo (Automa-style) goal scoring.
- Solo only: no competitive play (blocking, goal racing, tray/food denial).

## Suggested next steps
1. Implement threshold-based solo round-goal scoring so goals become a real
   optimization dimension (unblocks ~1/4 of the score).
2. Train a policy/value net on the best-line dataset (fast, generalizing
   engine; later guide deeper search).
3. Eventually fine-tune competitively against an opponent.
