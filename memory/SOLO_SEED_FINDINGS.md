# Solo single-seed optimization — findings

Approach: for each seed, build a fully deterministic solo Wingspan game
(fixed deck stack, fixed hand/bonus/goals, seeded birdfeeder stream) and
replay it ~150 times with a randomized heuristic policy on a cooling
temperature schedule, keeping the best-scoring line. Draft (keep-decision) is
a search dimension. Parallelized across seeds on Modal (one container/seed).

Harness: `backend/ml/solo_seed_optimizer.py` (+ `modal_solo.py`).
Canonical training set: `reports/ml/solo_seed/best_lines_4000_q250.jsonl`
(4000 best lines, 250 games/seed, fixed-target goal scoring; ~103k state→move
pairs, 25.8 moves/game). Earlier `best_lines_1000*.jsonl` are smaller/superseded.

### Scale / quality progression (all Oceania, goals active)
| dataset | seeds | games/seed | mean score | Nutcracker / Blue-Magpie |
|---|---|---|---|---|
| 1000@q150 | 1000 | 150 | 77.0 | #1 / #2 |
| **4000@q250** | 4000 | 250 | **78.7** | #1 (284×) / #2 (250×) |

Higher games-per-seed lifts target quality modestly (+1.7); the tier list and
~2.63/5 draft are stable across scales.

### Point breakdown (4000@q250, mean total 78.7)
Bird VPs 31.5 (40%) · Eggs 12.1 (15%) · Tucked cards 10.3 (13%, max 83) ·
EoR goals 9.8 (12.5%) · Cached food 5.9 (7.5%, max 86) · Nectar 4.7 (6%) ·
Bonus cards 4.3 (5.5%). Bird VP is the backbone; the high max tuck/cache values
confirm specialist all-in engines exist in the data.

### Goals-frozen vs goals-active (1000 seeds each)
| | mean score | round_goals | bird_vp | tucked | cached | eggs |
|---|---|---|---|---|---|---|
| goals frozen (old) | 90.1 | 22.0 (constant) | 31.1 | 10.2 | 6.7 | 11.2 |
| goals active (new) | 77.0 | 9.6 (0–22) | 30.7 | 9.8 | 6.1 | 12.1 |

The ~13-pt drop is entirely the goal change (free +22 → honestly-earned 9.6);
every other component is unchanged, so the engine strategy held. The optimizer
now earns ~9.6 of a possible 22 — it pursues goals when worthwhile but won't
wreck its engine for them. Goal distribution spans 0–22 (some specialist lines
take 0). Bird tier list is stable (Nutcracker #1, Sri Lanka Blue-Magpie #2).

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
- ~~Round goals score a flat 22 in solo mode~~ **FIXED**: solo round goals now
  use fixed-target scoring (`_compute_round_goal_scores_solo` in scoring.py) so
  goals are a live optimization dimension (mean 9.6, range 0–22). Kept modest by
  design so a specialist VP engine still wins while forfeiting goals/nectar.
- Solo only: no competitive play (blocking, goal racing, tray/food denial).

## Net training results (Option A: behavioral cloning)
- `solo_bc_dataset.py` replays best lines -> (state, factorized targets, value)
  pairs; `train_factorized_bc` trains; `solo_eval.py` plays real games.
- **Policy net works.** Greedy net beats the rule-based heuristic and
  generalizes to held-out seeds (4000+): ~+2 to +4 mean pts (52.2 vs 50.1 on
  25 seeds; 52.5 vs 48.6 on 100). Fast (numpy inference). This is the working
  deliverable for a quick solver.
- **Value head is NOT search-useful.** Greedy 1-ply selection by the value head
  is ~-24 vs the policy. Adding a value-only line spread (372k samples, value
  targets 18-99) cut the value *loss* but did not fix selection: policy+value
  1-ply is -2.2 over 25 seeds (a 5-seed +7.2 was noise). Root cause: MC value
  targets label every state in a game with the one final score, so the head
  can't finely rank moves from a position. Fixing it properly needs
  search-derived value targets (AlphaZero-style iteration), not raw MC labels.
- **Conclusion:** net-guided MCTS on this value head is NOT justified. The
  strongest mover we have is the deterministic per-seed *search* itself (~78);
  the policy net is the fast generalizing solver.

## Net-guided rollout search (the strong fast solver) — `solo_search.py`
Judges moves by *finishing the game* (rollouts), not the unreliable value head.
`fast_clone_game` (simulation.py) made per-move clones 8.5x cheaper
(4.55ms -> 0.53ms), which made this affordable. Three combinable knobs:
top-k (moves expanded ply 1), depth (plies of lookahead via a branching
schedule), n-drafts (search the top openings too).

Held-out seeds (net rollout), vs greedy net and the best-of-250 brute ceiling:
| config | mean score | speed |
|---|---|---|
| net greedy | ~51 | instant |
| top-k=5, depth=1, 1 draft (25 seeds) | 71.8 (+20) | 4.5s/game |
| **top-k=8, depth=2, 3 drafts (10 seeds)** | **91.5 (+44)** | ~60s/game |

The combined search **beats the best-of-250 brute search (~78)** and wins
10/10 vs the greedy net. On Modal (one container/seed) even the 60s config
finishes any number of seeds in ~1 min wall-clock.

## Flywheel iteration 2 (BC on search lines) — NULL RESULT
Regenerated 3000 lines with the net-guided search (mean 88.3 vs iter-1 78.6),
rebuilt BC (70.9k samples), retrained. Imitation val accuracy rose 0.693 ->
0.725, BUT end-to-end performance was flat on 10 held-out seeds:
| (same config) | iter-1 | iter-2 |
|---|---|---|
| net greedy | 47.2 | 46.6 |
| combined search | 91.5 | 90.3 |
Both within per-seed noise (60-125). One BC iteration did not compound. Likely
because: (a) at this heavy search budget the search dominates the policy prior,
so a better prior barely changes the searched output; (b) BC on a searcher's
*chosen moves* teaches surface moves, not the lookahead reasoning behind them,
so greedy play doesn't improve. Models kept (equivalent): solo_net_spread.npz
(iter-1), solo_net_iter2.npz.

## Search-budget sweep (50 held-out seeds, solo_sweep.py)
| config | mean | time/game | lever |
|---|---|---|---|
| d3 nd3 k8 | 91.4 | ~210s | depth-3 |
| d2 nd5 k8 | 90.7 | ~117s | +drafts (+4.4 from base) |
| d2 nd3 k12 | 88.7 | ~105s | +width (+2.4) |
| d2 nd3 k8 (base) | 86.3 | ~70s | — |
| d1 nd3 k8 | 84.2 | ~20s | shallow |

Verdict: **drafts > width > depth** for score-per-time. Depth-3 tops out (91.4)
but costs ~2x nd5 for +0.7 -- not worth it. The efficient levers are more
drafts and more width; combine them (nd5 + k12) rather than paying for depth.
(Note: the earlier 91.5 on 10 seeds was a lucky sample; base on 50 seeds = 86.)

## Suggested next steps
- Higher score now comes from MORE SEARCH budget (top-k / depth / n-drafts /
  multiple rollouts), not more BC iterations. Modal makes this fine at scale.
- A proper AlphaZero step would train on search VISIT distributions / values
  (not just the best move) -- may transfer better than plain BC, bigger build.
- Eventually fine-tune competitively against an opponent.
