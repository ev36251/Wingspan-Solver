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

Combined `d2 nd5 k12` = 91.6 (~175s/game) -- marginally beats depth-3 and nd5
alone. **The search saturates ~91-92 on 50 seeds regardless of knob**; levers
give diminishing returns once stacked. Recommended defaults:
- best value : `d2 nd5 k8` (top-k 8, depth 2, n-drafts 5) -> ~91 @ ~117s/game
- max        : `d2 nd5 k12` -> ~91.6 @ ~175s/game
- fast        : `d1 nd3 k8` -> ~84 @ ~20s/game
On Modal (one container/seed) any of these finishes N seeds in ~minutes.

## Multi-rollout (lever 2) + the deal-limit ceiling
Multiple stochastic rollouts (max-aggregated) per leaf: d2 nd5 k8 + 4 rollouts
= 91.9 on 50 seeds (only +1.2 over nd5). Saturated. Distribution: median 92,
**26% of seeds already >=100** (max 131), but a tail of hard deals (66-78)
holds the mean down. Those low seeds are likely deal-limited (a weak deal caps
the achievable score), so a *mean* of 100 across random deals is probably not
reachable by search at all. The engine already breaks 100 on good deals.

## Pushing search harder (50 seeds)
| config | mean | >=100 | max |
|---|---|---|---|
| d2 nd5 k8 R4 | 91.9 | 26% | 131 |
| d2 nd5 k8 R8 | 92.9 | 24% | 134 |
| **d2 nd8 k8 R4** | **94.3** | **30%** | 141 |
More drafts keeps being the best lever (nd5->nd8 R4 = +2.4); rollouts saturate.
Floor barely moves (min ~66-68) -> hard-deal tail is deal-limited. Mean
approaching a mid-90s ceiling with clear diminishing returns. Recommended max
config: `d2 nd8 k8 R4` (~94, 30% break 100).

## Suggested next steps
- Solo search is at its practical ceiling (~94 mean, 30% >=100, deal-limited
  tail). Further pushing yields <1-2 pts.
- Recommended: bank the solo engine and pivot to 2-player competitive (the
  original goal).

## Two-player pivot (two_player.py)
Differential rollout search: roll each candidate out for both seats with the
policy, score by (my_score - opp_score). 2-player games use real placement goal
+ nectar-majority scoring automatically. vs the rule-based heuristic (10 games,
alternating seats): **agent 101.0 vs heuristic 41.7, 10/10 wins.** Competitive
play emerged for free -- the heuristic scores ~70 vs another heuristic but
collapses to ~42 vs the search agent (active goal/tray/food suppression, no
hand-coded blocking).
Caveat: heuristic is a weak baseline. Next: stronger opponents (search vs
search; differential vs pure score-max ablation), faster/larger eval.

Ablation (--mode ablation, alternating seats): differential (my-opp) vs pure
score-max (my only), both single-greedy-rollout search, opponent modeled as net
policy in rollouts.
- n=20: differential 83.7 (12/20, 60%) vs score-max 79.3; p=0.25 (noise).
- **n=100 (Modal): score-max WINS. differential 76.7 (40/100, 40%) vs score-max
  78.8; p=0.98 against differential being better.** The small sample was noise;
  at scale pure score-max is clearly better.
WHY: the differential objective subtracts a *noisy, biased* opponent-score
estimate (opponent is a fixed net that can't even see its own board well), so
optimizing (my-opp) chases unreliable "denial" value and sometimes makes bad
blocking plays at the expense of its own engine. Score-max optimizes a clean,
low-variance target it controls directly. In Wingspan interaction is limited
(shared tray/feeder/goals), so a strong selfish engine beats naive denial.
ACTION: **make score-max (selfish) the default objective.** Competitive
reasoning only pays off with a GOOD opponent model -- which is exactly the case
for the opponent-board-aware net retrain (the eventual goal). Until then,
differential hurts.

Opponent-aware net (bootstrap step 1 DONE): trained a 2443-dim net (own board +
leading opponent's full board) on 41.6k BC samples from 2-player selfish search
(reports/ml/two_player/opp_aware_net.npz, val_acc 0.674). Re-ran the SAME n=100
ablation with this net as both agent and rollout opponent model:
- blind net (1693):     differential 40/100 (40%) -- denial HARMFUL.
- **opp-aware (2443):   differential 49/100 (49%), 72.6 vs 73.2, p=0.50 -- denial
  now BREAK-EVEN.**
Seeing the opponent removed denial's penalty (40->49), consistent with the
hypothesis that the differential term failed because the opponent score estimate
was garbage. BUT denial still doesn't win, and the 40->49 shift alone is only
~p=0.20 (two-proportion), so honest read = "moved to parity," not "improved."
WHY no win yet: the opp-aware net was cloned from SELFISH search, so its move
RANKING (top_k candidates) rarely proposes denial plays -- the agent can now
*evaluate* the opponent realistically but isn't *offered* denial moves to pick.
Denial-weight sweep (DECISIVE): opp-aware net, agent scores rollouts as
my - lambda*max(opp), vs pure-selfish baseline, n=100 per lambda:
    lambda=0.0  50% (by definition)
    lambda=0.1  49/100  p=0.54
    lambda=0.25 47/100  p=0.73
    lambda=0.5  52/100  p=0.31
    lambda=1.0  49/100  p=0.50
**Flat at 50% across the whole range -- NO denial weight significantly beats pure
score-max.** Denial is worth ~nothing in 2-player Wingspan. CONCLUSION: pure
score-max (lambda=0) is the final objective; do NOT pursue step 3 (training a
denial policy) -- the sweep already shows the ceiling is parity at every weight.
Honest caveat: the search only evaluates the net's top_k candidates, which are
selfish-trained, so it may not surface exotic denial plays; but full denial
(lambda=1) maximally rewards any denial move that IS in the candidate set and
still only breaks even, so the conclusion is robust. Matches the game-design
intuition: taking a bird to deny it costs you tempo/resources on an off-engine
card while the opponent would have spent their own turn playing it -- usually
-EV. Natural denial (taking strong birds you want anyway) is already captured by
score-max.

Strengthening attempts (--mode selfplay, alternating seats):
- Averaged stochastic rollouts (rollouts=4, temp=0.6) vs baseline (1 greedy
  rollout), both selfish objective:
    n=20:  improved 79.4 (11/20, 55%) vs 74.5; p=0.41 (noise).
    **n=100 (Modal): improved 88.4 (63/100, 63%) vs baseline 83.6; p=0.004 --
    SIGNIFICANT.** Averaged rollouts genuinely help; the n=20 was underpowered.
    ADOPT rollouts=4/temp=0.6 as the default search config.
- Deck determinization (--determinize): reshuffles unseen deck per rollout so the
  search plans over plausible futures instead of peeking at the true order.
  Honest agent (4 reshuffled rollouts) still beats heuristic 63-46. NOTE: in the
  fixed-seed harness the peeking baseline has an unfair edge, so determinize is
  about honest/robust play, not in-sim score.
Pattern: each search tweak (differential, averaged rollouts) buys ~+5 pts but
none is significant at n=20. Diminishing returns on rollout tweaks. Bigger levers:
larger Modal eval to resolve small effects, 2-ply search (model opponent's reply),
or retrain the net with opponent-board features. Lower temp (~0.3) worth a try.
- A proper AlphaZero step would train on search VISIT distributions / values
  (not just the best move) -- may transfer better than plain BC, bigger build.
- Eventually fine-tune competitively against an opponent.

Honesty tax (determinization cost), vs heuristic, selfish rollouts=4 temp=0.6,
n=100, solo_net_spread:
- PEEKING  (determinize=False, search clones the real deck): agent mean 89.9, 86/100.
- HONEST   (determinize=True, deck order hidden):            agent mean 88.7, 90/100.
**Honesty costs ~1.2 pts (within noise) and won MORE games.** A robust engine
doesn't need deck foreknowledge -> honest play already scores ~89. Determinization
is effectively free; make it default for fair 2-player play.

Honest-play tuning (determinize=True, vs heuristic, n=100), recovering peek score:
| rollouts | temp | agent mean | wins |
|---|---|---|---|
| 4 | 0.6 | 88.7 | 90 |
| 8 | 0.6 | 87.7 | 80 |
| 8 | 0.3 | 89.1 | 86 |
(peeking ceiling 89.9). LOWER TEMP is the lever: with determinization supplying
the future-deck variance, a sharper rollout policy (t=0.3) plays each imagined
deck better. More rollouts only help at low temp (8@0.6 was worse than 4@0.6 --
high-temp noise compounds). Best honest config r=8/t=0.3 = 89.1 ~ peek 89.9
(tied). Honesty gap effectively closed.

Completed the honest tuning grid (determinize, vs heuristic, n=100):
|        | t=0.6 | t=0.3 |
| r=4    | 88.7  | 86.4  |
| r=8    | 87.7  | 89.1  |
Low temp needs ENOUGH rollouts: at t=0.3 the playouts are near-greedy/identical,
so r=4 has nothing to average (86.4) but r=8 gets enough reshuffled-deck variety
to benefit (89.1). FINAL HONEST CONFIG: --determinize --rollouts 8 --temperature
0.3 -> 89.1, tied with the peeking ceiling (89.9). Cheaper fallback r=4 t=0.6 =
88.7. Honesty gap closed; pushing the MEAN clearly above ~90 is a separate
stronger-search task (depth/drafts), not a deck-knowledge issue.

Floor investigation (raising bad-deal scores):
- Mechanics audit (Oceania): nectar-spend-for-extra (egg/food/card) and
  spent-nectar per-habitat scoring are CORRECTLY modeled & generated. The
  reset_tray ("flip the tray") action was BUGGED: execute_draw_cards cleared the
  3 face-up cards but never refilled, forcing a blind deck draw -> reset was
  strictly bad and never used. FIXED (actions.py): discard 3 -> deal 3 fresh.
  Correctness proof: video-replay goldens' scripted-player divergence_count
  {2,0,4} -> {0,0,0} (engine now replays 3 real recorded games perfectly).
- BUT fixing the mechanic did NOT raise the floor. Paired n=100 (same seeds,
  heavy honest config r=12 k=10 t=0.3 det), broken vs fixed reset:
  mean 96.6 -> 92.0 (paired diff -4.6, sd 15.6); min 52 -> 51 (unchanged);
  98/100 games changed. The search isn't equipped to use the newly-available
  reset: its rollout policy is the net (trained when reset was unused), so it
  can't exploit a fresh tray; reset gets explored but underdelivers, and the
  added option cascades trajectory changes (slightly net-negative).
- TAKEAWAY: the reset fix is a correctness win (kept), not a free floor-raiser.
  The low tail is largely deal-limited. To actually capitalize on tray-cycling
  the SOLVER must learn it (regenerate search/BC data on the FIXED engine so the
  policy discovers reset-recovery), which is a separate retrain -- and given the
  heavy game-ending search already didn't benefit, the floor may be more
  deal-bound than the "humans always recover to 80+" intuition suggests.

Engine-correctness pass + retrain (matching the real Oceania board, per user):
FIXES (all tested, goldens updated, divergence stays 0):
1. reset_tray/feeder: refill the tray/feeder with fresh cards (was cleared, never
   refilled -> reset strictly bad).
2. reset payable with ANY food incl. NECTAR (was food-only); nectar spent on any
   action bonus (reset or wetland egg|nectar extra) now recorded as nectar_spent
   in that habitat row so it scores; move generator emits '[pay nectar]' variants
   so the SEARCH chooses food-vs-nectar.
3. round-goal scoring: a player with 0 of the goal no longer places (was wrongly
   getting the 2nd-place point). Tiers (4/1/0..7/4/3) and tie-division already ok.
   (Nectar majority 5/2/tie3/0 and the 7-component total were already correct.)
RETRAIN: regenerated 800-game opp-aware BC dataset on the corrected engine ->
opp_aware_net_v2.npz (2443-dim, val_acc 0.669).
FLOOR RESULT (heavy honest r=12 k=10 t=0.3 det, vs heuristic, n=100):
  net v2 / corrected engine: mean 90.4, min 59, median 90, <70=9%, >=100=26%.
  refs (solo_net): broken-engine 96.6/min52 (40% >=100); reset-fix-only 92.0/min51.
Floor lifted modestly (min 51->59, fewer sub-70 games) BUT mean dipped -- a net
confound: the BC opp-aware net is weaker at the TOP than solo-optimized solo_net
(26% vs 40% >=100). Can't cleanly separate engine-fix floor gain from net quality.
HONEST VERDICT: correct mechanics + a net trained to use them nudged the worst
hands up a little, but did not rescue them -- the bad-deal floor is largely
deal-bound. The real deliverable of this pass is a CORRECT engine.
TODO for a clean floor verdict: run solo_net on the fully-corrected engine (same
config) to isolate engine-fix effect from net quality.

CLEAN FLOOR VERDICT (same net solo_net_spread, same heavy-honest config + seeds,
only the engine differs):
  broken engine:          mean 96.6, min 52, bottom-10 ~52-61
  fully-corrected engine: mean 94.4, min 68, bottom-10 [68,68,68,71,74,78,78,79,80,80], <70=3%
**The ENGINE-CORRECTNESS FIXES RAISED THE FLOOR ~16 pts at the min (52->68);**
mean is flat (within noise). Correct mechanics (flip the tray to dig out of a bad
hand; burn use-it-or-lose-it nectar on resets/extras for free scoring) let the
search RECOVER bad deals -- confirms the user's intuition. The retrain (net v2)
was NOT the win and is a weaker mover (90.4/min59); use solo_net_spread on the
corrected engine. Deliverable: corrected engine + solo_net_spread = honest ~94
mean, floor ~68.

Solo-flywheel retrain on the CORRECTED engine (the mean-lift attempt) -- NULL:
Regenerated 2000 best lines on the corrected engine (mean 81.4 vs old 78.7),
built 181k-sample BC dataset, trained solo_net_corrected.npz (1693-dim).
Measured heavy honest (r=12 k=10 t=0.3 det) vs heuristic, n=100:
  solo_net_corrected: mean 92.3, min 65, 24% >=100
  solo_net_spread:    mean 94.4, min 68, 28% >=100  (still best)
Retrain is marginally WORSE -- matches the documented flywheel-iter-2 finding:
at heavy search budget the SEARCH dominates the policy prior, so a better/newer
prior barely changes (or slightly perturbs) the searched output. BOTH retrains
this session (BC opp-aware net v2, and this solo flywheel net) failed to beat
solo_net_spread. CONCLUSION: the net prior is NOT the lever at heavy search.
~94 mean / floor ~68 is near the honest ceiling at this budget; lifting the MEAN
further needs more SEARCH (depth/drafts/rollouts), not a new net. Keep
solo_net_spread on the corrected engine as the deployed mover.

## Promo UK Pack (25 new birds) — engine tier list
Added the promoUK set (471 birds total). Solo tiering over 1500 seeds (150
games/seed, corrected engine), ranked by appearances-in-best-line-drafts x avg
game score:

TOP (brown tuck/gain engines + high-value white) — the engine's favorites:
  European Shag 14x/84.8, Sandwich Tern 12x/82.8, Western Capercaillie 12x/80.3,
  Ruddy Turnstone 11x/83.5, European Golden Plover 10x/83.0, Lesser Spotted
  Woodpecker 8x/81.1, European Greenfinch 7x/81.6. Manx Shearwater 7x but HIGHEST
  avg 86.9 (tuck-per-star-nest). Eurasian Blackcap 3x/86.3 (low count, high avg).
MID: Whooper Swan, European Stonechat, Dartford Warbler, Meadow Pipit, Tawny Owl.
LOW (engine avoids): all 4 PINK birds (Eurasian Blue Tit 1x/62, Tundra Swan 2x,
  Song Thrush 3x, Eurasian Skylark 4x) + weak teal (Sand Martin 2x, W. Jackdaw 4x).

KEY READS / CAVEATS:
- The clear pattern: brown "when activated" tuck/gain engines win; pink birds are
  near-useless HERE because this is SOLO (no opponent to trigger them) -- in
  2-player the 4 pink birds (gain when opp gains seed/invert, react to opp
  draw/bonus) would be far better, so the solo tiering UNDERRATES them.
- Western Yellow Wagtail (bonus-card doubler) and Marsh Warbler (copy yellow) are
  currently NoPower approximations, so they're underrated (esp. the Wagtail).
- Sandwich Tern's power auto-parsed approximately (FlockingPower tuck; ignores the
  give-fish/lay-egg clauses) yet still ranks high on the tuck value alone.
- None crack the very top tier (Nutcracker ~99, Blue-Magpie ~98); the best promo
  birds are solid mid-tier engine pieces (~80-87 avg lines).

## Promo UK — 2-PLAYER tier list (4000 player-games, vs solo)
Played real 2-player rollout-search games (both seats), ranked promo birds by
appearances x avg player score (mean player score 84.2). Pink "react-to-opponent"
birds, dead last in SOLO, jump to the TOP -- confirming the solo tiering
structurally underrated them:

| bird (color) | SOLO appx/avg | 2-PLAYER appx/avg |
| Eurasian Skylark (pink) | 4x / 74.8  | **105x / 108.7** (highest avg!) |
| Song Thrush (pink)      | 3x / 75.0  | 53x / 97.5 |
| Eurasian Blue Tit (pink)| 1x / 62.0 (last) | 17x / 95.0 |
| Tundra Swan (pink)      | 2x / 70.0  | 4x / 78.5 (still weakest pink) |
| Western Capercaillie(w) | 12x / 80.3 | 111x / 91.5 (most-played) |
| European Shag (brown)   | 14x / 84.8 | 83x / 89.5 |
| Western Yellow Wagtail(y, now impl) | 5x / 78.2 | 33x / 87.0 |

TOP 2-player promo birds: Eurasian Skylark (bonus-draw-when-opp-draws-bonus),
Western Capercaillie (draw 2 bonus keep 1), Song Thrush + Eurasian Blue Tit
(gain food when opp gains it), and the brown tuck/gain engines (Shag, Golden
Plover, Sandwich Tern, Ruddy Turnstone). The properly-implemented Wagtail
(bonus-doubler) lands above average (87.0).
CAVEAT: appearances also reflect deal frequency + playability; avg score is the
cleaner "is it good" signal. Low-appearance birds (Tundra Swan 4x, Willow Warbler
6x) have noisy avgs.
TAKEAWAY: the engine LOVES bonus-card synergy in 2-player (Skylark + Capercaillie
top the list) and the opponent-reactive pink birds are genuinely strong -- play
them in 2-player, skip them solo.

## Depth-2 lookahead in 2-player search — NULL/NEGATIVE
Added opt-in depth-2 lookahead (--depth/--branch): after a root candidate, step
opponents one turn, then search my next move (branch-wide) before rolling out.
Equal-budget A/B vs heuristic, n=100, current engine (incl. promo birds):
  depth-1 (r8 x k10, width):            mean 92.3, min 63, max 159, 35% >=100
  depth-2 (r4 x k6 x branch-3, lookahead): mean 89.3, min 58, max 133, 22% >=100
Depth-2 is WORSE at equal compute. The 1-ply rollout already plays to game end
(captures long-horizon value), so explicit 2nd-move search is largely redundant
with it but steals rollout quality (8->4 rollouts, 10->6 width). Spend budget on
ROLLOUTS + WIDTH, not depth. Matches the solo sweep (depth saturates).

SEARCH LEVERS — final scorecard (2-player honest agent):
  + averaged stochastic rollouts: significant (+5, p=0.004)
  + determinization (honest play): ~free (~1 pt)
  - differential/denial objective: hurts (40% vs 60%)
  - net retraining (3 attempts): null (search dominates prior)
  - depth-2 lookahead: hurts at equal budget
Best config: depth-1, rollouts 8-12, top_k 10, temp 0.3, determinize -> ~92-95
honest mean. This is the practical ceiling for the rollout-search approach;
going higher needs a fundamentally better evaluator or far more absolute compute.

## Corrected-engine re-measure + config sweep (2-player honest, vs heuristic, n=100)
Every search config above was tuned on an EARLIER engine. Re-ran the baseline + a
6-cell config sweep on the LATEST fully-corrected engine (471 birds with exact
per-clause powers; tray/feeder reset deals fresh cards; reset payable with scoring
nectar; round-goal "must qualify >=1"; real bird discard pile; per-habitat
action-cube counting). Model solo_net_spread, honest --determinize, the SAME 100
seeds (0-99) across all configs (so comparisons are paired). Modal, 50 shards/config,
~7 min each.

| config (rollouts/top_k/temp) | mean | floor | median | %>=100 | wins/100 |
|---|---|---|---|---|---|
| r8  k10 t0.3 (task baseline)  | 89.3 | 61 | 90.0 | 17% | 88 |
| **r12 k10 t0.3 (best cell)**  | **91.7** | 64 | 90.5 | **28%** | **91** |
| r8  k10 t0.5                  | 90.4 | 58 | 91.5 | 24% | 82 |
| r12 k10 t0.5                  | 90.8 | **68** | 90.0 | 21% | 85 |
| r8  k8  t0.3                  | 89.2 | 67 | 88.0 | 18% | 87 |
| r12 k8  t0.3                  | 91.2 | 61 | 91.0 | 25% | 89 |
(all p<1e-4 vs the heuristic; heuristic mean ~68-70 throughout.)

READS:
- ROLLOUTS is the only directionally-consistent lever: r12 >= r8 on mean at all
  three (k,t) pairs (+2.4 / +0.4 / +2.0), and on win-rate and %>=100, at NO floor
  cost. top_k 10-vs-8 and temp 0.3-vs-0.5 are within-noise wobbles (k10 ~ k8;
  t0.3 best at r12, t0.5 best at r8 -- no clean winner).
- BUT the headline r12-vs-r8 lift is SUB-NOISE. Paired over the same 100 seeds the
  mean diff is only +2.35 (sd 14.8, t=1.59); sign test 51 up / 43 down / 6 tie,
  p=0.47. Per-seed variance swamps the shift. All six cells cluster in 89.2-91.7
  mean -> the grid is SATURATED; no config "clearly beats" another at n=100.
VERDICT: the corrected engine REPRODUCES the prior best region (rollouts 8-12,
top_k 10, temp 0.3, determinize) and its ~90-92 honest ceiling -- the re-tune did
NOT shift the optimum. r12 k10 t0.3 is the best-supported cell (91.7 / floor 64 /
28% break 100, 91/100 wins) and stays the recommended deployed config, but as
"directionally best within noise," not a significant gain over r8. The fresh
corrected-engine honest baseline to quote is r12 k10 t0.3 = ~92 mean / floor ~64 /
28% >=100 (seeds 0-99, n=100); this sits a touch below the older 94.4/68 figure,
consistent with the additional recent fixes (discard pile, action-cube counting)
making the game slightly more accurate/harder -- not a regression to chase.
NO code-default change adopted: the sweep win is sub-noise, and the two_player.py
argparse defaults feed the dataset/selfplay modes too, so perturbing them isn't
warranted. The lever for a real ceiling lift remains a learned search-quality
VALUE evaluator (AlphaZero-style), not more rollout-search tuning.

## Value-net GATE (AlphaZero-of-search-value, step 1 -- backend/ml/value_gate.py)
Before building any value self-play loop, ran the cheap go/no-go gate: is a
learned V(state) -> expected rollout return a better SINGLE-SHOT estimator of a
leaf's value than one actual playout? Logged, as a free byproduct of real
heuristic-mode agent play (search seat vs heuristic, best config k10/t0.3/det,
M=16 net-sampling determinized rollouts per candidate), every candidate leaf
state + its 16 rollout returns. 150 games on Modal (50 shards, ~18 min) ->
38,967 leaf states (dim 1693, M=16). Trained ridge + a small torch MLP on 80% to
predict the per-state mean return; gate metric on held-out 20%:

  predict-global-mean RMSE   = 15.9   (spread of true state values; no-info base)
  one-playout sigma (per-st) =  7.13  (RMSE of ONE rollout vs the state's truth)
  ridge V RMSE (raw)         =  4.70
  MLP   V RMSE (raw)         =  4.00  -> 3.58 bias-corrected for label noise s/sqrt(M)
  R* = (sigma/V_rmse)^2      =  3.96  ->  GREEN

READS:
- V explains ~95% of the variance in state value (R^2 = 1-(3.58/15.9)^2 = 0.95).
  The encoder + a tiny MLP already capture almost all of what the rollout
  estimates -- the evaluator is NOT at the information ceiling; it is learnable.
- One V-eval is worth ~4 full playouts of ACCURACY, at ~1/300 the cost (<1 ms vs
  ~4x80 ms). That is the compute lever the rollout-tuning grid could not move.
- HONEST limit: the current leaf AVERAGES rollouts=12 playouts, so its estimate
  has RMSE ~ sigma/sqrt(12) = 2.06 -- still sharper than a single V (3.58). So V
  does NOT by itself beat the 12-rollout leaf on accuracy; its win is COMPUTE
  (4x signal/playout) -> reinvest into more candidates/depth, or (better) use V to
  BOOTSTRAP truncated rollouts (roll H plies + V(leaf)) to undercut 12 raw
  rollouts' variance at lower cost. The MLP is a 40-epoch generic net, so 3.58 is
  a conservative upper bound on V error; a tuned/larger V will push R* higher.
VERDICT: GREEN -- proceed to step 3. Wire V into the search leaf as a depth-H
truncation bootstrap in two_player.make_search_chooser (replace _rollout_value's
play-to-end with H plies + V), then re-run the SAME n=100 honest harness at
MATCHED compute and compare mean/floor/%>=100 to the 91.7 baseline. A real win =
beats 91.7 at equal-or-less compute. Only then close the loop (agent-with-V plays
-> relabel -> retrain V). Dataset: reports/ml/value_gate/data.npz (force-added).
