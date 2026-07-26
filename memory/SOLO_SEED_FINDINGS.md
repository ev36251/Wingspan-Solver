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

## Value-net STEP 3 -- bootstrap leaf in live search (n=100, honest, same seeds)
Trained V (reports/ml/value_gate/value_v1.npz, val RMSE 4.0) and wired it into a
depth-1 search whose LEAF is V instead of a play-to-end rollout (value_gate.py
make_eval_chooser; leaf_mode v = roll H plies then V, H=0 = pure V; leaf_mode full
= the rollout baseline). Ran the agent vs heuristic, seeds 0-99, top_k 10, t0.3,
recording score AND wall-time/game. Full curve:

| leaf config              | mean | floor | %>=100 | s/game |
|--------------------------|------|-------|--------|--------|
| pure V (H0)              | 74.3 |  25   |  4%    |   1.1  |
| boot H8  M3              | 80.9 |  46   |  7%    |  30    |
| boot H16 M3              | 83.0 |  59   | 10%    |  50    |
| boot H8  M8              | 85.0 |  53   | 10%    |  77    |
| boot H16 M8              | 87.6 |  58   | 11%    | 133    |
| boot H24 M12             | 90.0 |  64   | 23%    | 234    |
| FULL rollout M12 (ctl)   | 93.3 |  70   | 22%    | 217    |

(full control through THIS harness = 93.3, reproduces the two_player 91.7 baseline
within noise -> no harness drift. All beat the heuristic p<1e-4 except pure V at
p=0.018.)

RESULT -- NEGATIVE for strength, POSITIVE for speed:
- The V-bootstrap is DOMINATED by full rollouts across the ENTIRE compute curve.
  At MATCHED compute (~220s, both M=12) full=93.3 vs boot=90.0: paired diff +3.31
  (t=2.24, sign p=0.064) in favor of full. The V curve only climbs toward full by
  adding more REAL simulation (bigger H / more M) -- i.e. by relying on V LESS.
- Pure V cratered (74.3, floor 25; wider top_k=40 made it WORSE, 27 on a probe).
  Cause = OPTIMIZER'S CURSE / distribution shift: argmax-over-candidates-by-V
  systematically selects states where V over-estimates. A rollout is a NOISY but
  UNBIASED sample, so argmax-over-rollout-means is safe; argmax-over-V is biased
  toward V's blind spots. The gate (passive RMSE) cannot see this; only live
  search can. This is exactly the failure AlphaZero's PUCT-MCTS is built to avoid.
- So this V is a SPEED lever (a ~85-pt agent at 1/3 compute, ~90 at matched, and
  it still beats the heuristic even at 1.1 s/game), NOT a ceiling lever.

WHAT WOULD ACTUALLY BEAT 93: (a) PUCT-MCTS with V as leaf + the policy net as
PRIOR (mcts.py already exists) -- the prior + visit-count averaging + backups make
it robust to the argmax exploitation that sank greedy-V; this is the real
AlphaZero recipe and the honest next experiment. (b) DAgger: retrain V on the
states V-search actually VISITS (not just on-policy agent states) to kill the
distribution shift, then re-test the bootstrap. (c) Accept ~92-93 as the practical
ceiling and move the lever elsewhere (richer encoder / move-gen / opponent model).
Recommendation: try (a) PUCT-MCTS before (b); if neither clears 93 at matched
compute, the ceiling is not the evaluator. Artifacts: value_v1.npz (force-added),
compare harness in value_gate.py.

## Value-net STEP 4 -- PUCT-MCTS with V leaf + policy prior (n=100, honest)
Built a determinized, score-maximizing PUCT (value_gate.py make_mcts_chooser):
policy-net softmax priors, our V at leaves, net-sampling opponent in-tree, per-sim
deck reshuffle, MinMax-normalized Q in the PUCT term, most-visited root action.
This is the AlphaZero recipe meant to survive the optimizer's curse that sank
greedy-V. (Inference uses Dirichlet root noise = 0; an initial run left it at 0.25
and was re-run noise-free.)

| MCTS config              | mean | floor | %>=100 | s/game |
|--------------------------|------|-------|--------|--------|
| n_sims 400 (noise 0.25)  | 68.2 |  27   |  2%    | 191    |
| n_sims 800 (noise 0.25)  | 70.0 |  36   |  3%    | 365    |
| n_sims 400 (noise 0)     | 71.6 |  45   |  1%    | 195    |
| n_sims 800 (noise 0)     | 74.0 |  30   |  2%    | 377    |
| (ref) bootstrap H24 M12  | 90.0 |  64   | 23%    | 234    |
| (ref) FULL rollout M12   | 93.3 |  70   | 22%    | 217    |

RESULT -- MCTS is DECISIVELY WORSE than rollout search here. Even noise-free and
at 1.7x full compute (377s) it tops out ~74 (about pure-V level), barely beating
the heuristic (60% wins). It scales with sims (72->74) but nowhere near 93.
WHY (structural, not a tuning miss): Wingspan has a HIGH BRANCHING FACTOR (often
20-50+ legal moves/decision) and an EXPENSIVE simulator (each sim replays the
descent: my moves + opponent turns from root). At the ~400-800 sims affordable
within full's compute, the tree is barely populated -- most nodes get <=1 visit,
so the most-visited root action collapses back to "policy prior + a thin layer of
V lookups." Rollout search spends the SAME compute on 120 full-game evaluations
that directly, unbiasedly estimate each candidate move's value -> far more
sample-efficient for this structure. MCTS shines when sims >> branching and the
simulator is cheap; both are false here.

## OVERARCHING CONCLUSION -- a learned value does NOT beat rollouts in this game
Three integration methods, all n=100 honest at matched compute vs the 93.3 rollout
baseline:
  - greedy V leaf (pure)      : 74.3  (optimizer's curse)
  - V bootstrap (H plies + V) : 90.0  (dominated; speed lever, not strength)
  - PUCT-MCTS (V + prior)     : 74.0  (sample-inefficient at this branching)
The passive gate was GREEN (V is accurate, R*~4, 95% of variance) yet NONE of the
active uses beat full rollouts. The rollout's unbiased full-game estimate is hard
to beat under selection/argmax pressure at an affordable sim budget. So ~92-93 is
NOT an evaluator/search-method problem fixable by a value net. The remaining
levers are structural: (1) richer state encoding / better features, (2) better or
PRUNED move generation (shrink the branching factor -> would also make MCTS
viable), (3) opponent modeling, or (4) accept ~93 as near the practical ceiling vs
this heuristic. The value net's one real use is SPEED (a ~85-pt agent at 1/3
compute via bootstrap; beats the heuristic even at 1.1 s/game), not the ceiling.
All harnesses + value_v1.npz live in value_gate.py / reports/ml/value_gate/.

## Branching-factor / move-pruning investigation (profiled, NOT a lever for rollouts)
Hypothesis: shrink the move set to speed search + make MCTS viable. Measured on
308 real decision states (medium-policy self-play):
  - moves/decision: mean 53.7, median 42, p90 108, MAX 232.
  - by action type (totals): GAIN_FOOD 10006, DRAW_CARDS 4602, LAY_EGGS 1347,
    PLAY_BIRD 581. The explosion is GAIN_FOOD (feeder food-type combos; mean 32,
    max 173/decision) and DRAW_CARDS, NOT bird-play payment variants (only 1.26x
    inflation over distinct (bird,habitat,slot)).
WHY PRUNING WON'T HELP THE DEPLOYED AGENT: profiled the rollout hot path. Each
rollout ply costs a FIXED encode+forward = 0.95 ms; score_move is ~0.009 ms/move
(negligible). Capping ~54 moves -> 16 changes a ply from ~1.43 to ~1.09 ms =
~1.3x rollout speedup at best (more at the rare 173-move states, ~1x at small
ones). Since r8->r12 rollouts (1.5x more) was already a sub-noise +2.4, a ~1.3x
speedup buys <~1 pt -> not worth the complexity/risk. The rollout bottleneck is
the per-ply NETWORK forward (encoder dim 1693), not the branching factor; the
top-level decision is ALREADY prior-pruned (top_k=10, the 93.3 config).
CONSEQUENCES: (1) move-gen pruning is a dead lever for rollout search. (2) It
WOULD help MCTS (thin tree <-> high branching) but MCTS is dead for other reasons
(step 4). (3) The only way pruning matters is if combined with a fundamentally
cheaper per-ply eval -- i.e. the real rollout speed lever is a SMALLER/faster
encoder+net or encode caching, not fewer moves. Net: confirms ~92-93 is not
reachable-past by search-efficiency tweaks; the open levers remain structural
(richer-but-cheaper encoder, opponent modeling) or accept the ceiling.

## Deterministic re-sweep + draft/engine-feature investigation (post hash-seed fix)
With games now reproducible (PYTHONHASHSEED pinned), re-ran the config sweep and
investigated two suspected levers. All three came back NEGATIVE.

CONFIG SWEEP (n=100, deterministic, vs heuristic):
  r8/k10/t0.3=90.1  r12/k10/t0.3=91.0  r16/k10/t0.3=91.8  r12/k10/t0.5=89.8
  r12/k8/t0.3=91.5  (r8/k10/t0.5 lost to a Modal hiccup)
  The means trend up with rollouts, BUT it is NOT paired-significant even with
  hash noise removed: r16-r8 = +1.68 (t=1.11, sign p=0.42, 44up/53dn); r12-r8
  p=0.76. Per-seed it's a coin flip. -> config tuning is a DEAD lever; the search
  is genuinely saturated at ~91, not noise-hidden. Clean baseline: r12/k10/t0.3
  = 91.0 (deterministic, reproducible).

DRAFT (#4): measured kept-vs-played across deterministic games. 2.58 birds kept,
  1.88 played to board, ~0.6 used as tuck/discard fuel, only 0.10 (4%) wasted
  (kept then left in hand all game). The draft is already efficient (analyze_setup
  is Monte-Carlo reranked + prefers 2-3 birds). No over-keeping; no change.

ENGINE FEATURES (#5): the 2-player agent's rollouts use the NET
  (make_net_sampling_chooser), NOT heuristics.py. So adding engine-awareness to
  the heuristic only reaches the SOLO agent (simulate_playout) and the heuristic
  opponent -- not the deployed 2p agent. And the heuristic already phase-scales
  engine_value/early_game_engine_bonus. Giving the 2p agent engine-awareness
  requires retraining the NET (overlaps #1/#3), not a heuristic tweak.

NET: the consistent session result holds -- search dominates, ~92 is the real
ceiling vs this heuristic, and the only remaining lever is a better learned
policy/value (expert iteration / opponent-aware retraining), which is a real
project with genuine null-risk, not a quick win.

## Net-retraining bet: existing opponent-aware nets evaluated (NULL, 4th confirmation)
Committed to "retrain the net." Found the opponent-aware BC pipeline already
exists (two_player_dataset.py: search teacher + opponent-board encoder, dim 2443)
AND two trained models (opp_aware_net.npz, opp_aware_net_v2.npz) -- never cleanly
evaluated (solo_net_spread still deployed). Evaluated them first.

Deterministic n=100 vs heuristic, r12/k10/t0.3, same seeds (paired):
  solo_net_spread (baseline) : 91.0
  opp_aware_net   (v1)       : 88.9  (paired -2.09, sign p=0.36)
  opp_aware_net_v2(v2)       : 90.2  (paired -0.87, sign p=0.61)  92% win rate
Both statistically EQUAL to the solo net (not better; v1 slightly worse).

This is the 4th independent confirmation that retraining the policy net does NOT
move the full-search 2p agent: (1) solo BC flywheel iter2 flat, (2) opp-aware v1,
(3) opp-aware v2, plus the value-net work (greedy-V 74, bootstrap 90, MCTS 74 --
all <= rollouts). ROOT CAUSE: at the deployed budget (top_k=10 x 12 rollouts) the
SEARCH dominates the policy prior, so a different/better prior washes out. Adding
opponent-board features (the one genuinely-new lever) did not help either.

CONCLUSION: the ~92 ceiling is a property of (rollout search + this game + this
heuristic opponent), not the policy/value net. Both halves of the AlphaZero
recipe (policy prior, value leaf) are confirmed dominated by the rollout search
here. Raising the ceiling is NOT an ML-on-top-of-search problem. A full
multi-iteration self-play loop with soft visit targets remains technically
untested but is low-probability given 4 nulls + the clear mechanism. Practical
ceiling vs this opponent is ~92; the agent is strong and now reproducible.

## Soft-target self-play loop: iter-1 net evaluated (NULL, 5th confirmation)
Built the self-play / expert-iteration loop the prior note flagged as the one
"technically untested" lever (selfplay_soft.py): both seats play the deployed
strong search (rollouts>1, determinize, temperature), and instead of argmax we
log the search's value-weighted softmax over the top_k candidates as factorized
SOFT policy targets -- the AlphaZero "improved policy" signal none of the earlier
hard-target BC attempts used. Generated soft_iter1.jsonl (15,600 rows, dim 2443,
opponent-board encoder) on Modal, trained soft_net1.npz (net_1), set an explicit
fail-fast gate before evaluating.

FAIL-FAST GATE (deterministic n=100 vs heuristic, r12/k10/t0.3, same seeds, paired):
  solo_net_spread (baseline) : 91.0
  soft_net1       (net_1)    : 86.0  (heuristic 70.1; win rate 80/100)
  paired diff = -4.98 (t=-2.90), 36 up / 62 dn, sign-test p=0.011
net_1 is SIGNIFICANTLY WORSE than the 91.0 baseline -> FAILS the gate decisively.
Per the pre-registered rule (beat 91.0 paired-significant -> iterate to net_2;
else stop), the loop STOPS at iter 1. Did NOT iterate to net_2.

Likely contributors: only 15,600 training rows for a 2443-dim net (val_acc ~0.59),
and -- the recurring mechanism -- the search dominating the prior so a noisier
prior actively hurts. This is the 5th independent null on net retraining
(after: solo BC flywheel iter2, opp-aware v1, opp-aware v2, value-net arc).

FINAL CONCLUSION: every branch of the AlphaZero recipe has now been tried and
confirmed dominated by the rollout search for this game/opponent -- policy prior
(hard BC x3, soft self-play x1), value leaf (greedy-V, bootstrap, MCTS), opponent
features, draft tuning, config re-sweep, move pruning. The ~91-92 ceiling vs this
heuristic is a property of (rollout search + game + opponent), not the net.
Raising it is not an ML-on-top-of-search problem. The deployed solo_net_spread
agent at r12/k10/t0.3 = 91.0 is the practical, reproducible champion. Stop here.

## Beyond the net: budget / algorithm / opponent sweep (the FIRST real lever)
After 5 net-retraining nulls, tested the three levers that are *different in kind*
from "another net": more search budget, a different search algorithm, a stronger
opponent. All n=100 deterministic vs heuristic, seeds 0-99, paired sign-test vs
the leaf=full k10/r12/t0.3/det baseline (which re-measured at mean 93.2 here).

  config                      mean   dVS base   up/dn   sign-p
  baseline  k10/r12           93.2     --        --       --
  A budget  k8/r6  (low)      87.5    -5.70     32/64    0.001 *
  A budget  k16/r24 (high)    96.2    +3.00     58/39    0.067
  A budget  k20/r36 (xhigh)   96.3    +3.11     59/40    0.070     (PLATEAU)
  B depth-2 lookahead         85.8    -7.33     24/75    0.000 *  (worse)
  B MCTS n_sims=200 (V leaf)  70.0   -23.18      7/92    0.000 *  (ties heuristic)
  C selfplay r12 (vs r1 opp)  91.6    -1.61     48/50    0.920     (robust)
  C denial ablation l=0.3     80.5      n/a       --      --       (see below)

KEY FINDING -- SEARCH BUDGET IS THE ONE LEVER THAT SCALES (TO A PLATEAU). The
ladder rises then flattens: r6=87.5 < r12=93.2 < r24=96.2 == r36=96.3. r24 and
r36 are statistically identical, so the diminishing-returns knee is ~k16/r24
(+3.0 over default); pushing to r36 (1.5x compute) buys +0.1 = nothing. The low point is significantly worse
(p=0.001) and the high point trends better (+3.0, 58/39, p=0.067). Independently
corroborated by selfplay: the strong agent (r12) beats a weak-search opponent
(r1) 66/100 (p=0.001). Three signals agree -- raw rollout budget buys strength,
at ~linear compute cost. This is the first thing in the entire project that moves
the number after ~6 net/value/algorithm nulls.

ALGORITHM (B) -- NULL/WORSE. A different search *shape* does not help:
  - depth-2 minimax-over-own-moves (k8/r2/b3): 85.8, significantly worse.
  - PUCT-MCTS with the trained V as leaf (n_sims=200): 70.0, barely ties the
    heuristic. Consistent with the RED value-gate -- V is a weak leaf, so any
    algorithm that leans on V instead of full rollouts collapses. Depth-1 rollout
    search remains the best shape; the leaf quality (full playout) is what matters.

OPPONENT (C) -- AGENT IS ROBUST; DENIAL STILL NULL. Against a real search
opponent (not the weak heuristic) the agent still scores 91.6, statistically
equal to its 93.2 vs-heuristic score (paired 48/50, p=0.92). So the ~92-96 level
is NOT an artifact of a weak opponent -- contention does not crush it. Denial-
awareness remains null: WITHIN the ablation run, differential (l=0.3) vs selfish
scored 80.5 vs 80.1 (55/100, p=0.18). (The -12.6 "dVS base" for the ablation row
is an apples-to-oranges pairing -- that config is a cheap single-rollout agent
playing a strong opponent, not comparable to the r12-vs-heuristic baseline.)

PRACTICAL CONCLUSION: to make the agent stronger, spend more rollouts -- it scales
monotonically and predictably. The net, the search algorithm, and the objective
are all saturated; budget is the free dial. Deployed default stays k10/r12 (93.2,
the speed/strength knee); bump to k16/r24 for ~+3 when compute allows. Higher
budget likely keeps helping with diminishing returns (untested above r24).

### Selectable strength presets (wired)
Because budget is the scaling lever, the 2p agent now takes a named `--strength`
preset (backend/ml/two_player.py BUDGET_PRESETS, shared into value_gate compare):
  default = k10/r12 (93.2, speed/strength knee)
  strong  = k16/r24 (96.2, ~+3, ~2x compute) <- value pick, the plateau knee
  max     = k20/r36 (96.3, == strong; highest measured, no gain for ~1.5x more)
Each pins temperature=0.3 + determinize (the config the numbers were measured at).
  python -m backend.ml.two_player --mode heuristic --seeds 0-99 --strength strong --use-modal
  python -m backend.ml.value_gate compare --seeds 0-99 --leaf-mode full --strength max --use-modal

## Headroom + value-bootstrap (is the engine improvable? what's fast?)
Two follow-up experiments to decide whether to keep improving the engine.

HEADROOM (is the agent near-optimal?). Solo regime, 40 seeds: compared a one-shot
deployed agent (top draft only, cheap depth-1 search k6/r1) to the heavy
multi-draft best line per deal (best_lines_4000_spread, the "ceiling"):
  agent one-shot : mean 73.3  median 72
  ceiling (best) : mean 78.4  median 78
  regret         : mean +5.2  median +6  (7% of ceiling); within 3pts on 38% of seeds
And this OVERSTATES the gap -- the agent here is deliberately cheap (k6/r1/1-draft)
vs the deployed K10/R12 + draft search, which would close most of it (true headroom
~2-3 pts). Several seeds the one-shot agent BEAT the stored ceiling, confirming the
"ceiling" is near the real optimum and the agent is close to it. CONCLUSION: the
engine is near its achievable ceiling -- consistent with the 2p budget plateau
(R24==R36). Chasing more engine points is low value; the product is the frontier.

VALUE-BOOTSTRAP LEAF (what's fast?). 2p vs heuristic, n=60, measured score + s/game:
  leaf=full (roll to game end), k10/r12        : 92.7  | 245 s/game
  leaf=v (roll 8 of my turns, then value net)  : 82.0  |  31 s/game
The truncated-rollout + value-net leaf is ~8x FASTER but 11 pts weaker, and the
bottleneck is the value net's quality (value_v1 is a known-weak leaf, RED gate).
So the value-bootstrap MECHANISM is the right lever for SPEED -- the missing piece
is a better-trained value net. This is the highest-value remaining ML work, but
its payoff is the PRODUCT (fast strong evaluation ~30s instead of 245s -> strong
score numbers + a snappy "deep" mode), not a higher score ceiling.

NET TAKEAWAY: stop chasing the score ceiling (it's near-optimal). The one ML lever
worth pursuing is a better value net to make the value-bootstrap leaf fast AND
strong -- which fixes the app's latency + weak-numbers problems, not the ceiling.

## Better value net: trained, but the bootstrap payoff is small (honest null-ish)
Acted on "train a better value net to make the value-bootstrap leaf fast AND
strong." Findings:

ESTIMATOR improved cleanly. A capacity sweep on the SAME data (39k states, M=16):
  net           val_rmse   gate R* (playouts worth)
  (256,64) v1     3.94        4.1
  (512,128)       3.65        5.0   <- chosen -> value_v2.npz (val_rmse 3.52)
  (768,256)       3.68        4.9
  (1024,256,64)   3.57        5.3
So value_v1 was just undertrained/too small; value_v2 is a strictly better
estimator (no new data needed -- and since capacity helped the estimator but NOT
play (below), more DATA won't help play either).

PLAY barely moved. value-bootstrap leaf, 2p vs heuristic, n=60 (score | s/game):
  full rollout (roll to end)        92.7 | 245
  value_v1, boot-8/r3               82.0 |  31
  value_v2, boot-8/r3               83.2 |  32   (+1.2 from the better net)
  value_v2, boot-12/r3              84.4 |  42   (+1.2 more from rolling further)
The value-bootstrap evaluator is ~6-8x FASTER but plateaus ~8 pts below full
rollout, and a better net + more bootstrap plies each buy only ~1 pt. ROOT CAUSE:
the net isn't the bottleneck -- inside search its residual errors get exploited
(optimizer's curse), and final-score-from-midgame variance is partly irreducible.

CONCLUSION: value_v2 is a better artifact (keep it), but value-bootstrap is a
"6x faster / ~8 pts weaker" tradeoff, NOT the hoped-for fast-AND-strong leaf.
Don't invest further in value nets or more value data. The app keeps full-rollout
quality for moves; if a fast "quick look" mode is wanted, value-bootstrap (~84)
is an option but won't match the strong agent.

## The FLOOR lever: draft quality (the first real engine win in a while)
User pushed on raising the worst-case score (salvaging bad starts). Investigation:

WHAT DRIVES SCORE (best-line data, worst vs best deals): board size + eggs + bonus.
  worst 40 deals: 6.0 birds, 8 eggs, 26 bird_vp, 2.6 bonus  (score ~57)
  best  40 deals: 9.4 birds, 16 eggs, 41 bird_vp, 8.0 bonus (score ~116)
On bad deals the agent ALREADY salvages (draws heavily ~8.8 draw-actions/game,
plays into wetland, values tray resets, even built a 47-card tuck engine on the
worst seed to reach 70). So the floor isn't a play-quality bug mid-game.

DRAFT IS THE FLOOR LEVER. The opening (which birds/food to keep) caps the whole
game. Deployed agent commits to ONE opening; searching 4 and keeping the best, on
the 22 worst deals:
  1 opening : mean 75.1  floor(min) 55
  4 openings: mean 81.0  floor(min) 65
  -> +5.9 mean, 17/22 improved (max +25), and the worst-case FLOOR rose 55 -> 65.

DATA CORRECTION: the agent beats the old best_lines_*_spread "ceiling" by 10-40
pts on many seeds -> that spread dataset is NOT a true max, so the earlier
headroom number understated real headroom. Draft search captures some of it.

DEPLOYED FIX: setup_advisor.rollout_draft_evaluation now reranks draft options
with the STRONG hero rollout policy (was the ~40-level "medium"), so the draft
advisor correctly values salvage openings (keep food over bad birds, placeholder
bird in wetland to churn the tray) instead of being fooled by weak playouts.
~12s for top-5 x 10 sims.

FLAG (correctness): the 47-tuck salvage line on seed 13 is suspiciously high and
worth a tuck-power audit -- if a bug it inflates scores; if legit, more headroom.

## Engine-in-draft + tuck audit
ENGINE-IN-DRAFT (deployed). Per user: the powerful net-guided search agent (the
~92 engine), not the heuristic, should make the DRAFT decision (the floor lever).
setup_advisor.rollout_draft_evaluation now takes model/encoder; when present it
plays each candidate opening out with make_search_chooser (hero=net rollout
search, selfish; opponents = cheap heuristic) instead of a heuristic playout.
routes_setup loads the deployed net and uses it by default (use_engine=True),
clamped to top-3 openings x 2 sims to fit a turn (~107s measured). Falls back to
the strong heuristic playout if no net. This is the draft_floor experiment's
configuration (net search + multi-opening) wired into the product, which lifted
the worst-deal mean 75->81 and floor 55->65.

TUCK AUDIT (seed 13, the 47-tuck flag): tucks concentrate on Red-Winged Blackbird
(27) + Common Chiffchaff (9) -- both per-activation card-tuckers in the wetland.
Final score was only 63 (NOT inflated), i.e. the agent traded everything else for
a wetland tuck engine on a bad hand. Looks like a LEGITIMATE tuck-engine salvage,
not a scoring bug. The high single-bird count (27) is plausible for a dedicated
wetland engine but a per-activation count trace would confirm with certainty.

## Draft search raises the MEAN, not just the floor (the biggest real win)
Q: does running the agent over many games with the improved (multi-opening) draft
raise the mean? Experiment: agent over 200 deals (solo), 1 opening vs 6 openings.
  metric      1-draft  6-draft  lift
  mean         77.2     84.9    +7.6
  median       77.0     84.5    +7.5
  floor p5     63       69      +6
  min          57       65      +8
  p25          70       76      +6
  p90          91       99      +8
  improved on 155/200 deals.

The lift is ~UNIFORM across the whole distribution (+6..+8 everywhere, incl. the
top p90) and the MEAN rises the full +7.6 -- NOT diluted as predicted. Reason:
even good hands have a non-obvious best opening, so searching 6 finds a better one
on almost every deal (155/200), not just the bad ones.

This makes the DRAFT the single biggest under-exploited lever in the project --
and a DIFFERENT axis from the in-game rollout budget (which plateaued at ~96 in
2p). Draft search (opening selection) is clearly not saturated: 1->6 openings adds
+7.6 to the mean. Also retroactively explains the bogus "small headroom" reading
(that used a spread dataset, not a true max). Deployed: the setup advisor uses the
net-search engine across N openings (engine_openings, default 4; up to 6 for max
floor at ~35s/opening).

## Draft saturation curve (where does the draft lift level off?)
Best-of-first-k ranked openings, mean over 150 deals (one pass, k=1..8):
  k:    1     2     3     4     5     6     7     8
  mean: 77.6  80.1  82.1  83.4  84.6  85.2  85.6  86.4
  gain:  -   +2.55 +1.97 +1.35 +1.13 +0.62 +0.42 +0.81*
Front-loaded with a long tail. Cumulative lift: 1->4 = +5.8 (~2/3 of the total),
1->6 = +7.6 (matches the broad mean-vs-floor run exactly), 1->8 = +8.8. Marginal
gain drops below ~1 pt/opening after k=4 and never fully saturates by 8 (the +0.81
at k=8 is tail noise). At ~35s/opening, the practical sweet spot is 4-6 openings;
the product default of 4 captures ~2/3 of the lift -- a good speed/quality knee,
with 6 available for max floor.

## Mid-game search IS a lever in the SOLO regime (prediction was wrong)
Probe: does wider/deeper in-game search help on top of a good draft? Fixed
n_drafts=2; varied the solo_search k_schedule over 120 deals:
  depth-1 k8  (current deployed) : mean 80.4
  depth-1 k16 (wider)            : mean 83.2  (+2.8)
  depth-2 k8x3 (deeper)          : mean 84.2  (+3.8)
Both wider AND deeper help, monotonically. This CONTRADICTS the predicted
"saturated" -- but it's a different agent: the earlier ~96 plateau (Exp A budget
ladder, Exp B depth-2 worse) was the 2-PLAYER leaf=full rollout agent vs the
heuristic, already at high budget + opponent contention. THIS is the SOLO agent
(solo_net_spread + solo_search recursive k_schedule), whose deployed config
(k8, depth-1) is genuinely UNDER-searched mid-game.

So the SOLO agent (the one the draft advisor / engine-in-draft uses) has a SECOND
headroom lever beyond the draft: deeper/wider mid-game search (+3-4). Caveats:
(1) it's a compute cost -- depth-2 (8,3) is ~3x/game, k16 ~2x; (2) not yet tested
whether it fully STACKS with the draft lift (likely partial overlap). Actionable:
the deployed solo/draft search could move to k_schedule=(8,3) or (16,) for ~+3,
trading latency. The 2p ceiling claim still stands for the 2p agent; "search is
saturated" was over-generalized from 2p to solo.

## The two levers STACK (draft + mid-game depth) -> +10.3
2x2 over 100 deals (solo agent):
  baseline   (d1, k8 depth-1) : 76.9
  draft only (d4, k8)         : 84.3  (+7.4)
  midgame    (d1, k8x3 depth2): 79.2  (+2.3)
  BOTH       (d4, k8x3)       : 87.2  (+10.3)
Independent sum +9.7; actual +10.3 -> they STACK (slightly super-additive). The
solo agent was under-searched on TWO independent axes (it wasn't searching the
draft at all, and was depth-1 mid-game); fixing both lifts it 76.9 -> 87.2.

DEPLOYMENT REALITY: the cheap/deployed win is the DRAFT lever (+7.4), already in
the engine-in-draft advisor (4 openings, depth-1 playouts). The mid-game depth
lever (+2-3) means deeper playouts (two_player make_search_chooser depth>=2),
which ~triples latency -- worth a "deep" mode but too slow as the default draft
advisor (would be ~105s/opening). These are compute-on-the-right-axes wins (solo,
not 2p): the 2p move-agent remains at its ~96 ceiling. Highest-value follow-up:
regenerate the best-line training dataset with d4+depth2 (now that it scores +10
higher) and retrain the solo net on the stronger lines.

## Retraining the solo net on stronger lines does NOT beat the champion (negative result)
Followed the "highest-value follow-up" above: regenerated 800 best-lines with the
+10 config (4-opening draft + depth-2 mid-game search) -> `best_lines_strong.jsonl`,
mean **87.5** / p10 74 / min 57 (vs the old training lines ~78.6 / p10 66). The
stronger DATA is real. Then retrained the solo net two ways and evaluated on
HELD-OUT seeds 800-959 (outside the 0-799 training range), KS=(8,), n_drafts=4,
vs the deployed `solo_net_spread.npz`:

  attempt                         | held-out mean | floor (p10) | wins/160 | delta
  --------------------------------|---------------|-------------|----------|------
  champion (solo_net_spread)      | 85.6-85.8     | 73          |   --     |  --
  warm-start fine-tune (strong)   | 82.2          | 66          | 49/160   | -3.7
  fresh train ("combined" x3)     | 84.8          | 69          | 71/160   | -0.7

Both LOSE to the champion. The fine-tune regressed hardest (catastrophic
forgetting: 731 strong lines / 19k samples pulled the net off the broad
distribution it learned from 4000 lines). The "combined" retrain was MEANT to mix
the original 4000 spread lines back in to prevent forgetting -- but the build
skipped 16205/18400 lines: **the original `best_lines_4000_spread.jsonl` no longer
replays through `solo_bc_dataset.replay_row`** (its `solo_seed_optimizer`
trajectory/description format differs from the `solo_search` format the replayer
now matches against; only ~2 of 16000 reproduced). So "combined" was effectively
strong-only x3 with duplicate train/val leakage -- not a true mix -- and still
came in -0.7.

CONCLUSION (now empirical, twice): **the solo net is not the bottleneck.** Better
training DATA does not yield a better policy net here, because the move pick is
search-dominated -- the net is only a rollout prior + draft ranker, and the
deployed `solo_net_spread` already priors those well enough that search washes out
policy differences. This matches every prior net-retrain null in this project (5x
in the 2p loop). The champion `solo_net_spread.npz` is UNCHANGED / not replaced.

The actual strength levers remain the two already documented and (for draft)
deployed: draft breadth (+7.4) and mid-game search depth (+2-3), which STACK to
+10. A genuinely stronger net would require regenerating a LARGE broad+strong
dataset entirely in `solo_search` format (so it replays) -- thousands of seeds of
Modal compute -- with no expectation it beats search-dominated play. Not worth it
unless the goal is a stronger STANDALONE (search-free) policy for faster draft
advice. Failed nets (`solo_net_strong.npz`, `solo_net_combined.npz`) are
gitignored and not promoted.

## Distilled fast rollout policy (the "student") — a real 2.4x SPEED lever, not a strength lever
Motivation: after the budget ladder showed rollout COUNT is the only scaling
lever, the remaining question was rollouts-per-second. Profiling the per-ply
policy cost (backend/ml/rollout_student.py): ~76% of a full playout is policy,
and of the ~2.2 ms/ply, the Python FEATURE ENCODING is 1.38 ms and per-move
scoring 0.66 ms — the numpy forward is only 0.15 ms. So a smaller net is
pointless; a cheaper ENCODER is the lever. (Also fixed en route: score_moves()
batch scorer — the move-value head was re-dotting the 1693-dim state per move;
computing it once per decision is numerically identical and ~10% of playout.)

BUILD: 43 cheap counter features (0.04 ms vs 1.38 ms, 34x) -> tiny MLP
(43->128->64->factorized heads, no value/move-value heads) distilled from
solo_net_spread by per-head soft-target CE on 41.5k states sampled from
big-net-policy playouts (the exact rollout distribution). Teacher-argmax
agreement: action_type 0.84, lay_eggs 0.92, draw 0.94, habitat 0.52.
Playout speed 172 -> 63 ms (2.74x); realized on full search games 2.43x.
Wired as make_search_chooser(rollout_model=...) — student plays ONLY the
rollouts; the big net keeps the root candidate ranking.

GATE (n=40 paired, seeds 0-39, r8/k10/t0.3/det vs heuristic, 4 local shards):
  baseline (big rollouts)   mean 92.83  floor 69  wins 37/40  197.5 s/game
  student SAME r8           mean 86.28  floor 55  wins 34/40   81.4 s/game
                            paired -6.55 (sd 16.4, t=-2.53)  <- real cost
  student MATCHED r22       mean 94.03  floor 70  wins 38/40  220.9 s/game
                            paired +1.20 (sd 16.4, t=0.46)   <- tie (noise)
READ: cheaper-but-weaker rollouts trade quality for quantity almost exactly
evenly here — the 6th confirmation that nothing beats full-quality rollouts
for STRENGTH. But unlike the value-bootstrap (8x faster / -8 pts), the student
FULLY RECOVERS baseline strength at matched wall-clock (and directionally
leads on mean/floor/wins), so it is deployable as a latency/throughput win:
  - same latency, ~2.4x the rollouts (deployed: advisor scales preset rollouts
    by MATCHED_ROLLOUT_SCALE=2.4 when rollout_student.npz is present), or
  - a fast mode at -6.5 pts and 2.4x speed if ever wanted (not deployed).
Artifacts: rollout_student.npz (force-added), gate JSON
(rollout_student_gate.json), dataset regenerable via
`python -m backend.ml.rollout_student gen`. Env kill-switch:
WINGSPAN_ROLLOUT_STUDENT=off.

## Rollout-student round 2: encoder caching + identity-aware v2 (DEPLOYED)
Follow-ups to the v1 gate, per user push to keep bird identities in rollouts.

1) ENCODER CACHING (exact-output engineering, kept regardless): the full
StateEncoder's per-bird blocks are pure functions of the bird — 33-dim bird
vector, regex-heavy 14-dim power vector, hashed identity buckets — now cached
by bird name, plus a per-encode memo for can_pay_food_cost keyed by food-cost
signature. Verified bit-identical over 156 real states. Full-identity playout
172 -> 126-129 ms (1.35x, stacking with the batch scorer). Helps every
consumer (root reads, projections, draft advisor).

2) STUDENT v2: same 43 counters + a 32-dim hashed bag of OWN board/hand bird
identities (cached hashes, ~zero encode cost). Distilled on a fresh 41.6k-state
dataset. Teacher-head agreement barely moved (habitat .52->.54) but GAMES
improved decisively vs v1.

GATE round 2 (same seeds 0-39, paired vs the stored r8 baseline; k10/t0.3/det):
  arm                          mean  floor  wins   s/game  paired-vs-base
  baseline (big rollouts r8)  92.83   69   37/40   197.5      --
  cached_full r10             91.92   59   34/40   173.2   -0.90 (t=-0.31)
  student v1 (no id) r22      94.03   70   38/40   220.9   +1.20 (t=0.46)
  student v2 (id)    r24      95.45   68   38/40   174.0   +2.62 (t=0.99)
  head-to-head v2 vs v1: +1.43 (t=0.65); v2 vs cached_full: +3.52 (t=1.53)
READS:
- Identity features fix the v1 weakness: v2 has the best mean and tied-best
  win rate of ANY arm while running 12% FASTER than baseline. Each individual
  diff is sub-noise (the ceiling story stands) but v2 dominates every axis
  simultaneously and costs nothing — deployed.
- cached_full r10 ties baseline (more full-quality rollouts at r8->r10 buys
  nothing measurable — consistent with the saturated ladder). Caching is kept
  as a free engine-wide speedup, not as the rollout policy strategy.
DEPLOYED: rollout_student.npz = v2 (identity-aware); MATCHED_ROLLOUT_SCALE=3.0
(advisor default preset now runs r36-equivalent rollouts at unchanged latency).
Honest headline: the app's engine pick now searches ~3x the playouts per
second at full parity-or-better quality; the strength ceiling itself is
unchanged (~92-96 band).

---

## Engine hot-path speedup — enum hashing, payment memoization, canonical order (2026-07-05)

Goal: make the analyzer faster with ZERO behavior change (the ceiling is
budget-bound, so cheaper playouts convert directly into more search at fixed
latency). Profiled one k10/r12+student root choice (cProfile): the top
self-time sink was `enum.Enum.__hash__` — 8.2M calls hashing member *name
strings*, because FoodType/Habitat/etc. key hot dicts everywhere. Move
generation (repeated per rollout ply) and `can_pay_food_cost` /
`find_food_payment_options` dominated the rest.

Changes (all bit-identical output — verified as an identical move *multiset*
over 174 real mid-game states, and identical golden-replay scores):
1. `__hash__ = object.__hash__` on the value-keyed enums (FoodType, Habitat,
   NestType, PowerColor, ActionType). Members are process-wide singletons and
   Enum equality is identity, so id-hash is consistent and ~free vs hashing the
   name string every call.
2. Memoize `can_pay_food_cost` and `find_food_payment_options` by
   (cost signature, effective supply) — a tiny key space hit millions of times
   per search. `find_food_payment_options` returns fresh dict copies per call
   (callers mutate them onto Move objects). Bounded caches (clear at 200k).
3. `FoodSupply.get` / `PlayerBoard.get_row`: branch chains instead of building
   a fresh dict every call. `_get_cached_food_pool`: iterate rows/slots
   directly (no all_slots() list alloc) and skip empty caches.
4. Bird.habitats: `frozenset` -> canonical `tuple` (forest,grassland,wetland).
   Move-generation iterates habitats, so this ALSO makes move order (and thus
   the heuristic opponent's argmax tie-break) deterministic and
   PYTHONHASHSEED-independent — previously it silently depended on hash order.

Result (honest, no profiler): full student rollout **70 -> 53 ms/game
(1.32x)**; single k10/r12+student root choice **5.2 -> 4.0 s**. Stacks
multiplicatively with the distilled student (so vs the pre-speedup big-net
baseline the effective playout throughput is ~4.5x). Ceiling unchanged — this
is throughput, spent as either a snappier advisor or more rollouts at fixed
latency.

Side effect: the `test_golden_replay_127.py` seed-20260211 golden was already
STALE on main (expected opp total 92; actual 91 with divergence still 0 — a
heuristic-opponent drift from an earlier merged power fix that CI never caught,
since the golden wasn't in the CI subset). Re-baselined to the true 91 and
ADDED to CI now that canonical habitat order makes it hashseed-stable. New
`test_payment_cache.py` pins the memoization invariants (cached == uncached
over a supply grid; fresh-copy mutation safety).

---

## Rollout scoring — lean factorized key (2026-07-05, follow-up)

After the enum/payment speedup, re-profiled a PURE student rollout (cloning via
fast_clone_game outside the timed loop, so no harness deepcopy contamination).
New #1 cost: the student scoring EVERY generated move —
`score_moves` -> `encode_factorized_targets`, ~25% of rollout time, ~35
moves/ply. The student has no move-value head, so that target encoding is the
only per-move cost. Only 19% of moves have a unique factorized key.

Key insight: `score_moves` only reads the heads in RELEVANT_HEADS_BY_ACTION for
each action (play_bird -> habitat/cost/color; gain_food -> food; lay_eggs ->
eggs; draw -> draw_mode), but `encode_factorized_targets` computed ALL 7 bins
per move — including the bird power-color lookup and two dict-sums — then
allocated a 7-entry dict. For the 3 non-play action types (the bulk of rollout
moves) almost all of that was wasted.

Fix: `encode_factorized_score_key(move, player)` computes ONLY the relevant
heads and returns the tuple key directly (no dict). score_moves uses it.
Bit-identical: verified 0/7320 key mismatches and 0.0 max score diff vs the
old full-targets key; a regression test pins the two together.

Clean apples-to-apples (fast_clone harness, best-of-3, 30 games):
  pre-speedup (before PR #11)          44.1 ms/game
  + enum/payment/canonical (PR #11)    32.7 ms/game  (1.35x)
  + lean factorized key (this)         28.5 ms/game  (1.15x; 1.55x cumulative)
All bit-identical, stacking on the distilled student. Ceiling unchanged; this
is pure throughput (snappier advisor, or more rollouts at fixed latency).
NOTE: the "70 -> 53 ms" figures in the PR #11 note were measured with a slower
deepcopy-in-loop harness — same 1.3x ratio, just a slower absolute baseline;
44 -> 28.5 are the clean fast-clone numbers.

---

## Process-parallel candidate rollouts (2026-07-06)

The advisor evaluated every candidate move's playouts sequentially on one
core. Candidates are independent, so they fan out across a spawn-context
worker pool (each worker loads registries + the distilled student once;
BLAS pinned to 1 thread to avoid oversubscription). Per-candidate seeds make
parallel results deterministic and scheduling-independent. The sequential
path is byte-identical when no pool is passed, so every gate/test keeps its
historical behavior; only the advisor route opts in (WINGSPAN_PARALLEL=off
kills it, WINGSPAN_PARALLEL_WORKERS overrides sizing).

Measured (fresh-game engine pick, k10/r36 student config, 3 determinizations,
4-core container, 3 workers): sequential 41.7s -> parallel 16.4s = 2.54x
(85% parallel efficiency), same move picked. Expect ~4x+ on an 8-core Mac
(6 workers). Distribution-equivalent by construction: same samplers, same
temperature; only the PRNG stream assignment changes (independent per-
candidate streams instead of one shared sequential stream).

This is the cores->budget converter: by the budget ladder, the freed
wall-clock IS score when spent on the heavier preset (strong at old-default
latency). First advisor call pays a one-time pool warm-up (~8s/worker,
overlapped).

---

## Parallel rollouts made OPT-IN (2026-07-26)

The process pool (PR #17) was on by default when >2 cores. A spawned
ProcessPoolExecutor can hang under macOS spawn + `uvicorn --reload`, and a
hung pool blocks the advisor's engine pick -> "Suggest a Move" never returns.
That's a core-feature regression traded for a speed optimization, so the
default is now SEQUENTIAL (the proven path). Enable the 2.5x speedup with
WINGSPAN_PARALLEL=on (auto-sizes) or WINGSPAN_PARALLEL_WORKERS=N. Kept because
it's a real, measured win on environments where spawn is healthy (Linux, or a
non-reload prod server); just no longer forced on.
