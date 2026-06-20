# v20 champion — bird-outcome ranking (which birds it plays, and which win)

**Source:** `backend/ml/champion_bird_outcomes.py` (committed). 120 champion-vs-
champion games, 40-sim MCTS (the champion's promotion-gate strength), Oceania
board → **240 player-games, 337 distinct birds played, overall mean 55.2.**
Regenerate: `python -m backend.ml.champion_bird_outcomes --model-path models/champions/v20_iter3_champion.npz --games 120 --mcts-sims 40` (raw JSON lands in the gitignored `reports/ml/champion_birds/`).

**Method.** Play full games with both seats = champion (MCTS). Log every bird
actually played (name / habitat). After each game, attribute that player's final
score to every bird they played. Per bird: `delta = mean score of games it
appeared in − overall mean`. Delta is **correlational** (a bird shows high delta
partly because the model already plays it in games that were going well), but
that is exactly the "which birds ride along with high-scoring games" signal.

## 1. Action tempo — the over-egg-laying is real
| action | share of all turns |
|---|---|
| draw_cards (wetland) | **38.7%** |
| lay_eggs (grassland) | **36.9%** |
| play_bird | 18.8% |
| **gain_food (forest)** | **5.6%** |

Birds are *placed* evenly (forest/grassland/wetland = 34/33/33%), but the champion
almost never spends a turn on the **forest gain-food action**. It funds bird-play
from the birdfeeder/nectar and pours its turns into grassland eggs + wetland draw.
This matches the human critique: too many bare egg-laying turns, no forest engine.

## 2. The core problem: most-played ≠ best
The champion leans hardest on birds that do **not** carry its good games:

| bird | plays | Δ | note |
|---|---|---|---|
| Ruby-Throated Hummingbird | 16 | +5.8 | birdfeeder die — mediocre payoff for #1 most-played |
| **Red Kite** | 12 | **−10.2** | "play on top" white — its **worst** high-use bird |
| **California Condor** | 11 | **−6.7** | "draw 2 bonus, keep 1" — also a loser |
| Black Vulture | 13 | −2.5 | reactive pink predator |
| Mississippi Kite | 10 | −2.5 | dice predator |

It over-relies on free/"play-on-top" tempo birds (Red Kite, Condor, Buzzard) and
reactive predators that don't compound.

## 3. The birds that actually win (min 10 games — reliable)
| bird | set | color | n | mean | Δ | win |
|---|---|---|---|---|---|---|
| Ring-Billed Gull | CORE | brown | 10 | 68.6 | **+13.4** | 0.65 | tuck from hand (wetland engine) |
| Red-Eyed Vireo | CORE | white | 12 | 66.2 | **+11.1** | 0.75 | **play an additional bird in forest** |
| American Crow | CORE | brown | 10 | 64.4 | +9.2 | 0.75 | discard egg → gain wild (egg→food economy) |
| Eurasian Hobby | EURO | white | 12 | 64.2 | +9.0 | 0.67 | free play (tempo) |
| Northern Flicker | CORE | white | 12 | 62.5 | +7.3 | 0.75 | gain all invertebrates in feeder |
| Common Starling | EURO | teal | 10 | 60.3 | +5.1 | 0.55 | feeder reset / food |

And with lighter support (n=8–9) the **engine birds the human flagged** rise to the top:
- **Eurasian Coot** (n=8, Δ**+14.7**) — *tuck up to 3 cards* (wetland tuck engine)
- **Grey Shrikethrush** (n=9, Δ+9.8, forest) — feeder reset → rodents (forest food engine)

## 4. The specific birds you named
| bird | n (support) | Δ | reading |
|---|---|---|---|
| **Common Raven** | 2 | **+21.8** | highest delta in the set — egg→2 wild economy. *Tiny support*, but points the right way. |
| **Common Chiffchaff** | 4 | **+10.1** | the wetland tuck engine — confirmed positive. |
| **Golden Pheasant** | 2 | **−9.2** | *counter to expectation* — the model plays it but scores poorly. Either it wastes the burst eggs, or n=2 noise. Worth a targeted look. |

Caveat: Raven/Pheasant are n=2 — anecdotal. The min-10 table is the trustworthy
ranking; the named-bird row is a directional check against human intuition.

## 5. Takeaway → prioritization target
The model's realized preferences are **misaligned with value** in a consistent way:
it over-plays free/play-on-top tempo birds (Red Kite, Condor) and grinds eggs,
while under-weighting the engines that win — **wetland tuckers** (Coot, Ring-Billed
Gull, Chiffchaff), **forest "play additional bird"** (Red-Eyed Vireo), and
**egg→food economy** (Raven, American Crow). Prioritization should push play toward
the high-delta engine birds and demote the high-use negative-delta ones.

**Next:** bias the agent toward high-delta birds. Candidate mechanism is a
bird-value prior injected into MCTS at play time (no retraining, reversible),
loaded from this ranking. See follow-up.
