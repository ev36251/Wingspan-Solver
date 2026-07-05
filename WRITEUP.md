# Six failed attempts to beat search with a neural network

*(and the two things that actually worked)*

This project began as "build an AI for the board game Wingspan" and turned into
a months-long, carefully measured argument between two ideas: **learned
evaluation** (train a network to judge positions) and **brute search** (just
play the game out and count the points). I went in expecting the network to
win. It lost six times in a row, each loss measured with paired experiments and
significance tests — and the levers that finally moved the score were search
budget, compute engineering, and rules-engine correctness.

This is the story, with the numbers.

## The problem

Wingspan is a good stress test for game AI: 471 unique bird powers across five
expansions, an economy of five food types plus nectar, hidden information (a
face-down deck, opponents' hands), and a 26-turn horizon where the best
strategies are *engines* — combinations that compound (tuck loops, round-end
food caches, egg mills) and only pay off many turns later. A greedy evaluator
that scores one move ahead literally cannot see why holding a card for two
rounds is good.

The foundation is a complete rules engine: every bird power implemented and
tested (~1,100 tests), validated by **golden replays** — three real recorded
games replayed move-for-move through the engine with zero divergences. Every
claim below sits on top of that, because an agent optimizing against a buggy
simulator learns to exploit the bugs, not the game.

## The baseline that refused to die

The core agent is embarrassingly simple: for each candidate move, **play the
whole game to the end** a dozen times with a fast policy, average the final
scores, and pick the move with the best average. To keep it honest under
hidden information, the unseen deck is reshuffled before every playout
(*determinization*), so the agent plans across plausible futures instead of
peeking at the real deck order.

This is worth roughly **+40 points over greedy play**, because full-game
playouts are what let the search *feel* compounding value: a tucking engine
looks mediocre for two turns and wins by 30 at the end, and only something that
simulates the end ever finds out. Averaging several noisy playouts instead of
trusting one was the first measured win (+5, p=0.004, n=100 paired games).

Then came the most important chart in the project — the **budget ladder**.
Holding everything fixed and only raising the number of playouts per decision:

| playouts per move | mean score |
|---|---|
| 6 | 87.5 |
| 12 | 93.2 |
| 24 | 96.2 |
| 36 | 96.3 (plateau) |

Rollout count was the *only* dial that scaled — and it eventually saturated.
That chart frames everything that follows: to beat ~93, a learned model had to
outperform "just run more playouts," and to beat ~96 it had to break through a
plateau that more compute couldn't.

## Six attempts to beat it with learning

Every attempt below was evaluated the same way: n=100 games on paired seeds
(both arms play the same deals), fanned across cloud containers, compared with
significance tests. Same harness, no cherry-picking.

1. **AlphaZero-style self-play loop.** Self-play data generation → factorized
   policy network → promotion gate, iterated for ~20 generations. The champions
   converged to a degenerate egg-spam strategy and the greedy network lost ~70%
   of games to a hand-written heuristic. Superseded.

2. **Behavior-cloning retrains on better data.** Once the search agent was
   strong, I retrained networks on its best games, expecting the student to
   bootstrap past the teacher. Multiple iterations: all within noise. The gains
   did not compound.

3. **A denial objective.** Train/search to maximize *my score minus yours*
   instead of my score. It lost decisively (40% win rate vs 60% for selfish
   play): in a low-interaction engine game, sacrificing your own engine to
   spite the opponent is just self-harm.

4. **A learned value function as the evaluator.** Replace playouts entirely
   with a trained V(s): **74.3 mean vs 93 for rollouts**, with a floor of 25.
   Classic optimizer's curse — argmax over a noisy learned function finds the
   states where the function is *wrong*, and a game about compounding engines
   punishes that hard.

5. **The value net as a rollout-truncation bootstrap.** Play out a few turns,
   then let V(s) estimate the rest — the standard compute-saving trick. ~90
   mean: dominated by full rollouts across the *entire* compute curve. The
   learned tail estimate lost more accuracy than the saved compute bought back.

6. **PUCT-MCTS with the value net at the leaves and the policy net as a
   prior** — the textbook AlphaZero remedy for exactly the failure in #4.
   It scored 74 against the plain rollout searcher's 93 at matched budget.

The pattern across all six: **at high playout counts, the search dominates the
prior.** A network's judgment matters when you can't afford to look; once you
can simulate the actual future thousands of times, approximate opinions about
the future stop adding value. I kept the negative results in the repo because
they're what make the positive numbers believable.

## What actually worked

**Making the playouts cheaper — with a network.** The one place learning paid
off was *inside* the search, not instead of it. Profiling showed each playout
step was dominated by feature encoding for the policy network, not by the math.
So I distilled the big policy net into a tiny student over ~75 cheap features
(including hashed bird identities, which turned out to matter), used **only for
the playouts** while the full network still ranks the root moves. Playouts got
~3× cheaper; spending that on more playouts at unchanged latency produced the
best configuration ever measured in the project — **95.45 mean vs the 92.83
baseline, at 12% less wall-clock**. Same idea as attempt #5, inverted: don't
replace the simulation with the network — use the network to buy *more*
simulation.

**Plain performance engineering.** A later profiling pass found the remaining
cost was Python plumbing: millions of enum hash calls (Python hashes an enum's
*name string* on every dict lookup), redundant food-payment legality checks,
and per-move encodings of features the scorer never reads. Fixing those —
identity hashing, memoized payment solving, lean per-action encoding — made
the engine another **1.55× faster with bit-identical outputs**, verified by
comparing the full move set on 174 real states and replaying the golden games.

**Correctness and the decisions nobody tunes.** Auditing the simulator against
the real game fixed several rules bugs and raised the worst-deal floor from 52
to 68 — the agent learned to recover from bad hands the way strong humans do.
And searching the *opening draft* (which keep-combination to start with) was
the single biggest late lever: +7.4 mean, stacking with deeper mid-game search
to +10.3. The least glamorous work bought the most points.

## Takeaways

- **Scale search before you model.** If simulation is cheap and the rules are
  known, more playouts beat better opinions. Learned evaluation earns its keep
  when simulation is expensive or the rules are hidden — know which regime
  you're in.
- **A model can still win — as a cost reducer.** Distillation inside the
  search loop was worth more than every attempt to replace the search.
- **Profile before you architect.** The single biggest per-playout cost in a
  "machine learning project" turned out to be string hashing in an enum.
- **Pay for trustworthy nulls.** Paired seeds, fixed n, significance tests, and
  golden-replay validation made it cheap to kill bad ideas quickly — six times.

*The engine, the experiments, and all of the numbers above live in this repo —
see the [README](README.md) for the tour and `memory/SOLO_SEED_FINDINGS.md`
for the full lab notebook.*
