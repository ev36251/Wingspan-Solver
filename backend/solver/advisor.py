"""Percentage-play advisor: turns rollout distributions into human "engine lines".

The solver already plays games to the end and produces a *distribution* of final
scores per candidate move (see solver.simulation.simulate_playout). This module
reframes that distribution as the two kinds of advice a player actually wants at
the table, with no raw "score = 91.0" point estimate:

  MAIN LINE   "If you play X, ~70% of the time you finish with >=Y points
               (typically ~M)."  -- percentiles of the recommended move's rollouts.

  UPSIDE LINE "~3% chance you draw bird B before the game ends; if you do, this
               line reaches ~W points."  -- a low-probability / high-ceiling card
               you do NOT hold yet, with its draw odds and conditional score.

Draw odds use a *smart unseen-pool* model (no per-turn discard logging): the deck
is treated as every bird in the active sets that you cannot currently see (your
hand, the tray, and every bird already played on every board are removed). A
one-time `active_sets` choice (which expansions are in play) tightens the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from backend.data.registries import get_bird_registry
from backend.models.bird import Bird
from backend.models.enums import GameSet, PowerColor
from backend.solver.simulation import deep_copy_game, execute_move_on_sim, simulate_playout

# Tunables (documented so the numbers are honest and adjustable) ------------- #
# Cards drawn per remaining player-turn, averaged over a game. A turn is a draw,
# food, lay, or play; draw turns net ~1.3 cards, and ~40% of turns are draws, so
# ~0.5 cards/turn is a defensible blended rate for "cards you'll still see".
DRAW_RATE_PER_TURN = 0.5
# Cubes (turns) per round in a standard game, rounds 1..4.
CUBES_BY_ROUND = {1: 8, 2: 7, 3: 6, 4: 5}
# An upside target must beat the main line's typical score by at least this many
# points to be worth mentioning (otherwise it's not really "upside").
UPSIDE_MIN_LIFT = 4
# Headline probability for the main line: "~P of the time you'll score >= Y".
MAIN_LINE_PROB = 0.70
# How many of YOUR OWN turns pass before a drawn upside card lands in hand. You
# can't have it instantly: the soonest a blind deck draw reaches your hand is a
# turn or two out. The conditional rollout only adds the bird after this many
# hero turns, so its score reflects the real remaining window to play it.
UPSIDE_ARRIVAL_TURNS = 2
# A draw only counts as "useful" if it lands with at least this many of your
# turns left to actually pay for and play the bird. Turns inside this tail buffer
# are removed from the draw-odds estimate (you'd see the card too late to use it).
UPSIDE_USE_BUFFER_TURNS = 3


# --------------------------------------------------------------------------- #
# Unseen-pool / draw-probability model
# --------------------------------------------------------------------------- #
def seen_bird_names(game) -> set[str]:
    """Every bird identity currently visible to the player and thus NOT a hidden
    deck draw: all players' boards, the face-up tray, the asking player's hand,
    and any known discarded cards. Opponent hands are unknown (stay in the pool)."""
    seen: set[str] = set()
    for p in game.players:
        for bird in p.board.all_birds():
            seen.add(bird.name)
        for bird in p.hand:  # known only for the companion's own player
            seen.add(bird.name)
    for bird in game.card_tray.face_up:
        seen.add(bird.name)
    for bird in getattr(game, "bird_discard_cards", []) or []:
        name = getattr(bird, "name", None)
        if name:
            seen.add(name)
    return seen


def unseen_pool(game, active_sets: set[GameSet] | None = None) -> list[Bird]:
    """Birds from the active sets that the player cannot see -> the hidden deck."""
    reg = get_bird_registry()
    if active_sets:
        birds = [b for b in reg.all_birds if b.game_set in active_sets]
    else:
        birds = list(reg.all_birds)
    seen = seen_bird_names(game)
    return [b for b in birds if b.name not in seen]


def remaining_hero_turns(game, player) -> int:
    """Total turns this player still has: this round's leftover cubes + the cubes
    of every later round."""
    rnd = max(1, int(getattr(game, "current_round", 1)))
    turns = max(0, int(getattr(player, "action_cubes_remaining", 0)))
    for r in range(rnd + 1, 5):
        turns += CUBES_BY_ROUND.get(r, 0)
    return turns


def estimate_future_draws(game, player, tail_buffer_turns: int = 0) -> float:
    """Expected number of cards this player still draws before the game ends.

    Sum of remaining player-turns (this round's cubes + all later rounds) times
    the blended draw rate, capped by what's left in the deck. If
    `tail_buffer_turns` > 0, the final that-many turns are excluded -- a card seen
    that late can't be paid for and played, so it shouldn't count as a "useful"
    draw."""
    remaining_turns = remaining_hero_turns(game, player)
    usable_turns = max(0, remaining_turns - max(0, tail_buffer_turns))
    draws = usable_turns * DRAW_RATE_PER_TURN
    deck_left = int(getattr(game, "deck_remaining", 0) or 0)
    return float(min(draws, deck_left)) if deck_left else float(draws)


def draw_probability(pool_size: int, expected_draws: float) -> float:
    """P(see one specific card from a `pool_size` deck given `expected_draws`
    draws), via the hypergeometric complement 1 - C(N-1,k)/C(N,k) = k/N for a
    single copy. Clamped to [0, 0.99]."""
    if pool_size <= 0 or expected_draws <= 0:
        return 0.0
    p = expected_draws / pool_size
    return float(max(0.0, min(0.99, p)))


# --------------------------------------------------------------------------- #
# Score-distribution framing
# --------------------------------------------------------------------------- #
def quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


@dataclass
class MainLine:
    move_description: str
    action_type: str
    typical: int          # median final score
    floor_prob: float     # MAIN_LINE_PROB
    floor_score: int      # ">= this" at floor_prob
    stretch_score: int    # top-10% outcome
    samples: int
    details: dict = field(default_factory=dict)

    def sentence(self) -> str:
        pct = round(self.floor_prob * 100)
        return (f"~{pct}% of the time this finishes with {self.floor_score}+ points "
                f"(typically around {self.typical}; best games reach {self.stretch_score}).")


def main_line(move_description: str, action_type: str, rollout_scores: list[int],
              details: dict | None = None, floor_prob: float = MAIN_LINE_PROB) -> MainLine:
    """Frame a move's rollout score samples as percentile-based advice."""
    s = sorted(float(x) for x in rollout_scores)
    typ = int(round(median(s))) if s else 0
    floor = int(round(quantile(s, 1.0 - floor_prob)))  # floor_prob of games >= this
    stretch = int(round(quantile(s, 0.90)))
    return MainLine(move_description=move_description, action_type=action_type,
                    typical=typ, floor_prob=floor_prob, floor_score=floor,
                    stretch_score=stretch, samples=len(s), details=details or {})


# --------------------------------------------------------------------------- #
# Upside-bird finder
# --------------------------------------------------------------------------- #
@dataclass
class UpsideLine:
    bird_name: str
    bird_vp: int
    draw_prob: float       # probability of seeing it before the game ends
    conditional_typical: int   # median final score in lines where you hold it
    lift_vs_main: int      # conditional_typical - main-line typical
    reason: str            # why it's high-leverage (bonus, big VP, ...)

    def sentence(self) -> str:
        pct = self.draw_prob * 100
        pct_str = f"{pct:.0f}%" if pct >= 1 else f"{pct:.1f}%"
        return (f"~{pct_str} chance you draw {self.bird_name} in time to use it; "
                f"if you do, this line can reach about {self.conditional_typical} points "
                f"(+{self.lift_vs_main}).")


def _candidate_leverage(game, player, bird: Bird) -> tuple[float, str]:
    """Cheap score for shortlisting unseen birds worth a conditional rollout.
    Returns (leverage, reason)."""
    score = 0.0
    reasons: list[str] = []
    # Completes / advances a bonus card the player holds.
    for bc in player.bonus_cards:
        if bc.name in bird.bonus_eligibility:
            score += 4.0
            reasons.append(f"works toward your {bc.name} bonus")
            break
    # Raw VP.
    score += bird.victory_points
    if bird.victory_points >= 5:
        reasons.append(f"{bird.victory_points} VP")
    # Brown engine power with rounds left to fire it.
    rounds_left = max(0, 5 - int(getattr(game, "current_round", 1)))
    if bird.color == PowerColor.BROWN and rounds_left >= 2:
        score += 2.0
        reasons.append("repeating brown engine power")
    reason = "; ".join(reasons) if reasons else f"{bird.victory_points} VP bird"
    return score, reason


def _conditional_scores(game, player_name: str, bird: Bird, sims: int,
                        arrival_turns: int = UPSIDE_ARRIVAL_TURNS) -> list[int]:
    """Median final score of playouts where `bird` arrives in the player's hand
    after `arrival_turns` of their own turns (i.e. 'if you draw this card a few
    turns from now'), not instantly. Uses the strong hero policy so the
    conditional score is comparable to the main line (both reflect competent
    play). If the game ends before the card arrives, the line simply gets no
    benefit -- which is the honest answer for a too-late draw."""
    out: list[int] = []
    for _ in range(max(1, sims)):
        result = simulate_playout(game, max_turns=260, rollout_policy="fast",
                                  hero_name=player_name, hero_policy="strong",
                                  inject_card=bird,
                                  inject_after_hero_turns=arrival_turns)
        if player_name in result:
            out.append(result[player_name])
    return out


def find_upside_lines(game, player, main_typical: int, active_sets: set[GameSet] | None,
                      shortlist: int = 8, sims_per_bird: int = 10,
                      max_lines: int = 3, deadline: float | None = None) -> list[UpsideLine]:
    """Find low-probability, high-ceiling birds the player does NOT hold.

    deadline (time.time() epoch) stops evaluating further shortlist birds once
    past it — each bird costs sims_per_bird full playouts, which are longest
    early in the game."""
    pool = unseen_pool(game, active_sets)
    if not pool:
        return []
    # Odds of drawing a specific card IN TIME to use it: discount the final few
    # turns, which are too late to pay for and play a freshly drawn bird.
    expected_draws = estimate_future_draws(game, player,
                                           tail_buffer_turns=UPSIDE_USE_BUFFER_TURNS)
    p_one = draw_probability(len(pool), expected_draws)
    if p_one <= 0:
        return []

    scored = sorted(
        ((*_candidate_leverage(game, player, b)[::-1], b) for b in pool),
        key=lambda t: -t[1],
    )[:shortlist]  # tuples of (reason, leverage, bird)

    import time as _time
    lines: list[UpsideLine] = []
    for i, (reason, _lev, bird) in enumerate(scored):
        if deadline is not None and i > 0 and _time.time() >= deadline:
            break
        cond = _conditional_scores(game, player.name, bird, sims_per_bird)
        if not cond:
            continue
        cond_typ = int(round(median(sorted(float(x) for x in cond))))
        lift = cond_typ - main_typical
        if lift < UPSIDE_MIN_LIFT:
            continue
        lines.append(UpsideLine(
            bird_name=bird.name, bird_vp=bird.victory_points, draw_prob=p_one,
            conditional_typical=cond_typ, lift_vs_main=lift, reason=reason,
        ))
    # Highest ceiling first.
    lines.sort(key=lambda l: -l.conditional_typical)
    return lines[:max_lines]
