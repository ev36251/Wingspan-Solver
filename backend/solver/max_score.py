"""Calculate theoretical maximum score from current game state.

Provides an optimistic upper bound assuming:
- All remaining turns are used optimally
- All birds in hand can be played
- All egg limits are filled
- All predator/cache/tuck powers succeed every activation
- 1st place in all remaining round goals
- Nectar majority in all habitats (Oceania)
"""

from dataclasses import dataclass

from backend.config import ACTIONS_PER_ROUND, ROUNDS
from backend.models.enums import PowerColor
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.scoring import (
    score_birds, score_eggs, score_cached_food, score_tucked_cards,
    score_bonus_cards, score_round_goals, score_nectar,
)
from backend.powers.registry import get_power
from backend.powers.base import NoPower


@dataclass
class MaxScoreBreakdown:
    """Breakdown of theoretical maximum score."""
    bird_vp: int = 0
    eggs: int = 0
    cached_food: int = 0
    tucked_cards: int = 0
    bonus_cards: int = 0
    round_goals: int = 0
    nectar: int = 0

    @property
    def total(self) -> int:
        return (self.bird_vp + self.eggs + self.cached_food +
                self.tucked_cards + self.bonus_cards + self.round_goals +
                self.nectar)

    def as_dict(self) -> dict[str, int]:
        return {
            "bird_vp": self.bird_vp,
            "eggs": self.eggs,
            "cached_food": self.cached_food,
            "tucked_cards": self.tucked_cards,
            "bonus_cards": self.bonus_cards,
            "round_goals": self.round_goals,
            "nectar": self.nectar,
            "total": self.total,
        }


def _remaining_actions(game: GameState, player: Player) -> int:
    """Total remaining actions for this player in the game."""
    total = player.action_cubes_remaining
    for r in range(game.current_round + 1, ROUNDS + 1):
        total += ACTIONS_PER_ROUND[r - 1]
    return total


def _max_bird_vp(player: Player, remaining_actions: int) -> int:
    """Max bird VP: current birds + best VP birds from hand."""
    current_vp = score_birds(player)
    empty_slots = 15 - player.total_birds
    max_playable = min(len(player.hand), remaining_actions, empty_slots)
    hand_sorted = sorted(player.hand, key=lambda b: -b.victory_points)
    future_vp = sum(b.victory_points for b in hand_sorted[:max_playable])
    return current_vp + future_vp


def _max_eggs(player: Player, playable_birds: list) -> int:
    """Max eggs: fill all current + future birds to their egg limits."""
    current_max = sum(
        slot.bird.egg_limit
        for row in player.board.all_rows()
        for slot in row.slots
        if slot.bird
    )
    future_max = sum(b.egg_limit for b in playable_birds)
    return current_max + future_max


def _max_cached_food(player: Player, remaining_actions: int) -> int:
    """Max cached food: current + estimate from cache powers."""
    current = score_cached_food(player)

    # Count birds with caching powers
    cache_activations = 0
    for row in player.board.all_rows():
        for slot in row.slots:
            if not slot.bird:
                continue
            power_text = (slot.bird.power_text or "").lower()
            if "cache" in power_text:
                cache_activations += 1

    # Optimistic: each cache bird caches 1 food per activation,
    # with ~remaining/3 activations per habitat
    future = cache_activations * max(1, remaining_actions // 3)
    return current + future


def _max_tucked_cards(player: Player, remaining_actions: int) -> int:
    """Max tucked cards: current + estimate from flocking birds."""
    current = score_tucked_cards(player)

    flocking_count = sum(
        1 for row in player.board.all_rows()
        for slot in row.slots
        if slot.bird and slot.bird.is_flocking
    )

    # Also count non-flocking birds with tuck powers
    tuck_power_count = sum(
        1 for row in player.board.all_rows()
        for slot in row.slots
        if slot.bird and not slot.bird.is_flocking
        and "tuck" in (slot.bird.power_text or "").lower()
    )

    total_tuckers = flocking_count + tuck_power_count
    future = total_tuckers * max(1, remaining_actions // 3)
    return current + future


def _max_bonus_cards(player: Player, playable_birds: list) -> int:
    """Max bonus score: score with all birds (board + playable hand)."""
    if not player.bonus_cards:
        return 0

    all_birds = player.board.all_birds() + playable_birds
    total = 0
    for bonus in player.bonus_cards:
        qualifying = sum(
            1 for bird in all_birds
            if bonus.name in bird.bonus_eligibility
        )
        total += bonus.score(qualifying)
    return total


def _max_round_goals(game: GameState, player: Player) -> int:
    """Max round goals: scored rounds + 1st place for unscored rounds."""
    current = score_round_goals(game, player)

    scored_rounds = set(game.round_goal_scores.keys())
    future = 0
    for r in range(1, ROUNDS + 1):
        if r in scored_rounds:
            continue
        idx = r - 1
        if idx < len(game.round_goals):
            goal = game.round_goals[idx]
            if goal.scoring:
                # 1st place is the last element (index 3)
                future += int(goal.scoring[-1])
        else:
            future += 5  # Default optimistic if no goal assigned
    return current + future


def _max_nectar(game: GameState, player: Player) -> int:
    """Max nectar: assume 1st in all 3 habitats = 15."""
    current = score_nectar(game, player)
    if game.num_players >= 2:
        return max(current, 15)  # 5 per habitat × 3
    return current  # Solo: keep current calculation


def calculate_max_score(game: GameState, player: Player) -> MaxScoreBreakdown:
    """Calculate the theoretical maximum score for a player.

    This is an optimistic upper bound — the actual achievable score
    will be lower in practice.
    """
    remaining = _remaining_actions(game, player)
    empty_slots = 15 - player.total_birds
    max_playable = min(len(player.hand), remaining, empty_slots)
    hand_sorted = sorted(player.hand, key=lambda b: -b.victory_points)
    playable_birds = hand_sorted[:max_playable]

    return MaxScoreBreakdown(
        bird_vp=_max_bird_vp(player, remaining),
        eggs=_max_eggs(player, playable_birds),
        cached_food=_max_cached_food(player, remaining),
        tucked_cards=_max_tucked_cards(player, remaining),
        bonus_cards=_max_bonus_cards(player, playable_birds),
        round_goals=_max_round_goals(game, player),
        nectar=_max_nectar(game, player),
    )
