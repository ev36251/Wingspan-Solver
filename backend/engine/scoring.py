"""Scoring engine: calculate final and in-progress scores."""

from dataclasses import dataclass
from backend.models.enums import Habitat
from backend.models.game_state import GameState
from backend.models.player import Player


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown for a player."""
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


def score_birds(player: Player) -> int:
    """Sum of VP values on all played birds."""
    return sum(
        slot.bird.victory_points
        for row in player.board.all_rows()
        for slot in row.slots
        if slot.bird
    )


def score_eggs(player: Player) -> int:
    """1 point per egg on the board."""
    return sum(
        slot.eggs
        for row in player.board.all_rows()
        for slot in row.slots
    )


def score_cached_food(player: Player) -> int:
    """1 point per cached food token."""
    return sum(
        slot.total_cached_food
        for row in player.board.all_rows()
        for slot in row.slots
    )


def score_tucked_cards(player: Player) -> int:
    """1 point per tucked card."""
    return sum(
        slot.tucked_cards
        for row in player.board.all_rows()
        for slot in row.slots
    )


def score_bonus_cards(player: Player) -> int:
    """Score all bonus cards based on board state.

    Uses the bonus_eligibility field on each bird to count qualifying birds.
    """
    total = 0
    for bonus in player.bonus_cards:
        qualifying = sum(
            1
            for bird in player.board.all_birds()
            if bonus.name in bird.bonus_eligibility
        )
        total += bonus.score(qualifying)
    return total


def score_round_goals(game_state: GameState, player: Player) -> int:
    """Sum of round goal scores for this player."""
    total = 0
    for round_num, scores in game_state.round_goal_scores.items():
        total += scores.get(player.name, 0)
    return total


def score_nectar(game_state: GameState, player: Player) -> int:
    """Oceania nectar majority scoring.

    At end of game, in each habitat:
    - 1st place in nectar spent: 5 points
    - 2nd place: 2 points
    - Ties split (round down)

    Also: leftover nectar in personal supply scores 0.
    """
    if game_state.num_players < 2:
        # Solo mode: simplified nectar scoring
        return _score_nectar_solo(player)

    total = 0
    for habitat in Habitat:
        # Gather nectar spent per player in this habitat
        nectar_by_player: dict[str, int] = {}
        for p in game_state.players:
            row = p.board.get_row(habitat)
            nectar_by_player[p.name] = row.nectar_spent

        # Rank players
        sorted_players = sorted(nectar_by_player.items(), key=lambda x: -x[1])

        if sorted_players[0][1] == 0:
            continue  # No nectar spent in this habitat

        # Determine placements with tie handling
        total += _nectar_points_for_player(sorted_players, player.name)

    return total


def _nectar_points_for_player(
    sorted_players: list[tuple[str, int]], player_name: str
) -> int:
    """Calculate nectar points for a specific player given rankings."""
    if not sorted_players:
        return 0

    # Group by nectar count
    groups: list[list[str]] = []
    current_count = None
    for name, count in sorted_players:
        if count != current_count:
            groups.append([name])
            current_count = count
        else:
            groups[-1].append(name)

    # Award points: 1st group gets 5pts, 2nd group gets 2pts
    # Ties split and round down
    points_pools = [5, 2]
    for group_idx, group in enumerate(groups):
        if group_idx >= len(points_pools):
            break
        if player_name in group:
            if group_idx == 0 and len(groups) > 1:
                # First place group
                return points_pools[0] // len(group)
            elif group_idx == 0 and len(groups) == 1:
                # Everyone tied for first — split 5+2=7
                total_pool = sum(points_pools[:min(len(points_pools), 1)])
                return total_pool // len(group)
            elif group_idx == 1:
                return points_pools[1] // len(group)
    return 0


def _score_nectar_solo(player: Player) -> int:
    """Solo nectar scoring: 1 point per nectar spent (simplified)."""
    return sum(row.nectar_spent for row in player.board.all_rows())


def calculate_score(game_state: GameState, player: Player) -> ScoreBreakdown:
    """Calculate the full score breakdown for a player."""
    return ScoreBreakdown(
        bird_vp=score_birds(player),
        eggs=score_eggs(player),
        cached_food=score_cached_food(player),
        tucked_cards=score_tucked_cards(player),
        bonus_cards=score_bonus_cards(player),
        round_goals=score_round_goals(game_state, player),
        nectar=score_nectar(game_state, player),
    )


def calculate_all_scores(game_state: GameState) -> dict[str, ScoreBreakdown]:
    """Calculate scores for all players."""
    return {
        player.name: calculate_score(game_state, player)
        for player in game_state.players
    }
