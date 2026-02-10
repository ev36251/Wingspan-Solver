"""Scoring engine: calculate final and in-progress scores."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from backend.models.enums import FoodType, Habitat, NestType, PowerColor, BeakDirection
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


def _count_breeding_manager(player: Player) -> int:
    """Birds with at least 4 eggs laid on them."""
    return sum(
        1 for _, _, slot in player.board.all_slots()
        if slot.bird and slot.eggs >= 4
    )


def _count_ecologist(player: Player) -> int:
    """Birds in habitat with the fewest birds."""
    counts = [row.bird_count for row in player.board.all_rows()]
    return min(counts) if counts else 0


def _count_oologist(player: Player) -> int:
    """Birds with at least 1 egg laid on them."""
    return sum(
        1 for _, _, slot in player.board.all_slots()
        if slot.bird and slot.eggs >= 1
    )


def _count_visionary_leader(player: Player) -> int:
    """Bird cards in hand at end of game."""
    return len(player.hand) + player.unknown_hand_count


def _count_behaviorist(player: Player) -> int:
    """Columns with 3 different power colors. No-power birds count as white."""
    count = 0
    for col_idx in range(5):
        colors = set()
        for row in player.board.all_rows():
            if col_idx < len(row.slots) and row.slots[col_idx].bird:
                color = row.slots[col_idx].bird.color
                if color == PowerColor.NONE:
                    color = PowerColor.WHITE
                colors.add(color)
        if len(colors) >= 3:
            count += 1
    return count


def _count_citizen_scientist(player: Player) -> int:
    """Birds with tucked cards."""
    return sum(
        1 for _, _, slot in player.board.all_slots()
        if slot.bird and slot.tucked_cards > 0
    )


def _count_ethologist(player: Player) -> int:
    """Max distinct power colors in any one habitat. No-power counts as white."""
    best = 0
    for row in player.board.all_rows():
        colors = set()
        for slot in row.slots:
            if slot.bird:
                color = slot.bird.color
                if color == PowerColor.NONE:
                    color = PowerColor.WHITE
                colors.add(color)
        best = max(best, len(colors))
    return best


def _longest_consecutive_sequence(values: list[int | None]) -> int:
    """Longest consecutive strictly ascending or descending run.

    None values break the sequence (e.g. flightless birds with no wingspan).
    """
    if len(values) <= 1:
        return len(values)

    best = 1

    # Ascending
    run = 1
    for i in range(1, len(values)):
        if values[i] is not None and values[i - 1] is not None and values[i] > values[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1

    # Descending
    run = 1
    for i in range(1, len(values)):
        if values[i] is not None and values[i - 1] is not None and values[i] < values[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1

    return best


def _count_data_analyst(player: Player, habitat: Habitat) -> int:
    """Consecutive birds with ascending or descending wingspans in a habitat."""
    row = player.board.get_row(habitat)
    wingspans = [bird.wingspan_cm for bird in row.birds()]
    return _longest_consecutive_sequence(wingspans)


def _count_ranger(player: Player, habitat: Habitat) -> int:
    """Consecutive birds with ascending or descending VP in a habitat."""
    row = player.board.get_row(habitat)
    scores = [bird.victory_points for bird in row.birds()]
    return _longest_consecutive_sequence(scores)


def _count_population_monitor(player: Player, habitat: Habitat) -> int:
    """Distinct nest types in a habitat. Star nests count as any type or 5th type."""
    row = player.board.get_row(habitat)
    non_wild = set()
    wild_count = 0
    for bird in row.birds():
        if bird.nest_type == NestType.WILD:
            wild_count += 1
        else:
            non_wild.add(bird.nest_type)
    return min(5, len(non_wild) + wild_count)


def _count_mechanical_engineer(player: Player) -> int:
    """Complete sets of 4 nest types (bowl, cavity, ground, platform).

    Star nests are wild — each can fill any one type per set.
    """
    nests: Counter[NestType] = Counter()
    wild_count = 0
    for bird in player.board.all_birds():
        if bird.nest_type == NestType.WILD:
            wild_count += 1
        else:
            nests[bird.nest_type] += 1

    required = [NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM]
    max_possible = (sum(nests.values()) + wild_count) // 4

    for k in range(max_possible, 0, -1):
        deficit = sum(max(0, k - nests.get(nt, 0)) for nt in required)
        if deficit <= wild_count:
            return k
    return 0


def _score_site_selection_expert(player: Player) -> int:
    """Score columns with matching nest pairs or trios.

    Per column: 2 matching nests = 1pt, 3 matching = 3pts.
    Star nests are wild but each counts only once.
    """
    total = 0
    for col_idx in range(5):
        nests = []
        for row in player.board.all_rows():
            if col_idx < len(row.slots) and row.slots[col_idx].bird:
                nests.append(row.slots[col_idx].bird.nest_type)

        if len(nests) < 2:
            continue

        wild_count = sum(1 for n in nests if n == NestType.WILD)
        type_counts = Counter(n for n in nests if n != NestType.WILD)
        max_freq = max(type_counts.values()) if type_counts else 0
        best_match = max_freq + wild_count

        if best_match >= 3:
            total += 3
        elif best_match >= 2:
            total += 1

    return total


def _count_avian_theriogenologist(player: Player) -> int:
    """Birds with completely full nests (eggs == egg_limit, egg_limit > 0)."""
    return sum(
        1 for _, _, slot in player.board.all_slots()
        if slot.bird and slot.bird.egg_limit > 0 and slot.eggs >= slot.bird.egg_limit
    )


def _count_pellet_dissector(player: Player) -> int:
    """Fish and rodent tokens cached on birds."""
    total = 0
    for _, _, slot in player.board.all_slots():
        total += slot.cached_food.get(FoodType.FISH, 0)
        total += slot.cached_food.get(FoodType.RODENT, 0)
    return total


def _count_winter_feeder(player: Player) -> int:
    """Total food remaining in supply at end of game."""
    return player.food_supply.total()


# Custom counter: bonus name → function(Player) → qualifying count
CUSTOM_BONUS_COUNTERS: dict[str, Callable[[Player], int]] = {
    "Breeding Manager": _count_breeding_manager,
    "Ecologist": _count_ecologist,
    "Oologist": _count_oologist,
    "Visionary Leader": _count_visionary_leader,
    "Behaviorist": _count_behaviorist,
    "Citizen Scientist": _count_citizen_scientist,
    "Ethologist": _count_ethologist,
    "Forest Data Analyst": lambda p: _count_data_analyst(p, Habitat.FOREST),
    "Grassland Data Analyst": lambda p: _count_data_analyst(p, Habitat.GRASSLAND),
    "Wetland Data Analyst": lambda p: _count_data_analyst(p, Habitat.WETLAND),
    "Mechanical Engineer": _count_mechanical_engineer,
    "Forest Population Monitor": lambda p: _count_population_monitor(p, Habitat.FOREST),
    "Grassland Population Monitor": lambda p: _count_population_monitor(p, Habitat.GRASSLAND),
    "Wetland Population Monitor": lambda p: _count_population_monitor(p, Habitat.WETLAND),
    "Forest Ranger": lambda p: _count_ranger(p, Habitat.FOREST),
    "Grassland Ranger": lambda p: _count_ranger(p, Habitat.GRASSLAND),
    "Wetland Ranger": lambda p: _count_ranger(p, Habitat.WETLAND),
    "Avian Theriogenologist": _count_avian_theriogenologist,
    "Pellet Dissector": _count_pellet_dissector,
    "Winter Feeder": _count_winter_feeder,
}

# Cards with fully custom scoring (bypass bonus.score())
CUSTOM_BONUS_FULL_SCORERS: dict[str, Callable[[Player], int]] = {
    "Site Selection Expert": _score_site_selection_expert,
}


def score_bonus_cards(player: Player) -> int:
    """Score all bonus cards based on board state.

    Uses custom scoring functions for cards that depend on game state
    (eggs, cached food, hand size, etc.), and falls back to spreadsheet
    eligibility columns for the remaining cards.
    """
    total = 0
    for bonus in player.bonus_cards:
        if bonus.name in CUSTOM_BONUS_FULL_SCORERS:
            total += CUSTOM_BONUS_FULL_SCORERS[bonus.name](player)
        elif bonus.name in CUSTOM_BONUS_COUNTERS:
            qualifying = CUSTOM_BONUS_COUNTERS[bonus.name](player)
            total += bonus.score(qualifying)
        else:
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


def _goal_progress_for_round(player: Player, goal) -> float:
    """Estimate round-goal progress for end-of-round placement scoring."""
    desc = goal.description.lower()
    board = player.board

    for hab_name, hab_enum in (("forest", Habitat.FOREST), ("grassland", Habitat.GRASSLAND),
                               ("wetland", Habitat.WETLAND)):
        if f"[bird] in [{hab_name}]" in desc:
            return board.get_row(hab_enum).bird_count
    if "total [bird]" in desc:
        return board.total_birds()
    if "[bird] in one row" in desc:
        return max(r.bird_count for r in board.all_rows())

    for hab_name, hab_enum in (("forest", Habitat.FOREST), ("grassland", Habitat.GRASSLAND),
                               ("wetland", Habitat.WETLAND)):
        if f"[egg] in [{hab_name}]" in desc:
            return board.get_row(hab_enum).total_eggs()
    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[egg] in [{nt}]" in desc:
            nest_map = {
                "bowl": NestType.BOWL,
                "cavity": NestType.CAVITY,
                "ground": NestType.GROUND,
                "platform": NestType.PLATFORM,
            }
            return sum(
                s.eggs for r in board.all_rows() for s in r.slots
                if s.bird and (
                    s.bird.nest_type == nest_map[nt]
                    or s.bird.nest_type == NestType.WILD
                )
            )

    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[{nt}] [bird]" in desc:
            nest_map = {
                "bowl": NestType.BOWL,
                "cavity": NestType.CAVITY,
                "ground": NestType.GROUND,
                "platform": NestType.PLATFORM,
            }
            return len(board.birds_with_nest(nest_map[nt]))

    if "[bird] worth >4 [feather]" in desc:
        return sum(1 for b in board.all_birds() if b.victory_points > 4)

    if "brown powers" in desc:
        return sum(1 for b in board.all_birds() if b.color == PowerColor.BROWN)
    if "white & no powers" in desc:
        return sum(1 for b in board.all_birds() if b.color in (PowerColor.WHITE, PowerColor.NONE))

    if "[bird_with_tucked_card]" in desc:
        return sum(1 for r in board.all_rows() for s in r.slots if s.bird and s.tucked_cards > 0)

    if "filled columns" in desc:
        return min(r.bird_count for r in board.all_rows())

    if "[beak_pointing_left]" in desc:
        return sum(1 for b in board.all_birds() if b.beak_direction == BeakDirection.LEFT)
    if "[beak_pointing_right]" in desc:
        return sum(1 for b in board.all_birds() if b.beak_direction == BeakDirection.RIGHT)

    return 0.0


def compute_round_goal_scores(game_state: GameState, round_num: int) -> dict[str, int]:
    """Compute per-player points for one round goal, with tie splitting."""
    if round_num < 1 or round_num > len(game_state.round_goals):
        return {}

    goal = game_state.round_goals[round_num - 1]
    if goal.description.lower() == "no goal":
        return {}

    progress = [(p.name, _goal_progress_for_round(p, goal)) for p in game_state.players]
    progress.sort(key=lambda x: -x[1])
    if not progress:
        return {}

    scores: dict[str, int] = {}
    pos = 1
    i = 0
    while i < len(progress):
        j = i + 1
        while j < len(progress) and progress[j][1] == progress[i][1]:
            j += 1
        tied = progress[i:j]
        placements = list(range(pos, min(pos + len(tied), 5)))
        pool = sum(goal.score_for_placement(pl) for pl in placements)
        share = int(pool // len(tied)) if tied else 0
        for name, _ in tied:
            scores[name] = share
        pos += len(tied)
        i = j

    return scores


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

        # Only players who spent >= 1 nectar qualify for placement
        qualifying = [(name, n) for name, n in nectar_by_player.items() if n > 0]
        if not qualifying:
            continue

        sorted_players = sorted(qualifying, key=lambda x: -x[1])
        total += _nectar_points_for_player(sorted_players, player.name)

    return total


def _nectar_points_for_player(
    sorted_players: list[tuple[str, int]], player_name: str
) -> int:
    """Calculate nectar points for a specific player given rankings.

    Only 1st place (5pts) and 2nd place (2pts). Ties share the pools for
    all positions the tied group occupies, rounded down.

    Examples (2 prizes: 5, 2):
    - 2 tied for 1st: share 5+2=7, each gets floor(7/2) = 3
    - 3 tied for 1st: share 5+2=7, each gets floor(7/3) = 2
    - 1st clear, 2 tied for 2nd: 1st gets 5, tied 2nd share 2, each gets 1
    """
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

    # Award points: tied groups absorb prizes for all positions they occupy
    points_pools = [5, 2]
    position = 0  # current placement position (0-based)

    for group in groups:
        if position >= len(points_pools):
            break  # No more prizes to award

        if player_name in group:
            # Share prizes for positions [position, position + group_size - 1]
            pool = sum(points_pools[position:position + len(group)])
            return pool // len(group)

        position += len(group)

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
