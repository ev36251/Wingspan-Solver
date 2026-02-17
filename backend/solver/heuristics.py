"""Static board evaluation and move ranking for the heuristic solver."""

import copy
import re
from dataclasses import dataclass, field

from backend.config import (
    ACTIONS_PER_ROUND, ROUNDS,
    EGG_COST_BY_COLUMN,
    get_action_column,
)
from backend.models.enums import (
    ActionType, FoodType, Habitat, NestType, PowerColor, BeakDirection,
)
from backend.models.bird import Bird
from backend.models.goal import Goal
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.scoring import calculate_score, ScoreBreakdown
from backend.powers.registry import get_power
from backend.powers.base import PowerContext, NoPower
from backend.solver.bird_priors import bird_prior_value
from backend.solver.move_generator import Move, generate_all_moves


# --- Tunable weights ---

@dataclass
class HeuristicWeights:
    """Weights for the static evaluation function."""
    # Direct scoring
    bird_vp: float = 1.0
    egg_points: float = 1.0
    cached_food_points: float = 1.0
    tucked_card_points: float = 1.0
    bonus_card_points: float = 1.0
    round_goal_points: float = 1.0
    nectar_points: float = 1.0

    # Engine value — expected future points from powers
    engine_value: float = 0.7

    # Resource value (not direct points, but enable future plays)
    food_in_supply: float = 0.3
    cards_in_hand: float = 0.35
    action_cubes: float = 0.2

    # Strategic modifiers
    habitat_diversity_bonus: float = 0.5
    early_game_engine_bonus: float = 0.3
    predator_penalty: float = 0.5
    goal_alignment: float = 1.2
    nectar_early_bonus: float = 0.4
    grassland_egg_synergy: float = 0.6
    play_another_bird_bonus: float = 2.5
    food_for_birds_bonus: float = 0.4


DEFAULT_WEIGHTS = HeuristicWeights()

# Per-player-count weights discovered via self-play evolutionary training
# Retrained with improved heuristics (bonus card pursuit, nectar majority,
# goal strategy, opponent blocking, tuck/cache early-game prioritization)
TRAINED_WEIGHTS: dict[int, HeuristicWeights] = {
    2: HeuristicWeights(
        bird_vp=1.878, egg_points=1.000, cached_food_points=1.588,
        tucked_card_points=1.000, bonus_card_points=0.577,
        round_goal_points=1.640, nectar_points=1.922,
        engine_value=0.404, food_in_supply=0.403, cards_in_hand=0.350,
        action_cubes=0.200, habitat_diversity_bonus=0.415,
        early_game_engine_bonus=0.177, predator_penalty=0.290,
        goal_alignment=1.200, nectar_early_bonus=0.380,
        grassland_egg_synergy=0.937, play_another_bird_bonus=2.440,
        food_for_birds_bonus=0.443,
    ),
    3: HeuristicWeights(
        bird_vp=0.95, egg_points=0.81, cached_food_points=1.07,
        tucked_card_points=1.68, bonus_card_points=0.76,
        round_goal_points=1.13, nectar_points=0.70,
        engine_value=0.36, food_in_supply=0.47, cards_in_hand=0.35,
        action_cubes=0.16, habitat_diversity_bonus=0.57,
        early_game_engine_bonus=0.19, predator_penalty=0.42,
        goal_alignment=0.62, nectar_early_bonus=0.76,
        grassland_egg_synergy=0.52, play_another_bird_bonus=1.79,
        food_for_birds_bonus=0.27,
    ),
    4: HeuristicWeights(
        bird_vp=0.58, egg_points=1.09, cached_food_points=0.78,
        tucked_card_points=0.66, bonus_card_points=1.00,
        round_goal_points=1.38, nectar_points=1.58,
        engine_value=0.70, food_in_supply=0.26, cards_in_hand=0.22,
        action_cubes=0.38, habitat_diversity_bonus=0.32,
        early_game_engine_bonus=0.47, predator_penalty=0.96,
        goal_alignment=0.96, nectar_early_bonus=0.54,
        grassland_egg_synergy=0.62, play_another_bird_bonus=2.14,
        food_for_birds_bonus=0.23,
    ),
}


def dynamic_weights(game: GameState,
                    base: HeuristicWeights | None = None) -> HeuristicWeights:
    """Compute phase-aware weights based on current game state.

    Automatically selects trained weights for the game's player count (2/3/4P).
    Then applies phase-based scaling for the current round:
      Round 1: Engine-building (brown powers, cards, food worth more)
      Round 2: Transition (balanced, start goal focus)
      Round 3: Scoring pivot (eggs, goals, bonus cards matter more)
      Round 4: Pure points (eggs dominant, cards/engine near worthless)

    If base is provided (for training), it overrides the player-count lookup.
    """
    # Pick the right trained weights for this player count
    if base is None:
        base = TRAINED_WEIGHTS.get(game.num_players)

    w = HeuristicWeights()
    rd = game.current_round
    rounds_left = max(0, ROUNDS - rd)
    phase = rounds_left / ROUNDS  # 1.0 at start, 0.0 at end

    # Engine value scales down
    # Keep a floor so early-round engine-building is never undervalued.
    w.engine_value = max(0.6, 0.4 + 0.6 * phase)
    w.early_game_engine_bonus = 0.5 * phase

    # Cards: valuable early for options, near-worthless late
    w.cards_in_hand = 0.15 + 0.4 * phase

    # Food: valuable when enabling bird plays
    w.food_in_supply = 0.15 + 0.3 * phase

    # Eggs: always 1pt but late-game = guaranteed safe points
    w.egg_points = 1.0 + 0.3 * (1.0 - phase)

    # Bonus cards: matters more as game progresses
    w.bonus_card_points = 0.8 + 0.4 * (1.0 - phase)

    # Goal alignment peaks mid-game
    if rd <= 2:
        w.goal_alignment = 1.4
    elif rd == 3:
        w.goal_alignment = 1.6
    else:
        w.goal_alignment = 0.8

    # Play-another-bird: huge early, pointless round 4
    w.play_another_bird_bonus = 3.0 * phase

    # Food-for-birds urgency
    w.food_for_birds_bonus = 0.3 + 0.3 * phase

    # Nectar majority: valuable early
    w.nectar_early_bonus = 0.5 * phase

    # Grassland egg synergy: more valuable mid-late
    w.grassland_egg_synergy = 0.4 + 0.4 * (1.0 - phase)

    # Apply trained/base weight scaling (player-count-aware or custom training)
    if base is not None:
        default = HeuristicWeights()
        for field_name in HeuristicWeights.__dataclass_fields__:
            default_val = getattr(default, field_name)
            base_val = getattr(base, field_name)
            scale = base_val / default_val if default_val != 0 else 1.0
            setattr(w, field_name, getattr(w, field_name) * scale)

    return w


# --- Goal alignment helpers ---

def _estimate_goal_contribution(bird: Bird, habitat: Habitat, goal: Goal) -> float:
    """Estimate how much playing this bird in this habitat contributes to a goal."""
    desc = goal.description.lower()

    # [bird] in [habitat] — count birds in specific habitat
    if "[bird] in [forest]" in desc:
        return 1.0 if habitat == Habitat.FOREST else 0.0
    if "[bird] in [grassland]" in desc:
        return 1.0 if habitat == Habitat.GRASSLAND else 0.0
    if "[bird] in [wetland]" in desc:
        return 1.0 if habitat == Habitat.WETLAND else 0.0

    # total [bird] — any bird counts
    if "total [bird]" in desc:
        return 1.0

    # [egg] in [habitat] — bird enables future eggs in that habitat
    for hab_name, hab_enum in (
        ("forest", Habitat.FOREST),
        ("grassland", Habitat.GRASSLAND),
        ("wetland", Habitat.WETLAND),
    ):
        if f"[egg] in [{hab_name}]" in desc and habitat == hab_enum and bird.egg_limit > 0:
            return bird.egg_limit * 0.25

    # [egg] in [nest_type] — eggs in specific nest type
    nest_val = bird.nest_type.value if bird.nest_type != NestType.WILD else None
    if nest_val:
        if f"[egg] in [{nest_val}]" in desc and bird.egg_limit > 0:
            return bird.egg_limit * 0.25
    if bird.nest_type == NestType.WILD:
        for nt in ("bowl", "cavity", "ground", "platform"):
            if f"[egg] in [{nt}]" in desc and bird.egg_limit > 0:
                return bird.egg_limit * 0.25

    # [nest] [bird] with [egg] — birds of nest type that can hold eggs
    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[{nt}] [bird] with [egg]" in desc:
            if (nest_val == nt or bird.nest_type == NestType.WILD) and bird.egg_limit > 0:
                return 1.0

    # VP threshold goals
    if "[bird] worth >4 [feather]" in desc:
        return 1.0 if bird.victory_points > 4 else 0.0
    if "\u22643" in desc:  # ≤3 unicode
        return 1.0 if bird.victory_points <= 3 else 0.0

    # Power color goals
    if "brown powers" in desc:
        return 1.0 if bird.color == PowerColor.BROWN else 0.0
    if "white & no powers" in desc:
        return 1.0 if bird.color in (PowerColor.WHITE, PowerColor.NONE) else 0.0

    # [bird] with no [egg] — newly played birds start with 0 eggs
    if "[bird] with no [egg]" in desc:
        return 1.0

    # [bird] in one row — habitat concentration
    if "[bird] in one row" in desc:
        return 0.3

    # filled columns — filling habitat rows
    if "filled columns" in desc:
        return 0.2

    # Beak direction goals
    if "[beak_pointing_left]" in desc:
        return 1.0 if bird.beak_direction == BeakDirection.LEFT else 0.0
    if "[beak_pointing_right]" in desc:
        return 1.0 if bird.beak_direction == BeakDirection.RIGHT else 0.0

    # Food cost sum goals
    if "[invertebrate] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft == FoodType.INVERTEBRATE)
    if "[fruit] + [seed] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft in (FoodType.FRUIT, FoodType.SEED))
    if "[rodent] + [fish] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft in (FoodType.RODENT, FoodType.FISH))
    if "food cost of played [bird]" in desc:
        return bird.food_cost.total * 0.3

    # Tucked cards goal
    if "[bird_with_tucked_card]" in desc and bird.is_flocking:
        return 0.5

    # Action cube goal (play bird = 1 cube on play-a-bird)
    if 'cubes on "play a bird"' in desc:
        return 0.5

    return 0.0


def _estimate_goal_progress(player: Player, goal: Goal) -> float:
    """Estimate a player's current progress toward a goal.

    Returns a rough count of qualifying items (birds, eggs, etc.).
    """
    desc = goal.description.lower()
    board = player.board

    # Bird count in habitat
    for hab_name, hab_enum in (("forest", Habitat.FOREST), ("grassland", Habitat.GRASSLAND),
                                ("wetland", Habitat.WETLAND)):
        if f"[bird] in [{hab_name}]" in desc:
            return board.get_row(hab_enum).bird_count
    if "total [bird]" in desc:
        return board.total_birds()
    if "[bird] in one row" in desc:
        return max(r.bird_count for r in board.all_rows())

    # Egg counts
    for hab_name, hab_enum in (("forest", Habitat.FOREST), ("grassland", Habitat.GRASSLAND),
                                ("wetland", Habitat.WETLAND)):
        if f"[egg] in [{hab_name}]" in desc:
            return board.get_row(hab_enum).total_eggs()
    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[egg] in [{nt}]" in desc:
            from backend.models.enums import NestType
            nest_map = {"bowl": NestType.BOWL, "cavity": NestType.CAVITY,
                        "ground": NestType.GROUND, "platform": NestType.PLATFORM}
            return sum(s.eggs for r in board.all_rows() for s in r.slots
                       if s.bird and (s.bird.nest_type == nest_map[nt]
                                      or s.bird.nest_type == NestType.WILD))

    # Nest-type bird counts
    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[{nt}] [bird]" in desc:
            from backend.models.enums import NestType
            nest_map = {"bowl": NestType.BOWL, "cavity": NestType.CAVITY,
                        "ground": NestType.GROUND, "platform": NestType.PLATFORM}
            return len(board.birds_with_nest(nest_map[nt]))

    # VP thresholds
    if "[bird] worth >4 [feather]" in desc:
        return sum(1 for b in board.all_birds() if b.victory_points > 4)

    # Power color goals
    if "brown powers" in desc:
        return sum(1 for b in board.all_birds() if b.color == PowerColor.BROWN)
    if "white & no powers" in desc:
        return sum(1 for b in board.all_birds()
                   if b.color in (PowerColor.WHITE, PowerColor.NONE))

    # Tucked cards
    if "[bird_with_tucked_card]" in desc:
        return sum(1 for r in board.all_rows() for s in r.slots
                   if s.bird and s.tucked_cards > 0)

    # Filled columns
    if "filled columns" in desc:
        min_birds = min(r.bird_count for r in board.all_rows())
        return min_birds

    # Beak direction
    if "[beak_pointing_left]" in desc:
        return sum(1 for b in board.all_birds() if b.beak_direction == BeakDirection.LEFT)
    if "[beak_pointing_right]" in desc:
        return sum(1 for b in board.all_birds() if b.beak_direction == BeakDirection.RIGHT)

    return 0.0


def _max_goal_contribution_per_action(goal: Goal) -> float:
    """Conservative upper bound for per-action progress toward a goal."""
    desc = goal.description.lower()
    if "[egg]" in desc:
        return 2.0
    if "facedown [card]" in desc:
        return 2.0
    if "[bird]" in desc:
        return 1.0
    if "[nest]" in desc:
        return 1.0
    return 1.0


def _should_concede_goal(
    game: GameState,
    player: Player,
    goal_index: int,
) -> tuple[bool, bool]:
    """Return (concede_first_place, pursue_second_place)."""
    if goal_index < 0 or goal_index >= len(game.round_goals):
        return False, False
    goal = game.round_goals[goal_index]
    opponents = [p for p in game.players if p.name != player.name]
    if not opponents:
        return False, False

    my_progress = _estimate_goal_progress(player, goal)
    opp_progress = sorted((_estimate_goal_progress(opp, goal) for opp in opponents), reverse=True)
    best_opp = opp_progress[0]
    gap_to_first = max(0.0, best_opp - my_progress)

    goal_round = goal_index + 1
    if goal_round < game.current_round:
        return True, False

    remaining_actions = max(0, player.action_cubes_remaining)
    for r in range(game.current_round + 1, min(goal_round, ROUNDS) + 1):
        remaining_actions += ACTIONS_PER_ROUND[r - 1]
    max_catchup = remaining_actions * _max_goal_contribution_per_action(goal)

    if gap_to_first <= max_catchup:
        return False, False

    if len(opp_progress) >= 2:
        gap_to_second = max(0.0, opp_progress[1] - my_progress)
    else:
        gap_to_second = gap_to_first
    if gap_to_second <= max_catchup:
        return True, True
    return True, False


def _goal_alignment_value(game: GameState, bird: Bird, habitat: Habitat,
                          weights: HeuristicWeights) -> float:
    """Calculate how much playing this bird helps with current and future round goals.

    Now opponent-aware: values contributions more when you're behind/close on a goal,
    less when you're already comfortably ahead.
    """
    if not game.round_goals:
        return 0.0

    player = game.current_player
    opponents = [p for p in game.players if p.name != player.name]

    total = 0.0
    for i in range(game.current_round - 1, min(len(game.round_goals), 4)):
        goal = game.round_goals[i]
        concede_first, pursue_second = _should_concede_goal(game, player, i)
        if concede_first and not pursue_second:
            continue
        contribution = _estimate_goal_contribution(bird, habitat, goal)
        if contribution <= 0:
            continue

        rounds_away = i - (game.current_round - 1)
        time_discount = 1.0 / (1.0 + rounds_away * 0.5)
        first_place_pts = max(goal.scoring[3], 0)

        # Opponent-aware multiplier: double down when ahead, push hard when close
        position_mult = 1.0
        if opponents:
            my_progress = _estimate_goal_progress(player, goal)
            best_opp = max(_estimate_goal_progress(opp, goal) for opp in opponents)
            gap = my_progress - best_opp
            if gap >= 3:
                # Comfortably ahead — maintain lead (still worth investing)
                position_mult = 0.8
            elif gap >= 1:
                # Slightly ahead — push to secure 1st place
                position_mult = 1.2
            elif gap >= -1:
                # Tied or slightly behind — high value to push ahead
                position_mult = 1.4
            else:
                # Far behind — contribution still valuable but catching up is hard
                position_mult = 0.7
        # Urgency: fewer actions left in round means goal contributions matter more
        if player.action_cubes_remaining <= 2:
            position_mult *= 1.25

        # Bonus if this contribution is likely to flip/tie the lead
        flip_bonus = 0.0
        if opponents:
            my_progress = _estimate_goal_progress(player, goal)
            best_opp = max(_estimate_goal_progress(opp, goal) for opp in opponents)
            if my_progress < best_opp and my_progress + contribution >= best_opp:
                flip_bonus = first_place_pts * 0.25
            elif my_progress == best_opp and contribution > 0:
                flip_bonus = first_place_pts * 0.15

        goal_value = first_place_pts * 0.4 * position_mult + flip_bonus
        if pursue_second:
            goal_value *= 0.4
        total += contribution * time_discount * goal_value

    return total * weights.goal_alignment


def _egg_goal_alignment(game: GameState) -> float:
    """Extra value of laying eggs when egg-related round goals are active."""
    if not game.round_goals:
        return 0.0

    bonus = 0.0
    for i in range(game.current_round - 1, min(len(game.round_goals), 4)):
        goal = game.round_goals[i]
        desc = goal.description.lower()
        if "[egg]" in desc:
            rounds_away = i - (game.current_round - 1)
            time_discount = 1.0 / (1.0 + rounds_away * 0.5)
            first_place_pts = max(goal.scoring[3], 0)
            bonus += time_discount * first_place_pts * 0.15
    return bonus


def _extract_power_count(text: str) -> int:
    """Extract an approximate count (1-5) from power text."""
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))
    for word, val in (
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
    ):
        if re.search(rf"\b{word}\b", text):
            return val
    return 1


def _build_power_context(
    game: GameState,
    player: Player,
    bird: Bird,
    slot_index: int,
    habitat: Habitat,
    action_cost: int = 0,
) -> PowerContext:
    """Create PowerContext with precomputed game-phase/resource fields."""
    rounds_remaining = max(0, ROUNDS - game.current_round)
    future_actions = sum(ACTIONS_PER_ROUND[r - 1] for r in range(game.current_round + 1, ROUNDS + 1))
    actions_remaining = max(0, player.action_cubes_remaining - action_cost) + future_actions
    food_total = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
    egg_space_total = sum(
        s.eggs_space()
        for row in player.board.all_rows()
        for s in row.slots
        if s.bird
    )
    tracker = getattr(game, "deck_tracker", None)
    tracker_count = (
        tracker.remaining_count
        if tracker is not None and tracker.remaining_count > 0
        else game.deck_remaining
    )
    return PowerContext(
        game_state=game,
        player=player,
        bird=bird,
        slot_index=slot_index,
        habitat=habitat,
        rounds_remaining=rounds_remaining,
        actions_remaining=actions_remaining,
        hand_size=player.hand_size,
        food_total=food_total,
        egg_space_total=egg_space_total,
        deck_remaining=tracker_count,
    )


def _expected_habitat_activations(
    game: GameState,
    player: Player,
    habitat: Habitat,
    slot_index: int | None = None,
    post_action: bool = False,
) -> float:
    """Estimate how often a habitat row will activate for this player."""
    if game.current_round > ROUNDS:
        return 0.0

    habitat_share = _expected_habitat_activation_shares(game, player).get(habitat, 1.0 / 3.0)
    total = 0.0

    current_round_actions = player.action_cubes_remaining - (1 if post_action else 0)
    total += max(0, current_round_actions) * habitat_share

    for round_num in range(game.current_round + 1, ROUNDS + 1):
        total += ACTIONS_PER_ROUND[round_num - 1] * habitat_share

    if slot_index is not None:
        position_activation_factor = max(0.3, 1.0 - slot_index * 0.15)
        total *= position_activation_factor
    return total


def _brown_row_power_density(game: GameState, player: Player, habitat: Habitat) -> float:
    """Estimated per-activation point density of a habitat's brown row."""
    row = player.board.get_row(habitat)
    density = 0.0
    for i, slot in enumerate(row.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=slot.bird,
            slot_index=i,
            habitat=habitat,
        )
        # Slightly favor left slots that trigger more often within the row.
        density += max(0.0, float(power.estimate_value(ctx))) * max(0.5, 1.0 - i * 0.1)
    return density


def _expected_habitat_activation_shares(game: GameState, player: Player) -> dict[Habitat, float]:
    """Allocate action share across habitats using brown-power density.

    Uses a floor per habitat so low-density rows are still considered.
    """
    habitats = (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND)
    raw = {hab: _brown_row_power_density(game, player, hab) for hab in habitats}
    total_raw = float(sum(raw.values()))

    if total_raw <= 1e-9:
        return {hab: 1.0 / len(habitats) for hab in habitats}

    floor = 0.2
    flex = max(0.0, 1.0 - floor * len(habitats))
    shares = {hab: floor + (raw[hab] / total_raw) * flex for hab in habitats}
    norm = float(sum(shares.values()))
    if norm <= 1e-9:
        return {hab: 1.0 / len(habitats) for hab in habitats}
    return {hab: shares[hab] / norm for hab in habitats}


def _is_zero_cost_point_generator(power) -> bool:
    """Brown powers that are near-deterministic point generation."""
    from backend.powers.templates.tuck_cards import TuckFromDeck
    from backend.powers.templates.special import FlockingPower
    from backend.powers.templates.cache_food import CacheFoodFromSupply

    return isinstance(power, (TuckFromDeck, CacheFoodFromSupply, FlockingPower))


def _brown_power_discount(power, bird: Bird, weights: HeuristicWeights) -> float:
    """Discount factor by brown power class and reliability."""
    from backend.powers.templates.tuck_cards import TuckFromDeck
    from backend.powers.templates.special import FlockingPower
    from backend.powers.templates.cache_food import CacheFoodFromSupply
    from backend.powers.templates.gain_food import (
        GainFoodFromSupply,
        GainFoodFromFeeder,
        GainFoodFromSupplyOrCache,
        ResetFeederGainFood,
    )
    from backend.powers.templates.draw_cards import DrawCards, DrawFromTray, DrawBonusCards
    from backend.powers.templates.lay_eggs import LayEggs, LayEggsEachBirdInRow

    if isinstance(power, (TuckFromDeck, CacheFoodFromSupply, FlockingPower)):
        return 0.95
    if bird.is_predator:
        return weights.predator_penalty
    if isinstance(
        power,
        (
            GainFoodFromSupply,
            GainFoodFromFeeder,
            GainFoodFromSupplyOrCache,
            ResetFeederGainFood,
            DrawCards,
            DrawFromTray,
            DrawBonusCards,
            LayEggs,
            LayEggsEachBirdInRow,
        ),
    ):
        return 0.5
    return 0.6


def _effective_brown_power_value(
    raw_power_value: float,
    total_expected_activations: float,
    power,
    bird: Bird,
    weights: HeuristicWeights,
) -> float:
    """Discounted brown-power value with a floor for zero-cost generators."""
    discounted = raw_power_value * total_expected_activations * _brown_power_discount(power, bird, weights)
    if _is_zero_cost_point_generator(power):
        # Zero-cost tuck/cache engines are close to deterministic points.
        discounted = max(discounted, total_expected_activations * 0.9)
    return discounted


def _engine_chain_value(
    game: GameState,
    player: Player,
    habitat: Habitat,
    new_bird: Bird,
    rounds_remaining: int,
) -> float:
    """Estimate compound engine value by simulating one brown activation chain."""
    row = player.board.get_row(habitat)
    birds_in_row = [slot.bird for slot in row.slots if slot.bird]
    birds_in_row.append(new_bird)
    if not any(b and b.color == PowerColor.BROWN for b in birds_in_row):
        return 0.0

    food_gained = 0.0
    cards_drawn = 0.0
    tuck_points = 0.0
    cache_points = 0.0
    egg_points = 0.0

    # Brown powers resolve from right-to-left in the activated habitat.
    for b in reversed(birds_in_row):
        if not b or b.color != PowerColor.BROWN:
            continue
        text = b.power_text.lower()
        count = _extract_power_count(text)

        if "gain" in text and "food" in text:
            food_gained += count
        if "draw" in text and "card" in text:
            cards_drawn += count

        if "tuck" in text:
            if "deck" in text or "top card" in text:
                tuck_points += count
            elif "hand" in text:
                tucks = min(cards_drawn, float(count))
                tuck_points += tucks
                cards_drawn -= tucks
            else:
                tuck_points += min(1.0, float(count))

        if "cache" in text:
            if "supply" in text:
                cache_points += count
            else:
                cached = min(food_gained, float(count))
                cache_points += cached
                food_gained -= cached

        if "lay" in text and "egg" in text:
            egg_points += count * 0.6

    points_per_activation = (
        tuck_points +
        cache_points +
        egg_points +
        food_gained * 0.25 +
        cards_drawn * 0.2
    )
    if points_per_activation <= 0:
        return 0.0

    expected_activations = _expected_habitat_activations(
        game=game,
        player=player,
        habitat=habitat,
        slot_index=None,
        post_action=True,
    )
    phase_multiplier = 0.8 + 0.4 * (rounds_remaining / ROUNDS if ROUNDS else 0.0)
    return points_per_activation * expected_activations * phase_multiplier


def _bonus_card_marginal_vp(player: Player, bird: Bird) -> float:
    """Calculate the marginal bonus card VP from playing this bird.

    Returns the total VP improvement across all bonus cards the player holds
    if this bird were added to their board.
    """
    total = 0.0
    for bc in player.bonus_cards:
        if bc.name not in bird.bonus_eligibility:
            continue
        current_qualifying = sum(
            1 for b in player.board.all_birds() if bc.name in b.bonus_eligibility
        )
        current_score = bc.score(current_qualifying)
        new_score = bc.score(current_qualifying + 1)
        total += (new_score - current_score)
    return total


def _bonus_card_threshold_urgency(player: Player, bird: Bird, rounds_remaining: int) -> float:
    """Extra urgency when a played bird crosses or approaches a bonus tier."""
    urgency = 0.0
    for bc in player.bonus_cards:
        if bc.name not in bird.bonus_eligibility:
            continue

        current_qualifying = sum(
            1 for b in player.board.all_birds() if bc.name in b.bonus_eligibility
        )
        score_now = bc.score(current_qualifying)
        score_plus1 = bc.score(current_qualifying + 1)
        score_plus2 = bc.score(current_qualifying + 2)

        # Immediate tier crossing: treat as full urgency (in addition to marginal value).
        immediate_delta = score_plus1 - score_now
        if immediate_delta > 0:
            urgency += immediate_delta
            continue

        # One-away from crossing after this play: apply probability-weighted urgency.
        near_delta = score_plus2 - score_plus1
        if near_delta <= 0:
            continue
        qualifying_in_hand = sum(
            1
            for b in player.hand
            if b.name != bird.name and bc.name in b.bonus_eligibility
        )
        hand_signal = qualifying_in_hand / max(1, player.hand_size)
        rounds_signal = rounds_remaining / ROUNDS if ROUNDS else 0.0
        probability = min(0.9, max(0.1, 0.25 + 0.45 * hand_signal + 0.35 * rounds_signal))
        urgency += near_delta * probability
    return urgency


def _is_play_bird_power(bird: Bird) -> bool:
    """Check if a bird has a white power that lets you play another bird."""
    if bird.color != PowerColor.WHITE:
        return False
    text = bird.power_text.lower()
    return "play" in text and "bird" in text


def detect_strategic_phase(game: GameState, player: Player) -> str:
    """Detect high-level strategic phase: engine, transition, or scoring."""
    rounds_remaining = max(0, ROUNDS - game.current_round)
    brown_birds = sum(
        1 for row in player.board.all_rows() for slot in row.slots
        if slot.bird and slot.bird.color == PowerColor.BROWN
    )
    point_generators = sum(
        1 for row in player.board.all_rows() for slot in row.slots
        if slot.bird and slot.bird.color == PowerColor.BROWN
        and any(k in slot.bird.power_text.lower() for k in ("tuck", "cache", "flock"))
    )

    if rounds_remaining >= 2 and brown_birds < 3 and point_generators < 2:
        return "engine"
    if rounds_remaining <= 1 or (point_generators >= 3 and rounds_remaining <= 2):
        return "scoring"
    return "transition"


def _move_play_bird(game: GameState, player: Player, move: Move) -> Bird | None:
    """Resolve bird object for a play-bird move without assuming registry only."""
    for b in player.hand:
        if b.name == move.bird_name:
            return b
    from backend.data.registries import get_bird_registry

    return get_bird_registry().get(move.bird_name)


# --- Position evaluation ---

def evaluate_position(game: GameState, player: Player,
                      weights: HeuristicWeights | None = None) -> float:
    """Evaluate a player's board position. Higher = better.

    Uses dynamic phase-aware weights by default.
    Combines actual score with estimated future value from engine and resources.
    """
    if weights is None:
        weights = dynamic_weights(game)
    # Current concrete score
    score = calculate_score(game, player)
    value = (
        score.bird_vp * weights.bird_vp +
        score.eggs * weights.egg_points +
        score.cached_food * weights.cached_food_points +
        score.tucked_cards * weights.tucked_card_points +
        score.bonus_cards * weights.bonus_card_points +
        score.round_goals * weights.round_goal_points +
        score.nectar * weights.nectar_points
    )

    # Engine value: expected future points from brown powers per activation
    engine_val = _estimate_engine_value(game, player, weights)
    rounds_remaining = max(0, ROUNDS - game.current_round)

    # Weight engine more in early game
    engine_weight = weights.engine_value
    if game.current_round <= 2:
        engine_weight += weights.early_game_engine_bonus

    value += engine_val * rounds_remaining * engine_weight

    # Resource value: food and cards enable future plays
    food_total = player.food_supply.total_non_nectar()
    value += min(food_total, 8) * weights.food_in_supply
    value += min(player.hand_size, 5) * weights.cards_in_hand

    # Late-game resource surplus penalty: excess food/cards with no way to use them
    if rounds_remaining == 0 and player.action_cubes_remaining <= 1:
        # Last turn of the game — food and cards in hand score 0 points
        playable = sum(1 for b in player.hand if food_total >= b.food_cost.total)
        if playable == 0:
            value -= food_total * 0.1  # Can't use this food
            value -= player.hand_size * 0.15  # Can't play these birds

    # Nectar timing: nectar is more valuable with more rounds remaining
    # But worthless on last turn of round since it clears at round end
    nectar_count = player.food_supply.get(FoodType.NECTAR)
    if nectar_count > 0:
        if player.action_cubes_remaining <= 1:
            # Last turn of round: nectar in supply is about to be lost
            # Strong penalty — each unspent nectar is a wasted resource
            value -= nectar_count * 2.0
        elif rounds_remaining > 0:
            value += nectar_count * weights.nectar_early_bonus * (rounds_remaining / ROUNDS)

    # Habitat diversity bonus (birds in all 3 habitats is strategically strong)
    habitats_with_birds = sum(
        1 for row in player.board.all_rows() if row.bird_count > 0
    )
    if habitats_with_birds >= 3:
        value += weights.habitat_diversity_bonus

    return value


def _estimate_engine_value(game: GameState, player: Player,
                           weights: HeuristicWeights | None = None) -> float:
    """Estimate the per-activation value of the player's engine.

    Sums the estimated value of all brown powers on the board,
    weighted by how often they'll be activated (rightmost birds activate more).
    Predator powers are discounted for unreliability.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = 0.0
    for row in player.board.all_rows():
        for i, slot in enumerate(row.slots):
            if not slot.bird:
                continue
            if slot.bird.color != PowerColor.BROWN:
                continue
            power = get_power(slot.bird)
            if isinstance(power, NoPower):
                continue
            ctx = _build_power_context(
                game=game,
                player=player,
                bird=slot.bird,
                slot_index=i,
                habitat=row.habitat,
            )
            raw_power_val = power.estimate_value(ctx)
            # Position-aware: earlier slots activate more per use of that habitat.
            activation_factor = max(0.3, 1.0 - i * 0.15)
            total += _effective_brown_power_value(
                raw_power_value=raw_power_val,
                total_expected_activations=activation_factor,
                power=power,
                bird=slot.bird,
                weights=weights,
            )
    return total


def _can_play_any_bird_now(player: Player) -> bool:
    """Approximate whether the player can legally play any current hand bird."""
    available_food = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
    total_eggs = player.board.total_eggs()

    for bird in player.hand:
        if available_food < bird.food_cost.total:
            continue
        for hab in bird.habitats:
            row = player.board.get_row(hab)
            slot_idx = row.next_empty_slot()
            if slot_idx is None:
                continue
            egg_cost = EGG_COST_BY_COLUMN[slot_idx] if slot_idx < len(EGG_COST_BY_COLUMN) else 0
            if total_eggs >= egg_cost:
                return True
    return False


def _tempo_penalty(game: GameState, player: Player, move: Move) -> float:
    """Penalty for tempo-wasting actions in low-conversion states."""
    penalty = 0.0

    if move.action_type == ActionType.GAIN_FOOD:
        total_food = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
        if total_food >= 5 and not _can_play_any_bird_now(player):
            penalty -= 1.0

    elif move.action_type == ActionType.DRAW_CARDS:
        birds_on_board = sum(row.bird_count for row in player.board.all_rows())
        hand_cap = max(0, 15 - birds_on_board)
        if player.hand_size >= hand_cap:
            penalty -= 0.5

    elif move.action_type == ActionType.PLAY_BIRD and game.current_round >= 4:
        bird = _move_play_bird(game, player, move)
        if bird and bird.color == PowerColor.BROWN:
            row = player.board.get_row(move.habitat)
            slot_idx = row.bird_count
            expected_activations = _expected_habitat_activations(
                game=game,
                player=player,
                habitat=move.habitat,
                slot_index=slot_idx,
                post_action=True,
            )
            if expected_activations <= 2.0:
                penalty -= 0.5

    return penalty


def _estimate_move_value(game: GameState, player: Player, move: Move,
                         weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Estimate the value of a specific move without full simulation.

    Uses a combination of immediate value and positional improvement.
    """
    strategic_phase = detect_strategic_phase(game, player)

    if move.action_type == ActionType.PLAY_BIRD:
        value = _evaluate_play_bird(game, player, move, weights)
        bird = _move_play_bird(game, player, move)
        if bird and bird.color == PowerColor.BROWN:
            text = bird.power_text.lower()
            is_tuck_cache = ("tuck" in text) or ("cache" in text)
            if strategic_phase == "engine":
                value *= 1.3
            elif strategic_phase == "scoring":
                if is_tuck_cache:
                    value *= 1.2
                else:
                    value *= 0.6
    elif move.action_type == ActionType.GAIN_FOOD:
        value = _evaluate_gain_food(game, player, move, weights)
        if strategic_phase == "engine":
            value *= 1.2
    elif move.action_type == ActionType.LAY_EGGS:
        value = _evaluate_lay_eggs(game, player, move, weights)
        if strategic_phase == "engine":
            value *= 0.7
        elif strategic_phase == "scoring":
            value *= 1.3
    elif move.action_type == ActionType.DRAW_CARDS:
        value = _evaluate_draw_cards(game, player, move, weights)
    else:
        value = 0.0

    value += _tempo_penalty(game, player, move)
    return value


def estimate_move_breakdown(
    game: GameState,
    player: Player,
    move: Move,
    weights: HeuristicWeights | None = None,
) -> dict:
    """Return a rough component breakdown for a move's heuristic score."""
    if weights is None:
        weights = dynamic_weights(game)

    breakdown: dict[str, float] = {}
    rounds_remaining = max(0, ROUNDS - game.current_round)

    if move.action_type == ActionType.PLAY_BIRD:
        from backend.data.registries import get_bird_registry
        bird = get_bird_registry().get(move.bird_name)
        if not bird:
            return {}

        # Base VP + egg capacity
        breakdown["bird_vp"] = round(bird.victory_points * weights.bird_vp, 2)
        eggs_likely = min(bird.egg_limit, max(1, rounds_remaining))
        breakdown["egg_capacity"] = round(eggs_likely * weights.egg_points * 0.8, 2)

        # Power value estimate
        power = get_power(bird)
        power_val = 0.0
        if bird.color == PowerColor.BROWN and not isinstance(power, NoPower):
            row = player.board.get_row(move.habitat)
            slot_idx = row.bird_count
            ctx = _build_power_context(
                game=game,
                player=player,
                bird=bird,
                slot_index=slot_idx,
                habitat=move.habitat,
                action_cost=1,
            )
            total_expected_activations = _expected_habitat_activations(
                game=game,
                player=player,
                habitat=move.habitat,
                slot_index=slot_idx,
                post_action=True,
            )
            raw_power_val = power.estimate_value(ctx)
            power_val = _effective_brown_power_value(
                raw_power_value=raw_power_val,
                total_expected_activations=total_expected_activations,
                power=power,
                bird=bird,
                weights=weights,
            )
            power_val += _engine_chain_value(game, player, move.habitat, bird, rounds_remaining)
        elif bird.color == PowerColor.WHITE and not isinstance(power, NoPower):
            ctx = _build_power_context(
                game=game,
                player=player,
                bird=bird,
                slot_index=0,
                habitat=move.habitat,
                action_cost=1,
            )
            power_val = power.estimate_value(ctx) * 0.8
            if _is_play_bird_power(bird):
                power_val += weights.play_another_bird_bonus * min(rounds_remaining, 2)
        elif bird.color == PowerColor.PINK:
            ctx = _build_power_context(
                game=game,
                player=player,
                bird=bird,
                slot_index=0,
                habitat=move.habitat,
                action_cost=1,
            )
            num_opponents = game.num_players - 1
            trigger_prob = 1.0 - (0.5 ** num_opponents) if num_opponents > 0 else 0.0
            from backend.config import ACTIONS_PER_ROUND
            actions_remaining = player.action_cubes_remaining
            future_triggers = rounds_remaining * ACTIONS_PER_ROUND[min(game.current_round, len(ACTIONS_PER_ROUND) - 1)] * trigger_prob
            current_triggers = actions_remaining * trigger_prob
            total_pink_triggers = current_triggers + future_triggers
            power_val = power.estimate_value(ctx) * total_pink_triggers * 0.5
        elif bird.color in (PowerColor.TEAL, PowerColor.YELLOW):
            ctx = _build_power_context(
                game=game,
                player=player,
                bird=bird,
                slot_index=0,
                habitat=move.habitat,
                action_cost=1,
            )
            power_val = power.estimate_value(ctx) * rounds_remaining * 0.6

        if power_val:
            breakdown["power_engine"] = round(power_val, 2)

        # Bonus and goal alignment
        bonus = (
            _bonus_card_marginal_vp(player, bird) +
            _bonus_card_threshold_urgency(player, bird, rounds_remaining)
        ) * weights.bonus_card_points
        if bonus:
            breakdown["bonus_cards"] = round(bonus, 2)
        goal_val = _goal_alignment_value(game, bird, move.habitat, weights)
        if goal_val:
            breakdown["round_goal"] = round(goal_val, 2)
        prior = bird_prior_value(bird.name, game.current_round)
        if prior:
            breakdown["tier_prior"] = round(prior, 2)

        # Costs
        nectar_in_payment = move.food_payment.get(FoodType.NECTAR, 0)
        non_nectar_spent = sum(v for ft, v in move.food_payment.items() if ft != FoodType.NECTAR)
        cost = non_nectar_spent * 0.15 + nectar_in_payment * 0.05
        row = player.board.get_row(move.habitat)
        slot_idx = row.next_empty_slot()
        if slot_idx is not None:
            cost += EGG_COST_BY_COLUMN[slot_idx] * 0.3
        if cost:
            breakdown["costs"] = round(-cost, 2)

    elif move.action_type == ActionType.GAIN_FOOD:
        bird_count = player.board.forest.bird_count
        column = get_action_column(game.board_type, Habitat.FOREST, bird_count)
        food_count = column.base_gain + move.bonus_count
        breakdown["food_gain"] = round(food_count * 0.5, 2)

        # Engine activation value
        engine = 0.0
        for i, slot in enumerate(player.board.forest.slots):
            if slot.bird and slot.bird.color == PowerColor.BROWN:
                power = get_power(slot.bird)
                if not isinstance(power, NoPower):
                    ctx = _build_power_context(
                        game=game,
                        player=player,
                        bird=slot.bird,
                        slot_index=i,
                        habitat=Habitat.FOREST,
                    )
                    val = power.estimate_value(ctx)
                    if slot.bird.is_predator:
                        val *= weights.predator_penalty
                    engine += val
        if engine:
            breakdown["engine_activation"] = round(engine, 2)

        # Unlock value
        food_available = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
        affordable_before = sum(1 for b in player.hand if food_available >= b.food_cost.total)
        affordable_after = sum(1 for b in player.hand if food_available + food_count >= b.food_cost.total)
        newly_affordable = affordable_after - affordable_before
        if newly_affordable > 0:
            breakdown["bird_unlocks"] = round(newly_affordable * 1.5, 2)

        # Nectar early value
        if move.food_choices and FoodType.NECTAR in move.food_choices:
            nectar_bonus = weights.nectar_early_bonus * (rounds_remaining / ROUNDS) if rounds_remaining > 0 else 0
            breakdown["nectar_value"] = round(nectar_bonus, 2) if nectar_bonus else 0.0

    elif move.action_type == ActionType.LAY_EGGS:
        bird_count = player.board.grassland.bird_count
        column = get_action_column(game.board_type, Habitat.GRASSLAND, bird_count)
        egg_count = column.base_gain + move.bonus_count
        eligible_space = sum(
            slot.eggs_space()
            for row in player.board.all_rows()
            for slot in row.slots
            if slot.bird
        )
        actual_eggs = min(egg_count, eligible_space)
        breakdown["eggs"] = round(actual_eggs * weights.egg_points, 2)

        # Engine activation
        engine = 0.0
        for i, slot in enumerate(player.board.grassland.slots):
            if slot.bird and slot.bird.color == PowerColor.BROWN:
                power = get_power(slot.bird)
                if not isinstance(power, NoPower):
                    ctx = _build_power_context(
                        game=game,
                        player=player,
                        bird=slot.bird,
                        slot_index=i,
                        habitat=Habitat.GRASSLAND,
                    )
                    engine += power.estimate_value(ctx)
        if engine:
            breakdown["engine_activation"] = round(engine, 2)

        goal_val = _egg_goal_alignment(game)
        if goal_val:
            breakdown["round_goal"] = round(goal_val, 2)

    elif move.action_type == ActionType.DRAW_CARDS:
        bird_count = player.board.wetland.bird_count
        column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
        card_count = column.base_gain + move.bonus_count
        phase = rounds_remaining / ROUNDS if ROUNDS else 0
        card_base_value = 0.2 + 0.1 * phase
        breakdown["cards"] = round(card_count * card_base_value, 2)

        tray_value = 0.0
        for idx in move.tray_indices:
            if idx < len(game.card_tray.face_up):
                tray_bird = game.card_tray.face_up[idx]
                tv = tray_bird.victory_points * 0.5
                tv += min(tray_bird.egg_limit, max(1, rounds_remaining)) * 0.3
                if tray_bird.color == PowerColor.BROWN and rounds_remaining >= 2:
                    tv += rounds_remaining * 0.4
                tray_value += tv
        if tray_value:
            breakdown["tray_value"] = round(tray_value, 2)

        deck_draws = move.deck_draws
        if deck_draws > 0:
            deck_per_card = 0.2 + 0.1 * phase
            breakdown["deck_value"] = round(deck_draws * deck_per_card, 2)

    # Total estimate
    total = _estimate_move_value(game, player, move, weights)
    breakdown["total_estimate"] = round(total, 2)
    return breakdown


def _nectar_majority_value(game: GameState, player: Player,
                           habitat: Habitat, nectar_count: int) -> float:
    """Estimate the marginal scoring improvement from spending nectar in a habitat.

    Nectar majority per habitat: 1st place = 5pts, 2nd = 2pts.
    Ties share prizes for all positions the tied group occupies (rounded down).
    Returns the marginal VP improvement from spending `nectar_count` more nectar.
    """
    from backend.config import BoardType
    if game.board_type != BoardType.OCEANIA or nectar_count <= 0:
        return 0.0

    row = player.board.get_row(habitat)
    my_current = row.nectar_spent
    my_new = my_current + nectar_count

    # Gather all opponent nectar counts in this habitat
    opponents = [p for p in game.players if p.name != player.name]
    opp_counts = [p.board.get_row(habitat).nectar_spent for p in opponents]

    def _estimate_points(my_nectar: int) -> float:
        if my_nectar <= 0:
            return 0.0
        # Count players above, tied with, and below us
        above = sum(1 for c in opp_counts if c > my_nectar)
        tied_with = sum(1 for c in opp_counts if c == my_nectar)
        # My position (0-based): number of opponents ahead of me
        position = above
        # Prize pools: 1st=5, 2nd=2
        prizes = [5, 2]
        if position >= len(prizes):
            return 0.0  # 3rd or worse, no points
        # Tied group occupies positions [position, position + tied_with]
        # (tied_with = opponents at same level; total group size = tied_with + 1 for me)
        group_size = tied_with + 1
        pool = sum(prizes[position:position + group_size])
        return pool // group_size

    score_before = _estimate_points(my_current)
    score_after = _estimate_points(my_new)
    return max(0.0, score_after - score_before)


def _nectar_spend_path_viable(game: GameState, player: Player, added_nectar: int = 1) -> bool:
    """Whether newly gained nectar can likely be spent before round end."""
    actions_left_after_gain = max(0, player.action_cubes_remaining - 1)
    if actions_left_after_gain <= 0:
        return False

    available_non_nectar = player.food_supply.total_non_nectar()
    nectar_after_gain = player.food_supply.get(FoodType.NECTAR) + max(0, added_nectar)
    eggs_total = player.board.total_eggs()

    for bird in player.hand:
        if bird.food_cost.total > available_non_nectar + nectar_after_gain:
            continue
        for hab in bird.habitats:
            row = player.board.get_row(hab)
            slot_idx = row.next_empty_slot()
            if slot_idx is None:
                continue
            egg_cost = EGG_COST_BY_COLUMN[slot_idx]
            if eggs_total >= egg_cost:
                return True
    return False


def _evaluate_play_bird(game: GameState, player: Player, move: Move,
                        weights: HeuristicWeights) -> float:
    """Evaluate a play-bird move."""
    from backend.data.registries import get_bird_registry
    bird_reg = get_bird_registry()
    bird = bird_reg.get(move.bird_name)
    if not bird:
        return 0.0

    value = 0.0
    rounds_remaining = max(0, ROUNDS - game.current_round)

    # Immediate VP value
    value += bird.victory_points * weights.bird_vp

    # Egg capacity value: each egg slot is worth ~1pt over the remaining game
    # A 5-egg bird in round 1 will realistically fill 3-4 eggs = 3-4 pts
    eggs_likely = min(bird.egg_limit, max(1, rounds_remaining))
    value += eggs_likely * weights.egg_points * 0.8

    # Power engine value (brown powers multiply over remaining rounds)
    power = get_power(bird)
    if bird.color == PowerColor.BROWN and not isinstance(power, NoPower):
        row = player.board.get_row(move.habitat)
        slot_idx = row.bird_count  # where this bird will be placed
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=bird,
            slot_index=slot_idx,
            habitat=move.habitat,
            action_cost=1,
        )
        total_expected_activations = _expected_habitat_activations(
            game=game,
            player=player,
            habitat=move.habitat,
            slot_index=slot_idx,
            post_action=True,
        )
        raw_power_val = power.estimate_value(ctx)
        power_val = _effective_brown_power_value(
            raw_power_value=raw_power_val,
            total_expected_activations=total_expected_activations,
            power=power,
            bird=bird,
            weights=weights,
        )

        # Engine chains compound: outputs from one brown power feed later powers.
        power_val += _engine_chain_value(game, player, move.habitat, bird, rounds_remaining)

        value += power_val

    elif bird.color == PowerColor.WHITE and not isinstance(power, NoPower):
        row = player.board.get_row(move.habitat)
        slot_idx = row.bird_count
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=bird,
            slot_index=slot_idx,
            habitat=move.habitat,
            action_cost=1,
        )
        power_val = power.estimate_value(ctx) * 0.8
        # Play-another-bird: compute value of the best follow-up bird from hand
        if _is_play_bird_power(bird):
            from backend.powers.templates.play_bird import PlayAdditionalBird
            play_power = get_power(bird)
            egg_disc = play_power.egg_discount if isinstance(play_power, PlayAdditionalBird) else 0
            food_disc = play_power.food_discount if isinstance(play_power, PlayAdditionalBird) else 0
            hab_filter = play_power.habitat_filter if isinstance(play_power, PlayAdditionalBird) else None

            remaining_hand = [b for b in player.hand if b.name != bird.name]
            food_after = player.food_supply.total_non_nectar() - sum(move.food_payment.values())
            nectar_after = player.food_supply.get(FoodType.NECTAR)
            eggs_after = player.board.total_eggs() - (EGG_COST_BY_COLUMN[slot_idx] if slot_idx is not None else 0)
            best_followup = 0.0
            for candidate in remaining_hand:
                # Check habitat restriction
                if hab_filter and hab_filter not in candidate.habitats:
                    continue
                # Check food affordability (with discount)
                effective_food_cost = max(0, candidate.food_cost.total - food_disc)
                if effective_food_cost > food_after + nectar_after:
                    continue
                # Check egg affordability for the target slot
                target_hab = hab_filter or (list(candidate.habitats)[0] if candidate.habitats else None)
                if target_hab:
                    target_row = player.board.get_row(target_hab)
                    # +1 because we're placing the first bird in move.habitat
                    target_slot = target_row.bird_count + (1 if target_hab == move.habitat else 0)
                    if target_slot < len(EGG_COST_BY_COLUMN):
                        followup_egg_cost = max(0, EGG_COST_BY_COLUMN[target_slot] - egg_disc)
                        if eggs_after < followup_egg_cost:
                            continue
                # Quick estimate: VP + egg capacity + power value
                fv = candidate.victory_points
                fv += min(candidate.egg_limit, max(1, rounds_remaining)) * 0.8
                if candidate.color == PowerColor.BROWN:
                    fv += rounds_remaining * 1.0
                best_followup = max(best_followup, fv)
            if best_followup > 0:
                # Playing a free/discounted bird is near full value
                power_val += best_followup
            else:
                power_val += weights.play_another_bird_bonus * min(rounds_remaining, 2)
        value += power_val

    elif bird.color == PowerColor.PINK:
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=move.habitat,
            action_cost=1,
        )
        # Pink powers trigger at most once between your turns.
        # More opponents = higher probability of triggering per window.
        # Your triggers per round = your action cubes (one window per turn).
        num_opponents = game.num_players - 1
        # Trigger probability: with N opponents, chance at least one does the
        # triggering action. Rough estimate: ~50% per opponent for common actions.
        trigger_prob = 1.0 - (0.5 ** num_opponents) if num_opponents > 0 else 0.0
        # Triggers per round ≈ actions_this_round * trigger_probability
        from backend.config import ACTIONS_PER_ROUND
        actions_remaining = player.action_cubes_remaining
        future_triggers = rounds_remaining * ACTIONS_PER_ROUND[min(game.current_round, len(ACTIONS_PER_ROUND) - 1)] * trigger_prob
        current_triggers = actions_remaining * trigger_prob
        total_pink_triggers = current_triggers + future_triggers
        value += power.estimate_value(ctx) * total_pink_triggers * 0.5

    elif bird.color in (PowerColor.TEAL, PowerColor.YELLOW):
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=move.habitat,
            action_cost=1,
        )
        value += power.estimate_value(ctx) * rounds_remaining * 0.6

    # Bonus card delta + urgency when this play crosses/approaches tier thresholds.
    bonus_marginal = _bonus_card_marginal_vp(player, bird)
    bonus_urgency = _bonus_card_threshold_urgency(player, bird, rounds_remaining)
    value += (bonus_marginal + bonus_urgency) * weights.bonus_card_points

    # Round goal alignment
    value += _goal_alignment_value(game, bird, move.habitat, weights)

    # Data-driven bird strength prior (tier + per-bird overrides).
    # Keep this additive and small so board context still dominates.
    value += bird_prior_value(bird.name, game.current_round)

    # Grassland egg synergy: high-egg birds in grassland benefit from lay-eggs engine
    if move.habitat == Habitat.GRASSLAND and bird.egg_limit >= 3:
        value += (bird.egg_limit - 2) * weights.grassland_egg_synergy

    # Food cost penalty (opportunity cost of food spent) — keep light to not discourage bird plays
    nectar_in_payment = move.food_payment.get(FoodType.NECTAR, 0)
    non_nectar_spent = sum(v for ft, v in move.food_payment.items() if ft != FoodType.NECTAR)
    value -= non_nectar_spent * 0.15

    if player.action_cubes_remaining <= 1:
        # Last turn of round: nectar is FREE — it will be lost at round end anyway
        # Strong bonus for spending nectar that would otherwise expire
        value += nectar_in_payment * 1.5
    else:
        # Nectar has scoring value when spent (majority), so minimal penalty
        value -= nectar_in_payment * 0.05

    # Nectar majority bonus: spending nectar in a habitat improves end-game scoring
    if nectar_in_payment > 0:
        value += _nectar_majority_value(game, player, move.habitat, nectar_in_payment) * 0.8

    # Column egg cost penalty — light penalty since eggs spent are themselves scored
    row = player.board.get_row(move.habitat)
    slot_idx = row.next_empty_slot()
    if slot_idx is not None:
        egg_cost = EGG_COST_BY_COLUMN[slot_idx]
        value -= egg_cost * 0.3

    # Hand constraint synergy: boost if playing this bird unlocks a much better bird
    # (e.g., spending food on a 3-VP bird now frees space for a 9-VP bird next turn)
    if rounds_remaining >= 1 and not _is_play_bird_power(bird):
        remaining_hand = [b for b in player.hand if b.name != bird.name]
        food_after = player.food_supply.total_non_nectar() - sum(
            v for ft, v in move.food_payment.items() if ft != FoodType.NECTAR
        )
        nectar_after = player.food_supply.get(FoodType.NECTAR) - move.food_payment.get(FoodType.NECTAR, 0)
        for candidate in remaining_hand:
            # Is this candidate significantly better than the bird we're playing?
            candidate_vp = candidate.victory_points
            if candidate_vp <= bird.victory_points + 2:
                continue  # not enough upside to warrant the "unlock" bonus
            # Can we afford the candidate after playing this bird?
            if candidate.food_cost.total > food_after + max(0, nectar_after):
                continue
            # Does a habitat have space for it?
            for hab in candidate.habitats:
                target_row = player.board.get_row(hab)
                # Account for the bird we're about to place
                count = target_row.bird_count + (1 if hab == move.habitat else 0)
                if count < 5:
                    target_egg_cost = EGG_COST_BY_COLUMN[count] if count < len(EGG_COST_BY_COLUMN) else 2
                    total_eggs = player.board.total_eggs() - (egg_cost if slot_idx is not None else 0)
                    if total_eggs >= target_egg_cost:
                        # This play unlocks the candidate — bonus proportional to VP gap
                        unlock_bonus = (candidate_vp - bird.victory_points) * 0.3
                        value += min(unlock_bonus, 3.0)
                        break

    # Large hand urgency: with many birds in hand, playing them is increasingly important
    # since unplayed birds at game end score 0 points
    if player.hand_size >= 8:
        value += (player.hand_size - 6) * 0.3  # +0.6 at 8, +3.0 at 16

    return value


def _evaluate_gain_food(game: GameState, player: Player, move: Move,
                        weights: HeuristicWeights) -> float:
    """Evaluate a gain-food move."""
    bird_count = player.board.forest.bird_count
    column = get_action_column(game.board_type, Habitat.FOREST, bird_count)
    food_count = column.base_gain + move.bonus_count

    # Base value: food tokens gained
    value = food_count * 0.5

    # Bonus cost: paying a card/egg for +1 food is a net trade
    if move.bonus_count > 0:
        # Extra food is worth ~0.5, but you pay a card (~0.4) or food (~0.5)
        # Net positive when you need food more than the payment resource
        value += move.bonus_count * 0.3  # Slight net positive for the extra

    # Reset feeder value: worth more when current feeder options are bad
    if move.reset_bonus:
        # Resetting feeder costs a food but gives fresh dice — speculative value
        value += 0.4

    # Engine value: brown powers in forest will activate
    for i, slot in enumerate(player.board.forest.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=slot.bird,
            slot_index=i,
            habitat=Habitat.FOREST,
        )
        power_val = power.estimate_value(ctx)
        # Predator penalty for forest predators
        if slot.bird.is_predator:
            power_val *= weights.predator_penalty
        value += power_val

    # Food-for-birds bonus: food that directly unlocks bird plays is very valuable
    food_available = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
    affordable_before = sum(1 for b in player.hand if food_available >= b.food_cost.total)
    affordable_after = sum(1 for b in player.hand if food_available + food_count >= b.food_cost.total)
    newly_affordable = affordable_after - affordable_before
    if newly_affordable > 0:
        # This food directly unlocks bird plays — high value
        value += newly_affordable * 1.5
    elif affordable_after > 0 and player.food_supply.total_non_nectar() < 3:
        value += weights.food_for_birds_bonus * affordable_after
    if player.hand_size > 0 and player.food_supply.total_non_nectar() == 0:
        value += 2.0  # Urgency: zero food with birds in hand

    # Bonus card pursuit: extra value when food unlocks bonus-qualifying birds
    if newly_affordable > 0:
        for b in player.hand:
            if (food_available + food_count >= b.food_cost.total
                    and food_available < b.food_cost.total):
                marginal = _bonus_card_marginal_vp(player, b)
                if marginal > 0:
                    value += marginal * 0.4  # Discounted: still need a turn to play it

    # Goal pursuit: extra value when food enables playing goal-aligned birds
    if newly_affordable > 0 and game.round_goals:
        best_goal_food_val = 0.0
        for b in player.hand:
            if (food_available + food_count >= b.food_cost.total
                    and food_available < b.food_cost.total):
                for hab in b.habitats:
                    gv = _goal_alignment_value(game, b, hab, weights)
                    best_goal_food_val = max(best_goal_food_val, gv)
        value += best_goal_food_val * 0.3

    # Phase-dependent food penalty: food gained in round 4 is less useful
    # because fewer turns remain to spend it on birds
    rounds_remaining = max(0, ROUNDS - game.current_round)
    if rounds_remaining == 0:
        # Last round: food is only useful if we can play a bird this round
        if affordable_after == 0:
            value -= food_count * 0.3  # Surplus food = wasted action
    elif rounds_remaining <= 1 and affordable_after == 0 and player.hand_size == 0:
        value -= food_count * 0.2  # No birds to play, food is near-worthless

    # Reset moves: food choices are speculative (dice will be rerolled),
    # so discount food-specific bonuses/penalties
    reset_discount = 0.3 if move.reset_bonus else 1.0

    # Nectar choice value: flexible food + contributes to nectar majority
    # BUT nectar clears at end of round — heavily penalize on last turn
    if move.food_choices and FoodType.NECTAR in move.food_choices:
        nectar_in_choices = sum(1 for ft in move.food_choices if ft == FoodType.NECTAR)
        if player.action_cubes_remaining <= 1 and not _nectar_spend_path_viable(game, player, nectar_in_choices):
            # Last turn and no spend path: nectar expires.
            value -= nectar_in_choices * 3.5 * reset_discount
        else:
            rounds_remaining = max(0, ROUNDS - game.current_round)
            if _nectar_spend_path_viable(game, player, nectar_in_choices):
                # Nectar that can be converted into a bird play this round is high-value.
                value += nectar_in_choices * 1.2 * reset_discount
            if rounds_remaining > 0:
                value += weights.nectar_early_bonus * (rounds_remaining / ROUNDS) * reset_discount
            # Nectar majority-aware valuation: gaining nectar is more valuable
            # when it helps win or maintain majority in a habitat
            best_majority = 0.0
            for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
                nv = _nectar_majority_value(game, player, hab, 1)
                best_majority = max(best_majority, nv)
            if best_majority > 0:
                value += nectar_in_choices * best_majority * 0.4 * reset_discount

    # Food deficit targeting: bonus when gaining food types needed by best hand birds
    if move.food_choices and player.hand:
        # Find the most valuable bird in hand and its food needs
        best_bird_value = 0.0
        best_bird_needs: dict[FoodType, int] = {}
        for b in player.hand:
            bv = b.victory_points + min(b.egg_limit, 3) * 0.4
            if b.color == PowerColor.BROWN:
                bv += 2.0
            if bv > best_bird_value:
                best_bird_value = bv
                # Compute food deficit for this bird
                if b.food_cost.is_or:
                    # OR cost: any single type works, prefer types we lack
                    best_bird_needs = {ft: 1 for ft in b.food_cost.distinct_types}
                else:
                    # AND cost: need all specific types
                    needs: dict[FoodType, int] = {}
                    for ft in b.food_cost.items:
                        needs[ft] = needs.get(ft, 0) + 1
                    for ft in list(needs.keys()):
                        have = player.food_supply.get(ft)
                        needs[ft] = max(0, needs[ft] - have)
                        if needs[ft] == 0:
                            del needs[ft]
                    best_bird_needs = needs

        # Score bonus for each chosen food that fills a deficit
        if best_bird_needs:
            deficit_filled = sum(
                1 for ft in move.food_choices if ft in best_bird_needs
            )
            if deficit_filled > 0:
                value += deficit_filled * 0.6 * (best_bird_value / 5.0) * reset_discount

    # Last turn nectar strategy: prefer moves that spend nectar before round end
    if player.action_cubes_remaining <= 1:
        nectar = player.food_supply.get(FoodType.NECTAR)
        if nectar > 0:
            if move.reset_bonus:
                # Reset feeder costs 1 food — can pay with nectar
                value += 1.5
            else:
                # Not spending nectar — penalty scales with nectar about to expire
                value -= nectar * 0.5

    return value


def _evaluate_lay_eggs(game: GameState, player: Player, move: Move,
                       weights: HeuristicWeights) -> float:
    """Evaluate a lay-eggs move."""
    bird_count = player.board.grassland.bird_count
    column = get_action_column(game.board_type, Habitat.GRASSLAND, bird_count)
    egg_count = column.base_gain + move.bonus_count

    # Eggs are worth 1 point each, limited by bird capacity
    eligible_space = sum(
        slot.eggs_space()
        for row in player.board.all_rows()
        for slot in row.slots
        if slot.bird
    )
    actual_eggs = min(egg_count, eligible_space)
    value = actual_eggs * weights.egg_points

    # Bonus cost: paying a food/card for +1 egg is usually net positive (egg = 1pt)
    if move.bonus_count > 0 and actual_eggs > column.base_gain:
        # Extra eggs actually laid beyond base (not wasted on full birds)
        extra_laid = actual_eggs - min(column.base_gain, eligible_space)
        if extra_laid > 0:
            value += extra_laid * 0.5  # Net gain: 1pt egg - ~0.5 cost

    # Engine value: brown powers in grassland
    for i, slot in enumerate(player.board.grassland.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=slot.bird,
            slot_index=i,
            habitat=Habitat.GRASSLAND,
        )
        value += power.estimate_value(ctx)

    # Late game bonus: eggs are guaranteed, zero-risk points
    if game.current_round >= 4:
        value += actual_eggs * 1.0  # Round 4: eggs are the best action
    elif game.current_round >= 3:
        value += actual_eggs * 0.5

    # Goal alignment: egg-related goals boost value
    value += _egg_goal_alignment(game)

    # Distribution-aware bonuses: value WHERE eggs go
    if move.egg_distribution and game.round_goals:
        for i in range(game.current_round - 1, min(len(game.round_goals), 4)):
            goal = game.round_goals[i]
            desc = goal.description.lower()
            rounds_away = i - (game.current_round - 1)
            time_discount = 1.0 / (1.0 + rounds_away * 0.5)
            # Eggs-in-habitat goals: reward concentrating eggs in that habitat
            for hab_name, hab_enum in (("forest", Habitat.FOREST),
                                        ("grassland", Habitat.GRASSLAND),
                                        ("wetland", Habitat.WETLAND)):
                if f"[egg] in [{hab_name}]" in desc:
                    eggs_in_target = sum(
                        c for (h, _), c in move.egg_distribution.items()
                        if h == hab_enum
                    )
                    value += eggs_in_target * 0.3 * time_discount
                    # Flip bonus for current goal
                    if i == game.current_round - 1:
                        opponents = [p for p in game.players if p.name != player.name]
                        if opponents:
                            my_progress = _estimate_goal_progress(player, goal)
                            best_opp = max(_estimate_goal_progress(opp, goal) for opp in opponents)
                            if my_progress < best_opp and my_progress + eggs_in_target >= best_opp:
                                value += max(goal.scoring[3], 0) * 0.25
            # Eggs-in-nest-type goals
            for nt in ("bowl", "cavity", "ground", "platform"):
                if f"[egg] in [{nt}]" in desc:
                    from backend.models.enums import NestType
                    nest_map = {"bowl": NestType.BOWL, "cavity": NestType.CAVITY,
                                "ground": NestType.GROUND, "platform": NestType.PLATFORM}
                    eggs_matching = 0
                    for (h, idx), c in move.egg_distribution.items():
                        slot = player.board.get_row(h).slots[idx]
                        if slot.bird and (slot.bird.nest_type == nest_map[nt]
                                          or slot.bird.nest_type == NestType.WILD):
                            eggs_matching += c
                    value += eggs_matching * 0.3 * time_discount
                    if i == game.current_round - 1:
                        opponents = [p for p in game.players if p.name != player.name]
                        if opponents and eggs_matching > 0:
                            my_progress = _estimate_goal_progress(player, goal)
                            best_opp = max(_estimate_goal_progress(opp, goal) for opp in opponents)
                            if my_progress < best_opp and my_progress + eggs_matching >= best_opp:
                                value += max(goal.scoring[3], 0) * 0.2

    # Bonus card awareness: filling birds to thresholds
    if move.egg_distribution:
        for bc in player.bonus_cards:
            bc_name = bc.name.lower()
            if "breeding manager" in bc_name:
                # Birds with >=4 eggs — value placing eggs to reach 4
                for (h, idx), c in move.egg_distribution.items():
                    slot = player.board.get_row(h).slots[idx]
                    if slot.bird:
                        eggs_after = slot.eggs + c
                        if slot.eggs < 4 <= eggs_after:
                            value += 1.0  # Crossing the 4-egg threshold
            elif "oologist" in bc_name:
                # Birds with >=1 egg — value spreading to new birds
                for (h, idx), c in move.egg_distribution.items():
                    slot = player.board.get_row(h).slots[idx]
                    if slot.bird and slot.eggs == 0 and c > 0:
                        value += 0.5  # New bird gets first egg
            elif "theriogenologist" in bc_name:
                # Full nests (eggs == egg_limit) — value filling birds
                for (h, idx), c in move.egg_distribution.items():
                    slot = player.board.get_row(h).slots[idx]
                    if slot.bird and slot.bird.egg_limit > 0:
                        eggs_after = slot.eggs + c
                        if eggs_after >= slot.bird.egg_limit and slot.eggs < slot.bird.egg_limit:
                            value += 1.0  # Completing a full nest

    # Last turn nectar strategy: prefer moves that spend nectar before round end
    if player.action_cubes_remaining <= 1:
        nectar = player.food_supply.get(FoodType.NECTAR)
        if nectar > 0:
            if move.bonus_count > 0:
                # Grassland column bonus costs 1 food per bonus — can pay with nectar
                value += move.bonus_count * 1.5
            elif move.reset_bonus:
                # Reset feeder costs 1 food — can pay with nectar
                value += 1.5
            else:
                # Not spending nectar — penalty scales with nectar about to expire
                value -= nectar * 0.5

    return value


def _evaluate_draw_cards(game: GameState, player: Player, move: Move,
                         weights: HeuristicWeights) -> float:
    """Evaluate a draw-cards move."""
    bird_count = player.board.wetland.bird_count
    column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
    card_count = column.base_gain + move.bonus_count

    rounds_remaining = max(0, ROUNDS - game.current_round)

    # Base value per card scales with rounds remaining (more turns = more chance to play)
    # Keep modest because tray cards are evaluated separately below.
    phase = rounds_remaining / ROUNDS if ROUNDS else 0
    card_base_value = 0.2 + 0.1 * phase
    value = card_count * card_base_value

    # Bonus cost: paying an egg/nectar for +1 card
    if move.bonus_count > 0:
        value += move.bonus_count * 0.2

    # Reset tray value: worth more when current tray options are poor
    if move.reset_bonus:
        value += 0.4

    # Known tray cards can be evaluated precisely — this is the key advantage of tray picks
    for idx in move.tray_indices:
        if idx < len(game.card_tray.face_up):
            tray_bird = game.card_tray.face_up[idx]
            # Full quick evaluation: VP + egg potential + engine + affordability
            tray_val = tray_bird.victory_points * 0.5
            tray_val += min(tray_bird.egg_limit, max(1, rounds_remaining)) * 0.3
            if tray_bird.color == PowerColor.BROWN and rounds_remaining >= 2:
                tray_val += rounds_remaining * 0.4
            elif tray_bird.color == PowerColor.WHITE and _is_play_bird_power(tray_bird):
                tray_val += 1.5  # Play-another-bird powers are very valuable to draw

            # Affordability bonus: a bird you can play NOW is worth more than one you can't
            food_available = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
            if food_available >= tray_bird.food_cost.total:
                tray_val += 0.8  # Can play immediately
            elif food_available + 2 >= tray_bird.food_cost.total:
                tray_val += 0.3  # Nearly affordable

            # Bonus card pursuit: actual marginal VP from adding this bird
            bonus_marginal = _bonus_card_marginal_vp(player, tray_bird)
            if bonus_marginal > 0:
                value += bonus_marginal * 0.6  # Discounted: need to draw AND play

            # Opponent denial: small bonus for taking cards opponents need for goals
            if game.round_goals and game.current_round <= len(game.round_goals):
                current_goal = game.round_goals[game.current_round - 1]
                opponents = [p for p in game.players if p.name != player.name]
                for opp in opponents:
                    opp_progress = _estimate_goal_progress(opp, current_goal)
                    my_progress = _estimate_goal_progress(player, current_goal)
                    if opp_progress >= my_progress:
                        for hab in tray_bird.habitats:
                            contrib = _estimate_goal_contribution(tray_bird, hab, current_goal)
                            if contrib > 0:
                                tray_val += 0.5  # Denial bonus
                                break
                        break  # One denial check is enough
            # Opponent denial: bonus card blocking if known
            opponents = [p for p in game.players if p.name != player.name]
            for opp in opponents:
                for bc in opp.bonus_cards:
                    if bc.name in tray_bird.bonus_eligibility:
                        tray_val += 0.4
                        break
            # Opponent tray value: deny high-value birds
            if opponents:
                for opp in opponents:
                    opp_food = opp.food_supply.total_non_nectar() + opp.food_supply.get(FoodType.NECTAR)
                    opp_val = tray_bird.victory_points * 0.4
                    if opp_food >= tray_bird.food_cost.total:
                        opp_val += 0.6
                    if tray_bird.color == PowerColor.BROWN and rounds_remaining >= 2:
                        opp_val += rounds_remaining * 0.3
                    if game.round_goals and game.current_round <= len(game.round_goals):
                        g = game.round_goals[game.current_round - 1]
                        best_contrib = 0.0
                        for hab in tray_bird.habitats:
                            best_contrib = max(best_contrib, _estimate_goal_contribution(tray_bird, hab, g))
                        opp_val += best_contrib * 0.6
                    if opp_val >= 3.5:
                        tray_val += min(opp_val * 0.2, 1.2)
                        break

            # Goal synergy
            if game.round_goals:
                best_goal_val = 0.0
                for hab in tray_bird.habitats:
                    gv = _goal_alignment_value(game, tray_bird, hab, weights)
                    best_goal_val = max(best_goal_val, gv)
                value += best_goal_val * 0.3

            value += tray_val

    # Deck draws: speculative value based on deck quality estimate
    deck_draws = move.deck_draws
    if deck_draws > 0:
        # Deck cards have uncertain value — average bird VP is ~3-4
        # Discount heavily since we can't see them
        deck_per_card = 0.2 + 0.1 * phase  # More valuable early, but uncertain
        if game.deck_tracker is not None and game.deck_tracker.remaining_count > 0:
            avg_vp = game.deck_tracker.avg_remaining_vp()
            # Higher remaining VP makes blind deck draws more attractive.
            deck_per_card += max(-0.08, min(0.25, (avg_vp - 3.5) * 0.07))
        value += deck_draws * deck_per_card

    # Engine value: brown powers in wetland
    for i, slot in enumerate(player.board.wetland.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=slot.bird,
            slot_index=i,
            habitat=Habitat.WETLAND,
        )
        value += power.estimate_value(ctx)

    # Hand size: urgency when empty, penalty when bloated
    if player.hand_size == 0:
        value += 2.0
    elif player.hand_size <= 2:
        value += 0.5
    elif player.hand_size >= 8:
        # Diminishing returns: you can't play them all
        # Remaining empty slots on the board is the real capacity
        empty_slots = sum(
            1 for row in player.board.all_rows()
            for slot in row.slots if not slot.bird
        )
        excess = player.hand_size - empty_slots
        if excess > 0:
            # More birds in hand than board slots — drawing more is waste
            value -= excess * 1.5
        elif player.hand_size > player.action_cubes_remaining * 2:
            # More birds than you could possibly play with remaining cubes
            value -= (player.hand_size - 6) * 0.5
    elif player.hand_size >= 4:
        # Moderate penalty once hand is already healthy
        value -= (player.hand_size - 3) * 0.4

    # Small penalty if few board slots remain
    empty_slots = sum(
        1 for row in player.board.all_rows()
        for slot in row.slots if not slot.bird
    )
    if empty_slots <= 2 and player.hand_size >= 3:
        value -= (3 - empty_slots) * 0.6

    # Late game penalty: drawing cards with few actions left is wasteful
    if game.current_round >= 4 and player.action_cubes_remaining <= 2:
        value -= 1.0

    # Last turn nectar strategy: prefer moves that spend nectar before round end
    if player.action_cubes_remaining <= 1:
        nectar = player.food_supply.get(FoodType.NECTAR)
        if nectar > 0:
            if move.bonus_count > 0:
                # Wetland column bonus costs 1 egg or 1 food — can pay with nectar
                value += move.bonus_count * 1.5
            elif move.reset_bonus:
                # Reset tray costs 1 food — can pay with nectar
                value += 1.5
            else:
                # Not spending nectar — penalty scales with nectar about to expire
                value -= nectar * 0.5

    return value


# --- Public solver interface ---

@dataclass
class RankedMove:
    """A move with its heuristic score."""
    move: Move
    score: float
    rank: int = 0
    reasoning: str = ""


def _activation_advice(game: GameState, player: Player, habitat: Habitat) -> list[str]:
    """Generate detailed activation advice for brown birds in a row.

    For each brown bird, uses describe_activation() for step-by-step instructions
    and skip_reason() for warnings when a power can't execute.
    For "all players" powers, compares your need vs opponents' need.
    """
    row = player.board.get_row(habitat)
    advice: list[str] = []
    opponents = [p for p in game.players if p.name != player.name]

    for i, slot in enumerate(row.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        name = slot.bird.name
        ctx = _build_power_context(
            game=game,
            player=player,
            bird=slot.bird,
            slot_index=i,
            habitat=habitat,
        )

        # Check if power can't execute at all
        skip = power.skip_reason(ctx)
        if skip:
            advice.append(f"SKIP {name} ({skip})")
            continue

        # For all-players powers, check if opponents benefit more
        is_all_players = getattr(power, 'all_players', False)
        if is_all_players and opponents:
            opp_skip = _should_skip_all_players(power, player, opponents)
            if opp_skip:
                advice.append(f"SKIP {name} ({opp_skip})")
                continue

        # Detailed activation description
        desc = power.describe_activation(ctx)
        advice.append(f"Activate {habitat.value}: {name} — {desc}")
    return advice


def _should_skip_all_players(power, player: Player, opponents: list[Player]) -> str | None:
    """Check if an all-players power benefits opponents more than you.

    Returns a skip reason string, or None if you benefit more.
    """
    from backend.powers.templates.gain_food import GainFoodFromSupply
    from backend.powers.templates.draw_cards import DrawCards
    from backend.powers.templates.lay_eggs import LayEggs

    if isinstance(power, GainFoodFromSupply):
        my_food = player.food_supply.total_non_nectar()
        for opp in opponents:
            opp_food = opp.food_supply.total_non_nectar()
            if my_food >= 4:
                return "you have enough food already"
            if opp_food <= my_food and opp_food < 3:
                if opp.hand_size > 0 or opp_food == 0:
                    return f"{opp.name} low on food ({opp_food})"

    elif isinstance(power, DrawCards):
        my_hand = player.hand_size
        for opp in opponents:
            opp_hand = opp.hand_size
            opp_wetland = opp.board.wetland.bird_count
            if opp_wetland < 2 and opp_hand <= 2:
                return f"{opp.name} needs cards (hand {opp_hand}, few wetland birds)"
            if opp_hand < my_hand and opp_hand <= 2:
                return f"{opp.name} needs cards more (hand {opp_hand})"

    elif isinstance(power, LayEggs):
        my_space = sum(
            s.eggs_space() for r in player.board.all_rows()
            for s in r.slots if s.bird
        )
        for opp in opponents:
            opp_space = sum(
                s.eggs_space() for r in opp.board.all_rows()
                for s in r.slots if s.bird
            )
            if my_space == 0 and opp_space > 0:
                return "you have no egg space"
            if opp_space > my_space and opp_space >= 2:
                return f"{opp.name} has more egg space ({opp_space})"

    return None


def _player_after_bonus(player: Player, move: Move,
                        column_bonus) -> Player:
    """Create a shallow copy of the player with bonus payment deducted.

    This is used so activation advice reflects the state *after* paying the
    bonus cost (e.g., if you discard a card for +1 food, tuck-from-hand
    powers should see the reduced hand size).
    """
    if not move.bonus_count or not column_bonus:
        return player
    # Determine what the player will pay
    cost_options = column_bonus.cost_options
    cards_lost = 0
    for cost in cost_options:
        if cost == "card" and player.hand:
            cards_lost = 1
            break
        elif cost == "food":
            # Food payment doesn't affect hand size
            break
        elif cost == "egg":
            break
        elif cost == "nectar":
            break
    if cards_lost == 0:
        return player
    # Create a copy with reduced hand
    adjusted = copy.copy(player)
    adjusted.hand = list(player.hand)
    for _ in range(cards_lost):
        if adjusted.hand:
            worst = min(adjusted.hand, key=lambda b: b.victory_points)
            adjusted.hand.remove(worst)
    return adjusted


def _recommend_bonus_payment(player: Player, cost_options: tuple[str, ...],
                             game: GameState) -> str:
    """Recommend which specific resource to discard for a bonus cost.

    Returns a string like 'discard your lowest bird' or 'spend 1 fruit'.
    """
    for cost in cost_options:
        if cost == "card" and player.hand:
            # Find the lowest-value card to discard
            worst = min(player.hand, key=lambda b: b.victory_points)
            return f"discard {worst.name} ({worst.victory_points}VP)"
        elif cost == "food":
            # Pick the food type you have the most surplus of
            # (least needed for birds in hand)
            needed: dict[FoodType, int] = {}
            for b in player.hand:
                if not b.food_cost.is_or:
                    for ft in b.food_cost.items:
                        needed[ft] = needed.get(ft, 0) + 1

            best_ft = None
            best_surplus = -999
            for ft in [FoodType.SEED, FoodType.INVERTEBRATE, FoodType.FRUIT,
                        FoodType.FISH, FoodType.RODENT]:
                have = player.food_supply.get(ft)
                if have <= 0:
                    continue
                surplus = have - needed.get(ft, 0)
                if surplus > best_surplus:
                    best_surplus = surplus
                    best_ft = ft

            if best_ft:
                return f"spend 1 {best_ft.value}"
            # No food available — try next cost option
        elif cost == "egg":
            # Find the bird with least strategic value to remove an egg from
            candidates = []
            for row in player.board.all_rows():
                for i, slot in enumerate(row.slots):
                    if slot.bird and slot.eggs > 0:
                        candidates.append((slot.bird.name, slot.eggs, row.habitat, i))
            if candidates:
                # Pick bird with most eggs (removing one hurts least proportionally)
                candidates.sort(key=lambda x: -x[1])
                name, eggs, _, _ = candidates[0]
                return f"remove egg from {name} ({eggs} eggs)"
            # No eggs available — try next cost option (e.g. nectar)
        elif cost == "nectar":
            if player.food_supply.get(FoodType.NECTAR) > 0:
                return "spend 1 nectar"
    return ""


def _generate_move_reasoning(game: GameState, player: Player, move: Move) -> str:
    """Generate a brief reasoning string explaining why a move is recommended."""
    reasons: list[str] = []
    rounds_remaining = max(0, ROUNDS - game.current_round)

    if move.action_type == ActionType.PLAY_BIRD:
        from backend.data.registries import get_bird_registry
        bird = get_bird_registry().get(move.bird_name)
        if not bird:
            return ""
        if bird.victory_points >= 5:
            reasons.append(f"high VP ({bird.victory_points})")
        elif bird.victory_points > 0:
            reasons.append(f"{bird.victory_points}VP")
        if player.hand_size >= 8:
            reasons.append(f"large hand ({player.hand_size} birds) — play birds now")
        # Column egg cost info
        row = player.board.get_row(move.habitat)
        slot_idx = row.next_empty_slot()
        if slot_idx is not None and slot_idx < len(EGG_COST_BY_COLUMN):
            egg_cost = EGG_COST_BY_COLUMN[slot_idx]
            if egg_cost > 0:
                # Recommend which eggs to remove
                egg_sources = []
                for r in player.board.all_rows():
                    for i, s in enumerate(r.slots):
                        if s.bird and s.eggs > 0:
                            egg_sources.append((s.eggs, s.bird.name))
                egg_sources.sort(key=lambda x: -x[0])
                if egg_sources:
                    source_names = [f"{n} ({e} eggs)" for e, n in egg_sources[:egg_cost]]
                    reasons.append(f"costs {egg_cost} egg — remove from {', '.join(source_names)}")
                else:
                    reasons.append(f"costs {egg_cost} egg")

        if bird.color == PowerColor.BROWN and rounds_remaining > 0:
            reasons.append("brown engine power")
        elif bird.color == PowerColor.WHITE:
            if _is_play_bird_power(bird):
                from backend.powers.templates.play_bird import PlayAdditionalBird
                play_power = get_power(bird)
                disc_parts = []
                if isinstance(play_power, PlayAdditionalBird):
                    if play_power.egg_discount:
                        disc_parts.append(f"-{play_power.egg_discount} egg cost")
                    if play_power.food_discount:
                        disc_parts.append(f"-{play_power.food_discount} food cost")
                disc_str = f" ({', '.join(disc_parts)})" if disc_parts else ""
                reasons.append(f"play-another-bird power{disc_str}")
            else:
                reasons.append("when-played power")
        elif bird.color == PowerColor.PINK:
            reasons.append("triggers on opponent turns")
        elif bird.color == PowerColor.TEAL:
            reasons.append("scores between turns")
        if bird.egg_limit >= 4:
            reasons.append(f"holds {bird.egg_limit} eggs")
        for bc in player.bonus_cards:
            if bc.name in bird.bonus_eligibility:
                reasons.append(f"bonus: {bc.name}")
                break
        if game.round_goals:
            idx = min(game.current_round - 1, len(game.round_goals) - 1)
            if idx >= 0:
                goal = game.round_goals[idx]
                contrib = _estimate_goal_contribution(bird, move.habitat, goal)
                if contrib > 0:
                    reasons.append("helps round goal")
        # Nectar majority scoring
        nectar_in_payment = move.food_payment.get(FoodType.NECTAR, 0)
        if nectar_in_payment > 0:
            my_spent = player.board.get_row(move.habitat).nectar_spent
            reasons.append(f"nectar in {move.habitat.value} -> {my_spent + nectar_in_payment} (majority scoring)")
            if player.action_cubes_remaining <= 1:
                reasons.append(f"SPEND NECTAR NOW — {player.food_supply.get(FoodType.NECTAR)} nectar expires at end of round")
        # Cached food that could help pay
        for row in player.board.all_rows():
            for slot in row.slots:
                if not slot.bird or slot.total_cached_food == 0:
                    continue
                for ft, cnt in slot.cached_food.items():
                    if cnt > 0 and ft in move.food_payment:
                        reasons.append(
                            f"cached {ft.value} on {slot.bird.name} could cover {cnt} (costs 1pt each)"
                        )
                        break  # one note per bird is enough

    elif move.action_type == ActionType.GAIN_FOOD:
        food_total = player.food_supply.total_non_nectar()
        nectar = player.food_supply.get(FoodType.NECTAR)
        if food_total < 2:
            reasons.append("low on food")
        # Detect free reroll opportunity (all dice show same face)
        feeder = game.birdfeeder
        if feeder.all_same_face() and feeder.count > 0:
            face = feeder.dice[0]
            face_name = face.value if not isinstance(face, tuple) else "/".join(f.value for f in face)
            reasons.append(f"REROLL AVAILABLE — all {feeder.count} dice show {face_name}")
        bird_count = player.board.forest.bird_count
        column = get_action_column(game.board_type, Habitat.FOREST, bird_count)
        gain = column.base_gain + move.bonus_count
        playable = sum(
            1 for b in player.hand
            if food_total + nectar + gain >= b.food_cost.total
        )
        if playable > 0:
            reasons.append(f"enables playing {playable} bird{'s' if playable > 1 else ''}")
        if move.food_choices and FoodType.NECTAR in move.food_choices:
            if player.action_cubes_remaining <= 1:
                reasons.append("WARNING: nectar clears at end of round — avoid gaining nectar on last turn")
        if move.bonus_count > 0 and column.bonus:
            payment = _recommend_bonus_payment(player, column.bonus.cost_options, game)
            reasons.append(f"+1 food ({payment})" if payment else "+1 food")
        if move.reset_bonus and column.reset_bonus:
            payment = _recommend_bonus_payment(player, column.reset_bonus.cost_options, game)
            reasons.insert(0, f"FIRST reset feeder ({payment}), then take food" if payment else "FIRST reset feeder, then take food")
            reasons.append("food shown is best-case after reroll — use After Reset for exact picks")
            if player.action_cubes_remaining <= 1 and player.food_supply.get(FoodType.NECTAR) > 0:
                reasons.append("SPEND NECTAR on feeder reset — nectar expires at end of round")
        brown_count = sum(1 for s in player.board.forest.slots if s.bird and s.bird.color == PowerColor.BROWN)
        if brown_count > 0:
            reasons.append(f"activates {brown_count} brown forest bird{'s' if brown_count > 1 else ''}")

    elif move.action_type == ActionType.LAY_EGGS:
        bird_count = player.board.grassland.bird_count
        column = get_action_column(game.board_type, Habitat.GRASSLAND, bird_count)
        egg_count = column.base_gain + move.bonus_count
        space = sum(
            s.eggs_space() for r in player.board.all_rows()
            for s in r.slots if s.bird
        )
        actual = min(egg_count, space)
        if actual > 0:
            reasons.append(f"{actual} eggs = {actual}pts")
        if game.current_round >= 3:
            reasons.append("guaranteed late-game points")
        brown_count = sum(1 for s in player.board.grassland.slots if s.bird and s.bird.color == PowerColor.BROWN)
        if brown_count > 0:
            reasons.append(f"activates {brown_count} brown grassland bird{'s' if brown_count > 1 else ''}")
        if move.bonus_count > 0 and column.bonus:
            payment = _recommend_bonus_payment(player, column.bonus.cost_options, game)
            label = f"+{move.bonus_count} egg{'s' if move.bonus_count > 1 else ''}"
            reasons.append(f"{label} ({payment})" if payment else label)
            if player.action_cubes_remaining <= 1 and player.food_supply.get(FoodType.NECTAR) > 0:
                reasons.append("SPEND NECTAR on bonus — nectar expires at end of round")
        if game.round_goals:
            idx = min(game.current_round - 1, len(game.round_goals) - 1)
            if idx >= 0 and "[egg]" in game.round_goals[idx].description.lower():
                reasons.append("helps egg goal")

    elif move.action_type == ActionType.DRAW_CARDS:
        if player.hand_size == 0:
            reasons.append("empty hand — need cards")
        elif player.hand_size <= 2:
            reasons.append("hand is low")
        elif player.hand_size >= 8:
            empty_slots = sum(
                1 for row in player.board.all_rows()
                for slot in row.slots if not slot.bird
            )
            if player.hand_size > empty_slots:
                reasons.insert(0, f"WARNING: {player.hand_size} birds in hand but only {empty_slots} board slots — play birds instead")
            else:
                reasons.insert(0, f"WARNING: {player.hand_size} birds in hand — consider playing birds instead")
        for idx in move.tray_indices:
            if idx < len(game.card_tray.face_up):
                tray_bird = game.card_tray.face_up[idx]
                reasons.append(f"picks {tray_bird.name}")
                break
        bird_count = player.board.wetland.bird_count
        column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
        if move.bonus_count > 0 and column.bonus:
            payment = _recommend_bonus_payment(player, column.bonus.cost_options, game)
            reasons.append(f"+1 card ({payment})" if payment else "+1 card")
            if player.action_cubes_remaining <= 1 and player.food_supply.get(FoodType.NECTAR) > 0:
                reasons.append("SPEND NECTAR on bonus — nectar expires at end of round")
        if move.reset_bonus and column.reset_bonus:
            payment = _recommend_bonus_payment(player, column.reset_bonus.cost_options, game)
            reasons.insert(0, f"FIRST reset tray ({payment}), then draw cards" if payment else "FIRST reset tray, then draw cards")
            if player.action_cubes_remaining <= 1 and player.food_supply.get(FoodType.NECTAR) > 0:
                reasons.append("SPEND NECTAR on tray reset — nectar expires at end of round")
        brown_count = sum(1 for s in player.board.wetland.slots if s.bird and s.bird.color == PowerColor.BROWN)
        if brown_count > 0:
            reasons.append(f"activates {brown_count} brown wetland bird{'s' if brown_count > 1 else ''}")

    return "; ".join(reasons) if reasons else "standard play"


def rank_moves(game: GameState, player: Player | None = None,
               weights: HeuristicWeights | None = None) -> list[RankedMove]:
    """Generate and rank all legal moves by heuristic value.

    Uses dynamic phase-aware weights by default. Pass explicit weights to override.
    Returns moves sorted by score (highest first) with rank assigned.
    """
    if player is None:
        player = game.current_player
    if weights is None:
        weights = dynamic_weights(game)

    moves = generate_all_moves(game, player)
    if not moves:
        return []

    ranked = [
        RankedMove(move=m, score=_estimate_move_value(game, player, m, weights))
        for m in moves
    ]

    # Sort by score descending
    ranked.sort(key=lambda r: -r.score)

    # Assign ranks (1-based) and generate reasoning
    for i, rm in enumerate(ranked):
        rm.rank = i + 1
        rm.reasoning = _generate_move_reasoning(game, player, rm.move)

    return ranked
