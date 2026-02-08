"""Static board evaluation and move ranking for the heuristic solver."""

import copy
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
    early_game_engine_bonus: float = 0.3  # Extra weight for engine in rounds 1-2
    predator_penalty: float = 0.5  # Discount predator power value by this factor
    goal_alignment: float = 1.2  # Weight for round goal contribution
    nectar_early_bonus: float = 0.4  # Extra value for nectar in early rounds
    grassland_egg_synergy: float = 0.6  # Bonus for high-egg birds in grassland
    play_another_bird_bonus: float = 2.5  # Extra value for play-another-bird powers
    food_for_birds_bonus: float = 0.4  # Extra value for food that enables playing hand birds


DEFAULT_WEIGHTS = HeuristicWeights()


def dynamic_weights(game: GameState) -> HeuristicWeights:
    """Compute phase-aware weights based on current game state.

    Round 1: Engine-building (brown powers, cards, food worth more)
    Round 2: Transition (balanced, start goal focus)
    Round 3: Scoring pivot (eggs, goals, bonus cards matter more)
    Round 4: Pure points (eggs dominant, cards/engine near worthless)
    """
    w = HeuristicWeights()
    rd = game.current_round
    rounds_left = max(0, ROUNDS - rd)
    phase = rounds_left / ROUNDS  # 1.0 at start, 0.0 at end

    # Engine value scales down
    w.engine_value = 0.4 + 0.6 * phase
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
        contribution = _estimate_goal_contribution(bird, habitat, goal)
        if contribution <= 0:
            continue

        rounds_away = i - (game.current_round - 1)
        time_discount = 1.0 / (1.0 + rounds_away * 0.5)
        first_place_pts = max(goal.scoring[3], 0)

        # Opponent-aware multiplier
        position_mult = 1.0
        if opponents:
            my_progress = _estimate_goal_progress(player, goal)
            best_opp = max(_estimate_goal_progress(opp, goal) for opp in opponents)
            gap = my_progress - best_opp
            if gap >= 3:
                # Comfortably ahead — don't overinvest
                position_mult = 0.4
            elif gap >= 1:
                # Slightly ahead — maintain but don't push hard
                position_mult = 0.7
            elif gap >= -1:
                # Tied or slightly behind — high value to push ahead
                position_mult = 1.3
            else:
                # Far behind — contribution still valuable but catching up is hard
                position_mult = 0.9

        goal_value = first_place_pts * 0.3 * position_mult
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


def _is_play_bird_power(bird: Bird) -> bool:
    """Check if a bird has a white power that lets you play another bird."""
    if bird.color != PowerColor.WHITE:
        return False
    text = bird.power_text.lower()
    return "play" in text and "bird" in text


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
    engine_val = _estimate_engine_value(game, player)
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

    # Nectar timing: nectar is more valuable with more rounds remaining
    # But worthless on last turn of round since it clears at round end
    nectar_count = player.food_supply.get(FoodType.NECTAR)
    if nectar_count > 0:
        if player.action_cubes_remaining <= 1:
            # Last turn of round: nectar in supply is about to be lost
            # Only value it if it can be spent this turn (playing a bird)
            value -= nectar_count * 0.5
        elif rounds_remaining > 0:
            value += nectar_count * weights.nectar_early_bonus * (rounds_remaining / ROUNDS)

    # Habitat diversity bonus (birds in all 3 habitats is strategically strong)
    habitats_with_birds = sum(
        1 for row in player.board.all_rows() if row.bird_count > 0
    )
    if habitats_with_birds >= 3:
        value += weights.habitat_diversity_bonus

    return value


def _estimate_engine_value(game: GameState, player: Player) -> float:
    """Estimate the per-activation value of the player's engine.

    Sums the estimated value of all brown powers on the board,
    weighted by how often they'll be activated (rightmost birds activate more).
    Predator powers are discounted for unreliability.
    """
    from backend.powers.templates.tuck_cards import TuckFromDeck
    from backend.powers.templates.special import FlockingPower
    from backend.powers.templates.cache_food import CacheFoodFromSupply

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
            ctx = PowerContext(
                game_state=game, player=player, bird=slot.bird,
                slot_index=i, habitat=row.habitat,
            )
            power_val = power.estimate_value(ctx)

            # Position-aware: earlier slots activate more per use of that habitat
            activation_factor = max(0.3, 1.0 - i * 0.15)
            power_val *= activation_factor

            # Zero-cost point generators get a boost
            if isinstance(power, (TuckFromDeck, FlockingPower, CacheFoodFromSupply)):
                power_val = max(power_val, 0.9 * activation_factor)

            # Predator penalty: ~50% success rate on dice
            if slot.bird.is_predator:
                power_val *= DEFAULT_WEIGHTS.predator_penalty
            total += power_val
    return total


def _estimate_move_value(game: GameState, player: Player, move: Move,
                         weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Estimate the value of a specific move without full simulation.

    Uses a combination of immediate value and positional improvement.
    """
    if move.action_type == ActionType.PLAY_BIRD:
        return _evaluate_play_bird(game, player, move, weights)
    elif move.action_type == ActionType.GAIN_FOOD:
        return _evaluate_gain_food(game, player, move, weights)
    elif move.action_type == ActionType.LAY_EGGS:
        return _evaluate_lay_eggs(game, player, move, weights)
    elif move.action_type == ActionType.DRAW_CARDS:
        return _evaluate_draw_cards(game, player, move, weights)
    return 0.0


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
    value += bird.victory_points

    # Egg capacity value (future egg points)
    value += min(bird.egg_limit, rounds_remaining) * 0.4

    # Power engine value (brown powers multiply over remaining rounds)
    power = get_power(bird)
    if bird.color == PowerColor.BROWN and not isinstance(power, NoPower):
        from backend.powers.templates.tuck_cards import TuckFromDeck
        from backend.powers.templates.special import FlockingPower
        from backend.powers.templates.cache_food import CacheFoodFromSupply
        row = player.board.get_row(move.habitat)
        slot_idx = row.bird_count  # where this bird will be placed
        ctx = PowerContext(
            game_state=game, player=player, bird=bird,
            slot_index=slot_idx, habitat=move.habitat,
        )
        # Position-aware activations: earlier slots activate more often
        # Estimate uses of this habitat over remaining rounds (assume ~1.5x per round avg)
        total_uses = rounds_remaining * 1.5
        # Each use activates all birds up to the number played at that time
        # Bird in slot 0 activates every time; slot N only if >=N+1 birds
        # Simplified: early slot = all uses, late slot = fewer
        activation_factor = max(0.3, 1.0 - slot_idx * 0.15)
        activations_estimate = total_uses * activation_factor

        power_val = power.estimate_value(ctx) * activations_estimate * 0.5

        # Extra positional bonus for zero-cost point generators (tuck from deck, cache from supply)
        if isinstance(power, (TuckFromDeck, FlockingPower, CacheFoodFromSupply)):
            # These generate 1pt per activation with no resource cost
            power_val = activations_estimate * 0.9  # near-guaranteed points

        # Predator penalty: predators are unreliable (~50% success)
        if bird.is_predator:
            power_val *= weights.predator_penalty
        value += power_val

    elif bird.color == PowerColor.WHITE and not isinstance(power, NoPower):
        ctx = PowerContext(
            game_state=game, player=player, bird=bird,
            slot_index=0, habitat=move.habitat,
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
                fv += min(candidate.egg_limit, rounds_remaining) * 0.4
                if candidate.color == PowerColor.BROWN:
                    fv += rounds_remaining * 0.5
                best_followup = max(best_followup, fv)
            if best_followup > 0:
                power_val += best_followup * 0.8
            else:
                power_val += weights.play_another_bird_bonus * min(rounds_remaining, 2)
        value += power_val

    elif bird.color == PowerColor.PINK:
        ctx = PowerContext(
            game_state=game, player=player, bird=bird,
            slot_index=0, habitat=move.habitat,
        )
        value += power.estimate_value(ctx) * rounds_remaining * 0.3

    elif bird.color in (PowerColor.TEAL, PowerColor.YELLOW):
        ctx = PowerContext(
            game_state=game, player=player, bird=bird,
            slot_index=0, habitat=move.habitat,
        )
        value += power.estimate_value(ctx) * rounds_remaining * 0.6

    # Bonus card delta: compute the actual marginal VP from each bonus card
    bonus_delta = 0.0
    for bc in player.bonus_cards:
        current_qualifying = sum(
            1 for b in player.board.all_birds() if bc.name in b.bonus_eligibility
        )
        current_score = bc.score(current_qualifying)
        if bc.name in bird.bonus_eligibility:
            new_score = bc.score(current_qualifying + 1)
            bonus_delta += (new_score - current_score)
    value += bonus_delta * weights.bonus_card_points

    # Round goal alignment
    value += _goal_alignment_value(game, bird, move.habitat, weights)

    # Grassland egg synergy: high-egg birds in grassland benefit from lay-eggs engine
    if move.habitat == Habitat.GRASSLAND and bird.egg_limit >= 3:
        value += (bird.egg_limit - 2) * weights.grassland_egg_synergy

    # Food cost penalty (opportunity cost of food spent)
    food_spent = sum(move.food_payment.values())
    value -= food_spent * 0.3

    # Column egg cost penalty
    row = player.board.get_row(move.habitat)
    slot_idx = row.next_empty_slot()
    if slot_idx is not None:
        egg_cost = EGG_COST_BY_COLUMN[slot_idx]
        value -= egg_cost * 0.5

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
        ctx = PowerContext(
            game_state=game, player=player, bird=slot.bird,
            slot_index=i, habitat=Habitat.FOREST,
        )
        power_val = power.estimate_value(ctx)
        # Predator penalty for forest predators
        if slot.bird.is_predator:
            power_val *= weights.predator_penalty
        value += power_val

    # Food-for-birds bonus: food is far more valuable when you have birds to play
    food_available = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
    affordable_soon = 0
    for b in player.hand:
        if food_available + food_count >= b.food_cost.total:
            affordable_soon += 1
    if affordable_soon > 0 and player.food_supply.total_non_nectar() < 3:
        value += weights.food_for_birds_bonus * affordable_soon
        value += 1.0  # Urgency bonus
    elif player.hand_size > 0 and player.food_supply.total_non_nectar() < 2:
        value += 0.5  # Some urgency even if can't quite afford

    # Nectar choice value: flexible food + contributes to nectar majority
    # BUT nectar clears at end of round — heavily penalize on last turn
    if move.food_choices and FoodType.NECTAR in move.food_choices:
        nectar_in_choices = sum(1 for ft in move.food_choices if ft == FoodType.NECTAR)
        if player.action_cubes_remaining <= 1:
            # Last turn of round: nectar will be lost, strong penalty
            value -= nectar_in_choices * 3.0
        else:
            rounds_remaining = max(0, ROUNDS - game.current_round)
            if rounds_remaining > 0:
                value += weights.nectar_early_bonus * (rounds_remaining / ROUNDS)

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
                value += deficit_filled * 0.6 * (best_bird_value / 5.0)

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
    value = actual_eggs * 1.0

    # Bonus cost: paying a food/card for +1 egg is usually net positive (egg = 1pt)
    if move.bonus_count > 0 and actual_eggs > column.base_gain:
        # Extra eggs actually laid beyond base (not wasted on full birds)
        extra_laid = actual_eggs - min(column.base_gain, eligible_space)
        if extra_laid > 0:
            value += extra_laid * 0.4  # Net gain: 1pt egg - ~0.5 cost = ~0.4 net

    # Engine value: brown powers in grassland
    for i, slot in enumerate(player.board.grassland.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = PowerContext(
            game_state=game, player=player, bird=slot.bird,
            slot_index=i, habitat=Habitat.GRASSLAND,
        )
        value += power.estimate_value(ctx)

    # Late game bonus: eggs are guaranteed points
    if game.current_round >= 3:
        value += actual_eggs * 0.3

    # Goal alignment: egg-related goals boost value
    value += _egg_goal_alignment(game)

    return value


def _evaluate_draw_cards(game: GameState, player: Player, move: Move,
                         weights: HeuristicWeights) -> float:
    """Evaluate a draw-cards move."""
    bird_count = player.board.wetland.bird_count
    column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
    card_count = column.base_gain + move.bonus_count

    # Cards are speculative value (might play them or tuck them)
    value = card_count * 0.4

    # Bonus cost: paying an egg/nectar for +1 card
    if move.bonus_count > 0:
        value += move.bonus_count * 0.2  # Cards are speculative, lower net than eggs

    # Reset tray value: worth more when current tray options are poor
    if move.reset_bonus:
        # Fresh tray gives new face-up options — costs a food
        value += 0.3

    # Known tray cards can be evaluated more precisely
    for idx in move.tray_indices:
        if idx < len(game.card_tray.face_up):
            tray_bird = game.card_tray.face_up[idx]
            # Quick evaluation of the tray bird's play value
            value += tray_bird.victory_points * 0.3

            # Bonus synergy
            for bc in player.bonus_cards:
                if bc.name in tray_bird.bonus_eligibility:
                    value += 0.8

            # Goal synergy: does this bird help with round goals?
            if game.round_goals:
                best_goal_val = 0.0
                for hab in tray_bird.habitats:
                    gv = _goal_alignment_value(game, tray_bird, hab, weights)
                    best_goal_val = max(best_goal_val, gv)
                value += best_goal_val * 0.3  # Discounted since we still need to play it

    # Engine value: brown powers in wetland
    for i, slot in enumerate(player.board.wetland.slots):
        if not slot.bird or slot.bird.color != PowerColor.BROWN:
            continue
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue
        ctx = PowerContext(
            game_state=game, player=player, bird=slot.bird,
            slot_index=i, habitat=Habitat.WETLAND,
        )
        value += power.estimate_value(ctx)

    # If hand is empty, drawing cards is more urgent
    if player.hand_size == 0:
        value += 2.0
    elif player.hand_size <= 2:
        value += 0.5

    # Late game penalty: drawing cards with few actions left is wasteful
    if game.current_round >= 4 and player.action_cubes_remaining <= 2:
        value -= 1.0

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
    """Generate activation advice for brown birds in a row.

    For "all players" powers, compares your need vs opponents' need for the
    resource to decide whether activating helps you more or helps them more.
    """
    from backend.powers.templates.gain_food import GainFoodFromSupply
    from backend.powers.templates.draw_cards import DrawCards
    from backend.powers.templates.lay_eggs import LayEggs

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
        is_all_players = getattr(power, 'all_players', False)

        if not is_all_players:
            advice.append(f"activate {name}")
            continue

        if not opponents:
            advice.append(f"activate {name}")
            continue

        # Analyze whether opponents benefit more than you
        should_skip = False
        reason = ""

        if isinstance(power, GainFoodFromSupply):
            my_food = player.food_supply.total_non_nectar()
            for opp in opponents:
                opp_food = opp.food_supply.total_non_nectar()
                # You already have plenty — giving to opponent is a net loss
                if my_food >= 4:
                    should_skip = True
                    reason = "you have enough food already"
                    break
                # Opponent is low on food and would benefit more
                if opp_food <= my_food and opp_food < 3:
                    if opp.hand_size > 0 or opp_food == 0:
                        should_skip = True
                        reason = f"{opp.name} low on food ({opp_food})"
                        break

        elif isinstance(power, DrawCards):
            my_hand = player.hand_size
            for opp in opponents:
                opp_hand = opp.hand_size
                opp_wetland = opp.board.wetland.bird_count
                # Opponent has few wetland birds — drawing is scarce for them
                if opp_wetland < 2 and opp_hand <= 2:
                    should_skip = True
                    reason = f"{opp.name} needs cards (hand {opp_hand}, few wetland birds)"
                    break
                # Opponent hand is lower than yours
                if opp_hand < my_hand and opp_hand <= 2:
                    should_skip = True
                    reason = f"{opp.name} needs cards more (hand {opp_hand})"
                    break

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
                # You have no egg space — pure gift to opponent
                if my_space == 0 and opp_space > 0:
                    should_skip = True
                    reason = "you have no egg space"
                    break
                # Opponent has more egg space, benefits more
                if opp_space > my_space and opp_space >= 2:
                    should_skip = True
                    reason = f"{opp.name} has more egg space ({opp_space})"
                    break

        if should_skip:
            advice.append(f"SKIP {name} ({reason})")
        else:
            advice.append(f"activate {name} (you benefit more)")
    return advice


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
            else:
                reasons.append("gains nectar for majority")
        if move.bonus_count > 0 and column.bonus:
            payment = _recommend_bonus_payment(player, column.bonus.cost_options, game)
            reasons.append(f"+1 food ({payment})" if payment else "+1 food")
        if move.reset_bonus and column.reset_bonus:
            payment = _recommend_bonus_payment(player, column.reset_bonus.cost_options, game)
            reasons.insert(0, f"FIRST reset feeder ({payment}), then take food" if payment else "FIRST reset feeder, then take food")
        if bird_count > 0:
            reasons.append(f"activates {bird_count} forest bird{'s' if bird_count > 1 else ''}")
            reasons.extend(_activation_advice(game, player, Habitat.FOREST))

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
        if bird_count > 0:
            reasons.append(f"activates {bird_count} grassland bird{'s' if bird_count > 1 else ''}")
            reasons.extend(_activation_advice(game, player, Habitat.GRASSLAND))
        if move.bonus_count > 0 and column.bonus:
            payment = _recommend_bonus_payment(player, column.bonus.cost_options, game)
            label = f"+{move.bonus_count} egg{'s' if move.bonus_count > 1 else ''}"
            reasons.append(f"{label} ({payment})" if payment else label)
        if game.round_goals:
            idx = min(game.current_round - 1, len(game.round_goals) - 1)
            if idx >= 0 and "[egg]" in game.round_goals[idx].description.lower():
                reasons.append("helps egg goal")

    elif move.action_type == ActionType.DRAW_CARDS:
        if player.hand_size == 0:
            reasons.append("empty hand — need cards")
        elif player.hand_size <= 2:
            reasons.append("hand is low")
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
        if move.reset_bonus and column.reset_bonus:
            payment = _recommend_bonus_payment(player, column.reset_bonus.cost_options, game)
            reasons.insert(0, f"FIRST reset tray ({payment}), then draw cards" if payment else "FIRST reset tray, then draw cards")
        if bird_count > 0:
            reasons.append(f"activates {bird_count} wetland bird{'s' if bird_count > 1 else ''}")
            reasons.extend(_activation_advice(game, player, Habitat.WETLAND))

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
