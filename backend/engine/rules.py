"""Legality checks and cost validation for Wingspan actions."""

from backend.config import EGG_COST_BY_COLUMN
from backend.models.bird import Bird, FoodCost
from backend.models.enums import FoodType, Habitat
from backend.models.game_state import GameState
from backend.models.player import Player


def can_play_bird(player: Player, bird: Bird, habitat: Habitat,
                  game_state: GameState) -> tuple[bool, str]:
    """Check if a player can legally play a bird into a habitat.

    Returns (is_legal, reason_if_not).
    """
    # Bird must be in hand
    if not player.has_bird_in_hand(bird.name):
        return False, "Bird not in hand"

    # Bird must be able to live in this habitat
    if not bird.can_live_in(habitat):
        return False, f"{bird.name} cannot live in {habitat.value}"

    # Habitat must have an empty slot
    row = player.board.get_row(habitat)
    slot_idx = row.next_empty_slot()
    if slot_idx is None:
        return False, f"{habitat.value} row is full"

    # Must be able to pay egg cost for the column
    egg_cost = EGG_COST_BY_COLUMN[slot_idx]
    if player.board.total_eggs() < egg_cost:
        return False, f"Need {egg_cost} egg(s) to play in column {slot_idx + 1}"

    # Must be able to pay food cost
    can_pay, food_reason = can_pay_food_cost(player, bird.food_cost)
    if not can_pay:
        return False, food_reason

    # Must have action cubes
    if player.action_cubes_remaining <= 0:
        return False, "No action cubes remaining"

    return True, ""


def can_pay_food_cost(player: Player, cost: FoodCost) -> tuple[bool, str]:
    """Check if a player can pay a bird's food cost.

    OR costs: player needs at least one of the listed types (total = 1 payment).
    Normal costs: player needs all listed food items.
    Wild food in cost: player can pay with any food type.
    Nectar (Oceania) can substitute for any food type.
    """
    if cost.total == 0:
        return True, ""

    if cost.is_or:
        # OR cost: need any ONE of the distinct types (nectar counts)
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                if player.food_supply.total() >= 1:
                    return True, ""
            elif player.food_supply.has(food_type):
                return True, ""
        # Nectar can substitute for any OR type
        if player.food_supply.has(FoodType.NECTAR):
            return True, ""
        return False, f"Cannot pay OR cost: need one of {[ft.value for ft in cost.distinct_types]}"

    # Normal cost: need to pay all items
    # Count required food by type
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Check specific types, allowing nectar to cover shortfalls
    total_shortfall = 0
    non_nectar_committed = 0
    for ft, count in required.items():
        have = player.food_supply.get(ft)
        covered = min(have, count)
        non_nectar_committed += covered
        total_shortfall += count - covered

    # Resources available to cover shortfall + wild slots:
    # surplus non-nectar food + all nectar
    non_nectar_surplus = max(0, player.food_supply.total_non_nectar() - non_nectar_committed)
    nectar_available = player.food_supply.get(FoodType.NECTAR)
    flexible = non_nectar_surplus + nectar_available

    total_needed = total_shortfall + wild_needed
    if flexible < total_needed:
        return False, f"Not enough food: need {total_needed} more (have {flexible} flexible)"

    return True, ""


_NON_NECTAR_TYPES = (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                     FoodType.FRUIT, FoodType.RODENT)
_ALL_PAYABLE_TYPES = _NON_NECTAR_TYPES + (FoodType.NECTAR,)


def find_food_payment_options(player: Player, cost: FoodCost) -> list[dict[FoodType, int]]:
    """Find all valid ways to pay a food cost.

    Returns a list of payment dicts {FoodType: count}.
    For OR costs, returns one option per payable type.
    For wild costs, returns options for each food type that could fill the wild slot.
    Nectar (Oceania) can substitute for any food type in all cases.
    """
    if cost.total == 0:
        return [{}]

    if cost.is_or:
        options = []
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                for ft in _ALL_PAYABLE_TYPES:
                    if player.food_supply.has(ft):
                        options.append({ft: 1})
            elif player.food_supply.has(food_type):
                options.append({food_type: 1})
        # Nectar can substitute for any OR type
        if player.food_supply.has(FoodType.NECTAR):
            nectar_opt = {FoodType.NECTAR: 1}
            if nectar_opt not in options:
                options.append(nectar_opt)
        return options

    # Normal cost: start with required specific foods
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Check if specific requirements can be met without nectar
    can_pay_specific = all(
        player.food_supply.has(ft, count) for ft, count in required.items()
    )

    options = []

    if can_pay_specific:
        if wild_needed == 0:
            options.append(dict(required))
        else:
            # Fill wild slots with available surplus food (including nectar)
            available: dict[FoodType, int] = {}
            for ft in _ALL_PAYABLE_TYPES:
                leftover = player.food_supply.get(ft) - required.get(ft, 0)
                if leftover > 0:
                    available[ft] = leftover

            for ft, leftover in available.items():
                if leftover >= wild_needed:
                    payment = dict(required)
                    payment[ft] = payment.get(ft, 0) + wild_needed
                    options.append(payment)

            # Combo fallback for wild_needed > 1
            if not options and wild_needed > 1:
                payment = dict(required)
                remaining = wild_needed
                for ft, leftover in available.items():
                    use = min(leftover, remaining)
                    if use > 0:
                        payment[ft] = payment.get(ft, 0) + use
                        remaining -= use
                    if remaining == 0:
                        break
                if remaining == 0:
                    options.append(payment)

    # Generate nectar-assisted options for shortfalls in specific types
    nectar_available = player.food_supply.get(FoodType.NECTAR)
    if nectar_available > 0 and not can_pay_specific:
        payment: dict[FoodType, int] = {}
        nectar_used = 0
        valid = True
        for ft, count in required.items():
            have = player.food_supply.get(ft)
            if have >= count:
                payment[ft] = count
            else:
                if have > 0:
                    payment[ft] = have
                shortfall = count - have
                nectar_used += shortfall
                if nectar_used > nectar_available:
                    valid = False
                    break

        if valid:
            if nectar_used > 0:
                payment[FoodType.NECTAR] = payment.get(FoodType.NECTAR, 0) + nectar_used

            # Handle wild slots with remaining resources
            if wild_needed > 0:
                remaining_nectar = nectar_available - nectar_used
                non_nectar_surplus = sum(
                    max(0, player.food_supply.get(ft) - payment.get(ft, 0))
                    for ft in _NON_NECTAR_TYPES
                )
                if non_nectar_surplus + remaining_nectar >= wild_needed:
                    remaining_wild = wild_needed
                    for ft in _NON_NECTAR_TYPES:
                        leftover = player.food_supply.get(ft) - payment.get(ft, 0)
                        if leftover > 0 and remaining_wild > 0:
                            use = min(leftover, remaining_wild)
                            payment[ft] = payment.get(ft, 0) + use
                            remaining_wild -= use
                    if remaining_wild > 0:
                        payment[FoodType.NECTAR] = payment.get(FoodType.NECTAR, 0) + remaining_wild
                else:
                    valid = False

            if valid:
                options.append(payment)

    return options


def can_gain_food(player: Player, game_state: GameState) -> tuple[bool, str]:
    """Check if the gain food (forest) action is legal."""
    if player.action_cubes_remaining <= 0:
        return False, "No action cubes remaining"
    if game_state.birdfeeder.is_empty:
        # Player can still take the action — feeder rerolls
        pass
    return True, ""


def can_lay_eggs(player: Player) -> tuple[bool, str]:
    """Check if the lay eggs (grassland) action is legal."""
    if player.action_cubes_remaining <= 0:
        return False, "No action cubes remaining"
    # Always legal even if no birds to lay on (action still used)
    return True, ""


def can_draw_cards(player: Player) -> tuple[bool, str]:
    """Check if the draw cards (wetland) action is legal."""
    if player.action_cubes_remaining <= 0:
        return False, "No action cubes remaining"
    return True, ""


def egg_cost_for_slot(slot_idx: int) -> int:
    """Egg cost to play a bird in a given column (0-indexed)."""
    return EGG_COST_BY_COLUMN[slot_idx]
