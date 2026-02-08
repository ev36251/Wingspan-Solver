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
    """
    if cost.total == 0:
        return True, ""

    if cost.is_or:
        # OR cost: need any ONE of the distinct types
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                # Wild can be paid with anything
                if player.food_supply.total_non_nectar() >= 1:
                    return True, ""
            elif player.food_supply.has(food_type):
                return True, ""
        return False, f"Cannot pay OR cost: need one of {[ft.value for ft in cost.distinct_types]}"

    # Normal cost: need to pay all items
    # Count required food by type
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Check each specific food type
    for ft, count in required.items():
        if not player.food_supply.has(ft, count):
            return False, f"Not enough {ft.value}: need {count}, have {player.food_supply.get(ft)}"

    # Wild food can be paid with any non-nectar food the player has
    # After accounting for specific costs
    if wild_needed > 0:
        remaining = player.food_supply.total_non_nectar()
        for ft, count in required.items():
            remaining -= count
        if remaining < wild_needed:
            return False, f"Not enough food for wild cost: need {wild_needed} more"

    return True, ""


def find_food_payment_options(player: Player, cost: FoodCost) -> list[dict[FoodType, int]]:
    """Find all valid ways to pay a food cost.

    Returns a list of payment dicts {FoodType: count}.
    For OR costs, returns one option per payable type.
    For wild costs, returns options for each food type that could fill the wild slot.
    """
    if cost.total == 0:
        return [{}]

    if cost.is_or:
        options = []
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                for ft in (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                           FoodType.FRUIT, FoodType.RODENT):
                    if player.food_supply.has(ft):
                        options.append({ft: 1})
            elif player.food_supply.has(food_type):
                options.append({food_type: 1})
        return options

    # Normal cost: start with required specific foods
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Check if specific requirements can be met
    for ft, count in required.items():
        if not player.food_supply.has(ft, count):
            return []

    if wild_needed == 0:
        return [dict(required)]

    # For wild slots, enumerate what food types could fill them
    # Simplified: just pick available food types not already fully committed
    available: dict[FoodType, int] = {}
    for ft in (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
               FoodType.FRUIT, FoodType.RODENT):
        have = player.food_supply.get(ft)
        committed = required.get(ft, 0)
        leftover = have - committed
        if leftover > 0:
            available[ft] = leftover

    # Generate one option per food type that can fill wild (simplified)
    options = []
    for ft, leftover in available.items():
        if leftover >= wild_needed:
            payment = dict(required)
            payment[ft] = payment.get(ft, 0) + wild_needed
            options.append(payment)

    # If no single type can fill all wild, try combinations (for wild_needed > 1)
    if not options and wild_needed > 1:
        # Fallback: greedily fill with whatever's available
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
