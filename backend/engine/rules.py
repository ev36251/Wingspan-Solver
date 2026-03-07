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


def _get_cached_food_pool(player: Player) -> dict[FoodType, int]:
    """Aggregate all cached food across the player's bird slots."""
    pool: dict[FoodType, int] = {}
    for _, _, slot in player.board.all_slots():
        for ft, count in slot.cached_food.items():
            if count > 0:
                pool[ft] = pool.get(ft, 0) + count
    return pool


def can_pay_food_cost(player: Player, cost: FoodCost) -> tuple[bool, str]:
    """Check if a player can pay a bird's food cost.

    OR costs: player needs at least one of the listed types (total = 1 payment).
    Normal costs: player needs all listed food items.
    Wild food in cost: player can pay with any food type.
    Nectar (Oceania) can substitute for any food type (1-for-1).
    2-for-1: any 2 food tokens substitute for 1 specific food you lack.
    Cached food on bird slots may be spent as payment.
    """
    if cost.total == 0:
        return True, ""

    cached = _get_cached_food_pool(player)

    def eff(ft: FoodType) -> int:
        """Effective food available = supply + cached on bird slots."""
        return player.food_supply.get(ft) + cached.get(ft, 0)

    if cost.is_or:
        # OR cost: need any ONE of the distinct types
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                if sum(eff(ft) for ft in _ALL_PAYABLE_TYPES) >= 1:
                    return True, ""
            elif eff(food_type) >= 1:
                return True, ""
        # Nectar can substitute for any OR type (1-for-1)
        if eff(FoodType.NECTAR) >= 1:
            return True, ""
        # 2-for-1: pay any 2 food tokens instead of the 1 required
        if sum(eff(ft) for ft in _ALL_PAYABLE_TYPES) >= 2:
            return True, ""
        return False, f"Cannot pay OR cost: need one of {[ft.value for ft in cost.distinct_types]}"

    # Normal cost: need to pay all items
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Compute shortfall using effective supply (supply + cached)
    total_shortfall = 0
    non_nectar_committed = 0
    for ft, count in required.items():
        have = eff(ft)
        covered = min(have, count)
        non_nectar_committed += covered
        total_shortfall += count - covered

    # Surplus non-nectar food (beyond what's directly committed)
    non_nectar_eff_total = sum(eff(ft) for ft in _NON_NECTAR_TYPES)
    non_nectar_surplus = max(0, non_nectar_eff_total - non_nectar_committed)
    nectar_available = eff(FoodType.NECTAR)

    # Cover shortfall: nectar is 1-for-1; any remaining needs 2-for-1
    nectar_for_shortfall = min(nectar_available, total_shortfall)
    remaining_shortfall = total_shortfall - nectar_for_shortfall
    remaining_nectar = nectar_available - nectar_for_shortfall

    # surplus_pool is available for 2-for-1 substitutions and wild slots
    surplus_pool = non_nectar_surplus + remaining_nectar
    two_for_one_cost = remaining_shortfall * 2

    if surplus_pool < two_for_one_cost:
        return False, (
            f"Not enough food: need {two_for_one_cost} surplus food for "
            f"2-for-1 substitution (have {surplus_pool})"
        )

    surplus_after_substitution = surplus_pool - two_for_one_cost
    if surplus_after_substitution < wild_needed:
        return False, (
            f"Not enough food for wild slots: need {wild_needed} "
            f"(have {surplus_after_substitution} after substitutions)"
        )

    return True, ""


_NON_NECTAR_TYPES = (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                     FoodType.FRUIT, FoodType.RODENT)
_ALL_PAYABLE_TYPES = _NON_NECTAR_TYPES + (FoodType.NECTAR,)


def find_food_payment_options(player: Player, cost: FoodCost) -> list[dict[FoodType, int]]:
    """Find all valid ways to pay a food cost.

    Returns a list of payment dicts {FoodType: count} representing food to deduct.
    Cached food on bird slots is included in the available pool.
    Nectar substitutes 1-for-1 for any specific food type.
    2-for-1: any 2 food tokens substitute for 1 specific food you lack.
    """
    if cost.total == 0:
        return [{}]

    cached = _get_cached_food_pool(player)

    def eff(ft: FoodType) -> int:
        """Effective supply = player supply + cached food on bird slots."""
        return player.food_supply.get(ft) + cached.get(ft, 0)

    if cost.is_or:
        options = []
        for food_type in cost.distinct_types:
            if food_type == FoodType.WILD:
                for ft in _ALL_PAYABLE_TYPES:
                    if eff(ft) >= 1:
                        options.append({ft: 1})
            elif eff(food_type) >= 1:
                options.append({food_type: 1})
        # Nectar can substitute for any OR type
        if eff(FoodType.NECTAR) >= 1:
            nectar_opt = {FoodType.NECTAR: 1}
            if nectar_opt not in options:
                options.append(nectar_opt)
        # 2-for-1 fallback: spend 2 food for the 1 required
        if not options:
            avail = sorted(
                [(ft, eff(ft)) for ft in _ALL_PAYABLE_TYPES if eff(ft) >= 1],
                key=lambda x: -x[1],
            )
            if avail:
                ft1, c1 = avail[0]
                if c1 >= 2:
                    options.append({ft1: 2})
                elif len(avail) >= 2:
                    ft2, _ = avail[1]
                    options.append({ft1: 1, ft2: 1})
        return options

    # Normal cost: start with required specific foods
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Compute what can be paid directly and what's in shortfall
    base_payment: dict[FoodType, int] = {}
    total_shortfall = 0
    for ft, count in required.items():
        have = eff(ft)
        covered = min(have, count)
        if covered > 0:
            base_payment[ft] = covered
        total_shortfall += count - covered

    # Cover shortfall with nectar (1-for-1)
    nectar_eff = eff(FoodType.NECTAR)
    nectar_committed = base_payment.get(FoodType.NECTAR, 0)
    nectar_free = nectar_eff - nectar_committed
    nectar_for_shortfall = min(nectar_free, total_shortfall)
    remaining_shortfall = total_shortfall - nectar_for_shortfall
    remaining_nectar = nectar_free - nectar_for_shortfall

    if nectar_for_shortfall > 0:
        base_payment[FoodType.NECTAR] = base_payment.get(FoodType.NECTAR, 0) + nectar_for_shortfall

    # Build surplus dict: food available beyond what's in base_payment
    surplus: dict[FoodType, int] = {}
    for ft in _NON_NECTAR_TYPES:
        leftover = eff(ft) - base_payment.get(ft, 0)
        if leftover > 0:
            surplus[ft] = leftover
    if remaining_nectar > 0:
        surplus[FoodType.NECTAR] = remaining_nectar

    # Sort surplus descending (greedy: use most-available type first)
    surplus_list = sorted(surplus.items(), key=lambda x: -x[1])
    surplus_total = sum(surplus.values())

    options = []

    if remaining_shortfall == 0:
        # Direct payment (+ nectar) covers all specific types; fill wild slots
        if wild_needed == 0:
            options.append(base_payment)
        else:
            if surplus_total >= wild_needed:
                # Generate single-type options (one per available surplus type)
                for ft, leftover in surplus_list:
                    if leftover >= wild_needed:
                        p = dict(base_payment)
                        p[ft] = p.get(ft, 0) + wild_needed
                        options.append(p)
                # Mixed combo fallback when no single type covers all wilds
                if not options and wild_needed > 1:
                    p = dict(base_payment)
                    remaining_w = wild_needed
                    for ft, leftover in surplus_list:
                        use = min(leftover, remaining_w)
                        if use > 0:
                            p[ft] = p.get(ft, 0) + use
                            remaining_w -= use
                        if remaining_w == 0:
                            break
                    if remaining_w == 0:
                        options.append(p)

    else:
        # Shortfall remains after nectar; try 2-for-1 substitution
        two_for_one_cost = remaining_shortfall * 2
        if surplus_total >= two_for_one_cost + wild_needed:
            # Spend 2× remaining shortfall from surplus pool (greedy)
            payment = dict(base_payment)
            sur = dict(surplus)
            remaining_2 = two_for_one_cost
            for ft, avail in surplus_list:
                use = min(sur.get(ft, 0), remaining_2)
                if use > 0:
                    payment[ft] = payment.get(ft, 0) + use
                    sur[ft] = sur.get(ft, 0) - use
                    remaining_2 -= use
                if remaining_2 == 0:
                    break

            if remaining_2 == 0:
                if wild_needed > 0:
                    # Fill wild slots from remaining surplus after 2-for-1
                    remaining_w = wild_needed
                    for ft, avail in sorted(sur.items(), key=lambda x: -x[1]):
                        use = min(avail, remaining_w)
                        if use > 0:
                            payment[ft] = payment.get(ft, 0) + use
                            remaining_w -= use
                        if remaining_w == 0:
                            break
                    if remaining_w == 0:
                        options.append(payment)
                else:
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
