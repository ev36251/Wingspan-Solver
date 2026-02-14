"""Execute the four core Wingspan actions."""

import random
from dataclasses import dataclass, field
from backend.config import (
    EGG_COST_BY_COLUMN,
    get_action_column,
)
from backend.models.bird import Bird
from backend.models.board import BirdSlot
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.rules import can_play_bird, can_gain_food, can_lay_eggs, can_draw_cards
from backend.powers.registry import get_power, assert_power_allowed_for_strict_mode
from backend.powers.base import PowerContext, PowerResult, NoPower
from backend.powers.choices import consume_power_activation_decision


@dataclass
class PowerActivation:
    """Record of a single bird power activation during row traversal."""
    bird_name: str
    slot_index: int
    result: PowerResult


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    action_type: ActionType
    message: str = ""
    # What happened during the action
    food_gained: dict[FoodType, int] = field(default_factory=dict)
    eggs_laid: int = 0
    cards_drawn: int = 0
    bird_played: str | None = None
    habitat: Habitat | None = None
    bonus_activated: int = 0
    power_activations: list[PowerActivation] = field(default_factory=list)


def _record_power_event(game_state: GameState, *, timing: str, player: Player, bird: Bird, executed: bool) -> None:
    recorder = getattr(game_state, "record_power_event", None)
    if callable(recorder):
        recorder(
            timing=timing,
            color=bird.color,
            player_name=player.name,
            bird_name=bird.name,
            executed=executed,
        )


def activate_row(game_state: GameState, player: Player,
                 habitat: Habitat, simulate: bool = True) -> list[PowerActivation]:
    """Activate brown powers in a habitat row, right to left.

    In Wingspan, after taking a habitat action, you traverse birds from
    right to left activating each brown power (optionally). For simulation,
    we activate all beneficial powers (skip all-players powers when opponents
    benefit more).

    Args:
        game_state: Current game state
        player: The acting player
        habitat: Which row to activate
        simulate: If True, auto-decide whether to activate each power.
                  If False, activate all powers unconditionally.
    """
    row = player.board.get_row(habitat)
    activations: list[PowerActivation] = []
    opponents = [p for p in game_state.players if p.name != player.name]

    # Iterate occupied slots right to left
    occupied = [(i, s) for i, s in enumerate(row.slots) if s.bird is not None]
    for i, slot in reversed(occupied):
        if slot.bird.color != PowerColor.BROWN:
            continue
        activate_choice = consume_power_activation_decision(game_state, player.name, slot.bird.name)
        if activate_choice is False:
            continue
        force_activate = activate_choice is True
        assert_power_allowed_for_strict_mode(game_state, slot.bird)
        power = get_power(slot.bird)
        if isinstance(power, NoPower):
            continue

        ctx = PowerContext(
            game_state=game_state, player=player, bird=slot.bird,
            slot_index=i, habitat=habitat,
        )

        # Smart activation: skip all-players powers that help opponents more
        if simulate and opponents and not force_activate:
            is_all_players = getattr(power, 'all_players', False)
            if is_all_players:
                val = power.estimate_value(ctx)
                # Skip if low net value (opponents gain roughly as much as you)
                if val < 1.5:
                    continue

        if power.can_execute(ctx):
            result = power.execute(ctx)
            _record_power_event(
                game_state,
                timing="brown",
                player=player,
                bird=slot.bird,
                executed=bool(result.executed),
            )
            if result.executed:
                activations.append(PowerActivation(
                    bird_name=slot.bird.name,
                    slot_index=i,
                    result=result,
                ))

    return activations


def _pay_bonus_cost(player: Player, cost_options: tuple[str, ...]) -> bool:
    """Try to pay one bonus cost. Returns True if successful.

    Tries each cost option in order and pays the first available one.
    """
    for cost in cost_options:
        if cost == "card" and player.hand:
            # Discard the lowest-VP card (least valuable to keep)
            worst_idx = min(range(len(player.hand)),
                            key=lambda i: player.hand[i].victory_points)
            player.hand.pop(worst_idx)
            return True
        elif cost == "food":
            for ft in [FoodType.SEED, FoodType.INVERTEBRATE, FoodType.FRUIT,
                        FoodType.FISH, FoodType.RODENT]:
                if player.food_supply.get(ft) > 0:
                    player.food_supply.spend(ft, 1)
                    return True
        elif cost == "egg":
            slots = _auto_select_egg_payment(player, 1)
            if slots:
                hab, si = slots[0]
                player.board.get_row(hab).slots[si].eggs -= 1
                return True
        elif cost == "nectar":
            if player.food_supply.get(FoodType.NECTAR) > 0:
                player.food_supply.spend(FoodType.NECTAR, 1)
                return True
    return False


def _draw_bird_from_deck(game_state: GameState):
    """Draw one bird identity from the game's deck model if available."""
    deck_cards = getattr(game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
        if game_state.deck_tracker is not None:
            game_state.deck_tracker.mark_drawn(card.name)
        return card

    # Fallback for games without initialized deck identities.
    if game_state.deck_remaining <= 0:
        return None
    from backend.data.registries import get_bird_registry
    pool = list(get_bird_registry().all_birds)
    if not pool:
        return None
    game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
    card = random.choice(pool)
    if game_state.deck_tracker is not None:
        game_state.deck_tracker.mark_drawn(card.name)
    return card


def execute_play_bird(
    game_state: GameState,
    player: Player,
    bird: Bird,
    habitat: Habitat,
    food_payment: dict[FoodType, int],
    egg_payment_slots: list[tuple[Habitat, int]] | None = None,
) -> ActionResult:
    """Play a bird card from hand onto the board.

    Args:
        game_state: Current game state
        player: The acting player
        bird: The bird to play
        habitat: Which habitat to place it in
        food_payment: How to pay food cost {FoodType: count}
        egg_payment_slots: Where to remove eggs from [(habitat, slot_idx), ...]
    """
    legal, reason = can_play_bird(player, bird, habitat, game_state)
    if not legal:
        return ActionResult(False, ActionType.PLAY_BIRD, reason)

    row = player.board.get_row(habitat)
    slot_idx = row.next_empty_slot()

    # Pay egg cost
    egg_cost = EGG_COST_BY_COLUMN[slot_idx]
    if egg_cost > 0:
        if not egg_payment_slots:
            # Auto-select: remove from birds with the most eggs
            egg_payment_slots = _auto_select_egg_payment(player, egg_cost)
        if egg_payment_slots is None:
            return ActionResult(False, ActionType.PLAY_BIRD, "Cannot pay egg cost")

        eggs_removed = 0
        for hab, si in egg_payment_slots:
            slot = player.board.get_row(hab).slots[si]
            if slot.eggs > 0:
                slot.eggs -= 1
                eggs_removed += 1
            if eggs_removed >= egg_cost:
                break

        if eggs_removed < egg_cost:
            return ActionResult(False, ActionType.PLAY_BIRD,
                                f"Could only remove {eggs_removed}/{egg_cost} eggs")

    # Pay food cost
    for food_type, count in food_payment.items():
        if not player.food_supply.spend(food_type, count):
            return ActionResult(False, ActionType.PLAY_BIRD,
                                f"Failed to spend {count} {food_type.value}")

    # Track nectar spent per habitat (Oceania)
    nectar_spent = food_payment.get(FoodType.NECTAR, 0)
    if nectar_spent > 0:
        row.nectar_spent += nectar_spent

    # Remove bird from hand and place on board
    player.remove_from_hand(bird.name)
    row.slots[slot_idx].bird = bird

    power_acts: list[PowerActivation] = []

    # Execute "when played" (white) powers immediately
    if bird.color == PowerColor.WHITE:
        assert_power_allowed_for_strict_mode(game_state, bird)
        power = get_power(bird)
        if not isinstance(power, NoPower):
            ctx = PowerContext(
                game_state=game_state, player=player, bird=bird,
                slot_index=slot_idx, habitat=habitat,
            )
            if power.can_execute(ctx):
                result = power.execute(ctx)
                _record_power_event(
                    game_state,
                    timing="white",
                    player=player,
                    bird=bird,
                    executed=bool(result.executed),
                )
                if result.executed:
                    power_acts.append(PowerActivation(
                        bird_name=bird.name,
                        slot_index=slot_idx,
                        result=result,
                    ))

    result = ActionResult(
        True, ActionType.PLAY_BIRD,
        f"Played {bird.name} in {habitat.value} slot {slot_idx + 1}",
        bird_played=bird.name, habitat=habitat,
        power_activations=power_acts,
    )
    # Track cubes spent on the "play a bird" action for round-goal scoring.
    player.play_bird_actions_this_round += 1
    player.action_types_used_this_round.add(ActionType.PLAY_BIRD)
    from backend.engine.timed_powers import trigger_between_turn_powers
    trigger_between_turn_powers(
        game_state,
        trigger_player=player,
        trigger_action=ActionType.PLAY_BIRD,
        trigger_result=result,
    )
    return result


def execute_play_bird_discounted(
    game_state: GameState,
    player: Player,
    bird: Bird,
    habitat: Habitat,
    food_payment: dict[FoodType, int],
    egg_discount: int = 0,
    egg_payment_slots: list[tuple[Habitat, int]] | None = None,
) -> ActionResult:
    """Play a bird with an egg cost discount (used for bonus plays)."""
    # Custom legality: same as can_play_bird, but egg cost is discounted
    if not player.has_bird_in_hand(bird.name):
        return ActionResult(False, ActionType.PLAY_BIRD, "Bird not in hand")
    if not bird.can_live_in(habitat):
        return ActionResult(False, ActionType.PLAY_BIRD,
                            f"{bird.name} cannot live in {habitat.value}")
    row = player.board.get_row(habitat)
    slot_idx = row.next_empty_slot()
    if slot_idx is None:
        return ActionResult(False, ActionType.PLAY_BIRD, f"{habitat.value} row is full")
    egg_cost = max(0, EGG_COST_BY_COLUMN[slot_idx] - egg_discount)
    if player.board.total_eggs() < egg_cost:
        return ActionResult(False, ActionType.PLAY_BIRD,
                            f"Need {egg_cost} egg(s) to play in column {slot_idx + 1}")
    # Validate food payment against supply (discounted payment provided)
    if any(player.food_supply.get(ft) < cnt for ft, cnt in food_payment.items()):
        return ActionResult(False, ActionType.PLAY_BIRD, "Not enough food to pay")
    row = player.board.get_row(habitat)
    slot_idx = row.next_empty_slot()

    # Pay discounted egg cost
    base_egg_cost = EGG_COST_BY_COLUMN[slot_idx]
    egg_cost = max(0, base_egg_cost - egg_discount)
    if egg_cost > 0:
        if not egg_payment_slots:
            egg_payment_slots = _auto_select_egg_payment(player, egg_cost)
        if egg_payment_slots is None:
            return ActionResult(False, ActionType.PLAY_BIRD, "Cannot pay egg cost")

        eggs_removed = 0
        for hab, si in egg_payment_slots:
            slot = player.board.get_row(hab).slots[si]
            if slot.eggs > 0:
                slot.eggs -= 1
                eggs_removed += 1
            if eggs_removed >= egg_cost:
                break

        if eggs_removed < egg_cost:
            return ActionResult(False, ActionType.PLAY_BIRD,
                                f"Could only remove {eggs_removed}/{egg_cost} eggs")

    # Pay food cost
    for food_type, count in food_payment.items():
        if not player.food_supply.spend(food_type, count):
            return ActionResult(False, ActionType.PLAY_BIRD,
                                f"Failed to spend {count} {food_type.value}")

    # Track nectar spent per habitat (Oceania)
    nectar_spent = food_payment.get(FoodType.NECTAR, 0)
    if nectar_spent > 0:
        row.nectar_spent += nectar_spent

    # Remove bird from hand and place on board
    player.remove_from_hand(bird.name)
    row.slots[slot_idx].bird = bird

    power_acts: list[PowerActivation] = []
    if bird.color == PowerColor.WHITE:
        assert_power_allowed_for_strict_mode(game_state, bird)
        power = get_power(bird)
        if not isinstance(power, NoPower):
            ctx = PowerContext(
                game_state=game_state, player=player, bird=bird,
                slot_index=slot_idx, habitat=habitat,
            )
            if power.can_execute(ctx):
                result = power.execute(ctx)
                _record_power_event(
                    game_state,
                    timing="white",
                    player=player,
                    bird=bird,
                    executed=bool(result.executed),
                )
                if result.executed:
                    power_acts.append(PowerActivation(
                        bird_name=bird.name,
                        slot_index=slot_idx,
                        result=result,
                    ))

    result = ActionResult(
        True, ActionType.PLAY_BIRD,
        f"Played {bird.name} in {habitat.value} slot {slot_idx + 1}",
        bird_played=bird.name, habitat=habitat,
        power_activations=power_acts,
    )
    from backend.engine.timed_powers import trigger_between_turn_powers
    trigger_between_turn_powers(
        game_state,
        trigger_player=player,
        trigger_action=ActionType.PLAY_BIRD,
        trigger_result=result,
    )
    return result


def _auto_select_egg_payment(player: Player, count: int) -> list[tuple[Habitat, int]] | None:
    """Auto-select eggs to remove, preferring birds with the most eggs."""
    candidates = []
    for habitat, idx, slot in player.board.all_slots():
        if slot.eggs > 0:
            candidates.append((slot.eggs, habitat.value, habitat, idx))

    # Sort by most eggs first (use habitat name as tiebreaker)
    candidates.sort(reverse=True)

    result = []
    remaining = count
    for _, _hab_str, hab, si in candidates:
        if remaining <= 0:
            break
        result.append((hab, si))
        remaining -= 1

    return result if remaining <= 0 else None


def execute_gain_food(
    game_state: GameState,
    player: Player,
    food_choices: list[FoodType],
    bonus_count: int = 0,
    reset_bonus: bool = False,
) -> ActionResult:
    """Execute the gain food (forest) action.

    Args:
        food_choices: Which food types to take from the birdfeeder.
        bonus_count: How many "extra" bonus trades to activate (0, 1, or 2).
        reset_bonus: Whether to activate the reset_feeder bonus (if available).
    """
    legal, reason = can_gain_food(player, game_state)
    if not legal:
        return ActionResult(False, ActionType.GAIN_FOOD, reason)

    bird_count = player.board.forest.bird_count
    column = get_action_column(game_state.board_type, Habitat.FOREST, bird_count)
    food_count = column.base_gain

    bonus_activated = 0

    # Handle reset_bonus (reset feeder before gaining food)
    if reset_bonus and column.reset_bonus:
        rb = column.reset_bonus
        if rb.bonus_type == "reset_feeder":
            if _pay_bonus_cost(player, rb.cost_options):
                game_state.birdfeeder.reroll()
                bonus_activated += 1

    # Handle "extra" bonus trades (+1 food per use)
    if bonus_count > 0 and column.bonus:
        bonus = column.bonus
        for _ in range(min(bonus_count, bonus.max_uses)):
            if bonus.bonus_type == "extra":
                if _pay_bonus_cost(player, bonus.cost_options):
                    food_count += 1
                    bonus_activated += 1

    # If birdfeeder needs reroll, do it
    if game_state.birdfeeder.should_reroll():
        game_state.birdfeeder.reroll()

    # Take food from birdfeeder
    gained: dict[FoodType, int] = {}
    taken = 0
    for food_type in food_choices:
        if taken >= food_count:
            break
        if game_state.birdfeeder.take_food(food_type):
            player.food_supply.add(food_type)
            gained[food_type] = gained.get(food_type, 0) + 1
            taken += 1

        # Reroll if feeder empties during action
        if game_state.birdfeeder.should_reroll() and taken < food_count:
            game_state.birdfeeder.reroll()

    # Activate brown powers in the forest row (right to left)
    power_acts = activate_row(game_state, player, Habitat.FOREST)

    result = ActionResult(
        True, ActionType.GAIN_FOOD,
        f"Gained {taken} food from birdfeeder",
        food_gained=gained, habitat=Habitat.FOREST,
        bonus_activated=bonus_activated,
        power_activations=power_acts,
    )
    player.gain_food_actions_this_round += 1
    player.action_types_used_this_round.add(ActionType.GAIN_FOOD)
    from backend.engine.timed_powers import trigger_between_turn_powers
    trigger_between_turn_powers(
        game_state,
        trigger_player=player,
        trigger_action=ActionType.GAIN_FOOD,
        trigger_result=result,
    )
    return result


def execute_lay_eggs(
    game_state: GameState,
    player: Player,
    egg_distribution: dict[tuple[Habitat, int], int],
    bonus_count: int = 0,
) -> ActionResult:
    """Execute the lay eggs (grassland) action.

    Args:
        egg_distribution: Where to place eggs {(habitat, slot_idx): count}
        bonus_count: How many bonus trades to activate (0, 1, or 2).
    """
    legal, reason = can_lay_eggs(player)
    if not legal:
        return ActionResult(False, ActionType.LAY_EGGS, reason)

    bird_count = player.board.grassland.bird_count
    column = get_action_column(game_state.board_type, Habitat.GRASSLAND, bird_count)
    egg_count = column.base_gain

    # Handle bonus trades
    bonus_activated = 0
    if bonus_count > 0 and column.bonus:
        bonus = column.bonus
        for _ in range(min(bonus_count, bonus.max_uses)):
            if bonus.bonus_type == "extra":
                if _pay_bonus_cost(player, bonus.cost_options):
                    egg_count += 1
                    bonus_activated += 1

    total_laid = 0
    for (hab, slot_idx), count in egg_distribution.items():
        slot = player.board.get_row(hab).slots[slot_idx]
        if slot.bird is None:
            continue
        for _ in range(count):
            if total_laid >= egg_count:
                break
            if slot.can_hold_more_eggs():
                slot.eggs += 1
                total_laid += 1

    # Activate brown powers in the grassland row (right to left)
    power_acts = activate_row(game_state, player, Habitat.GRASSLAND)

    result = ActionResult(
        True, ActionType.LAY_EGGS,
        f"Laid {total_laid} eggs",
        eggs_laid=total_laid, habitat=Habitat.GRASSLAND,
        bonus_activated=bonus_activated,
        power_activations=power_acts,
    )
    player.lay_eggs_actions_this_round += 1
    player.action_types_used_this_round.add(ActionType.LAY_EGGS)
    from backend.engine.timed_powers import trigger_between_turn_powers
    trigger_between_turn_powers(
        game_state,
        trigger_player=player,
        trigger_action=ActionType.LAY_EGGS,
        trigger_result=result,
    )
    return result


def execute_draw_cards(
    game_state: GameState,
    player: Player,
    from_tray_indices: list[int] | None = None,
    from_deck_count: int = 0,
    bonus_count: int = 0,
    reset_bonus: bool = False,
) -> ActionResult:
    """Execute the draw cards (wetland) action.

    Args:
        from_tray_indices: Indices of face-up cards to take (0-2)
        from_deck_count: Number of cards to draw from the deck
        bonus_count: How many "extra" bonus trades to activate (0, 1, or 2).
        reset_bonus: Whether to activate the reset_tray bonus (if available).
    """
    legal, reason = can_draw_cards(player)
    if not legal:
        return ActionResult(False, ActionType.DRAW_CARDS, reason)

    bird_count = player.board.wetland.bird_count
    column = get_action_column(game_state.board_type, Habitat.WETLAND, bird_count)
    card_count = column.base_gain

    bonus_activated = 0

    # Handle reset_bonus (reset tray before drawing)
    if reset_bonus and column.reset_bonus:
        rb = column.reset_bonus
        if rb.bonus_type == "reset_tray":
            if _pay_bonus_cost(player, rb.cost_options):
                game_state.card_tray.clear()
                bonus_activated += 1

    # Handle "extra" bonus trades (+1 card per use)
    if bonus_count > 0 and column.bonus:
        bonus = column.bonus
        for _ in range(min(bonus_count, bonus.max_uses)):
            if bonus.bonus_type == "extra":
                if _pay_bonus_cost(player, bonus.cost_options):
                    card_count += 1
                    bonus_activated += 1

    drawn = 0
    tray_indices = from_tray_indices or []

    # Take from tray first (highest index first to avoid shifting)
    for idx in sorted(tray_indices, reverse=True):
        if drawn >= card_count:
            break
        card = game_state.card_tray.take_card(idx)
        if card:
            player.hand.append(card)
            drawn += 1

    # Draw remaining from deck
    while drawn < card_count and from_deck_count > 0:
        card = _draw_bird_from_deck(game_state)
        if not card:
            break
        player.hand.append(card)
        from_deck_count -= 1
        drawn += 1

    # Activate brown powers in the wetland row (right to left)
    power_acts = activate_row(game_state, player, Habitat.WETLAND)

    result = ActionResult(
        True, ActionType.DRAW_CARDS,
        f"Drew {drawn} cards",
        cards_drawn=drawn, habitat=Habitat.WETLAND,
        bonus_activated=bonus_activated,
        power_activations=power_acts,
    )
    player.draw_cards_actions_this_round += 1
    player.action_types_used_this_round.add(ActionType.DRAW_CARDS)
    from backend.engine.timed_powers import trigger_between_turn_powers
    trigger_between_turn_powers(
        game_state,
        trigger_player=player,
        trigger_action=ActionType.DRAW_CARDS,
        trigger_result=result,
    )
    return result
