"""Enumerate all legal moves for the current player."""

from dataclasses import dataclass, field
from backend.config import (
    EGG_COST_BY_COLUMN,
    get_action_column,
)
from backend.models.enums import ActionType, FoodType, Habitat
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.rules import (
    can_play_bird, can_gain_food, can_lay_eggs, can_draw_cards,
    find_food_payment_options,
)


@dataclass
class Move:
    """A concrete legal move the player can make."""
    action_type: ActionType
    description: str

    # Play bird specifics
    bird_name: str | None = None
    habitat: Habitat | None = None
    food_payment: dict[FoodType, int] = field(default_factory=dict)

    # Gain food specifics
    food_choices: list[FoodType] = field(default_factory=list)

    # Lay eggs specifics — maps (habitat, slot_idx) -> egg count
    egg_distribution: dict[tuple[Habitat, int], int] = field(default_factory=dict)

    # Draw cards specifics
    tray_indices: list[int] = field(default_factory=list)
    deck_draws: int = 0

    # Bonus trade
    bonus_count: int = 0
    reset_bonus: bool = False


def generate_play_bird_moves(game: GameState, player: Player) -> list[Move]:
    """All legal play-bird moves with food payment options."""
    moves = []
    for bird in player.hand:
        for habitat in bird.habitats:
            legal, _ = can_play_bird(player, bird, habitat, game)
            if not legal:
                continue
            payment_options = find_food_payment_options(player, bird.food_cost)
            if not payment_options:
                # Free bird (0 cost)
                moves.append(Move(
                    action_type=ActionType.PLAY_BIRD,
                    description=f"Play {bird.name} in {habitat.value}",
                    bird_name=bird.name,
                    habitat=habitat,
                    food_payment={},
                ))
            else:
                # One move per payment option (solver picks the best)
                for payment in payment_options:
                    pay_desc = ", ".join(
                        f"{c} {ft.value}" for ft, c in payment.items()
                    )
                    moves.append(Move(
                        action_type=ActionType.PLAY_BIRD,
                        description=f"Play {bird.name} in {habitat.value} (pay {pay_desc})",
                        bird_name=bird.name,
                        habitat=habitat,
                        food_payment=payment,
                    ))
    return moves


def _can_pay_bonus(player: Player, cost_options: tuple[str, ...]) -> bool:
    """Check if the player can pay at least one bonus cost option."""
    for cost in cost_options:
        if cost == "card" and player.hand:
            return True
        elif cost == "food":
            for ft in [FoodType.SEED, FoodType.INVERTEBRATE, FoodType.FRUIT,
                        FoodType.FISH, FoodType.RODENT]:
                if player.food_supply.get(ft) > 0:
                    return True
        elif cost == "egg":
            if player.board.total_eggs() > 0:
                return True
        elif cost == "nectar":
            if player.food_supply.get(FoodType.NECTAR) > 0:
                return True
    return False


def _feeder_type_counts(feeder) -> dict[FoodType, int]:
    """Count how many dice of each food type are available in the feeder.

    Choice dice count toward both types they offer.
    """
    counts: dict[FoodType, int] = {}
    for die in feeder.dice:
        if isinstance(die, tuple):
            for ft in die:
                counts[ft] = counts.get(ft, 0) + 1
        else:
            counts[die] = counts.get(die, 0) + 1
    return counts


def _generate_food_combos(available_counts: dict[FoodType, int],
                          food_count: int) -> list[list[FoodType]]:
    """Generate all valid food-type combinations from available feeder dice.

    Each combo is a sorted list of food types, respecting the count of each
    type available. For food_count=1, this is just one of each available type.
    For food_count>=2, generates mixed combos.

    If food_count exceeds total available dice, generates combos using all
    available dice — the feeder will reroll mid-action to provide the rest.
    Also always includes single-type fallback moves (take N of one type)
    for cases where the feeder will reroll to provide more.
    """
    types = sorted(available_counts.keys(), key=lambda f: f.value)

    if food_count == 1:
        return [[ft] for ft in types]

    total_available = sum(available_counts.values())

    # Always include single-type moves (feeder rerolls can provide more of same type)
    combos: list[list[FoodType]] = []
    for ft in types:
        combos.append([ft] * food_count)

    # Generate mixed combos up to what's actually available
    effective_count = min(food_count, total_available)

    seen = set()
    mixed: list[list[FoodType]] = []

    def _build(remaining: int, start_idx: int, current: list[FoodType]):
        if remaining == 0:
            key = tuple(current)
            if key not in seen:
                seen.add(key)
                mixed.append(list(current))
            return
        for i in range(start_idx, len(types)):
            ft = types[i]
            already_used = current.count(ft)
            if already_used < available_counts[ft]:
                current.append(ft)
                _build(remaining - 1, i, current)
                current.pop()

    _build(effective_count, 0, [])

    # Add mixed combos that aren't already in the single-type list
    for combo in mixed:
        key = tuple(combo)
        if key not in {tuple(c) for c in combos}:
            combos.append(combo)

    return combos


def _food_combo_description(combo: list[FoodType]) -> str:
    """Build a description like '1 seed + 1 invertebrate'."""
    counts: dict[FoodType, int] = {}
    for ft in combo:
        counts[ft] = counts.get(ft, 0) + 1
    parts = [f"{c} {ft.value}" for ft, c in
             sorted(counts.items(), key=lambda x: x[0].value)]
    return " + ".join(parts)


def generate_gain_food_moves(game: GameState, player: Player) -> list[Move]:
    """All legal gain-food moves.

    Generates specific food type combinations from the feeder,
    with and without bonus trade activation (extra and/or reset_feeder).
    """
    legal, _ = can_gain_food(player, game)
    if not legal:
        return []

    bird_count = player.board.forest.bird_count
    column = get_action_column(game.board_type, Habitat.FOREST, bird_count)
    food_count = column.base_gain

    has_extra = column.bonus and column.bonus.bonus_type == "extra"
    can_extra = has_extra and _can_pay_bonus(player, column.bonus.cost_options)
    has_reset = column.reset_bonus and column.reset_bonus.bonus_type == "reset_feeder"
    can_reset = has_reset and _can_pay_bonus(player, column.reset_bonus.cost_options)

    feeder = game.birdfeeder
    available = feeder.available_food_types()

    if not available:
        moves = [Move(
            action_type=ActionType.GAIN_FOOD,
            description=f"Gain food (feeder will reroll, {food_count} food)",
            food_choices=[],
        )]
        if can_extra:
            moves.append(Move(
                action_type=ActionType.GAIN_FOOD,
                description=f"Gain food (feeder reroll, {food_count + 1} food, +1 food)",
                food_choices=[],
                bonus_count=1,
            ))
        if can_reset:
            moves.append(Move(
                action_type=ActionType.GAIN_FOOD,
                description=f"Gain food (reset feeder, {food_count} food)",
                food_choices=[],
                reset_bonus=True,
            ))
        return moves

    type_counts = _feeder_type_counts(feeder)

    def _make_food_moves(count: int, bonus: int = 0,
                         use_reset: bool = False) -> list[Move]:
        parts = []
        if bonus:
            parts.append("+1 food")
        if use_reset:
            parts.append("reset feeder")
        suffix = f" ({', '.join(parts)})" if parts else ""

        combos = _generate_food_combos(type_counts, count)
        result = []
        for combo in combos:
            desc = f"Gain {_food_combo_description(combo)}{suffix}"
            result.append(Move(
                action_type=ActionType.GAIN_FOOD,
                description=desc,
                food_choices=combo,
                bonus_count=bonus,
                reset_bonus=use_reset,
            ))
        return result

    moves = _make_food_moves(food_count)

    if can_extra:
        moves.extend(_make_food_moves(food_count + 1, bonus=1))

    if can_reset:
        moves.extend(_make_food_moves(food_count, use_reset=True))

    if can_extra and can_reset:
        moves.extend(_make_food_moves(food_count + 1, bonus=1, use_reset=True))

    return moves


def generate_lay_eggs_moves(game: GameState, player: Player) -> list[Move]:
    """All legal lay-eggs moves.

    Generate a default distribution that fills birds optimally,
    with and without bonus trade activation.
    """
    legal, _ = can_lay_eggs(player)
    if not legal:
        return []

    bird_count = player.board.grassland.bird_count
    column = get_action_column(game.board_type, Habitat.GRASSLAND, bird_count)
    egg_count = column.base_gain

    def _make_egg_move(count: int, bonus: int = 0) -> Move:
        eligible = []
        for row in player.board.all_rows():
            for i, slot in enumerate(row.slots):
                if slot.bird and slot.can_hold_more_eggs():
                    eligible.append((slot.eggs_space(), row.habitat, i, slot))

        if not eligible:
            suffix = " + bonus" if bonus else ""
            return Move(
                action_type=ActionType.LAY_EGGS,
                description=f"Lay eggs (no eligible birds, {count} available{suffix})",
                egg_distribution={},
                bonus_count=bonus,
            )

        eligible.sort(key=lambda x: -x[0])
        dist: dict[tuple[Habitat, int], int] = {}
        remaining = count
        for space, hab, idx, slot in eligible:
            if remaining <= 0:
                break
            to_lay = min(space, remaining)
            dist[(hab, idx)] = to_lay
            remaining -= to_lay

        desc_parts = []
        for (hab, idx), c in dist.items():
            slot = player.board.get_row(hab).slots[idx]
            desc_parts.append(f"{c} on {slot.bird.name}")
        desc = f"Lay eggs: {', '.join(desc_parts)}" if desc_parts else "Lay eggs"
        if bonus:
            desc += f" (+{bonus} bonus)"

        return Move(
            action_type=ActionType.LAY_EGGS,
            description=desc,
            egg_distribution=dist,
            bonus_count=bonus,
        )

    moves = [_make_egg_move(egg_count)]

    # Add bonus-trade variants
    if column.bonus and _can_pay_bonus(player, column.bonus.cost_options):
        bonus = column.bonus
        if bonus.bonus_type == "extra":
            moves.append(_make_egg_move(egg_count + 1, bonus=1))
            if bonus.max_uses >= 2 and _can_pay_bonus(player, bonus.cost_options):
                moves.append(_make_egg_move(egg_count + 2, bonus=2))

    return moves


def generate_draw_cards_moves(game: GameState, player: Player) -> list[Move]:
    """All legal draw-cards moves.

    Options: draw from tray (per face-up card) or from deck,
    with and without bonus trade activation (extra and/or reset_tray).
    """
    legal, _ = can_draw_cards(player)
    if not legal:
        return []

    bird_count = player.board.wetland.bird_count
    column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
    card_count = column.base_gain

    has_extra = column.bonus and column.bonus.bonus_type == "extra"
    can_extra = has_extra and _can_pay_bonus(player, column.bonus.cost_options)
    has_reset = column.reset_bonus and column.reset_bonus.bonus_type == "reset_tray"
    can_reset = has_reset and _can_pay_bonus(player, column.reset_bonus.cost_options)

    def _make_draw_moves(count: int, bonus: int = 0,
                         use_reset: bool = False) -> list[Move]:
        moves = []
        parts = []
        if bonus:
            parts.append("+1 card")
        if use_reset:
            parts.append("reset tray")
        suffix = f" ({', '.join(parts)})" if parts else ""

        # Option 1: draw all from deck
        if game.deck_remaining >= count:
            moves.append(Move(
                action_type=ActionType.DRAW_CARDS,
                description=f"Draw {count} from deck{suffix}",
                tray_indices=[],
                deck_draws=count,
                bonus_count=bonus,
                reset_bonus=use_reset,
            ))

        # Option 2: each face-up card from tray (take 1 from tray, rest from deck)
        for i, bird in enumerate(game.card_tray.face_up):
            deck_needed = count - 1
            if game.deck_remaining >= deck_needed:
                moves.append(Move(
                    action_type=ActionType.DRAW_CARDS,
                    description=f"Take {bird.name} from tray + {deck_needed} from deck{suffix}",
                    tray_indices=[i],
                    deck_draws=deck_needed,
                    bonus_count=bonus,
                    reset_bonus=use_reset,
                ))

        # If no moves yet (empty deck & tray), allow what's available
        if not moves:
            available_from_tray = min(count, game.card_tray.count)
            available_from_deck = min(count - available_from_tray, game.deck_remaining)
            moves.append(Move(
                action_type=ActionType.DRAW_CARDS,
                description=f"Draw {available_from_tray} from tray, {available_from_deck} from deck{suffix}",
                tray_indices=list(range(available_from_tray)),
                deck_draws=available_from_deck,
                bonus_count=bonus,
                reset_bonus=use_reset,
            ))

        return moves

    moves = _make_draw_moves(card_count)

    # Extra bonus variants (+1 card)
    if can_extra:
        moves.extend(_make_draw_moves(card_count + 1, bonus=1))

    # Reset tray variants
    if can_reset:
        moves.extend(_make_draw_moves(card_count, use_reset=True))

    # Both bonuses (dual-bonus columns like Oceania Wetland col 3)
    if can_extra and can_reset:
        moves.extend(_make_draw_moves(card_count + 1, bonus=1, use_reset=True))

    return moves


def generate_all_moves(game: GameState, player: Player | None = None) -> list[Move]:
    """Generate all legal moves for the current player."""
    if player is None:
        player = game.current_player

    if game.is_game_over or player.action_cubes_remaining <= 0:
        return []

    moves = []
    moves.extend(generate_play_bird_moves(game, player))
    moves.extend(generate_gain_food_moves(game, player))
    moves.extend(generate_lay_eggs_moves(game, player))
    moves.extend(generate_draw_cards_moves(game, player))
    return moves
