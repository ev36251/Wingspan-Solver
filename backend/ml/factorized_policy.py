"""Factorized policy target encoding for Wingspan moves."""

from __future__ import annotations

from dataclasses import dataclass

from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.player import Player
from backend.solver.move_generator import Move

ACTION_TYPES = [
    ActionType.PLAY_BIRD,
    ActionType.GAIN_FOOD,
    ActionType.LAY_EGGS,
    ActionType.DRAW_CARDS,
]
ACTION_TYPE_TO_ID = {a: i for i, a in enumerate(ACTION_TYPES)}

HABITAT_CLASSES = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND, None]
HABITAT_TO_ID = {h: i for i, h in enumerate(HABITAT_CLASSES)}

FOOD_CLASSES = [
    FoodType.INVERTEBRATE,
    FoodType.SEED,
    FoodType.FISH,
    FoodType.FRUIT,
    FoodType.RODENT,
    FoodType.NECTAR,
    None,
]
FOOD_TO_ID = {f: i for i, f in enumerate(FOOD_CLASSES)}

DRAW_MODE_CLASSES = ["deck_only", "tray_only", "mixed", "none"]
DRAW_MODE_TO_ID = {v: i for i, v in enumerate(DRAW_MODE_CLASSES)}

POWER_COLOR_CLASSES = [
    PowerColor.BROWN,
    PowerColor.WHITE,
    PowerColor.PINK,
    PowerColor.TEAL,
    PowerColor.YELLOW,
    PowerColor.NONE,
    None,
]
POWER_COLOR_TO_ID = {c: i for i, c in enumerate(POWER_COLOR_CLASSES)}

# Sub-heads that are relevant for each action_type target.
RELEVANT_HEADS_BY_ACTION = {
    0: ["play_habitat", "play_cost_bin", "play_power_color"],  # play_bird
    1: ["gain_food_primary"],
    2: ["lay_eggs_bin"],
    3: ["draw_mode"],
}


@dataclass
class HeadDims:
    action_type: int = 4
    play_habitat: int = 4
    gain_food_primary: int = 7
    draw_mode: int = 4
    lay_eggs_bin: int = 11
    play_cost_bin: int = 7
    play_power_color: int = 7


def _primary_food(move: Move) -> FoodType | None:
    if not move.food_choices:
        return None
    # Stable primary choice: lexicographically smallest present type.
    return sorted(move.food_choices, key=lambda f: f.value)[0]


def _draw_mode(move: Move) -> str:
    from_tray = len(move.tray_indices) > 0
    from_deck = move.deck_draws > 0
    if from_tray and from_deck:
        return "mixed"
    if from_tray:
        return "tray_only"
    if from_deck:
        return "deck_only"
    return "none"


def _play_power_color(move: Move, player: Player) -> PowerColor | None:
    if move.bird_name is None:
        return None
    for b in player.hand:
        if b.name == move.bird_name:
            return b.color
    return None


def encode_factorized_targets(move: Move, player: Player) -> dict[str, int]:
    """Encode a move into factorized target classes for supervised training."""
    t_action = ACTION_TYPE_TO_ID[move.action_type]

    habitat = move.habitat if move.habitat is not None else None
    t_hab = HABITAT_TO_ID.get(habitat, HABITAT_TO_ID[None])

    t_food = FOOD_TO_ID.get(_primary_food(move), FOOD_TO_ID[None])
    t_draw = DRAW_MODE_TO_ID[_draw_mode(move)]

    eggs_total = sum(max(0, int(v)) for v in move.egg_distribution.values())
    t_eggs = min(10, eggs_total)

    play_cost = sum(max(0, int(v)) for v in move.food_payment.values())
    t_cost = min(6, play_cost)

    pcolor = _play_power_color(move, player)
    t_color = POWER_COLOR_TO_ID.get(pcolor, POWER_COLOR_TO_ID[None])

    return {
        "action_type": t_action,
        "play_habitat": t_hab,
        "gain_food_primary": t_food,
        "draw_mode": t_draw,
        "lay_eggs_bin": t_eggs,
        "play_cost_bin": t_cost,
        "play_power_color": t_color,
    }
