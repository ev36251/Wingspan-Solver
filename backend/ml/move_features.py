"""Move-level feature encoder for factorized BC ranking/value scoring."""

from __future__ import annotations

from backend.models.enums import ActionType, FoodType, PowerColor
from backend.models.player import Player
from backend.solver.move_generator import Move


_ACTION_CLASSES = [
    ActionType.PLAY_BIRD,
    ActionType.GAIN_FOOD,
    ActionType.LAY_EGGS,
    ActionType.DRAW_CARDS,
]
_ACTION_TO_ID = {a: i for i, a in enumerate(_ACTION_CLASSES)}

_POWER_CLASSES = [
    PowerColor.BROWN,
    PowerColor.WHITE,
    PowerColor.PINK,
    PowerColor.TEAL,
    PowerColor.YELLOW,
    PowerColor.NONE,
    None,
]
_POWER_TO_ID = {c: i for i, c in enumerate(_POWER_CLASSES)}

_FOOD_CLASSES = [
    FoodType.INVERTEBRATE,
    FoodType.SEED,
    FoodType.FISH,
    FoodType.FRUIT,
    FoodType.RODENT,
    FoodType.NECTAR,
    None,
]
_FOOD_TO_ID = {f: i for i, f in enumerate(_FOOD_CLASSES)}

MOVE_FEATURE_DIM = 21


def _find_hand_bird(move: Move, player: Player):
    if move.bird_name is None:
        return None
    for b in player.hand:
        if b.name == move.bird_name:
            return b
    return None


def _primary_food(move: Move):
    if not move.food_choices:
        return None
    return sorted(move.food_choices, key=lambda f: f.value)[0]


def encode_move_features(move: Move, player: Player) -> list[float]:
    """Encode move-specific features for move-value scoring."""
    f = [0.0] * MOVE_FEATURE_DIM

    # action_type_one_hot[4]
    aidx = _ACTION_TO_ID.get(move.action_type)
    if aidx is not None:
        f[aidx] = 1.0

    # play block: vp/cost/egg_cap + power_color_one_hot[7]
    bird = _find_hand_bird(move, player)
    if move.action_type == ActionType.PLAY_BIRD and bird is not None:
        f[4] = float(max(0, bird.victory_points)) / 10.0
        f[5] = float(max(0, bird.food_cost.total)) / 6.0
        f[6] = float(max(0, bird.egg_limit)) / 6.0
        pidx = _POWER_TO_ID.get(bird.color, _POWER_TO_ID[None])
        f[7 + int(pidx)] = 1.0

    # gain-food block: primary food one_hot[7]
    if move.action_type == ActionType.GAIN_FOOD:
        food = _primary_food(move)
        food_idx = _FOOD_TO_ID.get(food, _FOOD_TO_ID[None])
        f[14 + int(food_idx)] = 1.0

    return f

