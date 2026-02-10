"""Stable action encoding for policy-network training data."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.models.enums import ActionType, FoodType, Habitat
from backend.solver.move_generator import Move


def _sorted_food_items(food_payment: dict[FoodType, int]) -> str:
    if not food_payment:
        return ""
    parts = [f"{ft.value}:{food_payment[ft]}" for ft in sorted(food_payment.keys(), key=lambda x: x.value)]
    return ",".join(parts)


def _sorted_food_list(foods: list[FoodType]) -> str:
    if not foods:
        return ""
    names = sorted(ft.value for ft in foods)
    return ",".join(names)


def _sorted_egg_distribution(dist: dict[tuple[Habitat, int], int]) -> str:
    if not dist:
        return ""
    parts: list[str] = []
    for (habitat, slot_idx), eggs in sorted(dist.items(), key=lambda x: (x[0][0].value, x[0][1])):
        parts.append(f"{habitat.value}:{slot_idx}:{eggs}")
    return ",".join(parts)


def action_signature(move: Move) -> str:
    """Build a deterministic string signature for a move."""
    if move.action_type == ActionType.PLAY_BIRD:
        return (
            f"PLAY_BIRD|bird={move.bird_name or ''}|hab={move.habitat.value if move.habitat else ''}"
            f"|pay={_sorted_food_items(move.food_payment)}"
        )

    if move.action_type == ActionType.GAIN_FOOD:
        return (
            f"GAIN_FOOD|foods={_sorted_food_list(move.food_choices)}|bonus={move.bonus_count}"
            f"|reset={int(move.reset_bonus)}"
        )

    if move.action_type == ActionType.LAY_EGGS:
        return (
            f"LAY_EGGS|dist={_sorted_egg_distribution(move.egg_distribution)}|bonus={move.bonus_count}"
        )

    if move.action_type == ActionType.DRAW_CARDS:
        tray = ",".join(str(i) for i in sorted(move.tray_indices))
        return (
            f"DRAW_CARDS|tray={tray}|deck={move.deck_draws}|bonus={move.bonus_count}"
            f"|reset={int(move.reset_bonus)}"
        )

    return f"UNKNOWN|type={move.action_type.value}"


@dataclass
class ActionCodec:
    """Maps move signatures to integer action IDs.

    IDs are assigned incrementally as signatures are observed during data generation.
    """

    signature_to_id: dict[str, int] = field(default_factory=dict)
    id_to_signature: list[str] = field(default_factory=list)

    def encode_signature(self, signature: str) -> int:
        if signature in self.signature_to_id:
            return self.signature_to_id[signature]
        action_id = len(self.id_to_signature)
        self.signature_to_id[signature] = action_id
        self.id_to_signature.append(signature)
        return action_id

    def encode_move(self, move: Move) -> int:
        return self.encode_signature(action_signature(move))

    def encode_moves(self, moves: list[Move]) -> list[int]:
        return [self.encode_move(m) for m in moves]

    def to_dict(self) -> dict:
        return {
            "num_actions": len(self.id_to_signature),
            "id_to_signature": self.id_to_signature,
        }
