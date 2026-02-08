"""Birdfeeder: 5 custom dice that provide food."""

import random
from dataclasses import dataclass, field
from .enums import BoardType, FoodType

# Base game die: 6 faces
BASE_DICE_FACES: list[FoodType | tuple[FoodType, FoodType]] = [
    FoodType.INVERTEBRATE,
    FoodType.SEED,
    FoodType.FISH,
    FoodType.FRUIT,
    FoodType.RODENT,
    (FoodType.INVERTEBRATE, FoodType.SEED),  # Player chooses one
]

# Oceania die: all 6 base faces + 2 nectar choice faces = 8 faces
OCEANIA_DICE_FACES: list[FoodType | tuple[FoodType, FoodType]] = [
    FoodType.INVERTEBRATE,
    FoodType.SEED,
    FoodType.FISH,
    FoodType.FRUIT,
    FoodType.RODENT,
    (FoodType.INVERTEBRATE, FoodType.SEED),  # Same as base
    (FoodType.NECTAR, FoodType.FRUIT),       # Player chooses nectar or berry
    (FoodType.NECTAR, FoodType.SEED),        # Player chooses nectar or grain
]

DICE_FACES_BY_BOARD = {
    BoardType.BASE: BASE_DICE_FACES,
    BoardType.OCEANIA: OCEANIA_DICE_FACES,
}

NUM_DICE = 5


def _roll_die(faces: list[FoodType | tuple[FoodType, FoodType]]) -> FoodType | tuple[FoodType, FoodType]:
    return random.choice(faces)


@dataclass
class Birdfeeder:
    """The shared birdfeeder with 5 dice.

    Each die shows a food type (or a choice of two types).
    Players take dice to gain food. Reroll when empty or all show same face.
    """
    dice: list[FoodType | tuple[FoodType, FoodType]] = field(default_factory=list)
    board_type: BoardType = BoardType.BASE

    def __post_init__(self):
        if not self.dice:
            self.reroll()

    @property
    def dice_faces(self) -> list[FoodType | tuple[FoodType, FoodType]]:
        return DICE_FACES_BY_BOARD.get(self.board_type, BASE_DICE_FACES)

    def reroll(self) -> None:
        """Roll all 5 dice using the correct faces for the board type."""
        faces = self.dice_faces
        self.dice = [_roll_die(faces) for _ in range(NUM_DICE)]

    @property
    def count(self) -> int:
        return len(self.dice)

    @property
    def is_empty(self) -> bool:
        return len(self.dice) == 0

    def available_food_types(self) -> set[FoodType]:
        """All distinct food types currently available (resolving choice dice)."""
        types = set()
        for die in self.dice:
            if isinstance(die, tuple):
                types.update(die)
            else:
                types.add(die)
        return types

    def all_same_face(self) -> bool:
        """Check if all remaining dice show the same face.

        Choice dice are considered different from single-food dice,
        so this only triggers if literally identical.
        """
        if len(self.dice) <= 1:
            return len(self.dice) == 1
        return all(d == self.dice[0] for d in self.dice)

    def should_reroll(self) -> bool:
        """Reroll if empty or all remaining dice show same face."""
        return self.is_empty or self.all_same_face()

    def take_food(self, food_type: FoodType) -> bool:
        """Remove one die showing the given food type.

        For choice dice, either option can be taken.
        Returns True if a die was successfully taken, False otherwise.
        """
        # Prefer exact match first
        for i, die in enumerate(self.dice):
            if die == food_type:
                self.dice.pop(i)
                return True

        # Check choice dice
        for i, die in enumerate(self.dice):
            if isinstance(die, tuple) and food_type in die:
                self.dice.pop(i)
                return True

        return False

    def can_take(self, food_type: FoodType) -> bool:
        """Check if a specific food type can be taken."""
        for die in self.dice:
            if die == food_type:
                return True
            if isinstance(die, tuple) and food_type in die:
                return True
        return False

    def take_any(self) -> FoodType | tuple[FoodType, FoodType] | None:
        """Take any available die (for automated play)."""
        if self.dice:
            return self.dice.pop(0)
        return None

    def set_dice(self, faces: list[FoodType | tuple[FoodType, FoodType]]) -> None:
        """Manually set dice faces (for state input from UI)."""
        self.dice = list(faces)
