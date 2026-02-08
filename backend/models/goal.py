from dataclasses import dataclass
from .enums import GameSet


@dataclass(frozen=True)
class Goal:
    """An end-of-round goal tile.

    scoring represents points by placement tier (4th, 3rd, 2nd, 1st)
    for competitive mode. Can be negative (European-style).
    """
    description: str
    game_set: GameSet
    scoring: tuple[float, float, float, float]  # (4th, 3rd, 2nd, 1st)
    reverse_description: str  # The other side of this goal tile

    def score_for_placement(self, placement: int) -> float:
        """Get score for a placement (1=first, 2=second, etc.)."""
        if placement < 1 or placement > 4:
            return 0.0
        # Index: 1st place -> index 3, 4th place -> index 0
        return self.scoring[4 - placement]

    def __str__(self) -> str:
        return f"{self.description} ({self.game_set.value})"


# Sentinel for rounds with no goal — subsequent rounds get +1 action
NO_GOAL = Goal(
    description="No Goal",
    game_set=GameSet.CORE,
    scoring=(0, 0, 0, 0),
    reverse_description="No Goal",
)
