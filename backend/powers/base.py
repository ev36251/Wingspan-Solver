"""Base class and context for bird power effects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from backend.models.enums import ActionType, FoodType, Habitat
from backend.models.bird import Bird


@dataclass
class PowerContext:
    """All contextual info a power needs to resolve."""
    game_state: "GameState"  # Forward ref to avoid circular import
    player: "Player"
    bird: Bird
    slot_index: int
    habitat: Habitat
    # For pink powers: the triggering player and action
    trigger_player: "Player | None" = None
    trigger_action: ActionType | None = None


@dataclass
class PowerResult:
    """What happened when a power resolved."""
    executed: bool = True
    food_gained: dict[FoodType, int] = field(default_factory=dict)
    eggs_laid: int = 0
    cards_drawn: int = 0
    cards_tucked: int = 0
    food_cached: dict[FoodType, int] = field(default_factory=dict)
    description: str = ""


class PowerEffect(ABC):
    """Base class for all bird power implementations."""

    @abstractmethod
    def execute(self, ctx: PowerContext) -> PowerResult:
        """Execute this power, modifying game state and returning result."""
        ...

    @abstractmethod
    def estimate_value(self, ctx: PowerContext) -> float:
        """Heuristic value estimate for the solver (0.0-5.0 scale).

        Represents expected point value of one activation of this power.
        """
        ...

    def can_execute(self, ctx: PowerContext) -> bool:
        """Check if this power can be meaningfully executed right now."""
        return True


class NoPower(PowerEffect):
    """Placeholder for birds with no power (high VP birds)."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        return PowerResult(executed=False, description="No power")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.0


class FallbackPower(PowerEffect):
    """Placeholder for unimplemented powers.

    Does not modify state but provides heuristic value estimates
    based on power color for solver use.
    """

    def __init__(self, color_hint: str = "brown"):
        self.color_hint = color_hint

    def execute(self, ctx: PowerContext) -> PowerResult:
        return PowerResult(executed=False, description="Power not yet implemented")

    def estimate_value(self, ctx: PowerContext) -> float:
        estimates = {
            "white": 2.0,   # One-time but often strong
            "brown": 1.5,   # Per activation
            "pink": 0.8,    # Depends on opponents
            "teal": 1.0,    # Once per round
            "yellow": 2.5,  # End of round/game
        }
        return estimates.get(self.color_hint, 1.0)
