"""Play additional bird power templates — covers ~18 white power birds."""

from backend.models.enums import Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class PlayAdditionalBird(PowerEffect):
    """When played, may play another bird (reduced cost).

    These are typically white powers. The actual bird-playing
    is handled by the engine — this template just records the
    discount/permission for the solver to evaluate.

    Examples:
    - "Play a second bird in your [wetland]. Pay its normal cost."
    - "Play a bird. You may ignore its habitat requirements."
    """

    def __init__(self, habitat_filter: Habitat | None = None,
                 food_discount: int = 0, egg_discount: int = 0,
                 ignore_habitat: bool = False):
        self.habitat_filter = habitat_filter
        self.food_discount = food_discount
        self.egg_discount = egg_discount
        self.ignore_habitat = ignore_habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        # This is a permission power — actual execution handled by engine
        # Just record that the player has an extra play available
        return PowerResult(
            description=f"May play additional bird"
                        f"{' in ' + self.habitat_filter.value if self.habitat_filter else ''}"
                        f"{' (-' + str(self.food_discount) + ' food)' if self.food_discount else ''}"
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Playing an extra bird is very valuable
        base = 3.0
        if self.food_discount:
            base += self.food_discount * 0.5
        if self.egg_discount:
            base += self.egg_discount * 0.5
        if self.habitat_filter:
            base -= 0.5  # Restriction reduces value
        return base


class PlaceBirdSideways(PowerEffect):
    """Play this bird so it covers 2 columns.

    This is a white power that affects the action cube spaces.
    """

    def __init__(self):
        pass

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Board effect handled at placement time
        return PowerResult(description="Bird placed sideways, covering 2 columns")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5  # Saves column space, affects action values
