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

    def describe_activation(self, ctx: PowerContext) -> str:
        parts = ["play another bird"]
        if self.habitat_filter:
            parts[0] += f" in {self.habitat_filter.value}"
        discounts = []
        if self.food_discount:
            discounts.append(f"-{self.food_discount} food cost")
        if self.egg_discount:
            discounts.append(f"-{self.egg_discount} egg cost")
        if discounts:
            parts.append(f"({', '.join(discounts)})")
        else:
            parts.append("(pay normal cost)")
        return " ".join(parts)

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

    This is a white power. After placement, marks the primary slot as sideways
    and blocks the next slot so no other bird can be played there.

    Birds: Common Blackbird, European Roller, Grey Heron, Long-Tailed Tit
    """

    def __init__(self):
        pass

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        row.slots[ctx.slot_index].is_sideways = True
        second_idx = ctx.slot_index + 1
        if second_idx < len(row.slots):
            row.slots[second_idx].is_sideways_blocked = True
        return PowerResult(description="Bird placed sideways, covering 2 columns")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5  # Saves column space, affects action values


class PlayOnTopPower(PowerEffect):
    """When played, may replace another bird for free (no food/egg cost).

    Birds: Common Buzzard, Red Kite, Eurasian Hobby, Montagu's Harrier
    "Instead of paying any costs, you may play this bird on top of another
    bird on your player mat. Discard any eggs and food from that bird.
    It becomes a tucked card."

    The actual replacement logic is handled in execute_play_bird() when
    play_on_top=True and target_slot is set.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        return PowerResult(description="Played on top of another bird (free, covered bird tucked)")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Free placement is valuable; more occupied slots = more options
        occupied = sum(
            1 for _, _, slot in ctx.player.board.all_slots()
            if slot.bird is not None and slot.bird.name != ctx.bird.name
        )
        return 2.0 + occupied * 0.2
