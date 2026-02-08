"""Cache food power templates — covers ~18 birds."""

from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class CacheFoodFromSupply(PowerEffect):
    """Cache food from the general supply onto this bird.

    Examples:
    - "Cache 1 [seed] from the supply on this bird."
    - "Cache 1 [fish] from the supply on this bird."
    """

    def __init__(self, food_type: FoodType, count: int = 1):
        self.food_type = food_type
        self.count = count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(self.food_type, self.count)
        return PowerResult(
            food_cached={self.food_type: self.count},
            description=f"Cached {self.count} {self.food_type.value}",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        return f"cache {self.count} {self.food_type.value} from supply onto {ctx.bird.name} ({self.count} pt)"

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.count * 0.9  # Each cached food = 1 point


class CacheFoodFromFeeder(PowerEffect):
    """Take food from feeder and cache directly on this bird.

    Example: "Gain all [fish] from the birdfeeder and cache them on this bird."
    """

    def __init__(self, food_type: FoodType):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        count = 0
        while feeder.take_food(self.food_type):
            count += 1

        if count > 0:
            slot.cache_food(self.food_type, count)

        return PowerResult(
            food_cached={self.food_type: count} if count else {},
            description=f"Cached {count} {self.food_type.value} from feeder",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        feeder = ctx.game_state.birdfeeder
        available = sum(1 for d in feeder.dice
                        if d == self.food_type or (isinstance(d, tuple) and self.food_type in d))
        return f"cache all {self.food_type.value} from feeder onto {ctx.bird.name} ({available} available)"

    def estimate_value(self, ctx: PowerContext) -> float:
        # Average ~1 matching die in feeder
        return 0.8


class TradeFood(PowerEffect):
    """Discard food to gain/cache different food.

    Example: "Discard 1 [egg] from this bird to cache 2 [seed] from the supply."
    Example: "Discard 1 food to gain 1 different food from the supply."
    """

    def __init__(self, discard_type: FoodType | None = None, discard_count: int = 1,
                 gain_type: FoodType | None = None, gain_count: int = 1,
                 discard_egg: bool = False, cache: bool = False):
        self.discard_type = discard_type
        self.discard_count = discard_count
        self.gain_type = gain_type
        self.gain_count = gain_count
        self.discard_egg = discard_egg
        self.cache = cache

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        # Pay cost
        if self.discard_egg:
            if slot.eggs < self.discard_count:
                return PowerResult(executed=False, description="Not enough eggs")
            slot.eggs -= self.discard_count
        elif self.discard_type:
            if not ctx.player.food_supply.spend(self.discard_type, self.discard_count):
                return PowerResult(executed=False, description="Not enough food")

        # Gain reward
        gained = {}
        cached = {}
        if self.gain_type:
            if self.cache:
                slot.cache_food(self.gain_type, self.gain_count)
                cached[self.gain_type] = self.gain_count
            else:
                ctx.player.food_supply.add(self.gain_type, self.gain_count)
                gained[self.gain_type] = self.gain_count

        return PowerResult(food_gained=gained, food_cached=cached,
                           description=f"Traded for {gained or cached}")

    def describe_activation(self, ctx: PowerContext) -> str:
        cost = f"{self.discard_count} egg" if self.discard_egg else f"{self.discard_count} {self.discard_type.value if self.discard_type else 'food'}"
        gain = f"{self.gain_count} {self.gain_type.value if self.gain_type else 'food'}"
        verb = "cache" if self.cache else "gain"
        return f"discard {cost} from {ctx.bird.name}, {verb} {gain}"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if self.discard_egg and slot.eggs < self.discard_count:
            return f"not enough eggs on {ctx.bird.name} ({slot.eggs})"
        if self.discard_type and not ctx.player.food_supply.has(self.discard_type, self.discard_count):
            return f"no {self.discard_type.value} to discard"
        return None

    def estimate_value(self, ctx: PowerContext) -> float:
        return (self.gain_count - self.discard_count) * 0.5 + 0.3
