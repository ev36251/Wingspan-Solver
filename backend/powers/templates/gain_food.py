"""Food gain power templates — covers ~90 birds."""

from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class GainFoodFromSupply(PowerEffect):
    """Gain specific food type(s) from the general supply.

    Examples:
    - "Gain 1 [invertebrate] from the supply."
    - "All players gain 1 [seed] from the supply."
    """

    def __init__(self, food_types: list[FoodType], count: int = 1,
                 all_players: bool = False, you_also: bool = False):
        self.food_types = food_types
        self.count = count
        self.all_players = all_players
        self.you_also = you_also  # If all_players, active player gains extra

    def execute(self, ctx: PowerContext) -> PowerResult:
        gained = {}
        if self.all_players:
            for p in ctx.game_state.players:
                for ft in self.food_types:
                    p.food_supply.add(ft, self.count)
            if self.you_also:
                for ft in self.food_types:
                    ctx.player.food_supply.add(ft, 1)
            for ft in self.food_types:
                gained[ft] = self.count
        else:
            for ft in self.food_types:
                ctx.player.food_supply.add(ft, self.count)
                gained[ft] = self.count

        return PowerResult(food_gained=gained,
                           description=f"Gained {gained} from supply")

    def describe_activation(self, ctx: PowerContext) -> str:
        food_names = " + ".join(f"{self.count} {ft.value}" for ft in self.food_types)
        prefix = "ALL PLAYERS: " if self.all_players else ""
        return f"{prefix}gain {food_names} from supply"

    def estimate_value(self, ctx: PowerContext) -> float:
        base = self.count * len(self.food_types) * 0.5
        if self.all_players:
            base *= 0.7  # Helping opponents reduces net value
        return min(base, 3.0)


class GainFoodFromFeeder(PowerEffect):
    """Gain food from the birdfeeder (specific type or any).

    Examples:
    - "Gain 1 [seed] from the birdfeeder, if available."
    - "Gain 1 [fish] from the birdfeeder, if available. You may cache it on this bird."
    """

    def __init__(self, food_types: list[FoodType] | None = None, count: int = 1,
                 may_cache: bool = False):
        self.food_types = food_types  # None = any type from feeder
        self.count = count
        self.may_cache = may_cache

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        gained = {}

        if feeder.should_reroll():
            feeder.reroll()

        for _ in range(self.count):
            taken = False
            if self.food_types:
                for ft in self.food_types:
                    if feeder.take_food(ft):
                        ctx.player.food_supply.add(ft)
                        gained[ft] = gained.get(ft, 0) + 1
                        taken = True
                        break
            else:
                # Take any available
                die = feeder.take_any()
                if die is not None:
                    ft = die if isinstance(die, FoodType) else die[0]
                    ctx.player.food_supply.add(ft)
                    gained[ft] = gained.get(ft, 0) + 1
                    taken = True

            if not taken:
                break

            if feeder.should_reroll():
                feeder.reroll()

        cached = {}
        if self.may_cache and gained:
            # Solver: auto-cache for simplicity (better heuristic value)
            for ft, cnt in gained.items():
                from backend.models.board import BirdSlot
                slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
                slot.cache_food(ft, cnt)
                ctx.player.food_supply.spend(ft, cnt)
                cached[ft] = cnt

        return PowerResult(food_gained=gained, food_cached=cached,
                           description=f"Gained {gained} from feeder")

    def describe_activation(self, ctx: PowerContext) -> str:
        if self.food_types:
            food_names = "/".join(ft.value for ft in self.food_types)
            base = f"gain {self.count} {food_names} from feeder (if available)"
        else:
            base = f"gain {self.count} food from feeder"
        if self.may_cache:
            base += f", may cache onto {ctx.bird.name}"
        return base

    def estimate_value(self, ctx: PowerContext) -> float:
        base = self.count * 0.6
        if self.may_cache:
            base += 0.3  # Cached food = extra point
        return base


class GainFoodFromSupplyOrCache(PowerEffect):
    """Gain food from supply and optionally cache it.

    Examples:
    - "Gain 1 [invertebrate] from the supply and cache it on this bird."
    """

    def __init__(self, food_types: list[FoodType], count: int = 1,
                 auto_cache: bool = True):
        self.food_types = food_types
        self.count = count
        self.auto_cache = auto_cache

    def execute(self, ctx: PowerContext) -> PowerResult:
        gained = {}
        cached = {}
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        for ft in self.food_types:
            if self.auto_cache:
                slot.cache_food(ft, self.count)
                cached[ft] = self.count
            else:
                ctx.player.food_supply.add(ft, self.count)
                gained[ft] = self.count

        return PowerResult(food_gained=gained, food_cached=cached,
                           description=f"Gained and cached {cached}")

    def describe_activation(self, ctx: PowerContext) -> str:
        food_names = " + ".join(f"{self.count} {ft.value}" for ft in self.food_types)
        return f"gain and cache {food_names} onto {ctx.bird.name}"

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.count * len(self.food_types) * 0.8


class ResetFeederGainFood(PowerEffect):
    """Reset the birdfeeder and gain food of a specific type.

    Example: "Reset the birdfeeder. If you do, gain all [fish] in the birdfeeder."
    """

    def __init__(self, food_type: FoodType | None = None, cache: bool = False):
        self.food_type = food_type
        self.cache = cache

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()

        gained = {}
        if self.food_type:
            # Gain all of that type
            count = 0
            while feeder.take_food(self.food_type):
                count += 1
            if count > 0:
                if self.cache:
                    slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
                    slot.cache_food(self.food_type, count)
                else:
                    ctx.player.food_supply.add(self.food_type, count)
                gained[self.food_type] = count

        return PowerResult(food_gained=gained,
                           description=f"Reset feeder, gained {gained}")

    def describe_activation(self, ctx: PowerContext) -> str:
        base = "reset birdfeeder"
        if self.food_type:
            base += f", gain all {self.food_type.value}"
            if self.cache:
                base += f", cache onto {ctx.bird.name}"
        return base

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5  # Depends heavily on dice rolls
