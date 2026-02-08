"""Special and miscellaneous power templates."""

from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class RepeatPower(PowerEffect):
    """Copy/repeat another brown power in the same habitat.

    Example: "Repeat a brown [when activated] power on another bird
    in this habitat."
    """

    def __init__(self):
        pass

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Would need to enumerate other brown powers in the row
        # For now, estimate as average brown power value
        return PowerResult(description="Repeat another brown power (simulated)")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Average brown power is ~1.5
        from backend.models.enums import PowerColor
        row = ctx.player.board.get_row(ctx.habitat)
        brown_count = sum(
            1 for s in row.slots
            if s.bird and s.bird.color == PowerColor.BROWN
            and s.bird.name != ctx.bird.name
        )
        return 1.5 if brown_count > 0 else 0.0


class MoveBird(PowerEffect):
    """Move a bird to a different habitat.

    Example: "Move 1 [bird] from one habitat to another."
    """

    def __init__(self):
        pass

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Bird movement is a complex choice — handled by solver
        return PowerResult(description="May move a bird between habitats")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0  # Situational


class DiscardEggForBenefit(PowerEffect):
    """Discard egg(s) from a bird to gain something.

    Example: "Discard 1 [egg] from any of your birds to gain 2 [food]."
    """

    def __init__(self, egg_cost: int = 1,
                 food_gain: int = 0, food_type: FoodType | None = None,
                 card_gain: int = 0):
        self.egg_cost = egg_cost
        self.food_gain = food_gain
        self.food_type = food_type
        self.card_gain = card_gain

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Find a bird with eggs to discard from
        for row in ctx.player.board.all_rows():
            for slot in row.slots:
                if slot.bird and slot.eggs >= self.egg_cost:
                    slot.eggs -= self.egg_cost

                    result = PowerResult()
                    if self.food_gain and self.food_type:
                        ctx.player.food_supply.add(self.food_type, self.food_gain)
                        result.food_gained = {self.food_type: self.food_gain}
                    if self.card_gain:
                        result.cards_drawn = self.card_gain
                        ctx.game_state.deck_remaining -= self.card_gain
                    result.description = f"Discarded {self.egg_cost} egg for benefit"
                    return result

        return PowerResult(executed=False, description="No eggs to discard")

    def estimate_value(self, ctx: PowerContext) -> float:
        gain = self.food_gain * 0.5 + self.card_gain * 0.4
        cost = self.egg_cost * 0.8
        return max(0, gain - cost)


class FlockingPower(PowerEffect):
    """Tuck card from deck behind this bird (flocking).

    Many flocking birds: "Tuck a [card] from the deck behind this bird."
    """

    def __init__(self, count: int = 1):
        self.count = count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(self.count, max(0, ctx.game_state.deck_remaining))
        slot.tucked_cards += tucked
        ctx.game_state.deck_remaining -= tucked
        return PowerResult(cards_tucked=tucked,
                           description=f"Flocking: tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.count * 0.9


class EndOfRoundLayEggs(PowerEffect):
    """Yellow power: lay eggs at end of round on matching birds.

    Example: "At end of round, lay 1 [egg] on each of your [ground] nesting birds."
    """

    def __init__(self, count_per_bird: int = 1,
                 nest_filter: str | None = None,
                 habitat_filter: Habitat | None = None):
        self.count_per_bird = count_per_bird
        self.nest_filter = nest_filter
        self.habitat_filter = habitat_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.models.enums import NestType
        laid = 0
        nest_map = {
            "bowl": NestType.BOWL, "cavity": NestType.CAVITY,
            "ground": NestType.GROUND, "platform": NestType.PLATFORM,
        }
        nest = nest_map.get(self.nest_filter) if self.nest_filter else None

        for row in ctx.player.board.all_rows():
            if self.habitat_filter and row.habitat != self.habitat_filter:
                continue
            for slot in row.slots:
                if not slot.bird:
                    continue
                if nest and slot.bird.nest_type != nest and slot.bird.nest_type != NestType.WILD:
                    continue
                for _ in range(self.count_per_bird):
                    if slot.can_hold_more_eggs():
                        slot.eggs += 1
                        laid += 1

        return PowerResult(eggs_laid=laid,
                           description=f"End of round: laid {laid} eggs")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Estimate based on matching birds on board
        count = 0
        for row in ctx.player.board.all_rows():
            for slot in row.slots:
                if slot.bird and slot.can_hold_more_eggs():
                    count += 1
        return min(count * self.count_per_bird * 0.7, 4.0)


class EndOfRoundCacheFood(PowerEffect):
    """Yellow power: cache food from supply at end of round.

    Example: "At end of round, cache 1 [invertebrate] from the supply on this bird."
    """

    def __init__(self, food_type: FoodType, count: int = 1):
        self.food_type = food_type
        self.count = count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(self.food_type, self.count)
        return PowerResult(
            food_cached={self.food_type: self.count},
            description=f"End of round: cached {self.count} {self.food_type.value}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Triggers once per round for remaining rounds
        return self.count * 0.9
