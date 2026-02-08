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

    def describe_activation(self, ctx: PowerContext) -> str:
        from backend.models.enums import PowerColor
        row = ctx.player.board.get_row(ctx.habitat)
        brown_names = [
            s.bird.name for s in row.slots
            if s.bird and s.bird.color == PowerColor.BROWN
            and s.bird.name != ctx.bird.name
        ]
        if brown_names:
            return f"repeat a brown power in {ctx.habitat.value} (options: {', '.join(brown_names)})"
        return f"repeat a brown power in {ctx.habitat.value} (none available)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        from backend.models.enums import PowerColor
        row = ctx.player.board.get_row(ctx.habitat)
        brown_count = sum(
            1 for s in row.slots
            if s.bird and s.bird.color == PowerColor.BROWN
            and s.bird.name != ctx.bird.name
        )
        if brown_count == 0:
            return "no other brown birds in this habitat"
        return None

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


class CopyNeighborBrownPower(PowerEffect):
    """Copy a brown power from a neighbor's habitat.

    Example: "Copy a brown power on one bird in the [forest] of the
    player to your right."
    """

    def __init__(self, direction: str = "right", target_habitat: Habitat = Habitat.FOREST):
        self.direction = direction
        self.target_habitat = target_habitat

    def _get_neighbor(self, ctx: PowerContext):
        """Get the neighbor player based on direction."""
        gs = ctx.game_state
        if self.direction == "right":
            return gs.player_to_right(ctx.player)
        return gs.player_to_left(ctx.player)

    def _neighbor_brown_birds(self, ctx: PowerContext) -> list:
        """Get all brown-powered birds in the neighbor's target habitat."""
        from backend.models.enums import PowerColor
        neighbor = self._get_neighbor(ctx)
        if not neighbor:
            return []
        row = neighbor.board.get_row(self.target_habitat)
        return [
            s.bird for s in row.slots
            if s.bird and s.bird.color == PowerColor.BROWN
        ]

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.registry import get_power
        from backend.powers.base import NoPower
        neighbor = self._get_neighbor(ctx)
        if not neighbor:
            return PowerResult(executed=False, description="No neighbor")

        brown_birds = self._neighbor_brown_birds(ctx)
        if not brown_birds:
            return PowerResult(
                executed=False,
                description=f"No brown birds in {neighbor.name}'s {self.target_habitat.value}",
            )

        # Execute the best brown power (by estimate_value)
        best_bird = None
        best_val = -1.0
        for bird in brown_birds:
            power = get_power(bird)
            if isinstance(power, NoPower):
                continue
            # Create a context as if we are the neighbor's bird
            neighbor_ctx = PowerContext(
                game_state=ctx.game_state, player=ctx.player, bird=bird,
                slot_index=0, habitat=self.target_habitat,
            )
            val = power.estimate_value(neighbor_ctx)
            if val > best_val:
                best_val = val
                best_bird = bird

        if not best_bird:
            return PowerResult(executed=False, description="No copyable brown power")

        # Execute the copied power on behalf of this player
        power = get_power(best_bird)
        copy_ctx = PowerContext(
            game_state=ctx.game_state, player=ctx.player, bird=best_bird,
            slot_index=ctx.slot_index, habitat=ctx.habitat,
        )
        result = power.execute(copy_ctx)
        result.description = f"Copied {best_bird.name}'s brown power from {neighbor.name}"
        return result

    def estimate_value(self, ctx: PowerContext) -> float:
        from backend.powers.registry import get_power
        from backend.powers.base import NoPower
        brown_birds = self._neighbor_brown_birds(ctx)
        if not brown_birds:
            return 0.0
        best_val = 0.0
        for bird in brown_birds:
            power = get_power(bird)
            if isinstance(power, NoPower):
                continue
            neighbor_ctx = PowerContext(
                game_state=ctx.game_state, player=ctx.player, bird=bird,
                slot_index=0, habitat=self.target_habitat,
            )
            best_val = max(best_val, power.estimate_value(neighbor_ctx))
        return best_val

    def describe_activation(self, ctx: PowerContext) -> str:
        target = self.best_copy_target(ctx)
        neighbor = self._get_neighbor(ctx)
        neighbor_name = neighbor.name if neighbor else "opponent"
        if target:
            from backend.powers.registry import get_power
            bird_obj = next((b for b in self._neighbor_brown_birds(ctx) if b.name == target), None)
            if bird_obj:
                power = get_power(bird_obj)
                copy_ctx = PowerContext(
                    game_state=ctx.game_state, player=ctx.player, bird=bird_obj,
                    slot_index=0, habitat=self.target_habitat,
                )
                power_desc = power.describe_activation(copy_ctx)
                return f"copy {target} from {neighbor_name}'s {self.target_habitat.value}: {power_desc}"
            return f"copy {target} from {neighbor_name}'s {self.target_habitat.value}"
        return f"copy brown power from {neighbor_name}'s {self.target_habitat.value} (none available)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        neighbor = self._get_neighbor(ctx)
        if not neighbor:
            return "no neighbor"
        brown_birds = self._neighbor_brown_birds(ctx)
        if not brown_birds:
            return f"no brown birds in {neighbor.name}'s {self.target_habitat.value}"
        return None

    def best_copy_target(self, ctx: PowerContext) -> str | None:
        """Return the name of the best bird to copy for solver advice."""
        from backend.powers.registry import get_power
        from backend.powers.base import NoPower
        brown_birds = self._neighbor_brown_birds(ctx)
        if not brown_birds:
            return None
        best_bird = None
        best_val = -1.0
        for bird in brown_birds:
            power = get_power(bird)
            if isinstance(power, NoPower):
                continue
            neighbor_ctx = PowerContext(
                game_state=ctx.game_state, player=ctx.player, bird=bird,
                slot_index=0, habitat=self.target_habitat,
            )
            val = power.estimate_value(neighbor_ctx)
            if val > best_val:
                best_val = val
                best_bird = bird
        return best_bird.name if best_bird else None


class MoveBird(PowerEffect):
    """Move a bird to a different habitat.

    Example: "Move 1 [bird] from one habitat to another."
    """

    def __init__(self):
        pass

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Bird movement is a complex choice — handled by solver
        return PowerResult(description="May move a bird between habitats")

    def describe_activation(self, ctx: PowerContext) -> str:
        return "move 1 bird from one habitat to another"

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

    def describe_activation(self, ctx: PowerContext) -> str:
        parts = [f"discard {self.egg_cost} egg from any bird"]
        if self.food_gain and self.food_type:
            parts.append(f"gain {self.food_gain} {self.food_type.value}")
        if self.card_gain:
            parts.append(f"draw {self.card_gain} card{'s' if self.card_gain > 1 else ''}")
        return ", then ".join(parts)

    def skip_reason(self, ctx: PowerContext) -> str | None:
        total_eggs = sum(
            s.eggs for row in ctx.player.board.all_rows()
            for s in row.slots if s.bird
        )
        if total_eggs < self.egg_cost:
            return "no eggs on any bird to discard"
        return None

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

    def describe_activation(self, ctx: PowerContext) -> str:
        return f"tuck {self.count} from deck behind {ctx.bird.name} ({self.count} pt)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if ctx.game_state.deck_remaining <= 0:
            return "deck is empty"
        return None

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
