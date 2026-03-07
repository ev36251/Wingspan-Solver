"""Special and miscellaneous power templates."""

import random

from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult


def _draw_one_bird(ctx: PowerContext):
    """Draw one concrete bird card from deck identity model if available."""
    deck_cards = getattr(ctx.game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        ctx.game_state.deck_remaining = max(0, ctx.game_state.deck_remaining - 1)
        if ctx.game_state.deck_tracker is not None:
            ctx.game_state.deck_tracker.mark_drawn(card.name)
        return card

    if ctx.game_state.deck_remaining <= 0:
        return None
    from backend.data.registries import get_bird_registry

    pool = list(get_bird_registry().all_birds)
    if not pool:
        return None
    ctx.game_state.deck_remaining = max(0, ctx.game_state.deck_remaining - 1)
    card = random.choice(pool)
    if ctx.game_state.deck_tracker is not None:
        ctx.game_state.deck_tracker.mark_drawn(card.name)
    return card


class RepeatPower(PowerEffect):
    """Repeat another brown 'when activated' power in the same habitat.

    Card text: "Repeat a brown [when activated] power on another bird in this habitat."
    Only birds in the same row as the repeater are eligible targets.

    Example birds: Gray Catbird, Northern Mockingbird
    """

    def __init__(self):
        pass

    def _brown_candidates(self, ctx: PowerContext) -> list[tuple[int, object, object]]:
        """Return (slot_idx, slot, power) for repeatable brown birds in this habitat."""
        from backend.models.enums import PowerColor
        from backend.powers.registry import get_power
        from backend.powers.base import NoPower, FallbackPower

        row = ctx.player.board.get_row(ctx.habitat)
        candidates = []
        for i, slot in enumerate(row.slots):
            if not slot.bird:
                continue
            if slot.bird.color != PowerColor.BROWN:
                continue
            if slot.bird.name == ctx.bird.name:
                continue  # don't repeat self
            power = get_power(slot.bird)
            if isinstance(power, (NoPower, FallbackPower, RepeatPower, CopyNeighborBrownPower)):
                continue  # skip no-ops and mutual-recursion guard
            candidates.append((i, slot, power))
        return candidates

    def execute(self, ctx: PowerContext) -> PowerResult:
        candidates = self._brown_candidates(ctx)
        if not candidates:
            return PowerResult(executed=False,
                               description="No other brown birds in this habitat to repeat")

        # Pick the candidate with the highest estimated value
        best = max(
            candidates,
            key=lambda c: c[2].estimate_value(
                PowerContext(
                    game_state=ctx.game_state, player=ctx.player,
                    bird=c[1].bird, slot_index=c[0], habitat=ctx.habitat,
                )
            ),
        )
        slot_idx, slot, power = best
        sub_ctx = PowerContext(
            game_state=ctx.game_state, player=ctx.player,
            bird=slot.bird, slot_index=slot_idx, habitat=ctx.habitat,
        )
        if not power.can_execute(sub_ctx):
            return PowerResult(executed=False,
                               description=f"Cannot repeat {slot.bird.name}'s power right now")
        result = power.execute(sub_ctx)
        result.description = f"Repeated {slot.bird.name}'s power: {result.description}"
        return result

    def describe_activation(self, ctx: PowerContext) -> str:
        candidates = self._brown_candidates(ctx)
        names = [c[1].bird.name for c in candidates]
        if names:
            return f"repeat a brown power in {ctx.habitat.value} (options: {', '.join(names)})"
        return f"repeat a brown power in {ctx.habitat.value} (none available)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if not self._brown_candidates(ctx):
            return f"no other brown birds in {ctx.habitat.value}"
        return None

    def estimate_value(self, ctx: PowerContext) -> float:
        candidates = self._brown_candidates(ctx)
        if not candidates:
            return 0.0
        best_val = max(
            c[2].estimate_value(
                PowerContext(
                    game_state=ctx.game_state, player=ctx.player,
                    bird=c[1].bird, slot_index=c[0], habitat=ctx.habitat,
                )
            )
            for c in candidates
        )
        return best_val * 0.9


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
            if isinstance(power, (NoPower, CopyNeighborBrownPower)):
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
            if isinstance(power, (NoPower, CopyNeighborBrownPower, RepeatPower)):
                continue  # skip no-ops and mutual-recursion guard
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
    """Migrate this bird to another habitat when it's the rightmost in its row.

    Card text: "If this bird is to the right of all other birds in its habitat,
    move it to another habitat."

    The entire slot (bird + eggs + cached food + tucked cards) moves to the next
    available slot in the chosen destination habitat. The action cube stays in
    the original row (FAQ clarification: the cube goes where the action was taken).

    Destination choice: the habitat with the most birds that still has an empty
    slot (maximises column bonus).
    """

    def __init__(self):
        pass

    def _is_rightmost(self, ctx: PowerContext) -> bool:
        """Return True if this bird has no birds to its right in the same row."""
        row = ctx.player.board.get_row(ctx.habitat)
        for i in range(ctx.slot_index + 1, len(row.slots)):
            if row.slots[i].bird is not None:
                return False
        return True

    def _score_destination(self, ctx: PowerContext, dest_hab: Habitat) -> float:
        """Score a destination habitat by the net column-gain delta.

        Adding this bird to `dest_hab` upgrades its column by +1 bird; removing
        it from the source row downgrades the source column by -1 bird.
        We compare actual base_gain values so that e.g. going from 0→1 birds in
        an empty row (which can be a big jump on the Oceania board) is valued
        correctly, not just "more birds = better".
        """
        from backend.config import get_action_column
        board = ctx.game_state.board_type
        src_count = ctx.player.board.get_row(ctx.habitat).bird_count
        src_gain_before = get_action_column(board, ctx.habitat, src_count).base_gain
        src_gain_after = get_action_column(board, ctx.habitat, max(0, src_count - 1)).base_gain
        src_loss = src_gain_before - src_gain_after

        dst_count = ctx.player.board.get_row(dest_hab).bird_count
        dst_gain_before = get_action_column(board, dest_hab, dst_count).base_gain
        dst_gain_after = get_action_column(board, dest_hab, dst_count + 1).base_gain
        dst_gain = dst_gain_after - dst_gain_before

        return dst_gain - src_loss

    def _best_destination(self, ctx: PowerContext) -> Habitat | None:
        """Return the destination habitat that maximises the net column-gain delta."""
        best_hab = None
        best_score = float("-inf")
        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            if hab == ctx.habitat:
                continue
            row = ctx.player.board.get_row(hab)
            if row.next_empty_slot() is None:
                continue  # full — no room
            score = self._score_destination(ctx, hab)
            if score > best_score:
                best_score = score
                best_hab = hab
        return best_hab

    def can_execute(self, ctx: PowerContext) -> bool:
        if not self._is_rightmost(ctx):
            return False
        return self._best_destination(ctx) is not None

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not self._is_rightmost(ctx):
            return PowerResult(executed=False,
                               description="Bird is not rightmost in its habitat")

        dest_hab = self._best_destination(ctx)
        if dest_hab is None:
            return PowerResult(executed=False,
                               description="No available slot in another habitat")

        src_row = ctx.player.board.get_row(ctx.habitat)
        dst_row = ctx.player.board.get_row(dest_hab)
        dst_slot_idx = dst_row.next_empty_slot()
        if dst_slot_idx is None:
            return PowerResult(executed=False, description="Destination row is full")

        # Move entire slot contents: bird, eggs, cached food, tucked cards
        src_slot = src_row.slots[ctx.slot_index]
        dst_slot = dst_row.slots[dst_slot_idx]

        dst_slot.bird = src_slot.bird
        dst_slot.eggs = src_slot.eggs
        dst_slot.cached_food = dict(src_slot.cached_food)
        dst_slot.tucked_cards = src_slot.tucked_cards
        dst_slot.counts_double = src_slot.counts_double

        # Clear source slot (also unblock adjacent sideways slot if present)
        src_slot.bird = None
        src_slot.eggs = 0
        src_slot.cached_food = {}
        src_slot.tucked_cards = 0
        src_slot.counts_double = False
        if src_slot.is_sideways:
            src_slot.is_sideways = False
            next_idx = ctx.slot_index + 1
            if next_idx < len(src_row.slots):
                src_row.slots[next_idx].is_sideways_blocked = False

        return PowerResult(
            description=(
                f"Migrated {dst_slot.bird.name} from {ctx.habitat.value} "
                f"to {dest_hab.value} slot {dst_slot_idx + 1}"
            )
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        if not self._is_rightmost(ctx):
            return f"migrate {ctx.bird.name} (not rightmost — cannot activate)"
        dest = self._best_destination(ctx)
        if dest is None:
            return f"migrate {ctx.bird.name} (no available destination)"
        return f"migrate {ctx.bird.name} from {ctx.habitat.value} to {dest.value}"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if not self._is_rightmost(ctx):
            return f"{ctx.bird.name} is not the rightmost bird in {ctx.habitat.value}"
        if self._best_destination(ctx) is None:
            return "all other habitats are full"
        return None

    def estimate_value(self, ctx: PowerContext) -> float:
        if not self.can_execute(ctx):
            return 0.0
        dest = self._best_destination(ctx)
        if dest is None:
            return 0.0
        # Net column-gain delta × expected remaining uses of that row
        net_delta = self._score_destination(ctx, dest)
        # Even a neutral or slightly negative delta can be worth doing to
        # reposition for future rounds — floor at 0.2 when the move is legal.
        return max(0.5 + net_delta * 0.4, 0.2)


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
                        drawn = 0
                        for _ in range(self.card_gain):
                            card = _draw_one_bird(ctx)
                            if card is None:
                                break
                            ctx.player.hand.append(card)
                            drawn += 1
                        result.cards_drawn = drawn
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
        tucked = 0
        for _ in range(self.count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
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
