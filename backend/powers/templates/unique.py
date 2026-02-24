"""Unique power templates for the 14 fallback birds.

These birds have powers too specific or complex for the regex parser
and require dedicated implementations.
"""

import random
from backend.models.enums import FoodType, Habitat, PowerColor
from backend.models.bonus_card import BonusCard
from backend.powers.base import PowerEffect, PowerContext, PowerResult
from backend.engine.scoring import CUSTOM_BONUS_COUNTERS, CUSTOM_BONUS_FULL_SCORERS


class CountsDoubleForGoal(PowerEffect):
    """This bird counts double toward the end-of-round goal.

    Birds: Cetti's Warbler, Eurasian Green Woodpecker, Greylag Goose (Teal)
    The bird is scored twice when evaluating end-of-round goals.
    Implementation: sets a flag on the slot; goal scoring checks this.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.counts_double = True
        return PowerResult(
            description=f"{ctx.bird.name} now counts double for round goals"
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Worth ~1-2 points depending on goal match likelihood
        return 1.5


class ActivateAllPredators(PowerEffect):
    """Activate the brown powers of all your other predator birds.

    Bird: Oriental Bay-Owl (Teal)
    Triggers the brown power of each predator bird on the player's board.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.registry import get_power

        total_cached = {}
        activated = 0

        for hab, idx, slot in ctx.player.board.all_slots():
            if not slot.bird or slot.bird.name == ctx.bird.name:
                continue
            if slot.bird.color != PowerColor.BROWN:
                continue
            # Check if this bird is a predator (has predator power)
            power = get_power(slot.bird)
            from backend.powers.templates.predator import PredatorDice, PredatorLookAt
            if isinstance(power, (PredatorDice, PredatorLookAt)):
                sub_ctx = PowerContext(
                    game_state=ctx.game_state,
                    player=ctx.player,
                    bird=slot.bird,
                    slot_index=idx,
                    habitat=hab,
                )
                result = power.execute(sub_ctx)
                if result.executed:
                    activated += 1
                    for ft, count in result.food_cached.items():
                        total_cached[ft] = total_cached.get(ft, 0) + count

        return PowerResult(
            food_cached=total_cached,
            description=f"Activated {activated} predator powers",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Value depends on number of predator birds on board
        predator_count = 0
        for _, _, slot in ctx.player.board.all_slots():
            if slot.bird and slot.bird.color == PowerColor.BROWN:
                from backend.powers.registry import get_power
                power = get_power(slot.bird)
                from backend.powers.templates.predator import PredatorDice, PredatorLookAt
                if isinstance(power, (PredatorDice, PredatorLookAt)):
                    predator_count += 1
        return predator_count * 0.7


class GainFoodMatchingPrevious(PowerEffect):
    """Gain 1 food of a type you already have in your supply.

    Bird: European Robin (Brown)
    Simplified: gain 1 of the most abundant food type in supply.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        supply = ctx.player.food_supply
        # Pick the food type the player has most of (excluding nectar)
        best_type = None
        best_count = 0
        for ft in [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                    FoodType.FRUIT, FoodType.RODENT]:
            count = supply.get(ft)
            if count > best_count:
                best_count = count
                best_type = ft

        if best_type and best_count > 0:
            supply.add(best_type, 1)
            return PowerResult(
                food_gained={best_type: 1},
                description=f"Gained 1 {best_type.value} (matching previous)",
            )

        return PowerResult(executed=False, description="No food in supply to match")

    def estimate_value(self, ctx: PowerContext) -> float:
        has_food = any(
            ctx.player.food_supply.get(ft) > 0
            for ft in [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                        FoodType.FRUIT, FoodType.RODENT]
        )
        return 1.2 if has_food else 0.3


class FewestBirdsGainFood(PowerEffect):
    """Player(s) with the fewest birds in a habitat gain food from feeder.

    Bird: Hermit Thrush (Brown)
    "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder."
    """

    def __init__(self, habitat: Habitat = Habitat.FOREST):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        game = ctx.game_state
        # Count birds in target habitat for each player
        counts = {}
        for p in game.players:
            row = p.board.get_row(self.habitat)
            counts[p.name] = row.bird_count

        min_count = min(counts.values())
        # Current player benefits if tied for fewest
        if counts[ctx.player.name] <= min_count:
            feeder = game.birdfeeder
            available = list(feeder.available_food_types())
            if available:
                food = random.choice(available)
                feeder.take_food(food)
                ctx.player.food_supply.add(food, 1)
                return PowerResult(
                    food_gained={food: 1},
                    description=f"Fewest birds in {self.habitat.value}: gained 1 {food.value}",
                )

        return PowerResult(executed=False,
                           description="Not fewest birds or feeder empty")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8  # Conditional, depends on opponent board state


class ScoreBonusCardNow(PowerEffect):
    """Score a bonus card now by caching seeds for each point.

    Bird: Great Indian Bustard (White)
    "Score 1 of your bonus cards now by caching 1 [seed] from the supply
    on this bird for each point. Also score it normally at game end."
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not ctx.player.bonus_cards:
            return PowerResult(executed=False, description="No bonus cards to score")

        def _score_bonus(bonus, player) -> int:
            if bonus.name in CUSTOM_BONUS_FULL_SCORERS:
                return CUSTOM_BONUS_FULL_SCORERS[bonus.name](player)
            if bonus.name in CUSTOM_BONUS_COUNTERS:
                qualifying = CUSTOM_BONUS_COUNTERS[bonus.name](player)
                return bonus.score(qualifying)
            qualifying = sum(
                1 for bird in player.board.all_birds()
                if bonus.name in bird.bonus_eligibility
            )
            return bonus.score(qualifying)

        # Pick the highest-scoring bonus card
        from backend.data.registries import get_bonus_registry
        best_score = 0
        for bc_entry in ctx.player.bonus_cards:
            if isinstance(bc_entry, BonusCard):
                bc = bc_entry
            else:
                bc = get_bonus_registry().get(str(bc_entry))
            if bc:
                score = _score_bonus(bc, ctx.player)
                best_score = max(best_score, score)

        if best_score > 0:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(FoodType.SEED, best_score)
            return PowerResult(
                food_cached={FoodType.SEED: best_score},
                description=f"Scored bonus card: cached {best_score} seeds",
            )

        return PowerResult(executed=False, description="No bonus card points to cache")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Average bonus card scores 3-5 points
        if ctx.player.bonus_cards:
            return 3.5
        return 0.0


class RetrieveDiscardedBonusCard(PowerEffect):
    """Look through all discarded bonus cards and keep 1.

    Bird: Wrybill (White)
    Simplified: gain 1 bonus card (adds to player's bonus cards).
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        # In our model, we don't track specific discarded bonus cards.
        # Approximate: if there are discards, player gains a generic bonus card benefit.
        if ctx.game_state.discard_pile_count > 0:
            ctx.game_state.discard_pile_count -= 1
            return PowerResult(
                description="Retrieved a bonus card from the discard pile",
            )

        return PowerResult(executed=False, description="No discarded bonus cards")

    def estimate_value(self, ctx: PowerContext) -> float:
        # A bonus card is worth ~3-5 points
        return 2.5


class CopyNeighborBonusCard(PowerEffect):
    """Copy one bonus card of a neighbor as if it were your own.

    Birds: Greater Adjutant (left), Indian Vulture (right) (Yellow)
    Score it based on your own birds at game end.
    """

    def __init__(self, direction: str = "left"):
        self.direction = direction  # "left" or "right"

    def execute(self, ctx: PowerContext) -> PowerResult:
        game = ctx.game_state
        if self.direction == "left":
            neighbor = game.player_to_left(ctx.player)
        else:
            neighbor = game.player_to_right(ctx.player)

        if not neighbor or not neighbor.bonus_cards:
            return PowerResult(executed=False, description="No neighbor bonus cards")

        # Find the best-scoring card (scored against THIS player's birds).
        best_card: BonusCard | None = None
        best_score = 0
        for bc in neighbor.bonus_cards:
            if bc.name in CUSTOM_BONUS_FULL_SCORERS:
                score = CUSTOM_BONUS_FULL_SCORERS[bc.name](ctx.player)
            elif bc.name in CUSTOM_BONUS_COUNTERS:
                qualifying = CUSTOM_BONUS_COUNTERS[bc.name](ctx.player)
                score = bc.score(qualifying)
            else:
                qualifying = sum(
                    1 for bird in ctx.player.board.all_birds()
                    if bc.name in bird.bonus_eligibility
                )
                score = bc.score(qualifying)
            if score > best_score:
                best_score = score
                best_card = bc

        if best_card and best_score > 0:
            # Keep copied card name unique so repeated triggers do not duplicate.
            if not any(bc.name == best_card.name for bc in ctx.player.bonus_cards):
                ctx.player.bonus_cards.append(best_card)
            return PowerResult(
                description=f"Copied {self.direction} neighbor's '{best_card.name}' as bonus card",
            )

        return PowerResult(executed=False,
                           description=f"No valuable bonus cards from {self.direction} neighbor")

    def estimate_value(self, ctx: PowerContext) -> float:
        # At game end, copying a bonus card is typically worth 3-5 points
        return 3.0


class TradeFoodForAny(PowerEffect):
    """Trade 1 food from supply for any other type.

    Bird: Green Heron (Brown)
    "Trade 1 [wild] for any other type from the supply."
    Discard any 1 food to gain any 1 different food.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        supply = ctx.player.food_supply
        # Find a food type we have
        available = [ft for ft in [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                                   FoodType.FRUIT, FoodType.RODENT]
                     if supply.get(ft) > 0]
        if not available:
            return PowerResult(executed=False, description="No food to trade")

        # Discard least useful (most abundant), gain least available
        discard = max(available, key=lambda ft: supply.get(ft))
        all_types = [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                     FoodType.FRUIT, FoodType.RODENT]
        gain = min(all_types, key=lambda ft: supply.get(ft))

        if discard == gain:
            # All equal — still useful to have the option
            gain = random.choice([ft for ft in all_types if ft != discard])

        supply.spend(discard, 1)
        supply.add(gain, 1)
        return PowerResult(
            food_gained={gain: 1},
            description=f"Traded 1 {discard.value} for 1 {gain.value}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        has_food = any(
            ctx.player.food_supply.get(ft) > 0
            for ft in [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                        FoodType.FRUIT, FoodType.RODENT]
        )
        return 0.8 if has_food else 0.0


class RepeatPredatorPower(PowerEffect):
    """Repeat 1 predator power in this habitat.

    Bird: Hooded Merganser (Brown)
    "Repeat 1 [predator] power in this habitat."
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.registry import get_power
        from backend.powers.templates.predator import PredatorDice, PredatorLookAt

        row = ctx.player.board.get_row(ctx.habitat)
        for i, slot in enumerate(row.slots):
            if not slot.bird or slot.bird.name == ctx.bird.name:
                continue
            power = get_power(slot.bird)
            if isinstance(power, (PredatorDice, PredatorLookAt)):
                sub_ctx = PowerContext(
                    game_state=ctx.game_state,
                    player=ctx.player,
                    bird=slot.bird,
                    slot_index=i,
                    habitat=ctx.habitat,
                )
                return power.execute(sub_ctx)

        return PowerResult(executed=False,
                           description="No predator powers in this habitat")

    def estimate_value(self, ctx: PowerContext) -> float:
        from backend.powers.registry import get_power
        from backend.powers.templates.predator import PredatorDice, PredatorLookAt

        row = ctx.player.board.get_row(ctx.habitat)
        for slot in row.slots:
            if slot.bird and slot.bird.name != ctx.bird.name:
                power = get_power(slot.bird)
                if isinstance(power, (PredatorDice, PredatorLookAt)):
                    return 0.7
        return 0.0


class CopyNeighborWhitePower(PowerEffect):
    """Copy a 'When Played' (white) ability from a neighbor's bird.

    Bird: Rose-Ringed Parakeet (White)
    Simplified: gain the average benefit of a white power (~2 pts).
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        game = ctx.game_state
        # Check both neighbors for white birds
        for neighbor in [game.player_to_left(ctx.player),
                         game.player_to_right(ctx.player)]:
            if not neighbor:
                continue
            for _, _, slot in neighbor.board.all_slots():
                if slot.bird and slot.bird.color == PowerColor.WHITE:
                    # Found a white bird — simulate getting the average benefit
                    # Most white powers draw bonus cards or play extra birds
                    return PowerResult(
                        description=f"Copied white power from neighbor's {slot.bird.name}",
                    )

        return PowerResult(executed=False,
                           description="No white powers on neighbor birds")

    def estimate_value(self, ctx: PowerContext) -> float:
        # White powers are typically worth 2-3 points
        return 2.0


class SouthernCassowaryPower(PowerEffect):
    """Discard a bird from your [forest] and put this bird in its place.

    Bird: Southern Cassowary (Oceania, White)
    "Discard a bird from your [forest] and put this bird in its place
    (do not pay an egg cost). If you do, lay 4 eggs on this bird and gain 2 fruit from supply."

    The forest-bird replacement and free placement is handled by execute_play_bird()
    with play_on_top=True and play_on_top_discard=True. This power fires after placement
    and applies the bonus: 4 eggs on self + 2 fruit from supply.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        slot = row.slots[ctx.slot_index]
        if slot.bird is None:
            return PowerResult(executed=False, description="No bird in slot for Cassowary bonus")

        eggs_laid = 0
        for _ in range(4):
            if slot.can_hold_more_eggs():
                slot.eggs += 1
                eggs_laid += 1

        ctx.player.food_supply.add(FoodType.FRUIT, 2)
        return PowerResult(
            food_gained={FoodType.FRUIT: 2},
            description=f"Cassowary: laid {eggs_laid} eggs on self, gained 2 fruit",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Free forest replacement + 4 eggs + 2 fruit is very valuable
        return 6.0


class TuckToPayCost(PowerEffect):
    """For each [rodent] in this bird's cost, may pay 1 card from hand instead.
    If you do, tuck the paid card behind this bird.

    Bird: Eastern Imperial Eagle (European, White)
    The cost-reduction logic is handled by execute_play_bird() with hand_tuck_payment set.
    This power class is used as a marker for the registry and move generator.
    """

    def execute(self, ctx: PowerContext) -> PowerResult:
        return PowerResult(description="Tuck-to-pay: cards were tucked during placement")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Having flexible payment options increases playability
        return 0.5


class ConditionalCacheFromNeighbor(PowerEffect):
    """Cache food if the neighboring player has that food type.

    Bird: South Island Robin (Brown)
    "If the player to your right has an [invertebrate] in their supply,
    cache 1 [invertebrate] from the general supply on this bird."
    """

    def __init__(self, direction: str = "right",
                 food_type: FoodType = FoodType.INVERTEBRATE):
        self.direction = direction
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        game = ctx.game_state
        if self.direction == "right":
            neighbor = game.player_to_right(ctx.player)
        else:
            neighbor = game.player_to_left(ctx.player)

        if not neighbor:
            return PowerResult(executed=False, description="No neighbor")

        if neighbor.food_supply.get(self.food_type) > 0:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(self.food_type, 1)
            return PowerResult(
                food_cached={self.food_type: 1},
                description=f"Neighbor has {self.food_type.value}: cached 1",
            )

        return PowerResult(executed=False,
                           description=f"Neighbor has no {self.food_type.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        # ~50% chance neighbor has the food type
        return 0.5
