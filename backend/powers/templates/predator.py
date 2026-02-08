"""Predator power templates — covers ~43 birds."""

import random
from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class PredatorDice(PowerEffect):
    """Roll dice not in feeder; cache food if match.

    Example: "Roll all dice not in the birdfeeder.
    If any match [rodent], cache 1 [rodent] on this bird."
    """

    def __init__(self, target_food: FoodType, cache_count: int = 1):
        self.target_food = target_food
        self.cache_count = cache_count

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.models.birdfeeder import NUM_DICE

        feeder = ctx.game_state.birdfeeder
        dice_out = NUM_DICE - feeder.count
        if dice_out <= 0:
            return PowerResult(executed=False, description="No dice outside feeder")

        # Roll dice outside the feeder (using correct faces for board type)
        hit = False
        for _ in range(dice_out):
            face = random.choice(feeder.dice_faces)
            if face == self.target_food:
                hit = True
                break
            if isinstance(face, tuple) and self.target_food in face:
                hit = True
                break

        cached = {}
        if hit:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(self.target_food, self.cache_count)
            cached[self.target_food] = self.cache_count

        return PowerResult(food_cached=cached,
                           description=f"Predator {'hit' if hit else 'miss'}")

    def estimate_value(self, ctx: PowerContext) -> float:
        # Probability depends on dice out and target food frequency
        # Each die has ~1/6 to 2/6 chance of matching
        from backend.models.birdfeeder import NUM_DICE
        feeder = ctx.game_state.birdfeeder
        dice_out = NUM_DICE - feeder.count
        # P(at least one match) = 1 - P(no match)^dice_out
        # Rough: each die ~1/3 chance for common types
        prob = 1 - (2/3) ** max(dice_out, 1)
        return prob * self.cache_count * 0.9


class PredatorLookAt(PowerEffect):
    """Look at top card of deck; tuck if wingspan < threshold.

    Example: "Look at a [card] from the deck. If its wingspan is
    less than 75cm, tuck it behind this bird. If not, discard it."
    """

    def __init__(self, wingspan_threshold: int = 75):
        self.wingspan_threshold = wingspan_threshold

    def execute(self, ctx: PowerContext) -> PowerResult:
        if ctx.game_state.deck_remaining <= 0:
            return PowerResult(executed=False, description="Deck empty")

        ctx.game_state.deck_remaining -= 1

        # Simulate: estimate probability based on bird database stats
        # Roughly 50% of birds have wingspan < 75cm
        success_prob = 0.5
        if self.wingspan_threshold >= 100:
            success_prob = 0.7
        elif self.wingspan_threshold <= 50:
            success_prob = 0.3

        tucked = 0
        if random.random() < success_prob:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.tucked_cards += 1
            tucked = 1

        return PowerResult(cards_tucked=tucked,
                           description=f"Predator look: {'tucked' if tucked else 'discarded'}")

    def estimate_value(self, ctx: PowerContext) -> float:
        success_prob = 0.5
        if self.wingspan_threshold >= 100:
            success_prob = 0.7
        elif self.wingspan_threshold <= 50:
            success_prob = 0.3
        return success_prob * 0.9
