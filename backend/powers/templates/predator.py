"""Predator power templates — covers ~43 birds."""

import random
from backend.models.enums import FoodType
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

    def describe_activation(self, ctx: PowerContext) -> str:
        from backend.models.birdfeeder import NUM_DICE
        dice_out = NUM_DICE - ctx.game_state.birdfeeder.count
        return f"roll {dice_out} dice outside feeder; if {self.target_food.value}, cache {self.cache_count} on {ctx.bird.name}"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        from backend.models.birdfeeder import NUM_DICE
        if NUM_DICE - ctx.game_state.birdfeeder.count <= 0:
            return "no dice outside feeder"
        return None

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
        card = _draw_one_bird(ctx)
        if card is None:
            return PowerResult(executed=False, description="Deck empty")

        tucked = 0
        if card.wingspan_cm is not None and card.wingspan_cm < self.wingspan_threshold:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.tucked_cards += 1
            tucked = 1

        return PowerResult(cards_tucked=tucked,
                           description=f"Predator look: {'tucked' if tucked else 'discarded'}")

    def describe_activation(self, ctx: PowerContext) -> str:
        prob_pct = int(self._success_prob(ctx) * 100)
        return f"look at top card; tuck behind {ctx.bird.name} if wingspan < {self.wingspan_threshold}cm (~{prob_pct}%)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if ctx.game_state.deck_remaining <= 0:
            return "deck is empty"
        return None

    def _success_prob(self, ctx: PowerContext) -> float:
        if ctx.game_state.deck_tracker is not None and ctx.game_state.deck_tracker.remaining_count > 0:
            return max(
                0.05,
                min(0.95, ctx.game_state.deck_tracker.predator_success_rate(self.wingspan_threshold)),
            )

        remaining = ctx.deck_remaining or ctx.game_state.deck_remaining
        # As the deck thins, predator hits become less reliable on average.
        deck_factor = max(0.5, min(1.0, remaining / 200.0))
        if self.wingspan_threshold >= 100:
            base = 0.7
        elif self.wingspan_threshold <= 50:
            base = 0.3
        else:
            base = 0.5
        return max(0.05, min(0.95, base * deck_factor))

    def estimate_value(self, ctx: PowerContext) -> float:
        return self._success_prob(ctx) * 0.9
