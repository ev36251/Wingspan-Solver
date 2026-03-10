"""Predator power templates — covers ~43 birds."""

import random
from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult


def _draw_one_bird(ctx: PowerContext):
    """Draw one concrete bird card from deck identity model if available."""
    if ctx.game_state.deck_remaining <= 0:
        return None

    deck_cards = getattr(ctx.game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        ctx.game_state.deck_remaining = max(0, ctx.game_state.deck_remaining - 1)
        if ctx.game_state.deck_tracker is not None:
            ctx.game_state.deck_tracker.mark_drawn(card.name)
        return card

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
    """Look at top card of deck; tuck if card matches configured predicate.

    Example: "Look at a [card] from the deck. If its wingspan is
    less than 75cm, tuck it behind this bird. If not, discard it."
    """

    def __init__(
        self,
        wingspan_threshold: int = 75,
        *,
        wingspan_cmp: str = "lt",
        habitat_filter: Habitat | None = None,
        food_cost_includes: set[FoodType] | None = None,
        cache_food_type: FoodType | None = None,
        cache_count: int = 0,
    ):
        self.wingspan_threshold = wingspan_threshold
        self.wingspan_cmp = wingspan_cmp
        self.habitat_filter = habitat_filter
        self.food_cost_includes = set(food_cost_includes or [])
        self.cache_food_type = cache_food_type
        self.cache_count = int(max(0, cache_count))

    def _matches(self, card) -> bool:
        if self.food_cost_includes:
            return any(ft in self.food_cost_includes for ft in card.food_cost.items)
        if self.habitat_filter is not None:
            return card.can_live_in(self.habitat_filter)
        if card.wingspan_cm is None:
            return False
        if self.wingspan_cmp == "gt":
            return card.wingspan_cm > self.wingspan_threshold
        return card.wingspan_cm < self.wingspan_threshold

    def execute(self, ctx: PowerContext) -> PowerResult:
        card = _draw_one_bird(ctx)
        if card is None:
            return PowerResult(executed=False, description="Deck empty")

        tucked = 0
        food_cached: dict[FoodType, int] = {}
        if self._matches(card):
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.tucked_cards += 1
            tucked = 1
            if self.cache_food_type and self.cache_count > 0:
                slot.cache_food(self.cache_food_type, self.cache_count)
                food_cached[self.cache_food_type] = self.cache_count

        return PowerResult(cards_tucked=tucked, food_cached=food_cached,
                           description=f"Predator look: {'tucked' if tucked else 'discarded'}")

    def describe_activation(self, ctx: PowerContext) -> str:
        prob_pct = int(self._success_prob(ctx) * 100)
        if self.food_cost_includes:
            foods = "/".join(sorted(ft.value for ft in self.food_cost_includes))
            cond = f"food cost includes {foods}"
        elif self.habitat_filter is not None:
            cond = f"can live in {self.habitat_filter.value}"
        else:
            op = "<" if self.wingspan_cmp != "gt" else ">"
            cond = f"wingspan {op} {self.wingspan_threshold}cm"
        extra = ""
        if self.cache_food_type and self.cache_count > 0:
            extra = f"; if tucked cache {self.cache_count} {self.cache_food_type.value}"
        return f"look at top card; tuck behind {ctx.bird.name} if {cond} (~{prob_pct}%){extra}"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if ctx.game_state.deck_remaining <= 0:
            return "deck is empty"
        return None

    def _success_prob(self, ctx: PowerContext) -> float:
        if self.food_cost_includes:
            # Roughly ~20% per target icon in mixed deck, capped.
            return max(0.05, min(0.95, 0.2 * max(1, len(self.food_cost_includes))))
        if self.habitat_filter is not None:
            # Typical habitat occupancy in mixed deck.
            return 0.35
        if ctx.game_state.deck_tracker is not None and ctx.game_state.deck_tracker.remaining_count > 0:
            base = max(
                0.05,
                min(0.95, ctx.game_state.deck_tracker.predator_success_rate(self.wingspan_threshold)),
            )
            return base if self.wingspan_cmp != "gt" else max(0.05, min(0.95, 1.0 - base))

        remaining = ctx.deck_remaining or ctx.game_state.deck_remaining
        # As the deck thins, predator hits become less reliable on average.
        deck_factor = max(0.5, min(1.0, remaining / 200.0))
        if self.wingspan_threshold >= 100:
            base = 0.7
        elif self.wingspan_threshold <= 50:
            base = 0.3
        else:
            base = 0.5
        if self.wingspan_cmp == "gt":
            base = 1.0 - base
        return max(0.05, min(0.95, base * deck_factor))

    def estimate_value(self, ctx: PowerContext) -> float:
        return self._success_prob(ctx) * (0.9 + (0.5 * self.cache_count if self.cache_food_type else 0.0))
