"""Card tucking power templates — covers ~53 birds."""

import random
from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


def _draw_one_bird(ctx: PowerContext):
    """Draw one concrete bird card from deck identity model if available."""
    deck_cards = getattr(ctx.game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        ctx.game_state.deck_remaining = max(0, ctx.game_state.deck_remaining - 1)
        return card

    if ctx.game_state.deck_remaining <= 0:
        return None
    from backend.data.registries import get_bird_registry
    pool = list(get_bird_registry().all_birds)
    if not pool:
        return None
    ctx.game_state.deck_remaining = max(0, ctx.game_state.deck_remaining - 1)
    return random.choice(pool)


class TuckFromHand(PowerEffect):
    """Tuck card(s) from hand behind this bird.

    Variants:
    - Tuck + draw: "Tuck a [card] from your hand behind this bird. If you do, draw 1 [card]."
    - Tuck + lay: "Tuck a [card] from your hand behind this bird. If you do, lay 1 [egg]."
    - Tuck + gain food: "Tuck a [card] from your hand behind this bird. If you do, gain 1 [food]."
    """

    def __init__(self, tuck_count: int = 1,
                 draw_count: int = 0,
                 lay_count: int = 0,
                 food_type: FoodType | None = None,
                 food_count: int = 0):
        self.tuck_count = tuck_count
        self.draw_count = draw_count
        self.lay_count = lay_count
        self.food_type = food_type
        self.food_count = food_count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        # Need cards in hand to tuck
        tucked = min(self.tuck_count, len(ctx.player.hand))
        if tucked == 0:
            return PowerResult(executed=False, description="No cards in hand to tuck")

        # Remove cards from hand and tuck
        for _ in range(tucked):
            if ctx.player.hand:
                ctx.player.hand.pop()  # Remove last card (choice in UI)
                slot.tucked_cards += 1

        result = PowerResult(cards_tucked=tucked)

        # Bonus effects
        if self.draw_count > 0 and tucked > 0:
            drawn = 0
            for _ in range(self.draw_count):
                card = _draw_one_bird(ctx)
                if card is None:
                    break
                ctx.player.hand.append(card)
                drawn += 1
            result.cards_drawn = drawn

        if self.lay_count > 0 and tucked > 0:
            if slot.can_hold_more_eggs():
                laid = min(self.lay_count, slot.eggs_space())
                slot.eggs += laid
                result.eggs_laid = laid

        if self.food_count > 0 and self.food_type and tucked > 0:
            ctx.player.food_supply.add(self.food_type, self.food_count)
            result.food_gained = {self.food_type: self.food_count}

        result.description = f"Tucked {tucked} cards"
        return result

    def describe_activation(self, ctx: PowerContext) -> str:
        parts = [f"tuck {self.tuck_count} card from hand behind {ctx.bird.name}"]
        if self.draw_count:
            parts.append(f"draw {self.draw_count} card{'s' if self.draw_count > 1 else ''}")
        if self.lay_count:
            parts.append(f"lay {self.lay_count} egg on {ctx.bird.name}")
        if self.food_count and self.food_type:
            parts.append(f"gain {self.food_count} {self.food_type.value}")
        return ", then ".join(parts)

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if len(ctx.player.hand) < self.tuck_count:
            return "no cards in hand to tuck"
        return None

    def estimate_value(self, ctx: PowerContext) -> float:
        # Tucked card = 1 point, but costs a card from hand
        base = self.tuck_count * 0.5  # Net ~0.5 after losing the card
        base += self.draw_count * 0.4
        base += self.lay_count * 0.8
        base += self.food_count * 0.5
        return base


class TuckFromDeck(PowerEffect):
    """Draw from deck and tuck behind this bird.

    Example: "Draw 1 [card] from the deck and tuck it behind this bird."
    """

    def __init__(self, count: int = 1):
        self.count = count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(self.count, max(0, ctx.game_state.deck_remaining))
        slot.tucked_cards += tucked
        ctx.game_state.deck_remaining -= tucked

        return PowerResult(cards_tucked=tucked,
                           description=f"Tucked {tucked} from deck")

    def describe_activation(self, ctx: PowerContext) -> str:
        return f"tuck {self.count} from deck behind {ctx.bird.name} ({self.count} pt)"

    def skip_reason(self, ctx: PowerContext) -> str | None:
        if ctx.game_state.deck_remaining <= 0:
            return "deck is empty"
        return None

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.count * 0.9  # Pure point gain, no hand cost


class DiscardToTuck(PowerEffect):
    """Discard a card (from hand or deck) and tuck if condition met.

    Example: "Draw and discard 1 [card] from the deck. If its food cost has
    [invertebrate], tuck it and gain 1 [invertebrate]."
    """

    def __init__(self, draw_count: int = 1, tuck_probability: float = 0.4,
                 food_type: FoodType | None = None, food_on_success: int = 0):
        self.draw_count = draw_count
        self.tuck_probability = tuck_probability
        self.food_type = food_type
        self.food_on_success = food_on_success

    def execute(self, ctx: PowerContext) -> PowerResult:
        # Simulation: use probability to determine success
        import random
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        tucked = 0
        for _ in range(self.draw_count):
            ctx.game_state.deck_remaining -= 1
            if random.random() < self.tuck_probability:
                slot.tucked_cards += 1
                tucked += 1

        result = PowerResult(cards_tucked=tucked)
        if tucked > 0 and self.food_type and self.food_on_success:
            ctx.player.food_supply.add(self.food_type, self.food_on_success * tucked)
            result.food_gained = {self.food_type: self.food_on_success * tucked}

        result.description = f"Drew {self.draw_count}, tucked {tucked}"
        return result

    def describe_activation(self, ctx: PowerContext) -> str:
        prob_pct = int(self.tuck_probability * 100)
        base = f"draw {self.draw_count} from deck, tuck if match (~{prob_pct}%)"
        if self.food_type and self.food_on_success:
            base += f", gain {self.food_on_success} {self.food_type.value} if tucked"
        return base

    def estimate_value(self, ctx: PowerContext) -> float:
        expected_tucks = self.draw_count * self.tuck_probability
        base = expected_tucks * 0.9
        base += expected_tucks * self.food_on_success * 0.5
        return base
