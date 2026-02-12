"""Card drawing power templates — covers ~70 birds."""

import random
from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


def _draw_context_factor(ctx: PowerContext) -> float:
    deck_remaining = ctx.deck_remaining or ctx.game_state.deck_remaining
    deck_factor = max(0.15, min(1.0, deck_remaining / 200.0))
    hand_size = ctx.hand_size or len(ctx.player.hand)
    if hand_size >= 12:
        hand_factor = 0.55
    elif hand_size >= 8:
        hand_factor = 0.75
    else:
        hand_factor = 1.0
    return deck_factor * hand_factor


def _draw_one_bird(game_state):
    """Draw one concrete bird card from deck identity model if available."""
    deck_cards = getattr(game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
        if game_state.deck_tracker is not None:
            game_state.deck_tracker.mark_drawn(card.name)
        return card

    if game_state.deck_remaining <= 0:
        return None
    from backend.data.registries import get_bird_registry
    pool = list(get_bird_registry().all_birds)
    if not pool:
        return None
    game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
    card = random.choice(pool)
    if game_state.deck_tracker is not None:
        game_state.deck_tracker.mark_drawn(card.name)
    return card


class DrawCards(PowerEffect):
    """Draw cards from the deck, optionally keeping some.

    Examples:
    - "Draw 2 [card] from the deck."
    - "Draw 3 [card], then discard 1."
    - "All players draw 1 [card] from the deck."
    """

    def __init__(self, draw: int = 1, keep: int | None = None,
                 all_players: bool = False):
        self.draw = draw
        self.keep = keep if keep is not None else draw
        self.all_players = all_players

    def execute(self, ctx: PowerContext) -> PowerResult:
        kept_for_actor = 0

        def _resolve_for_player(p) -> int:
            seen = []
            for _ in range(self.draw):
                card = _draw_one_bird(ctx.game_state)
                if card is None:
                    break
                seen.append(card)
            keep_n = min(self.keep, len(seen))
            if keep_n > 0:
                p.hand.extend(seen[:keep_n])
            discarded = max(0, len(seen) - keep_n)
            if discarded > 0:
                ctx.game_state.discard_pile_count += discarded
            return keep_n

        if self.all_players:
            for p in ctx.game_state.players:
                kept = _resolve_for_player(p)
                if p.name == ctx.player.name:
                    kept_for_actor = kept
        else:
            kept_for_actor = _resolve_for_player(ctx.player)

        return PowerResult(
            cards_drawn=kept_for_actor,
            description=f"Drew {kept_for_actor} cards",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        prefix = "ALL PLAYERS: " if self.all_players else ""
        if self.keep < self.draw:
            return f"{prefix}draw {self.draw}, keep {self.keep} card{'s' if self.keep > 1 else ''}"
        return f"{prefix}draw {self.draw} card{'s' if self.draw > 1 else ''}"

    def estimate_value(self, ctx: PowerContext) -> float:
        base = self.keep * 0.4  # Cards are potential, not points
        if self.all_players:
            base *= 0.7
        return base * _draw_context_factor(ctx)


class DrawFromTray(PowerEffect):
    """Draw face-up cards with optional filtering.

    Examples:
    - "Draw 1 [card] from the tray whose food cost includes [invertebrate]."
    """

    def __init__(self, count: int = 1, food_filter: FoodType | None = None):
        self.count = count
        self.food_filter = food_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        drawn = 0
        tray = ctx.game_state.card_tray

        for _ in range(self.count):
            for i, bird in enumerate(tray.face_up):
                if self.food_filter:
                    if self.food_filter not in bird.food_cost.distinct_types:
                        continue
                card = tray.take_card(i)
                if card:
                    ctx.player.hand.append(card)
                    drawn += 1
                    break

        return PowerResult(cards_drawn=drawn,
                           description=f"Drew {drawn} from tray")

    def describe_activation(self, ctx: PowerContext) -> str:
        if self.food_filter:
            return f"draw {self.count} from tray (matching {self.food_filter.value} cost)"
        return f"draw {self.count} from tray"

    def estimate_value(self, ctx: PowerContext) -> float:
        hand_size = ctx.hand_size or len(ctx.player.hand)
        hand_factor = 0.65 if hand_size >= 10 else 1.0
        return self.count * 0.35 * hand_factor


class DrawBonusCards(PowerEffect):
    """Draw bonus cards from the bonus deck.

    Example: "Draw 2 bonus cards, then discard 1."
    """

    def __init__(self, draw: int = 2, keep: int = 1):
        self.draw = draw
        self.keep = keep

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.data.registries import get_bonus_registry

        reg = get_bonus_registry()
        all_cards = list(reg.all_cards)
        kept = 0
        # Deterministic fallback when no explicit draft choice is modeled:
        # keep the highest draft-value bonus cards available.
        ranked = sorted(
            all_cards,
            key=lambda bc: (bc.draft_value_pct if bc.draft_value_pct is not None else 0.0),
            reverse=True,
        )
        for bc in ranked[: max(0, self.keep)]:
            ctx.player.bonus_cards.append(bc)
            kept += 1

        return PowerResult(
            cards_drawn=kept,
            description=f"Drew {self.draw} bonus cards, kept {kept}",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        if self.keep < self.draw:
            return f"draw {self.draw} bonus cards, keep {self.keep}"
        return f"draw {self.draw} bonus card{'s' if self.draw > 1 else ''}"

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.keep * 1.5  # Bonus cards are high value
