"""Card drawing power templates — covers ~70 birds."""

from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


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
        drawn = 0

        if self.all_players:
            for p in ctx.game_state.players:
                amount = min(self.keep, ctx.game_state.deck_remaining)
                ctx.game_state.deck_remaining -= self.draw
                if p.name == ctx.player.name:
                    drawn = amount
        else:
            drawn = min(self.keep, ctx.game_state.deck_remaining)
            ctx.game_state.deck_remaining -= self.draw

        # In state-input mode, actual cards drawn come from UI
        # For simulation, we just track counts
        return PowerResult(cards_drawn=drawn,
                           description=f"Drew {drawn} cards")

    def describe_activation(self, ctx: PowerContext) -> str:
        prefix = "ALL PLAYERS: " if self.all_players else ""
        if self.keep < self.draw:
            return f"{prefix}draw {self.draw}, keep {self.keep} card{'s' if self.keep > 1 else ''}"
        return f"{prefix}draw {self.draw} card{'s' if self.draw > 1 else ''}"

    def estimate_value(self, ctx: PowerContext) -> float:
        base = self.keep * 0.4  # Cards are potential, not points
        if self.all_players:
            base *= 0.7
        return base


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
        return self.count * 0.35


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
