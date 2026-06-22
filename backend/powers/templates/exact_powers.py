"""Exact implementations for non-promo birds whose auto-parsed mappings dropped
clauses (per-condition counts, two-part bonus powers, give-then-draw, etc.)."""

from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult
from backend.powers.templates.tuck_cards import _draw_one_bird
from backend.powers.templates.draw_cards import (
    _draw_one_bonus, _best_bonus_subset_by_score, _init_bonus_deck_if_needed,
)
from backend.powers.templates.promo_uk import _gain_one_from_feeder


def _draw_n_keep_best_bonus(ctx, draw_n, keep_n):
    _init_bonus_deck_if_needed(ctx.game_state)
    drawn = [bc for bc in (_draw_one_bonus(ctx.game_state) for _ in range(draw_n)) if bc]
    if not drawn:
        return []
    kept = _best_bonus_subset_by_score(ctx, drawn, min(keep_n, len(drawn)))
    for bc in kept:
        ctx.player.bonus_cards.append(bc)
    return kept


class GiveCardThenDrawTwo(PowerEffect):
    """Give 1 [card] from your hand to another player; if you do, draw 2 [card].
    (Red Avadavat.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        opp = (ctx.game_state.player_to_left(pl)
               or next((p for p in ctx.game_state.players if p is not pl), None))
        if not pl.hand or opp is None:
            return PowerResult(executed=False, description="No card to give / no other player")
        opp.hand.append(pl.hand.pop())
        drawn = 0
        for _ in range(2):
            c = _draw_one_bird(ctx)
            if c is None:
                break
            pl.hand.append(c)
            drawn += 1
        return PowerResult(cards_drawn=drawn, description=f"Gave 1 card, drew {drawn}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0 if (ctx.player.hand and ctx.game_state.num_players > 1) else 0.0


class DrawCardsPerConditionKeepOne(PowerEffect):
    """Draw 1 [card] for each bird matching a condition, keep 1, discard the rest.
    condition="wetland_birds_with_egg" (Common Sandpiper)."""

    def __init__(self, condition: str):
        self.condition = condition

    def _count(self, ctx: PowerContext) -> int:
        pl = ctx.player
        if self.condition == "wetland_birds_with_egg":
            return sum(1 for _, s in pl.board.wetland.occupied_slots() if s.eggs > 0)
        return 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        n = self._count(ctx)
        kept = 0
        for i in range(n):
            c = _draw_one_bird(ctx)
            if c is None:
                break
            if i == 0:  # keep the first, discard the rest
                ctx.player.hand.append(c)
                kept = 1
            else:
                ctx.game_state.discard_pile_count += 1
        return PowerResult(cards_drawn=kept, description=f"Drew {n}, kept {kept}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0 if self._count(ctx) > 0 else 0.0


class DrawBonusPerConditionKeepOne(PowerEffect):
    """Draw 1 bonus card for each bird matching a condition, keep 1, discard rest.
    condition="grassland_birds" (Plains-Wanderer)."""

    def __init__(self, condition: str):
        self.condition = condition

    def _count(self, ctx: PowerContext) -> int:
        if self.condition == "grassland_birds":
            return ctx.player.board.grassland.bird_count
        return 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        n = self._count(ctx)
        kept = _draw_n_keep_best_bonus(ctx, n, 1) if n > 0 else []
        return PowerResult(description=f"Drew {n} bonus, kept {len(kept)}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2 if self._count(ctx) > 0 else 0.0


class DrawBonusThenDrawCardsKeep(PowerEffect):
    """Draw 1 new bonus card. Then draw `draw` [card] and keep `keep` of them.
    (Black-Tailed Godwit, Red Knot.)"""

    def __init__(self, draw: int = 3, keep: int = 1):
        self.draw = draw
        self.keep = keep

    def execute(self, ctx: PowerContext) -> PowerResult:
        _draw_n_keep_best_bonus(ctx, 1, 1)
        drawn = [c for c in (_draw_one_bird(ctx) for _ in range(self.draw)) if c]
        kept = 0
        for i, c in enumerate(drawn):
            if i < self.keep:
                ctx.player.hand.append(c)
                kept += 1
            else:
                ctx.game_state.discard_pile_count += 1
        return PowerResult(cards_drawn=kept, description=f"Drew bonus + {len(drawn)} cards, kept {kept}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.0


class DrawBonusThenGainDie(PowerEffect):
    """Draw 1 new bonus card, then gain 1 [die] from the birdfeeder.
    (Corsican Nuthatch.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        _draw_n_keep_best_bonus(ctx, 1, 1)
        _gain_one_from_feeder(ctx.game_state, ctx.player)
        return PowerResult(description="Drew bonus + gained a die")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8


class DrawBonusThenChoice(PowerEffect):
    """Draw 1 new bonus card, then choose one: gain 1 [die] / lay 1 [egg] on any
    bird / draw 1 [card]. (European Turtle Dove.) Picks the highest-value option."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        _draw_n_keep_best_bonus(ctx, 1, 1)
        pl = ctx.player
        # Prefer laying an egg if there's space (1 VP), else gain a die, else draw.
        for _, s in [(h, s) for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND)
                     for _, s in pl.board.get_row(h).occupied_slots()]:
            if s.can_hold_more_eggs():
                s.eggs += 1
                return PowerResult(eggs_laid=1, description="Drew bonus + laid an egg")
        if _gain_one_from_feeder(ctx.game_state, pl):
            return PowerResult(description="Drew bonus + gained a die")
        c = _draw_one_bird(ctx)
        if c is not None:
            pl.hand.append(c)
            return PowerResult(cards_drawn=1, description="Drew bonus + drew a card")
        return PowerResult(description="Drew bonus")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.9


class EachPlayerMayDiscardEggDraw(PowerEffect):
    """Each player may discard 1 [egg] to draw 1 [card] from the deck.
    (Sarus Crane.) Each player does it when it's worthwhile (has a spare egg)."""

    def _discard_one_egg(self, player) -> bool:
        for row in player.board.all_rows():
            for _, s in row.occupied_slots():
                if s.eggs > 0:
                    s.eggs -= 1
                    return True
        return False

    def execute(self, ctx: PowerContext) -> PowerResult:
        drawn_self = 0
        # Active player: trade an egg only if holding a comfortable surplus.
        for p in ctx.game_state.players:
            total_eggs = sum(s.eggs for r in p.board.all_rows() for _, s in r.occupied_slots())
            if total_eggs >= 2 and self._discard_one_egg(p):
                c = _draw_one_bird(ctx)
                if c is not None:
                    p.hand.append(c)
                    if p is ctx.player:
                        drawn_self = 1
        return PowerResult(cards_drawn=drawn_self, description="Players traded eggs for cards")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class DrawTwoKeepOneTuckOrHand(PowerEffect):
    """Draw 2 [card] (from the discard pile, approximated by the deck), keep 1
    (tuck it or add to hand), discard the other. (Australian Ibis.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        c1 = _draw_one_bird(ctx)
        c2 = _draw_one_bird(ctx)
        cards = [c for c in (c1, c2) if c is not None]
        if not cards:
            return PowerResult(executed=False, description="Nothing to draw")
        # Keep the higher-VP card; tuck it (1 guaranteed VP) behind this bird.
        keep = max(cards, key=lambda c: c.victory_points)
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        for c in cards:
            if c is not keep:
                ctx.game_state.discard_pile_count += 1
        return PowerResult(cards_tucked=1, description=f"Drew {len(cards)}, tucked 1")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class DrawBonusKeepOneGiveOne(PowerEffect):
    """Draw 2 new bonus cards, keep 1, give 1 to another player.
    (Western Capercaillie.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        _init_bonus_deck_if_needed(ctx.game_state)
        drawn = [bc for bc in (_draw_one_bonus(ctx.game_state) for _ in range(2)) if bc]
        if not drawn:
            return PowerResult(executed=False, description="No bonus cards")
        keep = _best_bonus_subset_by_score(ctx, drawn, 1)
        kept = keep[0] if keep else drawn[0]
        ctx.player.bonus_cards.append(kept)
        opp = (ctx.game_state.player_to_left(ctx.player)
               or next((p for p in ctx.game_state.players if p is not ctx.player), None))
        for bc in drawn:
            if bc is not kept and opp is not None:
                opp.bonus_cards.append(bc)
                break
        return PowerResult(description="Drew 2 bonus, kept 1, gave 1")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5
