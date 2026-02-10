"""Strict, per-card power implementations for high-fidelity rules paths."""

from __future__ import annotations

import random

from backend.models.enums import BoardType, FoodType, Habitat, NestType
from backend.powers.base import PowerContext, PowerEffect, PowerResult
from backend.powers.choices import consume_power_choice


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


def _refill_tray_from_deck(ctx: PowerContext) -> None:
    needed = ctx.game_state.card_tray.needs_refill()
    if needed <= 0:
        return
    for _ in range(needed):
        card = _draw_one_bird(ctx)
        if card is None:
            break
        ctx.game_state.card_tray.add_card(card)


def _parse_food_choice(raw: str | None) -> FoodType | None:
    if not raw:
        return None
    norm = raw.strip().lower()
    for ft in (
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
        FoodType.NECTAR,
    ):
        if norm in {ft.value, ft.name.lower()}:
            return ft
    return None


def _all_food_types() -> list[FoodType]:
    return [
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
        FoodType.NECTAR,
    ]


class DrawThenDiscardFromHand(PowerEffect):
    """Draw N cards, then discard M cards from hand."""

    def __init__(self, draw: int, discard: int):
        self.draw = draw
        self.discard = discard

    def execute(self, ctx: PowerContext) -> PowerResult:
        drew = 0
        for _ in range(self.draw):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            ctx.player.hand.append(card)
            drew += 1

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        discard_names = choice.get("discard_names", [])

        # Explicit discard first (by card names in order), then fallback.
        discarded = 0
        for name in discard_names:
            if discarded >= self.discard:
                break
            idx = next((i for i, c in enumerate(ctx.player.hand) if c.name == name), None)
            if idx is None:
                continue
            ctx.player.hand.pop(idx)
            discarded += 1

        # Deterministic fallback: drop lowest VP first.
        for _ in range(min(self.discard, len(ctx.player.hand))):
            if discarded >= self.discard:
                break
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            discarded += 1
        ctx.game_state.discard_pile_count += discarded

        kept = max(0, drew - discarded)
        return PowerResult(cards_drawn=kept, description=f"Drew {drew}, discarded {discarded}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return max(0.0, (self.draw - self.discard) * 0.4)


class AllPlayersLayEggsSelfBonus(PowerEffect):
    """All players lay X eggs; active player lays +Y additional eggs."""

    def __init__(self, all_players_count: int, self_bonus_count: int):
        self.all_players_count = all_players_count
        self.self_bonus_count = self_bonus_count

    def _lay_any(self, player, count: int) -> int:
        laid = 0
        # Deterministic placement: most available capacity first.
        slots = []
        for _, _, slot in player.board.all_slots():
            if slot.bird and slot.can_hold_more_eggs():
                slots.append((slot.eggs_space(), slot))
        slots.sort(key=lambda x: -x[0])
        for _, slot in slots:
            while laid < count and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
            if laid >= count:
                break
        return laid

    def execute(self, ctx: PowerContext) -> PowerResult:
        self_laid = 0
        for p in ctx.game_state.players:
            laid = self._lay_any(p, self.all_players_count)
            if p.name == ctx.player.name:
                self_laid += laid
        self_laid += self._lay_any(ctx.player, self.self_bonus_count)
        return PowerResult(eggs_laid=self_laid, description=f"Laid {self_laid} eggs (self)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.0


class TuckThenCacheFromSupply(PowerEffect):
    """Tuck one card from hand; if tucked, cache one matching food from supply."""

    def __init__(self, cache_food_type: FoodType):
        self.cache_food_type = cache_food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        tuck_name = choice.get("tuck_name")
        do_cache = choice.get("cache", True)

        # Explicit tuck selection first; fallback lowest VP.
        idx = next((i for i, c in enumerate(ctx.player.hand) if c.name == tuck_name), None) if tuck_name else None
        if idx is None:
            idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
        ctx.player.hand.pop(idx)
        slot.tucked_cards += 1
        cached = 0
        if do_cache and ctx.player.food_supply.spend(self.cache_food_type, 1):
            slot.cache_food(self.cache_food_type, 1)
            cached = 1

        return PowerResult(
            cards_tucked=1,
            food_cached={self.cache_food_type: cached} if cached else {},
            description="Tucked 1 and cached 1" if cached else "Tucked 1",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class EndRoundMandarinDuck(PowerEffect):
    """Round-end: draw 5, keep 1, tuck 1, give 1 to left, discard rest."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        seen = []
        for _ in range(5):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            seen.append(card)
        if not seen:
            return PowerResult(executed=False, description="No cards drawn")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        keep_name = choice.get("keep_name")
        tuck_name = choice.get("tuck_name")
        give_name = choice.get("give_name")

        # Keep explicit card if present; fallback highest VP.
        seen.sort(key=lambda b: b.victory_points, reverse=True)
        keep_idx = next((i for i, c in enumerate(seen) if c.name == keep_name), 0)
        keep = seen.pop(keep_idx)
        ctx.player.hand.append(keep)

        # Tuck explicit card if present; fallback next best.
        tucked = 0
        if seen:
            tuck_idx = next((i for i, c in enumerate(seen) if c.name == tuck_name), 0)
            seen.pop(tuck_idx)
            slot.tucked_cards += 1
            tucked = 1

        # Give one explicit card if present; fallback next best.
        left = ctx.game_state.player_to_left(ctx.player)
        if left is not None and seen:
            give_idx = next((i for i, c in enumerate(seen) if c.name == give_name), 0)
            left.hand.append(seen.pop(give_idx))

        # Rest discarded.
        ctx.game_state.discard_pile_count += len(seen)
        return PowerResult(cards_drawn=1, cards_tucked=tucked, description="Mandarin Duck resolved")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.0


class SnowyOwlBonusThenChoice(PowerEffect):
    """Draw 1 bonus card, then gain 1 card or lay 1 egg."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.templates.draw_cards import DrawBonusCards

        draw_bonus = DrawBonusCards(draw=1, keep=1).execute(ctx)
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        prefer = choice.get("prefer")  # "egg" or "card"
        egg_target = choice.get("egg_target_bird")

        if prefer != "card":
            # Explicit egg target first.
            if egg_target:
                for _, _, slot in ctx.player.board.all_slots():
                    if slot.bird and slot.bird.name == egg_target and slot.can_hold_more_eggs():
                        slot.eggs += 1
                        return PowerResult(
                            cards_drawn=draw_bonus.cards_drawn,
                            eggs_laid=1,
                            description="Drew bonus and laid 1 egg",
                        )
            # Fallback egg placement.
            for _, _, slot in ctx.player.board.all_slots():
                if slot.bird and slot.can_hold_more_eggs():
                    slot.eggs += 1
                    return PowerResult(
                        cards_drawn=draw_bonus.cards_drawn,
                        eggs_laid=1,
                        description="Drew bonus and laid 1 egg",
                    )

        card = _draw_one_bird(ctx)
        if card is not None:
            ctx.player.hand.append(card)
            return PowerResult(cards_drawn=draw_bonus.cards_drawn + 1, description="Drew bonus and 1 bird")
        return PowerResult(cards_drawn=draw_bonus.cards_drawn, description="Drew bonus")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8


class CommonNightingaleChooseFoodAllPlayers(PowerEffect):
    """Choose a food type; all players gain 1 from supply."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        chosen = _parse_food_choice(choice.get("food_type"))
        if chosen is None:
            chosen = FoodType.NECTAR if ctx.game_state.board_type == BoardType.OCEANIA else FoodType.SEED

        for p in ctx.game_state.players:
            p.food_supply.add(chosen, 1)
        return PowerResult(food_gained={chosen: 1}, description=f"All players gained 1 {chosen.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class RedBelliedWoodpeckerSeedFromFeeder(PowerEffect):
    """Gain 1 seed from feeder if available; may cache it on this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        if not feeder.can_take(FoodType.SEED):
            return PowerResult(executed=False, description="No seed available in feeder")

        feeder.take_food(FoodType.SEED)
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        do_cache = bool(choice.get("cache", False))
        if do_cache:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(FoodType.SEED, 1)
            return PowerResult(food_cached={FoodType.SEED: 1}, description="Cached 1 seed")

        ctx.player.food_supply.add(FoodType.SEED, 1)
        return PowerResult(food_gained={FoodType.SEED: 1}, description="Gained 1 seed")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class WillieWagtailDrawTrayNest(PowerEffect):
    """Draw 1 tray card with bowl/wild nest; may reset tray before drawing."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        if choice.get("reset_tray", False):
            discarded = ctx.game_state.card_tray.clear()
            ctx.game_state.discard_pile_count += len(discarded)
            _refill_tray_from_deck(ctx)

        preferred_name = choice.get("draw_name")
        candidates = []
        for i, bird in enumerate(ctx.game_state.card_tray.face_up):
            if bird.nest_type not in {NestType.BOWL, NestType.WILD}:
                continue
            candidates.append((i, bird))
        if not candidates:
            return PowerResult(executed=False, description="No bowl/wild nest card in tray")

        idx = None
        if preferred_name:
            idx = next((i for i, b in candidates if b.name == preferred_name), None)
        if idx is None:
            idx = max(candidates, key=lambda x: x[1].victory_points)[0]

        card = ctx.game_state.card_tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray draw failed")
        ctx.player.hand.append(card)
        return PowerResult(cards_drawn=1, description=f"Drew {card.name} from tray")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class WhiteStorkResetTrayThenDraw(PowerEffect):
    """Discard all tray cards and refill tray, then draw 1 face-up card."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        discarded = ctx.game_state.card_tray.clear()
        ctx.game_state.discard_pile_count += len(discarded)
        _refill_tray_from_deck(ctx)
        if not ctx.game_state.card_tray.face_up:
            return PowerResult(executed=False, description="No cards available after refill")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = choice.get("draw_name")
        idx = next((i for i, c in enumerate(ctx.game_state.card_tray.face_up) if c.name == preferred), None)
        if idx is None:
            idx = max(range(len(ctx.game_state.card_tray.face_up)), key=lambda i: ctx.game_state.card_tray.face_up[i].victory_points)
        card = ctx.game_state.card_tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray draw failed")
        ctx.player.hand.append(card)
        return PowerResult(cards_drawn=1, description=f"Reset tray and drew {card.name}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class PinkEaredDuckDrawKeepGive(PowerEffect):
    """Draw 2 from deck, keep 1, give 1 to another player."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        seen = []
        for _ in range(2):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            seen.append(card)
        if not seen:
            return PowerResult(executed=False, description="No cards drawn")
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        keep_name = choice.get("keep_name")
        give_player = choice.get("give_player")

        keep_idx = next((i for i, c in enumerate(seen) if c.name == keep_name), None)
        if keep_idx is None:
            keep_idx = max(range(len(seen)), key=lambda i: seen[i].victory_points)
        keep = seen.pop(keep_idx)
        ctx.player.hand.append(keep)

        if seen:
            recipient = next((p for p in ctx.game_state.players if p.name == give_player and p.name != ctx.player.name), None)
            if recipient is None:
                recipient = ctx.game_state.player_to_left(ctx.player)
            if recipient is not None:
                recipient.hand.append(seen.pop(0))
            else:
                ctx.player.hand.extend(seen)
                seen.clear()

        if seen:
            ctx.game_state.discard_pile_count += len(seen)

        return PowerResult(cards_drawn=1, description=f"Kept {keep.name} and gave 1 card")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class CrestedLarkDiscardSeedLayEgg(PowerEffect):
    """Discard 1 seed to lay 1 egg on this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not slot.can_hold_more_eggs():
            return PowerResult(executed=False, description="No egg space on bird")
        if not ctx.player.food_supply.spend(FoodType.SEED, 1):
            return PowerResult(executed=False, description="No seed to discard")
        slot.eggs += 1
        return PowerResult(eggs_laid=1, description="Discarded 1 seed to lay 1 egg")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class BlackHeadedGullStealWild(PowerEffect):
    """Steal 1 food from another player; that player gains 1 from feeder."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        others = [p for p in ctx.game_state.players if p.name != ctx.player.name]
        if not others:
            return PowerResult(executed=False, description="No opponent")

        # Prefer opponent with most stealable food.
        others.sort(key=lambda p: p.food_supply.total_non_nectar(), reverse=True)
        target = others[0]
        steal_order = [
            FoodType.INVERTEBRATE,
            FoodType.SEED,
            FoodType.FISH,
            FoodType.FRUIT,
            FoodType.RODENT,
            FoodType.NECTAR,
        ]
        stolen = None
        for ft in steal_order:
            if target.food_supply.get(ft) > 0 and target.food_supply.spend(ft, 1):
                ctx.player.food_supply.add(ft, 1)
                stolen = ft
                break

        feeder = ctx.game_state.birdfeeder
        gained = None
        for ft in _all_food_types():
            if feeder.take_food(ft):
                target.food_supply.add(ft, 1)
                gained = ft
                break

        if stolen is None and gained is None:
            return PowerResult(executed=False, description="No food transferred")
        return PowerResult(
            food_gained={stolen: 1} if stolen else {},
            description=f"Stole {stolen.value if stolen else 'none'}; target gained {gained.value if gained else 'none'}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class BlackDrongoDiscardTrayLayEgg(PowerEffect):
    """Discard any tray cards then refill; if any discarded had grassland habitat, lay 1 egg."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        discard_names = list(choice.get("discard_names", []))
        tray = ctx.game_state.card_tray

        to_discard = []
        if discard_names:
            for name in discard_names:
                idx = next((i for i, c in enumerate(tray.face_up) if c.name == name), None)
                if idx is not None:
                    to_discard.append(idx)
            to_discard = sorted(set(to_discard), reverse=True)
        else:
            # Deterministic default: discard lowest-VP tray card.
            if tray.face_up:
                min_idx = min(range(len(tray.face_up)), key=lambda i: tray.face_up[i].victory_points)
                to_discard = [min_idx]

        discarded_birds = []
        for idx in to_discard:
            card = tray.take_card(idx)
            if card is not None:
                discarded_birds.append(card)
        ctx.game_state.discard_pile_count += len(discarded_birds)
        _refill_tray_from_deck(ctx)

        laid = 0
        if any(Habitat.GRASSLAND in b.habitats for b in discarded_birds):
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            if slot.can_hold_more_eggs():
                slot.eggs += 1
                laid = 1

        return PowerResult(eggs_laid=laid, description=f"Discarded {len(discarded_birds)} tray cards")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7
