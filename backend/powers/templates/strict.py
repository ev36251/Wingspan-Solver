"""Strict, per-card power implementations for high-fidelity rules paths."""

from __future__ import annotations

import random

from backend.models.enums import ActionType, BoardType, FoodType, Habitat, NestType
from backend.powers.base import PowerContext, PowerEffect, PowerResult
from backend.powers.choices import consume_power_choice


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


def _non_wild_food_types_for_board(board_type: BoardType) -> list[FoodType]:
    foods = [
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
    ]
    if board_type == BoardType.OCEANIA:
        foods.append(FoodType.NECTAR)
    return foods


def _choose_wild_food_from_supply(ctx: PowerContext) -> FoodType:
    foods = _non_wild_food_types_for_board(ctx.game_state.board_type)
    # Deterministic default: pick the currently scarcest food in hand supply.
    return min(foods, key=lambda ft: (ctx.player.food_supply.get(ft), ft.value))


def _draw_many_birds(ctx: PowerContext, count: int) -> list:
    drew = []
    for _ in range(max(0, count)):
        card = _draw_one_bird(ctx)
        if card is None:
            break
        ctx.player.hand.append(card)
        drew.append(card)
    return drew


def _used_all_action_types_this_round(player) -> bool:
    required = {
        ActionType.PLAY_BIRD,
        ActionType.GAIN_FOOD,
        ActionType.LAY_EGGS,
        ActionType.DRAW_CARDS,
    }
    return required.issubset(player.action_types_used_this_round)


def _merge_power_results(base: PowerResult, other: PowerResult) -> PowerResult:
    if not other.executed:
        return base
    base.eggs_laid += other.eggs_laid
    base.cards_drawn += other.cards_drawn
    base.cards_tucked += other.cards_tucked
    for ft, cnt in other.food_gained.items():
        base.food_gained[ft] = base.food_gained.get(ft, 0) + cnt
    for ft, cnt in other.food_cached.items():
        base.food_cached[ft] = base.food_cached.get(ft, 0) + cnt
    return base


def _auto_remove_eggs(player, count: int) -> bool:
    if count <= 0:
        return True
    if player.board.total_eggs() < count:
        return False
    candidates = []
    for hab, idx, slot in player.board.all_slots():
        if slot.bird is None or slot.eggs <= 0:
            continue
        candidates.append((slot.eggs, hab, idx))
    candidates.sort(key=lambda t: (t[0], t[1].value, t[2]), reverse=True)
    remaining = count
    for _, hab, idx in candidates:
        if remaining <= 0:
            break
        slot = player.board.get_row(hab).slots[idx]
        removable = min(slot.eggs, remaining)
        slot.eggs -= removable
        remaining -= removable
    return remaining == 0


def _select_grassland_target_player(ctx: PowerContext):
    others = [p for p in ctx.game_state.players if p.name != ctx.player.name]
    if not others:
        return None
    return max(others, key=lambda p: (p.lay_eggs_actions_this_round, p.name))


def _best_bonus_play_candidate(ctx: PowerContext, habitat_filter: Habitat | None = None, egg_discount: int = 0):
    from backend.config import EGG_COST_BY_COLUMN
    from backend.engine.rules import find_food_payment_options

    best = None
    for bird in list(ctx.player.hand):
        habitats = [habitat_filter] if habitat_filter else list(bird.habitats)
        for habitat in habitats:
            if habitat is None or not bird.can_live_in(habitat):
                continue
            row = ctx.player.board.get_row(habitat)
            slot_idx = row.next_empty_slot()
            if slot_idx is None:
                continue
            egg_cost = max(0, EGG_COST_BY_COLUMN[slot_idx] - egg_discount)
            if ctx.player.board.total_eggs() < egg_cost:
                continue
            payment_options = find_food_payment_options(ctx.player, bird.food_cost)
            if not payment_options:
                continue
            payment = min(
                payment_options,
                key=lambda p: (sum(p.values()), p.get(FoodType.NECTAR, 0), sorted((k.value, v) for k, v in p.items())),
            )
            score = (
                bird.victory_points,
                bird.egg_limit,
                -egg_cost,
                -sum(payment.values()),
            )
            candidate = (score, bird, habitat, slot_idx, egg_cost, payment)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def _play_bonus_bird(ctx: PowerContext, habitat_filter: Habitat | None = None, egg_discount: int = 0) -> PowerResult:
    from backend.powers.base import NoPower
    from backend.powers.registry import get_power, assert_power_allowed_for_strict_mode

    candidate = _best_bonus_play_candidate(ctx, habitat_filter=habitat_filter, egg_discount=egg_discount)
    if candidate is None:
        return PowerResult(executed=False, description="No playable bonus bird")

    _, bird, habitat, slot_idx, egg_cost, payment = candidate
    if not _auto_remove_eggs(ctx.player, egg_cost):
        return PowerResult(executed=False, description="Failed to pay egg cost")
    for ft, cnt in payment.items():
        if not ctx.player.food_supply.spend(ft, cnt):
            return PowerResult(executed=False, description="Failed to pay food cost")

    row = ctx.player.board.get_row(habitat)
    row.nectar_spent += payment.get(FoodType.NECTAR, 0)
    ctx.player.remove_from_hand(bird.name)
    row.slots[slot_idx].bird = bird

    out = PowerResult(description=f"Played bonus bird {bird.name}")
    if bird.color.value in {"white", "yellow"}:
        assert_power_allowed_for_strict_mode(ctx.game_state, bird)
        power = get_power(bird)
        if not isinstance(power, NoPower):
            sub_ctx = PowerContext(
                game_state=ctx.game_state,
                player=ctx.player,
                bird=bird,
                slot_index=slot_idx,
                habitat=habitat,
            )
            if power.can_execute(sub_ctx):
                out = _merge_power_results(out, power.execute(sub_ctx))
    return out


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


class ConditionalAllActionsPlayBird(PowerEffect):
    """If all 4 action types were used this round, play 1 bonus bird."""

    def __init__(self, habitat_filter: Habitat | None = None):
        self.habitat_filter = habitat_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not _used_all_action_types_this_round(ctx.player):
            return PowerResult(executed=False, description="All 4 action types were not used")
        return _play_bonus_bird(ctx, habitat_filter=self.habitat_filter, egg_discount=0)

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.6 if _used_all_action_types_this_round(ctx.player) else 0.3


class ConditionalAllActionsGainWild(PowerEffect):
    """If all 4 action types were used this round, gain N wild food."""

    def __init__(self, count: int):
        self.count = count

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not _used_all_action_types_this_round(ctx.player):
            return PowerResult(executed=False, description="All 4 action types were not used")
        gained: dict[FoodType, int] = {}
        for _ in range(self.count):
            ft = _choose_wild_food_from_supply(ctx)
            ctx.player.food_supply.add(ft, 1)
            gained[ft] = gained.get(ft, 0) + 1
        return PowerResult(food_gained=gained, description=f"Gained {self.count} wild food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0 if _used_all_action_types_this_round(ctx.player) else 0.2


class ConditionalAllActionsFoodEggCard(PowerEffect):
    """If all 4 action types were used this round: gain 1 wild, lay 1 egg, draw 1 card."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not _used_all_action_types_this_round(ctx.player):
            return PowerResult(executed=False, description="All 4 action types were not used")

        out = PowerResult()
        ft = _choose_wild_food_from_supply(ctx)
        ctx.player.food_supply.add(ft, 1)
        out.food_gained = {ft: 1}

        # Lay 1 egg on first available bird, preferring this row.
        laid = 0
        row = ctx.player.board.get_row(ctx.habitat)
        for slot in row.slots:
            if slot.bird and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid = 1
                break
        if laid == 0:
            for _, _, slot in ctx.player.board.all_slots():
                if slot.bird and slot.can_hold_more_eggs():
                    slot.eggs += 1
                    laid = 1
                    break
        out.eggs_laid = laid

        drawn = _draw_many_birds(ctx, 1)
        out.cards_drawn = len(drawn)
        out.description = "Gained 1 wild, laid 1 egg, and drew 1 card"
        return out

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8 if _used_all_action_types_this_round(ctx.player) else 0.3


class ChooseEgglessHabitatLayEggEachBird(PowerEffect):
    """Choose a habitat with no eggs; lay 1 egg on each bird in that habitat."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        candidates = []
        for row in ctx.player.board.all_rows():
            if row.bird_count == 0:
                continue
            if row.total_eggs() == 0:
                candidates.append(row)
        if not candidates:
            return PowerResult(executed=False, description="No habitat with zero eggs")

        row = max(candidates, key=lambda r: r.bird_count)
        laid = 0
        for slot in row.slots:
            if slot.bird and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} eggs in {row.habitat.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class LayEggAdjacentToSelf(PowerEffect):
    """Lay 1 egg on birds immediately left and right of this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        laid = 0
        for idx in (ctx.slot_index - 1, ctx.slot_index + 1):
            if idx < 0 or idx >= len(row.slots):
                continue
            slot = row.slots[idx]
            if slot.bird and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} adjacent egg(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.1


class LayEggOnSelfPerColumnEggBird(PowerEffect):
    """For each other bird in this column with an egg, lay 1 egg on this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        count = 0
        for row in ctx.player.board.all_rows():
            if row.habitat == ctx.habitat:
                continue
            slot = row.slots[ctx.slot_index]
            if slot.bird and slot.eggs > 0:
                count += 1
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not found")
        laid = min(count, target.eggs_space())
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on this bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class LayEggOnSelfPerOtherCavityBird(PowerEffect):
    """Lay 1 egg on this bird for each other cavity/wild nest bird you have."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not found")
        count = 0
        for _, _, slot in ctx.player.board.all_slots():
            if not slot.bird or slot.bird.name == ctx.bird.name:
                continue
            if slot.bird.nest_type in {NestType.CAVITY, NestType.WILD}:
                count += 1
        laid = min(count, target.eggs_space())
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on this bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class LayEggOnSelfPerBirdToLeft(PowerEffect):
    """Lay 1 egg on this bird for each bird to its left in this row."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        target = row.slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not found")
        left_count = sum(1 for i in range(ctx.slot_index) if row.slots[i].bird is not None)
        laid = min(left_count, target.eggs_space())
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on this bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.1


class DiscardFoodToTuckFromDeck(PowerEffect):
    """Discard up to N food; for each discarded, tuck 1 from deck."""

    def __init__(self, max_discard: int = 5, food_type: FoodType | None = None):
        self.max_discard = max_discard
        self.food_type = food_type

    def _spend_one(self, ctx: PowerContext) -> bool:
        if self.food_type is not None:
            return ctx.player.food_supply.spend(self.food_type, 1)
        # Wild discard: spend the most abundant non-wild type first.
        candidates = _non_wild_food_types_for_board(ctx.game_state.board_type)
        candidates.sort(key=lambda ft: (ctx.player.food_supply.get(ft), ft.value), reverse=True)
        for ft in candidates:
            if ctx.player.food_supply.spend(ft, 1):
                return True
        return False

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        discarded = 0
        for _ in range(self.max_discard):
            if not self._spend_one(ctx):
                break
            discarded += 1
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
        if discarded == 0:
            return PowerResult(executed=False, description="No discardable food")
        return PowerResult(cards_tucked=tucked, description=f"Discarded {discarded}, tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class DrawThenTuckFromHand(PowerEffect):
    """Draw cards to hand, then tuck up to N cards from hand behind this bird."""

    def __init__(self, draw_count: int, max_tuck: int):
        self.draw_count = draw_count
        self.max_tuck = max_tuck

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        drawn = _draw_many_birds(ctx, self.draw_count)
        tucked = min(self.max_tuck, len(ctx.player.hand))
        for _ in range(tucked):
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            slot.tucked_cards += 1
        return PowerResult(cards_drawn=len(drawn), cards_tucked=tucked, description=f"Drew {len(drawn)} and tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8


class TuckFromHandThenDrawEqual(PowerEffect):
    """Tuck up to N cards from hand, then draw an equal number of cards."""

    def __init__(self, max_tuck: int):
        self.max_tuck = max_tuck

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(self.max_tuck, len(ctx.player.hand))
        for _ in range(tucked):
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            slot.tucked_cards += 1
        drawn = _draw_many_birds(ctx, tucked)
        if tucked == 0:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        return PowerResult(cards_tucked=tucked, cards_drawn=len(drawn), description=f"Tucked {tucked}, drew {len(drawn)}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.7


class DrawKeepTuckDiscard(PowerEffect):
    """Draw N, keep K in hand, tuck T on this bird, discard the rest."""

    def __init__(self, draw_count: int = 5, keep_count: int = 1, tuck_count: int = 1):
        self.draw_count = draw_count
        self.keep_count = keep_count
        self.tuck_count = tuck_count

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        seen = []
        for _ in range(self.draw_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            seen.append(card)
        if not seen:
            return PowerResult(executed=False, description="No cards drawn")

        seen.sort(key=lambda b: b.victory_points, reverse=True)
        keep = seen[: self.keep_count]
        rest = seen[self.keep_count:]
        for card in keep:
            ctx.player.hand.append(card)

        tucked = min(self.tuck_count, len(rest))
        for _ in range(tucked):
            rest.pop(0)
            slot.tucked_cards += 1

        ctx.game_state.discard_pile_count += len(rest)
        return PowerResult(cards_drawn=len(keep), cards_tucked=tucked, description="Resolved draw/keep/tuck/discard")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.9


class PerThreeEggsInHabitatDrawThenTuck(PowerEffect):
    """For every 3 eggs in a habitat, draw cards then tuck up to a limit from hand."""

    def __init__(self, habitat: Habitat, max_tuck: int):
        self.habitat = habitat
        self.max_tuck = max_tuck

    def execute(self, ctx: PowerContext) -> PowerResult:
        eggs = ctx.player.board.get_row(self.habitat).total_eggs()
        draw_count = eggs // 3
        if draw_count <= 0:
            return PowerResult(executed=False, description="Not enough eggs for conversion")
        drawn = _draw_many_birds(ctx, draw_count)
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(self.max_tuck, len(ctx.player.hand))
        for _ in range(tucked):
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            slot.tucked_cards += 1
        return PowerResult(cards_drawn=len(drawn), cards_tucked=tucked, description=f"Drew {len(drawn)}, tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5


class PerThreeEggsInHabitatGainSeedOrInvertThenCache(PowerEffect):
    """For every 3 eggs in a habitat, gain seed/invertebrate and optionally cache up to N."""

    def __init__(self, habitat: Habitat, max_cache: int = 2):
        self.habitat = habitat
        self.max_cache = max_cache

    def execute(self, ctx: PowerContext) -> PowerResult:
        eggs = ctx.player.board.get_row(self.habitat).total_eggs()
        gain_count = eggs // 3
        if gain_count <= 0:
            return PowerResult(executed=False, description="Not enough eggs for conversion")

        gained: dict[FoodType, int] = {}
        choices = [FoodType.INVERTEBRATE, FoodType.SEED]
        for _ in range(gain_count):
            ft = min(choices, key=lambda f: (ctx.player.food_supply.get(f), f.value))
            ctx.player.food_supply.add(ft, 1)
            gained[ft] = gained.get(ft, 0) + 1

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        to_cache = min(self.max_cache, gain_count)
        cached: dict[FoodType, int] = {}
        for ft in (FoodType.INVERTEBRATE, FoodType.SEED):
            if to_cache <= 0:
                break
            available = min(gained.get(ft, 0), to_cache)
            if available <= 0:
                continue
            if ctx.player.food_supply.spend(ft, available):
                slot.cache_food(ft, available)
                cached[ft] = cached.get(ft, 0) + available
                to_cache -= available

        return PowerResult(food_gained=gained, food_cached=cached, description=f"Gained {gain_count} then cached up to {self.max_cache}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5


class CacheRodentPerPredator(PowerEffect):
    """Choose a player; cache 1 rodent for each predator they have."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        players = list(ctx.game_state.players)
        if not players:
            return PowerResult(executed=False, description="No players found")

        def predator_count(player) -> int:
            count = 0
            for _, _, slot in player.board.all_slots():
                if slot.bird and slot.bird.is_predator:
                    count += 1
            return count

        target = max(players, key=lambda p: (predator_count(p), p.name))
        count = predator_count(target)
        if count <= 0:
            return PowerResult(executed=False, description="No predator birds to count")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(FoodType.RODENT, count)
        return PowerResult(food_cached={FoodType.RODENT: count}, description=f"Cached {count} rodent(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class LayEggOnSelfPerOpponentGrasslandCubes(PowerEffect):
    """Choose another player; lay 1 egg on this bird per grassland action cube used."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        target_player = _select_grassland_target_player(ctx)
        if target_player is None:
            return PowerResult(executed=False, description="No opponent")
        count = target_player.lay_eggs_actions_this_round
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if count <= 0 or not target.bird:
            return PowerResult(executed=False, description="No grassland cubes to convert")
        laid = min(count, target.eggs_space())
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) from opponent cubes")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class CacheWildOnAnyBirdPerOpponentGrasslandCubes(PowerEffect):
    """Choose another player; cache 1 wild on any of your birds per grassland action cube used."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        target_player = _select_grassland_target_player(ctx)
        if target_player is None:
            return PowerResult(executed=False, description="No opponent")
        count = target_player.lay_eggs_actions_this_round
        if count <= 0:
            return PowerResult(executed=False, description="No grassland cubes to convert")

        own_slots = [slot for _, _, slot in ctx.player.board.all_slots() if slot.bird]
        if not own_slots:
            return PowerResult(executed=False, description="No birds to cache on")
        preferred = own_slots[0]
        cached: dict[FoodType, int] = {}
        for _ in range(count):
            ft = _choose_wild_food_from_supply(ctx)
            preferred.cache_food(ft, 1)
            cached[ft] = cached.get(ft, 0) + 1
        return PowerResult(food_cached=cached, description=f"Cached {count} wild food token(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class TuckHandThenDrawEqualPerOpponentGrasslandCubes(PowerEffect):
    """Choose another player; for each grassland cube, tuck 1 from hand then draw 1."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        target_player = _select_grassland_target_player(ctx)
        if target_player is None:
            return PowerResult(executed=False, description="No opponent")
        limit = target_player.lay_eggs_actions_this_round
        if limit <= 0:
            return PowerResult(executed=False, description="No grassland cubes to convert")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(limit, len(ctx.player.hand))
        for _ in range(tucked):
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            slot.tucked_cards += 1
        drawn = _draw_many_birds(ctx, tucked)
        if tucked == 0:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        return PowerResult(cards_tucked=tucked, cards_drawn=len(drawn), description=f"Tucked {tucked}, drew {len(drawn)}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class CacheWildOnEachBirdInRow(PowerEffect):
    """Cache 1 wild on each bird in this row."""

    def __init__(self, include_self: bool):
        self.include_self = include_self

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        cached: dict[FoodType, int] = {}
        total = 0
        for idx, slot in enumerate(row.slots):
            if not slot.bird:
                continue
            if not self.include_self and idx == ctx.slot_index:
                continue
            ft = _choose_wild_food_from_supply(ctx)
            slot.cache_food(ft, 1)
            cached[ft] = cached.get(ft, 0) + 1
            total += 1
        if total == 0:
            return PowerResult(executed=False, description="No eligible birds in row")
        return PowerResult(food_cached=cached, description=f"Cached {total} food token(s) in row")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.1


class CacheWildOnEachOtherBird(PowerEffect):
    """Cache 1 wild on each of your other birds."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        cached: dict[FoodType, int] = {}
        total = 0
        for hab, idx, slot in ctx.player.board.all_slots():
            if not slot.bird:
                continue
            if hab == ctx.habitat and idx == ctx.slot_index:
                continue
            ft = _choose_wild_food_from_supply(ctx)
            slot.cache_food(ft, 1)
            cached[ft] = cached.get(ft, 0) + 1
            total += 1
        if total == 0:
            return PowerResult(executed=False, description="No other birds to cache on")
        return PowerResult(food_cached=cached, description=f"Cached {total} food token(s) on other birds")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class PhilippineEagleRodentBonus(PowerEffect):
    """Roll 5 dice; up to 2 rerolls. If >=3 rodent showing, draw 2 bonus and keep 1; then reset feeder."""

    @staticmethod
    def _rodent_count(dice: list) -> int:
        count = 0
        for die in dice:
            if die == FoodType.RODENT:
                count += 1
            elif isinstance(die, tuple) and FoodType.RODENT in die:
                count += 1
        return count

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()
        for _ in range(2):
            if self._rodent_count(feeder.dice) >= 3:
                break
            new_dice = []
            for die in feeder.dice:
                has_rodent = die == FoodType.RODENT or (isinstance(die, tuple) and FoodType.RODENT in die)
                if has_rodent:
                    new_dice.append(die)
                else:
                    new_dice.append(random.choice(feeder.dice_faces))
            feeder.dice = new_dice

        out = PowerResult(description="Philippine Eagle resolved")
        if self._rodent_count(feeder.dice) >= 3:
            from backend.powers.templates.draw_cards import DrawBonusCards
            bonus = DrawBonusCards(draw=2, keep=1).execute(ctx)
            out = _merge_power_results(out, bonus)

        feeder.reroll()
        return out

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


def _iter_birds_in_row_with_indices(player, habitat: Habitat):
    row = player.board.get_row(habitat)
    for idx, slot in enumerate(row.slots):
        if slot.bird:
            yield idx, slot


def _remove_eggs_from_habitat(player, habitat: Habitat, count: int) -> bool:
    if count <= 0:
        return True
    row = player.board.get_row(habitat)
    candidates = [(slot.eggs, idx) for idx, slot in enumerate(row.slots) if slot.bird and slot.eggs > 0]
    candidates.sort(reverse=True)
    remaining = count
    for _, idx in candidates:
        if remaining <= 0:
            break
        slot = row.slots[idx]
        take = min(slot.eggs, remaining)
        slot.eggs -= take
        remaining -= take
    return remaining == 0


def _lay_eggs_on_matching_nest(player, nest_types: set[NestType]) -> int:
    laid = 0
    for _, _, slot in player.board.all_slots():
        if not slot.bird or not slot.can_hold_more_eggs():
            continue
        nest = slot.bird.nest_type
        # Wild nests count as any nest type for matching conditions.
        if nest in nest_types or (nest == NestType.WILD and NestType.WILD not in nest_types):
            slot.eggs += 1
            laid += 1
    return laid


def _lay_eggs_on_matching_wingspan(player, *, min_cm: int | None = None, max_cm: int | None = None) -> int:
    laid = 0
    for _, _, slot in player.board.all_slots():
        if not slot.bird or not slot.can_hold_more_eggs():
            continue
        wingspan = slot.bird.wingspan_cm
        if wingspan is None:
            continue
        if min_cm is not None and wingspan <= min_cm:
            continue
        if max_cm is not None and wingspan >= max_cm:
            continue
        slot.eggs += 1
        laid += 1
    return laid


def _longest_contiguous_group_same_nest(player):
    best: list[tuple[Habitat, int]] = []

    def compatible(slot, target: NestType) -> bool:
        if not slot.bird:
            return False
        return slot.bird.nest_type == target or slot.bird.nest_type == NestType.WILD

    targets = [NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM]

    # Scan each habitat row.
    for habitat in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
        row = player.board.get_row(habitat)
        for target in targets:
            start = 0
            while start < len(row.slots):
                while start < len(row.slots) and not compatible(row.slots[start], target):
                    start += 1
                end = start
                while end < len(row.slots) and compatible(row.slots[end], target):
                    end += 1
                group = [(habitat, idx) for idx in range(start, end) if row.slots[idx].bird]
                if len(group) > len(best):
                    best = group
                start = end + 1

    # Scan columns.
    for col in range(5):
        for target in targets:
            column_slots = [
                (Habitat.FOREST, player.board.forest.slots[col]),
                (Habitat.GRASSLAND, player.board.grassland.slots[col]),
                (Habitat.WETLAND, player.board.wetland.slots[col]),
            ]
            start = 0
            while start < len(column_slots):
                while start < len(column_slots) and not compatible(column_slots[start][1], target):
                    start += 1
                end = start
                while end < len(column_slots) and compatible(column_slots[end][1], target):
                    end += 1
                group = [(column_slots[idx][0], col) for idx in range(start, end) if column_slots[idx][1].bird]
                if len(group) > len(best):
                    best = group
                start = end + 1

    return best


class EndGameTuckPerBirdInHabitat(PowerEffect):
    """Tuck one from deck behind this bird for each bird in a habitat row."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        count = sum(1 for _ in _iter_birds_in_row_with_indices(ctx.player, self.habitat))
        if count <= 0:
            return PowerResult(executed=False, description="No birds in target habitat")
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        for _ in range(count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
        return PowerResult(cards_tucked=tucked, description=f"Tucked {tucked} cards from deck")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class EndGameDiscardEggsRowColumnCacheSeeds(PowerEffect):
    """Discard eggs on row/column (excluding this bird); cache 2 seed per discarded egg."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        own_row = ctx.player.board.get_row(ctx.habitat)
        affected: set[tuple[Habitat, int]] = set()
        for idx in range(len(own_row.slots)):
            if idx != ctx.slot_index:
                affected.add((ctx.habitat, idx))
        for habitat in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            if habitat != ctx.habitat:
                affected.add((habitat, ctx.slot_index))

        discarded = 0
        for habitat, idx in affected:
            slot = ctx.player.board.get_row(habitat).slots[idx]
            if slot.bird and slot.eggs > 0:
                slot.eggs -= 1
                discarded += 1

        if discarded <= 0:
            return PowerResult(executed=False, description="No eggs to discard")

        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        cache_count = discarded * 2
        target.cache_food(FoodType.SEED, cache_count)
        return PowerResult(food_cached={FoodType.SEED: cache_count}, description=f"Discarded {discarded} eggs; cached {cache_count} seed")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5


class EndGameCacheFromSupplyOnSelf(PowerEffect):
    """Cache up to N wild food from supply on this bird."""

    def __init__(self, max_count: int):
        self.max_count = max_count

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        cached: dict[FoodType, int] = {}
        remaining = self.max_count
        foods = _non_wild_food_types_for_board(ctx.game_state.board_type)
        while remaining > 0:
            available = [ft for ft in foods if ctx.player.food_supply.get(ft) > 0]
            if not available:
                break
            ft = max(available, key=lambda f: (ctx.player.food_supply.get(f), f.value))
            if not ctx.player.food_supply.spend(ft, 1):
                break
            target.cache_food(ft, 1)
            cached[ft] = cached.get(ft, 0) + 1
            remaining -= 1
        total = sum(cached.values())
        if total <= 0:
            return PowerResult(executed=False, description="No food available to cache")
        return PowerResult(food_cached=cached, description=f"Cached {total} food from supply")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class EndGameLayEggOnMatchingNest(PowerEffect):
    """Lay one egg on each bird matching nest criteria."""

    def __init__(self, nest_types: set[NestType]):
        self.nest_types = nest_types

    def execute(self, ctx: PowerContext) -> PowerResult:
        laid = _lay_eggs_on_matching_nest(ctx.player, self.nest_types)
        if laid <= 0:
            return PowerResult(executed=False, description="No matching birds with egg space")
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} eggs on matching nests")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class EndGameLayEggOnMatchingWingspan(PowerEffect):
    """Lay one egg on each bird matching wingspan threshold."""

    def __init__(self, *, min_cm: int | None = None, max_cm: int | None = None):
        self.min_cm = min_cm
        self.max_cm = max_cm

    def execute(self, ctx: PowerContext) -> PowerResult:
        laid = _lay_eggs_on_matching_wingspan(ctx.player, min_cm=self.min_cm, max_cm=self.max_cm)
        if laid <= 0:
            return PowerResult(executed=False, description="No matching birds with egg space")
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} wingspan eggs")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class EndGameLayEggOnEachBirdInHabitat(PowerEffect):
    """Lay one egg on each bird in habitat."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        laid = 0
        for _, slot in _iter_birds_in_row_with_indices(ctx.player, self.habitat):
            if slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
        if laid <= 0:
            return PowerResult(executed=False, description="No birds with egg space in habitat")
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} eggs in {self.habitat.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.3


class EndGameLayEggOnSelfPerTwoEggsInHabitat(PowerEffect):
    """For every 2 eggs in habitat, lay 1 egg on this bird."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        source_eggs = ctx.player.board.get_row(self.habitat).total_eggs()
        count = source_eggs // 2
        if count <= 0:
            return PowerResult(executed=False, description="Not enough source eggs")
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        laid = min(count, target.eggs_space())
        if laid <= 0:
            return PowerResult(executed=False, description="No egg space on this bird")
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on this bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class EndGameLayContiguousSameNest(PowerEffect):
    """Lay 1 egg on a contiguous group of same-nest birds (wild matches any)."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        group = _longest_contiguous_group_same_nest(ctx.player)
        if not group:
            return PowerResult(executed=False, description="No contiguous nest group found")
        laid = 0
        for habitat, idx in group:
            slot = ctx.player.board.get_row(habitat).slots[idx]
            if slot.bird and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
        if laid <= 0:
            return PowerResult(executed=False, description="No egg space in contiguous group")
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} contiguous nest egg(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.3


class EndGameDrawBonusKeep(PowerEffect):
    """Draw N bonus cards, keep K."""

    def __init__(self, draw: int, keep: int):
        self.draw = draw
        self.keep = keep

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.templates.draw_cards import DrawBonusCards

        result = DrawBonusCards(draw=self.draw, keep=self.keep).execute(ctx)
        if not result.executed:
            return result
        result.description = f"Drew {self.draw} bonus cards, kept {self.keep}"
        return result

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8


class EndGamePlayBird(PowerEffect):
    """Play one bird at game end with optional habitat filter and egg discount."""

    def __init__(self, habitat_filter: Habitat | None = None, egg_discount: int = 0):
        self.habitat_filter = habitat_filter
        self.egg_discount = egg_discount

    def execute(self, ctx: PowerContext) -> PowerResult:
        return _play_bonus_bird(
            ctx,
            habitat_filter=self.habitat_filter,
            egg_discount=self.egg_discount,
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.2


class EndGameMagpieLark(PowerEffect):
    """Discard 2 eggs from forest, then play 1 bird in grassland (ignore egg cost)."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not _remove_eggs_from_habitat(ctx.player, Habitat.FOREST, 2):
            return PowerResult(executed=False, description="Could not discard 2 eggs from forest")
        out = _play_bonus_bird(
            ctx,
            habitat_filter=Habitat.GRASSLAND,
            egg_discount=99,
        )
        if out.executed:
            out.description = "Discarded 2 forest eggs and played 1 grassland bird"
            return out
        return PowerResult(executed=False, description="Discarded eggs but no playable grassland bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.4


def _slot_can_take_egg(slot, extra_over_limit: int = 0) -> bool:
    if slot.bird is None:
        return False
    return slot.eggs < (slot.bird.egg_limit + max(0, extra_over_limit))


def _pink_common_trigger_guard(ctx: PowerContext) -> bool:
    return ctx.trigger_player is not None and ctx.trigger_player.name != ctx.player.name


class PinkLayEggOnTrigger(PowerEffect):
    """Pink: when another player takes lay-eggs action, lay 1 egg on matching bird."""

    def __init__(
        self,
        *,
        nest_types: set[NestType] | None = None,
        wingspan_lt: int | None = None,
        on_self: bool = False,
        require_another_bird: bool = False,
        target_overage: int = 0,
    ):
        self.nest_types = nest_types
        self.wingspan_lt = wingspan_lt
        self.on_self = on_self
        self.require_another_bird = require_another_bird
        self.target_overage = target_overage

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        return ctx.trigger_action == ActionType.LAY_EGGS

    def _eligible(self, ctx: PowerContext, hab: Habitat, idx: int, slot) -> bool:
        if slot.bird is None:
            return False
        if self.on_self:
            return hab == ctx.habitat and idx == ctx.slot_index
        if self.require_another_bird and hab == ctx.habitat and idx == ctx.slot_index:
            return False
        if self.nest_types is not None:
            if slot.bird.nest_type not in self.nest_types and slot.bird.nest_type != NestType.WILD:
                return False
        if self.wingspan_lt is not None:
            wingspan = slot.bird.wingspan_cm
            if wingspan is None or wingspan >= self.wingspan_lt:
                return False
        return True

    def execute(self, ctx: PowerContext) -> PowerResult:
        candidates: list[tuple[int, Habitat, int, object]] = []
        for hab, idx, slot in ctx.player.board.all_slots():
            if not self._eligible(ctx, hab, idx, slot):
                continue
            overage = self.target_overage if not self.on_self else 0
            if not _slot_can_take_egg(slot, overage):
                continue
            space = (slot.bird.egg_limit + overage) - slot.eggs
            candidates.append((space, hab, idx, slot))
        if not candidates:
            return PowerResult(executed=False, description="No eligible target for pink egg")
        # Deterministic: pick target with most space.
        candidates.sort(key=lambda t: (t[0], t[1].value, t[2]), reverse=True)
        _, hab, idx, slot = candidates[0]
        slot.eggs += 1
        target_desc = "this bird" if (hab == ctx.habitat and idx == ctx.slot_index) else "another bird"
        return PowerResult(eggs_laid=1, description=f"Laid 1 egg on {target_desc}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class PinkGainFoodFromSupplyOnPlayInHabitat(PowerEffect):
    """Pink: when another player plays in habitat, gain 1 food from supply."""

    def __init__(self, trigger_habitat: Habitat, food_type: FoodType):
        self.trigger_habitat = trigger_habitat
        self.food_type = food_type

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        if ctx.trigger_action != ActionType.PLAY_BIRD:
            return False
        played = (ctx.trigger_meta or {}).get("played_habitat")
        return played == self.trigger_habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        ctx.player.food_supply.add(self.food_type, 1)
        return PowerResult(food_gained={self.food_type: 1}, description=f"Gained 1 {self.food_type.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class PinkGainFoodFromFeederOnGainFood(PowerEffect):
    """Pink: when another player gains food, take one matching die from feeder."""

    def __init__(self, food_types: list[FoodType]):
        self.food_types = list(food_types)

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        return ctx.trigger_action == ActionType.GAIN_FOOD

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        for ft in self.food_types:
            if feeder.take_food(ft):
                ctx.player.food_supply.add(ft, 1)
                return PowerResult(food_gained={ft: 1}, description=f"Gained 1 {ft.value} from feeder")
        return PowerResult(executed=False, description="No matching feeder die")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class PinkPredatorSuccessGainDie(PowerEffect):
    """Pink: when another player's predator succeeds, gain one die from feeder."""

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        meta = ctx.trigger_meta or {}
        return int(meta.get("predator_successes", 0) or 0) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        choices = sorted(feeder.available_food_types(), key=lambda ft: ft.value)
        if not choices:
            return PowerResult(executed=False, description="No die available in feeder")
        # Prefer non-nectar to preserve nectar scarcity signal in Oceania.
        choices.sort(key=lambda ft: (ft == FoodType.NECTAR, ft.value))
        for ft in choices:
            if feeder.take_food(ft):
                ctx.player.food_supply.add(ft, 1)
                return PowerResult(food_gained={ft: 1}, description="Gained 1 die from feeder")
        return PowerResult(executed=False, description="Failed to take feeder die")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class PinkCacheRodentOnOpponentRodentGain(PowerEffect):
    """Pink: when another player gains rodent via gain-food action, cache 1 rodent."""

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        if ctx.trigger_action != ActionType.GAIN_FOOD:
            return False
        food = (ctx.trigger_meta or {}).get("food_gained", {})
        return int(food.get(FoodType.RODENT, 0) or 0) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(FoodType.RODENT, 1)
        return PowerResult(food_cached={FoodType.RODENT: 1}, description="Cached 1 rodent")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class PinkTuckFromDeckOnOpponentTuck(PowerEffect):
    """Pink: when another player tucks, tuck 1 from deck behind this bird."""

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        return int((ctx.trigger_meta or {}).get("cards_tucked", 0) or 0) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        card = _draw_one_bird(ctx)
        if card is None:
            return PowerResult(executed=False, description="Deck empty")
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description="Tucked 1 from deck")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class PinkTuckFromHandOnPlayInHabitat(PowerEffect):
    """Pink: when another player plays in habitat, tuck 1 from hand."""

    def __init__(self, trigger_habitat: Habitat):
        self.trigger_habitat = trigger_habitat

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        if ctx.trigger_action != ActionType.PLAY_BIRD:
            return False
        played = (ctx.trigger_meta or {}).get("played_habitat")
        if played != self.trigger_habitat:
            return False
        return len(ctx.player.hand) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
        ctx.player.hand.pop(worst_idx)
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description="Tucked 1 from hand")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class PinkTuckFromHandThenDrawOnOpponentTuck(PowerEffect):
    """Pink: when another player tucks, tuck 1 from hand and draw 1."""

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        if int((ctx.trigger_meta or {}).get("cards_tucked", 0) or 0) <= 0:
            return False
        return len(ctx.player.hand) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
        ctx.player.hand.pop(worst_idx)
        slot.tucked_cards += 1
        drawn = _draw_many_birds(ctx, 1)
        return PowerResult(cards_tucked=1, cards_drawn=len(drawn), description="Tucked 1 from hand, drew 1")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class PinkGainNectarOnOpponentNectarGain(PowerEffect):
    """Pink: when another player gains nectar, gain 1 nectar."""

    def can_execute(self, ctx: PowerContext) -> bool:
        if not _pink_common_trigger_guard(ctx):
            return False
        return int((ctx.trigger_meta or {}).get("nectar_gained", 0) or 0) > 0

    def execute(self, ctx: PowerContext) -> PowerResult:
        ctx.player.food_supply.add(FoodType.NECTAR, 1)
        return PowerResult(food_gained={FoodType.NECTAR: 1}, description="Gained 1 nectar")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7
