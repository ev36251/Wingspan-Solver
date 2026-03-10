"""Strict, per-card power implementations for high-fidelity rules paths."""

from __future__ import annotations

import random

from backend.models.enums import ActionType, BoardType, FoodType, Habitat, NestType
from backend.powers.base import PowerContext, PowerEffect, PowerResult
from backend.powers.choices import consume_power_activation_decision, consume_power_choice


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


def _schedule_end_turn_hand_discard(
    ctx: PowerContext,
    count: int,
    preferred_names: list[str] | None = None,
) -> None:
    if count <= 0:
        return
    pending = getattr(ctx.game_state, "pending_end_turn_hand_discards", None)
    if not isinstance(pending, dict):
        pending = {}
        setattr(ctx.game_state, "pending_end_turn_hand_discards", pending)
    pending[ctx.player.name] = int(pending.get(ctx.player.name, 0)) + int(count)

    if preferred_names:
        name_map = getattr(ctx.game_state, "pending_end_turn_discard_names", None)
        if not isinstance(name_map, dict):
            name_map = {}
            setattr(ctx.game_state, "pending_end_turn_discard_names", name_map)
        queue = name_map.setdefault(ctx.player.name, [])
        queue.extend(str(n) for n in preferred_names if isinstance(n, str) and n)


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


def _best_bonus_play_candidate(
    ctx: PowerContext,
    habitat_filter: Habitat | None = None,
    egg_discount: int = 0,
    *,
    allow_nectar: bool = True,
):
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
            if not allow_nectar:
                payment_options = [p for p in payment_options if int(p.get(FoodType.NECTAR, 0) or 0) == 0]
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


def _play_bonus_bird(
    ctx: PowerContext,
    habitat_filter: Habitat | None = None,
    egg_discount: int = 0,
    *,
    allow_nectar: bool = True,
) -> PowerResult:
    from backend.powers.base import NoPower
    from backend.powers.registry import get_power, assert_power_allowed_for_strict_mode

    candidate = _best_bonus_play_candidate(
        ctx,
        habitat_filter=habitat_filter,
        egg_discount=egg_discount,
        allow_nectar=allow_nectar,
    )
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


class DrawThenEndTurnDiscardFromHand(PowerEffect):
    """Draw N now, then discard M from hand at end of turn."""

    def __init__(self, draw: int, discard: int):
        self.draw = int(max(0, draw))
        self.discard = int(max(0, discard))

    def execute(self, ctx: PowerContext) -> PowerResult:
        drew = _draw_many_birds(ctx, self.draw)
        if drew and self.discard > 0:
            choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
            discard_names = list(choice.get("discard_names", [])) if isinstance(choice, dict) else []
            _schedule_end_turn_hand_discard(ctx, self.discard, preferred_names=discard_names)
        return PowerResult(
            cards_drawn=len(drew),
            description=f"Drew {len(drew)}; discard {self.discard} at end of turn",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return max(0.0, (self.draw - self.discard) * 0.4)


class DrawPerEmptySlotThenKeepOneAtEndTurn(PowerEffect):
    """Draw 1 per empty slot in this row; keep 1 and discard the rest at end of turn."""

    def __init__(self, keep: int = 1):
        self.keep = int(max(0, keep))

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        draw = sum(1 for slot in row.slots if slot.is_available)
        drew = _draw_many_birds(ctx, draw)
        discard_later = max(0, len(drew) - self.keep)
        if discard_later > 0:
            _schedule_end_turn_hand_discard(ctx, discard_later)
        return PowerResult(
            cards_drawn=len(drew),
            description=f"Drew {len(drew)}; keep {self.keep} at end of turn",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        row = ctx.player.board.get_row(ctx.habitat)
        draw = sum(1 for slot in row.slots if slot.is_available)
        return max(0.0, min(self.keep, draw) * 0.4)


class DrawPerColumnBirdWithEggKeepOne(PowerEffect):
    """For each bird in this column with an egg, draw 1 card; keep 1 and discard the rest."""

    def __init__(self, keep: int = 1):
        self.keep = int(max(0, keep))

    def execute(self, ctx: PowerContext) -> PowerResult:
        draw_count = 0
        for row in ctx.player.board.all_rows():
            slot = row.slots[ctx.slot_index]
            if slot.bird and slot.eggs > 0:
                draw_count += 1

        seen = []
        for _ in range(draw_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            seen.append(card)

        if not seen:
            return PowerResult(executed=False, description="No cards drawn")

        keep_n = min(self.keep, len(seen))
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        keep_name = choice.get("keep_name") if isinstance(choice, dict) else None

        kept = []
        remaining = list(seen)
        if isinstance(keep_name, str) and keep_name:
            idx = next((i for i, c in enumerate(remaining) if c.name == keep_name), None)
            if idx is not None:
                kept.append(remaining.pop(idx))

        if len(kept) < keep_n:
            remaining.sort(key=lambda c: (c.victory_points, c.name), reverse=True)
            need = keep_n - len(kept)
            kept.extend(remaining[:need])
            keep_ids = {id(c) for c in kept}
            remaining = [c for c in remaining if id(c) not in keep_ids]

        for card in kept:
            ctx.player.hand.append(card)

        discarded = max(0, len(seen) - len(kept))
        if discarded > 0:
            ctx.game_state.discard_pile_count += discarded

        return PowerResult(
            cards_drawn=len(kept),
            description=f"Drew {len(seen)} from egged column birds, kept {len(kept)}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        draw_count = 0
        for row in ctx.player.board.all_rows():
            slot = row.slots[ctx.slot_index]
            if slot.bird and slot.eggs > 0:
                draw_count += 1
        return max(0.0, min(self.keep, draw_count) * 0.45)


class AllPlayersLayEggsSelfBonus(PowerEffect):
    """All players lay X eggs; active player lays +Y additional eggs."""

    def __init__(
        self,
        all_players_count: int,
        self_bonus_count: int,
        nest_filter: NestType | None = None,
        self_bonus_must_be_distinct_bird: bool = False,
    ):
        self.all_players_count = all_players_count
        self.self_bonus_count = self_bonus_count
        self.nest_filter = nest_filter
        self.self_bonus_must_be_distinct_bird = bool(self_bonus_must_be_distinct_bird)

    def _lay_any(
        self,
        player,
        count: int,
        *,
        exclude_slot_ids: set[int] | None = None,
        max_one_per_slot: bool = False,
    ) -> tuple[int, set[int]]:
        laid = 0
        used_slot_ids: set[int] = set()
        blocked = exclude_slot_ids or set()
        # Deterministic placement: most available capacity first.
        slots = []
        for _, _, slot in player.board.all_slots():
            if not slot.bird or not slot.can_hold_more_eggs():
                continue
            if self.nest_filter is not None and slot.bird.nest_type not in {self.nest_filter, NestType.WILD}:
                continue
            if id(slot) in blocked:
                continue
            slots.append((slot.eggs_space(), slot))
        slots.sort(key=lambda x: -x[0])
        for _, slot in slots:
            while laid < count and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1
                used_slot_ids.add(id(slot))
                if max_one_per_slot:
                    break
            if laid >= count:
                break
        return laid, used_slot_ids

    def execute(self, ctx: PowerContext) -> PowerResult:
        self_laid = 0
        self_base_used: set[int] = set()
        for p in ctx.game_state.players:
            laid, used = self._lay_any(p, self.all_players_count)
            if p.name == ctx.player.name:
                self_laid += laid
                self_base_used = used
        bonus_laid, _ = self._lay_any(
            ctx.player,
            self.self_bonus_count,
            exclude_slot_ids=self_base_used if self.self_bonus_must_be_distinct_bird else None,
            max_one_per_slot=self.self_bonus_must_be_distinct_bird,
        )
        self_laid += bonus_laid
        return PowerResult(eggs_laid=self_laid, description=f"Laid {self_laid} eggs (self)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.0


class LayEggOnSelfIfTotalEggsBelow(PowerEffect):
    """Lay eggs on this bird only if total eggs across all your birds is below threshold."""

    def __init__(self, threshold: int, count: int = 1):
        self.threshold = int(threshold)
        self.count = int(max(0, count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        total_eggs = ctx.player.board.total_eggs()
        if total_eggs >= self.threshold:
            return PowerResult(
                executed=False,
                description=f"Total eggs {total_eggs} is not below {self.threshold}",
            )

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if slot.bird is None:
            return PowerResult(executed=False, description="No bird in source slot")

        laid = 0
        while laid < self.count and slot.can_hold_more_eggs():
            slot.eggs += 1
            laid += 1

        if laid <= 0:
            return PowerResult(executed=False, description="No egg capacity on this bird")

        return PowerResult(
            eggs_laid=laid,
            description=(
                f"Laid {laid} egg(s) on this bird because total eggs "
                f"{total_eggs} was below {self.threshold}"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        if ctx.player.board.total_eggs() >= self.threshold:
            return 0.0
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if slot.bird is None:
            return 0.0
        return min(self.count, slot.eggs_space()) * 0.8


class AllPlayersGainFoodThenSelfFoodOrEggBonus(PowerEffect):
    """All players gain base food; active player gets an additional bonus."""

    def __init__(
        self,
        base_food_type: FoodType,
        base_count: int = 1,
        self_food_type: FoodType | None = None,
        self_food_count: int = 0,
        self_egg_bonus: int = 0,
    ):
        self.base_food_type = base_food_type
        self.base_count = int(max(0, base_count))
        self.self_food_type = self_food_type
        self.self_food_count = int(max(0, self_food_count))
        self.self_egg_bonus = int(max(0, self_egg_bonus))

    @staticmethod
    def _lay_any_eggs(player, count: int) -> int:
        laid = 0
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
        for p in ctx.game_state.players:
            p.food_supply.add(self.base_food_type, self.base_count)

        gained_actor = {self.base_food_type: self.base_count}
        eggs_laid = 0

        if self.self_food_type is not None and self.self_food_count > 0:
            ctx.player.food_supply.add(self.self_food_type, self.self_food_count)
            gained_actor[self.self_food_type] = gained_actor.get(self.self_food_type, 0) + self.self_food_count
        if self.self_egg_bonus > 0:
            eggs_laid = self._lay_any_eggs(ctx.player, self.self_egg_bonus)

        return PowerResult(
            food_gained=gained_actor,
            eggs_laid=eggs_laid,
            description="All players gained food; active player bonus applied",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class AllPlayersDrawCardsWithSelfBonus(PowerEffect):
    """All players draw N cards; active player draws additional cards, optional all-player food gain."""

    def __init__(
        self,
        all_players_draw: int,
        self_extra_draw: int = 0,
        all_players_food_type: FoodType | None = None,
        all_players_food_count: int = 0,
    ):
        self.all_players_draw = int(max(0, all_players_draw))
        self.self_extra_draw = int(max(0, self_extra_draw))
        self.all_players_food_type = all_players_food_type
        self.all_players_food_count = int(max(0, all_players_food_count))

    @staticmethod
    def _draw_for_player(ctx: PowerContext, player, count: int) -> int:
        drawn = 0
        for _ in range(max(0, count)):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            player.hand.append(card)
            drawn += 1
        return drawn

    def execute(self, ctx: PowerContext) -> PowerResult:
        actor_drawn = 0
        for p in ctx.game_state.players:
            d = self._draw_for_player(ctx, p, self.all_players_draw)
            if p.name == ctx.player.name:
                actor_drawn += d
            if self.all_players_food_type is not None and self.all_players_food_count > 0:
                p.food_supply.add(self.all_players_food_type, self.all_players_food_count)
        actor_drawn += self._draw_for_player(ctx, ctx.player, self.self_extra_draw)

        gained = {}
        if self.all_players_food_type is not None and self.all_players_food_count > 0:
            gained[self.all_players_food_type] = self.all_players_food_count
        return PowerResult(cards_drawn=actor_drawn, food_gained=gained, description="Resolved all-player draw bonus")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class AmericanOystercatcherDraftPlayersPlusOneClockwise(PowerEffect):
    """Draw players+1 cards, then draft clockwise from active player; active keeps extra."""

    @staticmethod
    def _clockwise_players(ctx: PowerContext) -> list:
        players = list(ctx.game_state.players)
        if not players:
            return []
        actor_idx = next((i for i, p in enumerate(players) if p.name == ctx.player.name), 0)
        return players[actor_idx:] + players[:actor_idx]

    @staticmethod
    def _pick_card(ctx: PowerContext, chooser, pool: list):
        choice = consume_power_choice(ctx.game_state, chooser.name, ctx.bird.name) or {}
        if isinstance(choice, dict):
            preferred_names: list[str] = []
            pick_name = choice.get("pick_name")
            pick_names = choice.get("pick_names")
            if isinstance(pick_name, str) and pick_name:
                preferred_names.append(pick_name)
            if isinstance(pick_names, list):
                preferred_names.extend(str(name) for name in pick_names if isinstance(name, str) and name)

            if preferred_names:
                lowered = [name.lower() for name in preferred_names]
                for wanted in lowered:
                    chosen = next((card for card in pool if card.name.lower() == wanted), None)
                    if chosen is not None:
                        return chosen

        # Deterministic fallback for both self and opponents.
        return max(
            pool,
            key=lambda card: (
                float(getattr(card, "draft_value_pct", 0.0) or 0.0),
                int(getattr(card, "victory_points", 0) or 0),
                card.name,
            ),
        )

    def execute(self, ctx: PowerContext) -> PowerResult:
        clockwise = self._clockwise_players(ctx)
        draw_count = len(clockwise) + 1
        pool = []
        for _ in range(max(0, draw_count)):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            pool.append(card)

        if not pool:
            return PowerResult(executed=False, description="No cards available to draft")

        actor_kept = 0
        for chooser in clockwise:
            if not pool:
                break
            chosen = self._pick_card(ctx, chooser, pool)
            pool.remove(chosen)
            chooser.hand.append(chosen)
            if chooser.name == ctx.player.name:
                actor_kept += 1

        if pool:
            extra = self._pick_card(ctx, ctx.player, pool)
            pool.remove(extra)
            ctx.player.hand.append(extra)
            actor_kept += 1

        if pool:
            ctx.game_state.discard_pile_count += len(pool)

        return PowerResult(
            cards_drawn=actor_kept,
            description=f"Drafted cards clockwise; active player kept {actor_kept}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.6


class TuckFromHandThenAllPlayersGainFood(PowerEffect):
    """Tuck 1 from hand; if tucked, all players gain specified food."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
        ctx.player.hand.pop(worst_idx)
        slot.tucked_cards += 1
        for p in ctx.game_state.players:
            p.food_supply.add(self.food_type, 1)
        return PowerResult(cards_tucked=1, food_gained={self.food_type: 1}, description="Tucked 1; all players gained food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class EachPlayerGainDieFromFeederStartingChoice(PowerEffect):
    """Each player gains 1 die from feeder, starting with a chosen player."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        if feeder.should_reroll():
            feeder.reroll()

        players = list(ctx.game_state.players)
        if not players:
            return PowerResult(executed=False, description="No players")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        start_name = choice.get("start_player") if isinstance(choice, dict) else None
        name_to_idx = {p.name: i for i, p in enumerate(players)}
        start_idx = name_to_idx.get(start_name, name_to_idx.get(ctx.player.name, 0))
        order = players[start_idx:] + players[:start_idx]

        actor_gained: dict[FoodType, int] = {}
        gained_total = 0
        for p in order:
            die = feeder.take_any()
            if die is None:
                continue
            ft = die if isinstance(die, FoodType) else die[0]
            p.food_supply.add(ft, 1)
            gained_total += 1
            if p.name == ctx.player.name:
                actor_gained[ft] = actor_gained.get(ft, 0) + 1
            if feeder.should_reroll():
                feeder.reroll()

        if gained_total <= 0:
            return PowerResult(executed=False, description="No food available in feeder")
        return PowerResult(
            food_gained=actor_gained,
            description=f"{gained_total} players gained 1 die from feeder",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


def _choose_wild_food_for_player(player, board_type: BoardType) -> FoodType:
    foods = _non_wild_food_types_for_board(board_type)
    return min(foods, key=lambda ft: (player.food_supply.get(ft), ft.value))


def _lay_any_egg_for_player(player, nest_filter: NestType | None = None) -> int:
    for _, _, slot in player.board.all_slots():
        if not slot.bird or not slot.can_hold_more_eggs():
            continue
        if nest_filter is not None and slot.bird.nest_type not in {nest_filter, NestType.WILD}:
            continue
        slot.eggs += 1
        return 1
    return 0


class AllPlayersMayDiscardEggFromHabitatForWild(PowerEffect):
    """Each player may discard 1 egg from a habitat bird to gain 1 wild-equivalent food."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        actor_gained: dict[FoodType, int] = {}
        for p in ctx.game_state.players:
            row = p.board.get_row(self.habitat)
            target_slot = next((s for s in row.slots if s.bird and s.eggs > 0), None)
            if target_slot is None:
                continue
            target_slot.eggs -= 1
            ft = _choose_wild_food_for_player(p, ctx.game_state.board_type)
            p.food_supply.add(ft, 1)
            if p.name == ctx.player.name:
                actor_gained[ft] = actor_gained.get(ft, 0) + 1
        if not actor_gained:
            return PowerResult(executed=False, description="No players discarded eggs")
        return PowerResult(food_gained=actor_gained, description="Players discarded eggs for wild food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class RollDiceCacheThenAllPlayersMayDiscardCardGainFood(PowerEffect):
    """Roll N dice, cache on hit, then players may discard 1 card to gain target food."""

    def __init__(self, dice_count: int, target_food: FoodType):
        self.dice_count = int(max(0, dice_count))
        self.target_food = target_food

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        hit = False
        if feeder.dice_faces:
            for _ in range(self.dice_count):
                face = random.choice(feeder.dice_faces)
                if face == self.target_food or (isinstance(face, tuple) and self.target_food in face):
                    hit = True
                    break

        cached = {}
        if hit:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(self.target_food, 1)
            cached[self.target_food] = 1

        gained_actor: dict[FoodType, int] = {}
        for p in ctx.game_state.players:
            activate = consume_power_activation_decision(ctx.game_state, p.name, ctx.bird.name)
            if activate is False:
                continue
            if not p.hand:
                continue

            choice = consume_power_choice(ctx.game_state, p.name, ctx.bird.name) or {}
            discard_name = choice.get("discard_name") if isinstance(choice, dict) else None
            if discard_name:
                idx = next((i for i, c in enumerate(p.hand) if c.name == discard_name), None)
                if idx is None:
                    idx = min(range(len(p.hand)), key=lambda i: p.hand[i].victory_points)
            else:
                idx = min(range(len(p.hand)), key=lambda i: p.hand[i].victory_points)

            p.hand.pop(idx)
            p.food_supply.add(self.target_food, 1)
            if p.name == ctx.player.name:
                gained_actor[self.target_food] = gained_actor.get(self.target_food, 0) + 1

        return PowerResult(food_cached=cached, food_gained=gained_actor, description="Resolved roll/cache/discard-card gain")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class AllPlayersMayDiscardFoodToLayEgg(PowerEffect):
    """Each player may discard one food token to lay one egg."""

    def __init__(self, discard_food_type: FoodType, nest_filter: NestType | None = None):
        self.discard_food_type = discard_food_type
        self.nest_filter = nest_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        actor_laid = 0
        for p in ctx.game_state.players:
            if not p.food_supply.spend(self.discard_food_type, 1):
                continue
            laid = _lay_any_egg_for_player(p, nest_filter=self.nest_filter)
            if laid == 0:
                # Refund if no legal egg target.
                p.food_supply.add(self.discard_food_type, 1)
                continue
            if p.name == ctx.player.name:
                actor_laid += laid
        if actor_laid == 0 and all(p.name == ctx.player.name or not p.food_supply.has(self.discard_food_type, 1) for p in ctx.game_state.players):
            return PowerResult(executed=False, description="No players discarded food")
        return PowerResult(eggs_laid=actor_laid, description="Players discarded food to lay eggs")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


def _auto_discard_two_resources_for_bonus(player, nectar_habitat: Habitat) -> bool:
    """Deterministically discard two resources (food/eggs/cards) if possible."""
    total_resources = int(player.food_supply.total()) + int(player.board.total_eggs()) + len(player.hand)
    if total_resources < 2:
        return False

    paid = 0

    # Prefer spending non-nectar food first.
    for ft in (
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
    ):
        while paid < 2 and player.food_supply.get(ft) > 0:
            player.food_supply.spend(ft, 1)
            paid += 1
        if paid >= 2:
            return True

    # Then discard eggs from birds with the most eggs.
    while paid < 2:
        best_slot = None
        best_eggs = 0
        for _, _, slot in player.board.all_slots():
            if slot.bird is None or slot.eggs <= 0:
                continue
            if slot.eggs > best_eggs:
                best_slot = slot
                best_eggs = slot.eggs
        if best_slot is None:
            break
        best_slot.eggs -= 1
        paid += 1
    if paid >= 2:
        return True

    # Then discard lowest-VP cards from hand.
    while paid < 2 and player.hand:
        worst_idx = min(range(len(player.hand)), key=lambda i: player.hand[i].victory_points)
        player.hand.pop(worst_idx)
        paid += 1
    if paid >= 2:
        return True

    # Finally, spend nectar and track it in the current habitat's spent nectar.
    row = player.board.get_row(nectar_habitat)
    while paid < 2 and player.food_supply.get(FoodType.NECTAR) > 0:
        if not player.food_supply.spend(FoodType.NECTAR, 1):
            break
        row.nectar_spent += 1
        paid += 1

    return paid >= 2


class SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo(PowerEffect):
    """Draw 2 bonus cards, keep 1; other players may discard 2 resources to do the same."""

    def __init__(self, draw: int = 2, keep: int = 1):
        self.draw = int(max(1, draw))
        self.keep = int(max(1, keep))

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.templates.draw_cards import DrawBonusCards

        actor_result = DrawBonusCards(draw=self.draw, keep=self.keep).execute(ctx)

        other_paid = 0
        for p in ctx.game_state.players:
            if p.name == ctx.player.name:
                continue

            activate = consume_power_activation_decision(
                ctx.game_state,
                p.name,
                ctx.bird.name,
            )
            if activate is False:
                continue
            if not _auto_discard_two_resources_for_bonus(p, ctx.habitat):
                continue

            other_paid += 1
            other_ctx = PowerContext(
                game_state=ctx.game_state,
                player=p,
                bird=ctx.bird,
                slot_index=ctx.slot_index,
                habitat=ctx.habitat,
            )
            DrawBonusCards(draw=self.draw, keep=self.keep).execute(other_ctx)

        return PowerResult(
            executed=bool(actor_result.executed or other_paid > 0),
            cards_drawn=actor_result.cards_drawn,
            description=(
                f"Drew {self.draw} bonus cards, kept {self.keep}; "
                f"{other_paid} other player(s) paid 2 resources to do the same"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.8


class KeaDrawBonusDiscardFoodForMoreKeepOne(PowerEffect):
    """Draw 1 bonus; may discard any food for more bonus draws; keep 1 total."""

    draw: int = 1
    keep: int = 1

    @staticmethod
    def _spend_one_food(ctx: PowerContext, preferred: FoodType | None = None) -> FoodType | None:
        foods = _non_wild_food_types_for_board(ctx.game_state.board_type)
        if preferred is not None and preferred in foods and ctx.player.food_supply.spend(preferred, 1):
            return preferred

        # Deterministic fallback: spend most abundant non-nectar first; nectar last.
        ordered = sorted(
            foods,
            key=lambda ft: (
                ft == FoodType.NECTAR,
                -ctx.player.food_supply.get(ft),
                ft.value,
            ),
        )
        for ft in ordered:
            if ctx.player.food_supply.spend(ft, 1):
                return ft
        return None

    def execute(self, ctx: PowerContext) -> PowerResult:
        from backend.powers.templates.draw_cards import _draw_one_bonus
        from backend.engine.scoring import score_bonus_cards

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        discard_food_count_raw = choice.get("discard_food_count") if isinstance(choice, dict) else None
        discard_food_types_raw = choice.get("discard_food_types") if isinstance(choice, dict) else None

        requested_count: int | None = None
        if discard_food_count_raw is not None:
            try:
                requested_count = max(0, int(discard_food_count_raw))
            except (TypeError, ValueError):
                requested_count = 0

        typed_requests: list[FoodType] = []
        if isinstance(discard_food_types_raw, list):
            for raw in discard_food_types_raw:
                ft = _parse_food_choice(str(raw)) if raw is not None else None
                if ft == FoodType.WILD:
                    ft = None
                typed_requests.append(ft)
            if requested_count is None:
                requested_count = len(typed_requests)

        if requested_count is None:
            requested_count = 0

        spent_foods: list[FoodType] = []
        # Spend explicitly requested food types first.
        for ft in typed_requests:
            if len(spent_foods) >= requested_count:
                break
            spent = self._spend_one_food(ctx, preferred=ft)
            if spent is None:
                break
            spent_foods.append(spent)

        # Fill remaining discards with deterministic fallback.
        while len(spent_foods) < requested_count:
            spent = self._spend_one_food(ctx, preferred=None)
            if spent is None:
                break
            spent_foods.append(spent)

        if spent_foods:
            row = ctx.player.board.get_row(ctx.habitat)
            row.nectar_spent += sum(1 for ft in spent_foods if ft == FoodType.NECTAR)

        drawn = []
        draw_total = 1 + len(spent_foods)
        for _ in range(draw_total):
            card = _draw_one_bonus(ctx.game_state)
            if card is None:
                break
            drawn.append(card)

        if not drawn:
            return PowerResult(executed=False, description="No bonus cards available")

        keep_name = None
        keep_names = None
        if isinstance(choice, dict):
            keep_name = choice.get("keep_bonus_name")
            keep_names = choice.get("keep_bonus_names")

        chosen = None
        preferred_names = []
        if isinstance(keep_name, str) and keep_name:
            preferred_names.append(keep_name)
        if isinstance(keep_names, list):
            preferred_names.extend(str(n) for n in keep_names if isinstance(n, str) and n)
        for name in preferred_names:
            chosen = next((bc for bc in drawn if bc.name == name), None)
            if chosen is not None:
                break

        if chosen is None:
            base_cards = list(ctx.player.bonus_cards)
            base_score = score_bonus_cards(ctx.player)
            best_key = None
            for bc in drawn:
                ctx.player.bonus_cards = base_cards + [bc]
                delta = score_bonus_cards(ctx.player) - base_score
                tie = float(getattr(bc, "draft_value_pct", 0.0) or 0.0)
                key = (delta, tie, bc.name)
                if best_key is None or key > best_key:
                    best_key = key
                    chosen = bc
            ctx.player.bonus_cards = base_cards

        if chosen is None:
            chosen = drawn[0]

        ctx.player.bonus_cards.append(chosen)
        discard_pile = getattr(ctx.game_state, "_bonus_discard_cards", None)
        if not isinstance(discard_pile, list):
            discard_pile = []
            setattr(ctx.game_state, "_bonus_discard_cards", discard_pile)
        discard_pile.extend([bc for bc in drawn if bc is not chosen])

        return PowerResult(
            cards_drawn=1,
            description=(
                f"Drew {len(drawn)} bonus cards (spent {len(spent_foods)} food), kept 1"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        food_total = ctx.player.food_supply.total()
        extra = min(3, max(0, food_total // 3))
        return 1.4 + 0.2 * extra


class GainSeedFromSupplyOrTuckFromDeck(PowerEffect):
    """Gain 1 seed from supply OR tuck 1 card from deck behind this bird."""

    def __init__(self, food_type: FoodType = FoodType.SEED):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        mode = choice.get("mode") if isinstance(choice, dict) else None

        can_tuck = ctx.game_state.deck_remaining > 0
        # Default deterministic behavior: tuck if possible; otherwise gain food.
        if mode not in {"gain", "tuck"}:
            mode = "tuck" if can_tuck else "gain"

        if mode == "tuck":
            card = _draw_one_bird(ctx)
            if card is not None:
                slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
                slot.tucked_cards += 1
                return PowerResult(cards_tucked=1, description="Tucked 1 card from deck")
            # If deck is empty, fallback to gain.

        ctx.player.food_supply.add(self.food_type, 1)
        return PowerResult(food_gained={self.food_type: 1}, description=f"Gained 1 {self.food_type.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class BlackNoddyResetFeederGainFishOptionalDiscardToTuck(PowerEffect):
    """Reset feeder, gain all fish, then optionally discard gained fish to tuck equal cards."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()

        gained_fish = 0
        while feeder.take_food(FoodType.FISH):
            gained_fish += 1

        if gained_fish > 0:
            ctx.player.food_supply.add(FoodType.FISH, gained_fish)

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested_discard = choice.get("discard_fish_for_tuck") if isinstance(choice, dict) else None
        try:
            requested_discard_int = int(requested_discard) if requested_discard is not None else None
        except (TypeError, ValueError):
            requested_discard_int = None

        discard_count = requested_discard_int if requested_discard_int is not None else 0
        discard_count = max(0, min(discard_count, gained_fish))

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        while tucked < discard_count:
            if not ctx.player.food_supply.spend(FoodType.FISH, 1):
                break
            card = _draw_one_bird(ctx)
            if card is None:
                ctx.player.food_supply.add(FoodType.FISH, 1)
                break
            slot.tucked_cards += 1
            tucked += 1

        kept_fish = max(0, gained_fish - tucked)
        food_gained = {FoodType.FISH: kept_fish} if kept_fish > 0 else {}
        return PowerResult(
            cards_tucked=tucked,
            food_gained=food_gained,
            description=f"Reset feeder; gained {gained_fish} fish, discarded {tucked} to tuck {tucked}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class ResetFeederGainAllMayCacheAny(PowerEffect):
    """Reset feeder, gain all of one food type, optionally cache any subset."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()

        gained = 0
        while feeder.take_food(self.food_type):
            gained += 1

        if gained <= 0:
            return PowerResult(description=f"Reset feeder; no {self.food_type.value} gained")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested_cache = choice.get("cache_count") if isinstance(choice, dict) else None
        try:
            requested_cache_int = int(requested_cache) if requested_cache is not None else None
        except (TypeError, ValueError):
            requested_cache_int = None

        # Default deterministic behavior: keep gained food in supply unless explicitly queued.
        cache_count = requested_cache_int if requested_cache_int is not None else 0
        cache_count = max(0, min(cache_count, gained))
        keep_count = gained - cache_count

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if cache_count > 0:
            slot.cache_food(self.food_type, cache_count)
        if keep_count > 0:
            ctx.player.food_supply.add(self.food_type, keep_count)

        return PowerResult(
            food_gained={self.food_type: keep_count} if keep_count > 0 else {},
            food_cached={self.food_type: cache_count} if cache_count > 0 else {},
            description=(
                f"Reset feeder; gained {gained} {self.food_type.value}, "
                f"cached {cache_count}, kept {keep_count}"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class ResetFeederGainOneFromOptions(PowerEffect):
    """Reset feeder, then gain/cache one food from a configured option set if available."""

    def __init__(self, food_types: list[FoodType], cache: bool = False):
        self.food_types = list(food_types)
        self.cache = bool(cache)

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()

        available = [ft for ft in self.food_types if feeder.can_take(ft)]
        if not available:
            return PowerResult(executed=False, description="No eligible food in feeder")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = _parse_food_choice(choice.get("food_type")) if isinstance(choice, dict) else None
        selected = preferred if preferred in available else min(
            available,
            key=lambda ft: (ctx.player.food_supply.get(ft), ft.value),
        )

        if not feeder.take_food(selected):
            return PowerResult(executed=False, description="Selected food no longer available")

        if self.cache:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(selected, 1)
            return PowerResult(food_cached={selected: 1}, description=f"Reset feeder; cached 1 {selected.value}")

        ctx.player.food_supply.add(selected, 1)
        return PowerResult(food_gained={selected: 1}, description=f"Reset feeder; gained 1 {selected.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class GreatCormorantMoveFishThenOptionalRoll(PowerEffect):
    """May move 1 cached fish to supply, then may roll 2 dice; on fish result cache 1 fish."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}

        requested_move = choice.get("move_fish_to_supply") if isinstance(choice, dict) else None
        requested_roll = choice.get("roll_dice") if isinstance(choice, dict) else None
        requested_indices = choice.get("roll_indices") if isinstance(choice, dict) else None

        # "You may move 1 fish..." => optional. Default deterministic behavior: move if available.
        do_move = bool(requested_move) if requested_move is not None else slot.cached_food.get(FoodType.FISH, 0) > 0
        moved = 0
        if do_move and slot.cached_food.get(FoodType.FISH, 0) > 0:
            slot.cached_food[FoodType.FISH] -= 1
            if slot.cached_food[FoodType.FISH] <= 0:
                slot.cached_food.pop(FoodType.FISH, None)
            ctx.player.food_supply.add(FoodType.FISH, 1)
            moved = 1

        # The roll step is also optional. Default deterministic behavior: roll if possible.
        if requested_roll is None:
            do_roll = len(feeder.dice) > 0
        else:
            do_roll = bool(requested_roll)

        rolled_any = 0
        hit_fish = False
        if do_roll and feeder.dice:
            dice_count = min(2, len(feeder.dice))
            indices: list[int] = []
            if isinstance(requested_indices, list):
                for raw in requested_indices:
                    try:
                        idx = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(feeder.dice) and idx not in indices:
                        indices.append(idx)
                    if len(indices) >= dice_count:
                        break
            if len(indices) < dice_count:
                for idx in range(len(feeder.dice)):
                    if idx not in indices:
                        indices.append(idx)
                    if len(indices) >= dice_count:
                        break

            for idx in indices[:dice_count]:
                face = random.choice(feeder.dice_faces)
                feeder.dice[idx] = face
                rolled_any += 1
                if face == FoodType.FISH or (isinstance(face, tuple) and FoodType.FISH in face):
                    hit_fish = True

        cached = 0
        if hit_fish:
            slot.cache_food(FoodType.FISH, 1)
            cached = 1

        executed = moved > 0 or rolled_any > 0
        if not executed:
            return PowerResult(executed=False, description="No fish moved and no dice rolled")

        return PowerResult(
            executed=True,
            food_gained={FoodType.FISH: moved} if moved > 0 else {},
            food_cached={FoodType.FISH: cached} if cached > 0 else {},
            description=(
                f"Moved {moved} fish to supply; rolled {rolled_any} die/dice; "
                f"{'cached 1 fish' if cached else 'no fish rolled'}"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        has_cached = slot.cached_food.get(FoodType.FISH, 0) > 0
        # Rough EV for 2-die fish hit in base/oceania pools.
        fish_hit_ev = 0.55
        return (0.4 if has_cached else 0.0) + fish_hit_ev


class WillowTitCacheOneFromFeederOptionalResetOnAllSame(PowerEffect):
    """Cache 1 invertebrate/seed/fruit from feeder; may reset first only when all dice show same face."""

    _ALLOWED = [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FRUIT]

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}

        can_reset = feeder.all_same_face()
        requested_reset = bool(choice.get("reset_feeder")) if isinstance(choice, dict) else False

        # Deterministic default: if reset is legal and no eligible food exists, reset for best chance.
        available_before = [ft for ft in self._ALLOWED if feeder.can_take(ft)]
        should_reset = can_reset and (requested_reset or not available_before)
        if should_reset:
            feeder.reroll()

        available = [ft for ft in self._ALLOWED if feeder.can_take(ft)]
        if not available:
            return PowerResult(executed=False, description="No invertebrate/seed/fruit available in feeder")

        preferred = _parse_food_choice(choice.get("food_type")) if isinstance(choice, dict) else None
        selected = preferred if preferred in available else min(
            available,
            key=lambda ft: (ctx.player.food_supply.get(ft), ft.value),
        )
        if not feeder.take_food(selected):
            return PowerResult(executed=False, description="Selected food not available")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(selected, 1)
        return PowerResult(
            food_cached={selected: 1},
            description=f"{'Reset feeder, then ' if should_reset else ''}cached 1 {selected.value}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class ResetFeederGainRodentMayGiveLayUpToThree(PowerEffect):
    """Reset feeder; gain 1 rodent if available; may give it to opponent to lay up to 3 eggs on this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        feeder.reroll()

        if not feeder.take_food(FoodType.RODENT):
            return PowerResult(executed=False, description="Reset feeder; no rodent available")

        # Gain from feeder first (die removed), then optionally give it away.
        ctx.player.food_supply.add(FoodType.RODENT, 1)
        actor_gained = {FoodType.RODENT: 1}

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target_name = choice.get("give_player") if isinstance(choice, dict) else None
        target = None
        if isinstance(target_name, str) and target_name:
            target = next(
                (p for p in ctx.game_state.players if p.name == target_name and p.name != ctx.player.name),
                None,
            )

        if target is None:
            return PowerResult(food_gained=actor_gained, description="Reset feeder; gained 1 rodent")

        if not ctx.player.food_supply.spend(FoodType.RODENT, 1):
            return PowerResult(food_gained=actor_gained, description="Reset feeder; gained 1 rodent")

        target.food_supply.add(FoodType.RODENT, 1)
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        requested = choice.get("lay_eggs") if isinstance(choice, dict) else None
        try:
            requested_int = int(requested) if requested is not None else 3
        except (TypeError, ValueError):
            requested_int = 3
        lay_count = max(0, min(3, requested_int, slot.eggs_space()))
        if lay_count > 0:
            slot.eggs += lay_count

        return PowerResult(
            food_gained=actor_gained,
            eggs_laid=lay_count,
            description=f"Reset feeder; gained 1 rodent, gave to {target.name}, laid {lay_count} egg(s)",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.1


class NoisyMinerTuckLayOthersLayEgg(PowerEffect):
    """Tuck 1 from hand; lay up to 2 on self; all other players may lay 1 egg."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")

        worst_idx = min(
            range(len(ctx.player.hand)),
            key=lambda i: (ctx.player.hand[i].victory_points, ctx.player.hand[i].name),
        )
        ctx.player.hand.pop(worst_idx)
        slot.tucked_cards += 1

        laid_self = min(2, slot.eggs_space())
        if laid_self > 0:
            slot.eggs += laid_self

        others_laid = 0
        for p in ctx.game_state.players:
            if p.name == ctx.player.name:
                continue
            others_laid += _lay_any_egg_for_player(p)

        return PowerResult(
            cards_tucked=1,
            eggs_laid=laid_self,
            description=f"Tucked 1; laid {laid_self} on self; others laid {others_laid}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class NorthIslandBrownKiwiDiscardBonusDrawKeep(PowerEffect):
    """Discard 1 bonus card; if you do, draw 4 bonus cards and keep 2."""

    def __init__(self, draw: int = 4, keep: int = 2):
        self.draw = int(max(1, draw))
        self.keep = int(max(1, keep))

    def _pick_discard_index(self, ctx: PowerContext) -> int | None:
        if not ctx.player.bonus_cards:
            return None

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        if isinstance(choice, dict):
            name = choice.get("discard_bonus_name")
            if isinstance(name, str) and name:
                idx = next((i for i, bc in enumerate(ctx.player.bonus_cards) if bc.name == name), None)
                if idx is not None:
                    return idx

        # Deterministic fallback: discard the card with least scoring utility.
        from backend.engine.scoring import score_bonus_cards

        original = list(ctx.player.bonus_cards)
        best_idx = 0
        best_key = None
        for i in range(len(original)):
            remaining = original[:i] + original[i + 1:]
            ctx.player.bonus_cards = remaining
            score = score_bonus_cards(ctx.player)
            draft = sum(float(getattr(bc, "draft_value_pct", 0.0) or 0.0) for bc in remaining)
            names = tuple(sorted(bc.name for bc in remaining))
            key = (score, draft, names)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = i
        ctx.player.bonus_cards = original
        return best_idx

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not ctx.player.bonus_cards:
            return PowerResult(executed=False, description="No bonus card available to discard")

        discard_idx = self._pick_discard_index(ctx)
        if discard_idx is None:
            return PowerResult(executed=False, description="No bonus card available to discard")

        discard_pile = getattr(ctx.game_state, "_bonus_discard_cards", None)
        if not isinstance(discard_pile, list):
            discard_pile = []
            setattr(ctx.game_state, "_bonus_discard_cards", discard_pile)
        discarded = ctx.player.bonus_cards.pop(discard_idx)
        discard_pile.append(discarded)

        from backend.powers.templates.draw_cards import DrawBonusCards

        result = DrawBonusCards(draw=self.draw, keep=self.keep).execute(ctx)
        if result.executed:
            result.description = f"Discarded 1 bonus; drew {self.draw}, kept {self.keep}"
            return result
        return PowerResult(
            executed=True,
            cards_drawn=0,
            description="Discarded 1 bonus; no bonus cards available to draw",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 2.1


class AllPlayersMayCacheFoodInHabitat(PowerEffect):
    """Each player may cache one specified food on a bird in the target habitat."""

    def __init__(self, food_type: FoodType, habitat: Habitat):
        self.food_type = food_type
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        actor_cached = 0
        for p in ctx.game_state.players:
            row = p.board.get_row(self.habitat)
            target_slot = next((s for s in row.slots if s.bird is not None), None)
            if target_slot is None or not p.food_supply.spend(self.food_type, 1):
                continue
            target_slot.cache_food(self.food_type, 1)
            if p.name == ctx.player.name:
                actor_cached += 1
        if actor_cached == 0:
            return PowerResult(executed=False, description="No eligible cache action")
        return PowerResult(food_cached={self.food_type: actor_cached}, description="Players cached food in habitat")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class AllPlayersMayTuckAndOrCacheInHabitat(PowerEffect):
    """Each player may tuck from hand and/or cache one food on a bird in habitat."""

    def __init__(self, cache_food_type: FoodType, habitat: Habitat):
        self.cache_food_type = cache_food_type
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        actor_tucked = 0
        actor_cached = 0
        for p in ctx.game_state.players:
            row = p.board.get_row(self.habitat)
            target_slot = next((s for s in row.slots if s.bird is not None), None)
            if target_slot is None:
                continue
            # Tuck first if possible.
            if p.hand:
                worst_idx = min(range(len(p.hand)), key=lambda i: p.hand[i].victory_points)
                p.hand.pop(worst_idx)
                target_slot.tucked_cards += 1
                if p.name == ctx.player.name:
                    actor_tucked += 1
            # Cache if possible.
            if p.food_supply.spend(self.cache_food_type, 1):
                target_slot.cache_food(self.cache_food_type, 1)
                if p.name == ctx.player.name:
                    actor_cached += 1

        if actor_tucked == 0 and actor_cached == 0:
            return PowerResult(executed=False, description="No player resolved tuck/cache")
        return PowerResult(
            cards_tucked=actor_tucked,
            food_cached={self.cache_food_type: actor_cached} if actor_cached > 0 else {},
            description="Players tucked and/or cached in habitat",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


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
    """Draw 1 tray card with selected nest types; may reset/refill tray first."""

    def __init__(
        self,
        allowed_nests: set[NestType] | None = None,
        allow_reset: bool = True,
        allow_refill: bool = False,
    ):
        self.allowed_nests = set(allowed_nests or {NestType.BOWL, NestType.WILD})
        self.allow_reset = allow_reset
        self.allow_refill = allow_refill

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}

        tray_action = str(choice.get("tray_action", "")).strip().lower()
        do_reset = bool(choice.get("reset_tray", False))
        do_refill = bool(choice.get("refill_tray", False))
        if tray_action == "reset":
            do_reset = True
            do_refill = False
        elif tray_action == "refill":
            do_refill = True
            do_reset = False

        if do_reset and self.allow_reset:
            discarded = ctx.game_state.card_tray.clear()
            ctx.game_state.discard_pile_count += len(discarded)
            _refill_tray_from_deck(ctx)
        elif do_refill and self.allow_refill:
            _refill_tray_from_deck(ctx)

        preferred_name = choice.get("draw_name")
        candidates = []
        for i, bird in enumerate(ctx.game_state.card_tray.face_up):
            if bird.nest_type not in self.allowed_nests:
                continue
            candidates.append((i, bird))
        if not candidates:
            return PowerResult(executed=False, description="No eligible nest card in tray")

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


class DrawAllFaceUpTrayCards(PowerEffect):
    """Draw all currently face-up tray cards."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        tray = ctx.game_state.card_tray
        drew = 0
        while tray.face_up:
            card = tray.take_card(0)
            if card is None:
                break
            ctx.player.hand.append(card)
            drew += 1
        if drew <= 0:
            return PowerResult(executed=False, description="No tray cards to draw")
        return PowerResult(cards_drawn=drew, description=f"Drew {drew} tray cards")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class DrawMiddleTrayCard(PowerEffect):
    """Draw the card in the tray's middle slot."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        tray = ctx.game_state.card_tray
        if len(tray.face_up) < 2:
            return PowerResult(executed=False, description="No middle tray card available")
        card = tray.take_card(1)
        if card is None:
            return PowerResult(executed=False, description="Middle tray draw failed")
        ctx.player.hand.append(card)
        return PowerResult(cards_drawn=1, description=f"Drew middle tray card {card.name}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class DrawTrayCardByHabitat(PowerEffect):
    """Draw 1 tray card that can live in the given habitat."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = choice.get("draw_name")
        candidates = [
            (i, b) for i, b in enumerate(ctx.game_state.card_tray.face_up)
            if b.can_live_in(self.habitat)
        ]
        if not candidates:
            return PowerResult(executed=False, description="No eligible habitat card in tray")

        idx = None
        if preferred:
            idx = next((i for i, b in candidates if b.name == preferred), None)
        if idx is None:
            idx = max(candidates, key=lambda x: x[1].victory_points)[0]

        card = ctx.game_state.card_tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray draw failed")
        ctx.player.hand.append(card)
        return PowerResult(cards_drawn=1, description=f"Drew {card.name} from tray")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class TuckSmallestTrayBird(PowerEffect):
    """Tuck the smallest wingspan bird from tray behind this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        tray = ctx.game_state.card_tray
        if not tray.face_up:
            return PowerResult(executed=False, description="No tray cards to tuck")

        # Flightless birds have wingspan=None. Treat them as very large so
        # numeric wingspans always win "smallest wingspan" comparisons.
        def _wingspan_sort_value(card) -> int:
            w = card.wingspan_cm
            return int(w) if isinstance(w, int) else 10_000

        idx = min(
            range(len(tray.face_up)),
            key=lambda i: (
                _wingspan_sort_value(tray.face_up[i]),
                tray.face_up[i].victory_points,
                tray.face_up[i].name,
            ),
        )
        card = tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray tuck failed")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description=f"Tucked smallest tray bird {card.name}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class DiscardSeedTuckTrayCard(PowerEffect):
    """Discard 1 seed to choose/tuck 1 tray card behind this bird."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        if not ctx.player.food_supply.spend(FoodType.SEED, 1):
            return PowerResult(executed=False, description="No seed to discard")
        tray = ctx.game_state.card_tray
        if not tray.face_up:
            return PowerResult(executed=False, description="No tray cards to tuck")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = choice.get("draw_name")
        idx = next((i for i, c in enumerate(tray.face_up) if c.name == preferred), None)
        if idx is None:
            idx = max(range(len(tray.face_up)), key=lambda i: tray.face_up[i].victory_points)
        card = tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray tuck failed")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description=f"Discarded seed and tucked {card.name}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class TuckTrayCardIfFoodCostIncludes(PowerEffect):
    """Tuck 1 tray card whose food cost includes a target type."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        tray = ctx.game_state.card_tray
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = choice.get("draw_name")
        candidates = [
            (i, b) for i, b in enumerate(tray.face_up)
            if self.food_type in b.food_cost.distinct_types
        ]
        if not candidates:
            return PowerResult(executed=False, description="No matching food-cost card in tray")

        idx = None
        if preferred:
            idx = next((i for i, b in candidates if b.name == preferred), None)
        if idx is None:
            idx = max(candidates, key=lambda x: x[1].victory_points)[0]
        card = tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray tuck failed")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description=f"Tucked {card.name} from tray")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class TuckTrayCardByMaxWingspan(PowerEffect):
    """Tuck 1 tray card with wingspan below a threshold."""

    def __init__(self, max_wingspan_cm: int):
        self.max_wingspan_cm = max_wingspan_cm

    def execute(self, ctx: PowerContext) -> PowerResult:
        tray = ctx.game_state.card_tray
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        preferred = choice.get("draw_name")
        candidates = [
            (i, b) for i, b in enumerate(tray.face_up)
            if (b.wingspan_cm is not None and b.wingspan_cm < self.max_wingspan_cm)
        ]
        if not candidates:
            return PowerResult(executed=False, description="No eligible wingspan card in tray")

        idx = None
        if preferred:
            idx = next((i for i, b in candidates if b.name == preferred), None)
        if idx is None:
            idx = max(candidates, key=lambda x: x[1].victory_points)[0]
        card = tray.take_card(idx)
        if card is None:
            return PowerResult(executed=False, description="Tray tuck failed")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        return PowerResult(cards_tucked=1, description=f"Tucked {card.name} from tray")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class RedWattledLapwingDiscardTrayLayEgg(PowerEffect):
    """Discard any tray cards, refill, lay 1 egg if any discarded card is predator."""

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
        if any(getattr(b, "is_predator", False) for b in discarded_birds):
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            if slot.can_hold_more_eggs():
                slot.eggs += 1
                laid = 1

        return PowerResult(eggs_laid=laid, description=f"Discarded {len(discarded_birds)} tray cards")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


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


class StealSpecificFoodCacheThenTargetGainDieFromFeeder(PowerEffect):
    """Steal one specific food from an opponent, cache it, then that opponent gains one feeder die."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        others = [p for p in ctx.game_state.players if p.name != ctx.player.name]
        if not others:
            return PowerResult(executed=False, description="No opponent")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested_target = choice.get("target_player") if isinstance(choice, dict) else None

        candidates = [p for p in others if p.food_supply.get(self.food_type) > 0]
        if not candidates:
            return PowerResult(executed=False, description=f"No opponent has {self.food_type.value} to steal")

        target = next((p for p in candidates if p.name == requested_target), None)
        if target is None:
            target = max(candidates, key=lambda p: (p.food_supply.get(self.food_type), p.name))

        if not target.food_supply.spend(self.food_type, 1):
            return PowerResult(executed=False, description=f"Failed to steal {self.food_type.value}")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(self.food_type, 1)

        feeder = ctx.game_state.birdfeeder
        if feeder.should_reroll():
            feeder.reroll()

        gained_food = None
        available = sorted(feeder.available_food_types(), key=lambda ft: ft.value)
        if available:
            target_choice = consume_power_choice(ctx.game_state, target.name, ctx.bird.name) or {}
            preferred = _parse_food_choice(target_choice.get("food_type")) if isinstance(target_choice, dict) else None
            selected = preferred if preferred in available else min(
                available,
                key=lambda ft: (target.food_supply.get(ft), ft.value),
            )
            if feeder.take_food(selected):
                target.food_supply.add(selected, 1)
                gained_food = selected

        return PowerResult(
            food_cached={self.food_type: 1},
            description=(
                f"Stole 1 {self.food_type.value} from {target.name} and cached it; "
                f"{target.name} gained {gained_food.value if gained_food else 'no food'} from feeder"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


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


class LayEggsOnEachOtherBirdInColumn(PowerEffect):
    """Lay N eggs on each other bird in this column."""

    def __init__(self, eggs_per_bird: int = 1):
        self.eggs_per_bird = int(max(0, eggs_per_bird))

    def execute(self, ctx: PowerContext) -> PowerResult:
        if self.eggs_per_bird <= 0:
            return PowerResult(executed=False, description="No eggs configured to lay")

        laid = 0
        for row in ctx.player.board.all_rows():
            if row.habitat == ctx.habitat:
                continue
            slot = row.slots[ctx.slot_index]
            if not slot.bird:
                continue
            add = min(self.eggs_per_bird, slot.eggs_space())
            if add > 0:
                slot.eggs += add
                laid += add

        if laid <= 0:
            return PowerResult(executed=False, description="No other birds in this column with egg space")
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on other birds in this column")

    def estimate_value(self, ctx: PowerContext) -> float:
        targets = 0
        for row in ctx.player.board.all_rows():
            if row.habitat == ctx.habitat:
                continue
            slot = row.slots[ctx.slot_index]
            if slot.bird and slot.eggs_space() > 0:
                targets += 1
        return targets * self.eggs_per_bird * 0.8


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


class DiscardAllEggsFromOneNestThenTuckDouble(PowerEffect):
    """Discard all eggs from one matching nest bird; tuck twice that many from deck on this bird."""

    def __init__(self, nest_types: set[NestType], include_wild: bool = True):
        self.nest_types = set(nest_types)
        self.include_wild = bool(include_wild)

    def _is_matching_nest(self, nest: NestType) -> bool:
        return nest in self.nest_types or (self.include_wild and nest == NestType.WILD)

    def _parse_habitat(self, raw: str | None) -> Habitat | None:
        if not isinstance(raw, str) or not raw:
            return None
        norm = raw.strip().lower()
        for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            if norm in {h.value, h.name.lower()}:
                return h
        return None

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}

        candidates: list[tuple[int, Habitat, int, object]] = []
        for hab, idx, slot in ctx.player.board.all_slots():
            if not slot.bird:
                continue
            if not self._is_matching_nest(slot.bird.nest_type):
                continue
            candidates.append((slot.eggs, hab, idx, slot))

        if not candidates:
            return PowerResult(executed=False, description="No matching nest birds")

        target_slot = None
        target_hab = self._parse_habitat(choice.get("target_habitat")) if isinstance(choice, dict) else None
        target_idx_raw = choice.get("target_slot") if isinstance(choice, dict) else None
        try:
            target_idx = int(target_idx_raw) if target_idx_raw is not None else None
        except (TypeError, ValueError):
            target_idx = None
        if target_hab is not None and target_idx is not None:
            for _, hab, idx, slot in candidates:
                if hab == target_hab and idx == target_idx:
                    target_slot = slot
                    break

        if target_slot is None:
            # Deterministic fallback: choose matching bird with most eggs.
            candidates.sort(key=lambda t: (t[0], t[1].value, t[2]), reverse=True)
            target_slot = candidates[0][3]

        discarded_eggs = int(max(0, target_slot.eggs))
        target_slot.eggs = 0

        source_slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        for _ in range(discarded_eggs * 2):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            source_slot.tucked_cards += 1
            tucked += 1

        return PowerResult(
            executed=True,
            cards_tucked=tucked,
            description=f"Discarded {discarded_eggs} egg(s); tucked {tucked} card(s)",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        best_eggs = 0
        for _, _, slot in ctx.player.board.all_slots():
            if not slot.bird:
                continue
            if not self._is_matching_nest(slot.bird.nest_type):
                continue
            best_eggs = max(best_eggs, int(slot.eggs))
        return min(4.0, best_eggs * 1.2)


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

    def __init__(self, max_discard: int = 5, food_type: FoodType | None = None, tuck_per_discard: int = 1):
        self.max_discard = max_discard
        self.food_type = food_type
        self.tuck_per_discard = max(1, int(tuck_per_discard))

    def _spend_one(self, ctx: PowerContext) -> bool:
        if self.food_type is not None and self.food_type != FoodType.WILD:
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
            for _ in range(self.tuck_per_discard):
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


class PushYourLuckPredatorDiceCache(PowerEffect):
    """Roll N dice up to M times; each hit caches; miss busts this activation."""

    def __init__(
        self,
        dice_count: int,
        target_foods: set[FoodType],
        cache_food: FoodType,
        max_rolls: int = 3,
        cache_per_hit: int = 1,
    ):
        self.dice_count = int(max(1, dice_count))
        self.target_foods = set(target_foods)
        self.cache_food = cache_food
        self.max_rolls = int(max(1, max_rolls))
        self.cache_per_hit = int(max(1, cache_per_hit))

    def _roll_hits(self, feeder_faces) -> bool:
        for _ in range(self.dice_count):
            face = random.choice(feeder_faces)
            if isinstance(face, tuple):
                if any(ft in self.target_foods for ft in face):
                    return True
            elif face in self.target_foods:
                return True
        return False

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        if not feeder.dice_faces:
            return PowerResult(executed=False, description="No dice faces available")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested_rolls = choice.get("roll_attempts") if isinstance(choice, dict) else None
        try:
            attempts = int(requested_rolls) if requested_rolls is not None else self.max_rolls
        except (TypeError, ValueError):
            attempts = self.max_rolls
        attempts = max(1, min(self.max_rolls, attempts))

        hits = 0
        busted = False
        for _ in range(attempts):
            if self._roll_hits(feeder.dice_faces):
                hits += 1
            else:
                hits = 0
                busted = True
                break

        if hits <= 0:
            return PowerResult(
                executed=True,
                food_cached={},
                description="Predator push-your-luck missed and returned cached food from this activation",
            )

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        cached_total = hits * self.cache_per_hit
        slot.cache_food(self.cache_food, cached_total)
        return PowerResult(
            executed=True,
            food_cached={self.cache_food: cached_total},
            description=(
                f"Predator push-your-luck {'busted' if busted else 'stopped'}; "
                f"cached {cached_total} {self.cache_food.value}"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Coarse EV estimate for 3-roll push-your-luck sequence.
        p_hit = max(0.05, min(0.95, 0.2 * max(1, len(self.target_foods))))
        ev_hits = p_hit + (p_hit * p_hit) + (p_hit * p_hit * p_hit)
        return ev_hits * self.cache_per_hit * 0.8


class PushYourLuckDrawByWingspanTotalThenTuck(PowerEffect):
    """Draw up to N cards; if total wingspan < threshold when stopping, tuck them; else discard."""

    def __init__(self, max_draws: int = 3, wingspan_threshold: int = 110):
        self.max_draws = int(max(1, max_draws))
        self.wingspan_threshold = int(max(1, wingspan_threshold))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested = choice.get("draw_attempts") if isinstance(choice, dict) else None
        try:
            attempts = int(requested) if requested is not None else self.max_draws
        except (TypeError, ValueError):
            attempts = self.max_draws
        attempts = max(1, min(self.max_draws, attempts))
        wild_choices = choice.get("wild_wingspans") if isinstance(choice, dict) else None
        wild_idx = 0

        def _resolve_wild_wingspan(card_name: str) -> int:
            nonlocal wild_idx
            # Optional deterministic selection for wild-wingspan cards.
            # Accept either draw-order list (wild_wingspans=[...]) or
            # name map (wild_wingspans={"Bird Name": value}).
            if isinstance(wild_choices, dict):
                raw = wild_choices.get(card_name)
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    return 1
                return max(1, val)
            if isinstance(wild_choices, list):
                raw = wild_choices[wild_idx] if wild_idx < len(wild_choices) else None
                wild_idx += 1
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    return 1
                return max(1, val)
            return 1

        drawn = []
        total_wingspan = 0
        for _ in range(attempts):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            drawn.append(card)
            wingspan = card.wingspan_cm
            if wingspan is None:
                wingspan = _resolve_wild_wingspan(card.name)
            total_wingspan += max(1, int(wingspan))

            # Push-your-luck fail is checked immediately after each draw.
            if total_wingspan >= self.wingspan_threshold:
                ctx.game_state.discard_pile_count += len(drawn)
                return PowerResult(
                    cards_tucked=0,
                    description=(
                        f"Drew {len(drawn)}; total wingspan {total_wingspan} >= {self.wingspan_threshold}; discarded all"
                    ),
                )

        if not drawn:
            return PowerResult(executed=False, description="Deck empty")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += len(drawn)
        return PowerResult(
            cards_tucked=len(drawn),
            description=(
                f"Drew {len(drawn)}; total wingspan {total_wingspan} < {self.wingspan_threshold}; tucked all"
            ),
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        # Conservative expected value for uncertain wingspan draw sequence.
        return 1.0


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

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested_name = choice.get("target_player") if isinstance(choice, dict) else None
        target = next((p for p in players if p.name == requested_name), None)
        if target is None:
            target = max(players, key=lambda p: (predator_count(p), p.name))
        count = predator_count(target)
        if count <= 0:
            return PowerResult(executed=False, description="No predator birds to count")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.cache_food(FoodType.RODENT, count)
        return PowerResult(
            food_cached={FoodType.RODENT: count},
            description=f"Cached {count} rodent(s) from {target.name}'s predator count",
        )

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


class TuckHandThenDrawEqualPerOpponentWetlandCubes(PowerEffect):
    """Choose another player; for each wetland cube, tuck 1 from hand then draw 1."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        others = [p for p in ctx.game_state.players if p.name != ctx.player.name]
        if not others:
            return PowerResult(executed=False, description="No opponent")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target_name = choice.get("target_player") if isinstance(choice, dict) else None
        target_player = next((p for p in others if p.name == target_name), None)
        if target_player is None:
            target_player = max(others, key=lambda p: (p.draw_cards_actions_this_round, p.name))

        limit = int(max(0, target_player.draw_cards_actions_this_round))
        if limit <= 0:
            return PowerResult(executed=False, description="No wetland cubes to convert")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = min(limit, len(ctx.player.hand))
        for _ in range(tucked):
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            slot.tucked_cards += 1
        drawn = _draw_many_birds(ctx, tucked)

        if tucked == 0:
            return PowerResult(executed=False, description="No cards in hand to tuck")
        return PowerResult(
            cards_tucked=tucked,
            cards_drawn=len(drawn),
            description=f"Tucked {tucked}, drew {len(drawn)} from wetland cubes",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class ChooseBirdsInHabitatTuckFromHandEach(PowerEffect):
    """Choose birds in this habitat; tuck 1 hand card behind each chosen bird."""

    def __init__(self, max_targets: int = 5, draw_if_tucked: int = 0):
        self.max_targets = int(max(1, max_targets))
        self.draw_if_tucked = int(max(0, draw_if_tucked))

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        candidates = [(idx, slot) for idx, slot in enumerate(row.slots) if slot.bird is not None]
        if not candidates:
            return PowerResult(executed=False, description="No birds in habitat")
        if not ctx.player.hand:
            return PowerResult(executed=False, description="No cards in hand to tuck")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        selected_indices: list[int] = []

        if isinstance(choice, dict):
            raw_slots = choice.get("target_slots")
            if isinstance(raw_slots, list):
                for raw in raw_slots:
                    try:
                        idx = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if idx not in selected_indices:
                        selected_indices.append(idx)

            raw_birds = choice.get("target_birds")
            if isinstance(raw_birds, list):
                for name in raw_birds:
                    if not isinstance(name, str):
                        continue
                    idx = next(
                        (i for i, s in candidates if s.bird is not None and s.bird.name == name and i not in selected_indices),
                        None,
                    )
                    if idx is not None:
                        selected_indices.append(idx)

        valid_indices = [i for i, _ in candidates]
        selected_indices = [i for i in selected_indices if i in valid_indices]
        if not selected_indices:
            selected_indices = valid_indices

        max_tucks = min(self.max_targets, len(selected_indices), len(ctx.player.hand))
        if max_tucks <= 0:
            return PowerResult(executed=False, description="No cards in hand to tuck")

        tucked = 0
        for idx in selected_indices[:max_tucks]:
            worst_idx = min(range(len(ctx.player.hand)), key=lambda i: ctx.player.hand[i].victory_points)
            ctx.player.hand.pop(worst_idx)
            row.slots[idx].tucked_cards += 1
            tucked += 1

        cards_drawn = 0
        if tucked > 0 and self.draw_if_tucked > 0:
            cards_drawn = len(_draw_many_birds(ctx, self.draw_if_tucked))

        return PowerResult(
            cards_tucked=tucked,
            cards_drawn=cards_drawn,
            description=f"Tucked {tucked} card(s) across chosen birds, drew {cards_drawn}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        hand_size = len(ctx.player.hand)
        return min(self.max_targets, hand_size) * 0.7


class ChooseBirdsInHabitatCacheFoodEach(PowerEffect):
    """Choose birds in this habitat; cache 1 specified food on each chosen bird."""

    def __init__(self, food_type: FoodType, max_targets: int = 5):
        self.food_type = food_type
        self.max_targets = int(max(1, max_targets))

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(ctx.habitat)
        candidates = [(idx, slot) for idx, slot in enumerate(row.slots) if slot.bird is not None]
        if not candidates:
            return PowerResult(executed=False, description="No birds in habitat")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        selected_indices: list[int] = []

        if isinstance(choice, dict):
            raw_slots = choice.get("target_slots")
            if isinstance(raw_slots, list):
                for raw in raw_slots:
                    try:
                        idx = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if idx not in selected_indices:
                        selected_indices.append(idx)

            raw_birds = choice.get("target_birds")
            if isinstance(raw_birds, list):
                for name in raw_birds:
                    if not isinstance(name, str):
                        continue
                    idx = next(
                        (i for i, s in candidates if s.bird is not None and s.bird.name == name and i not in selected_indices),
                        None,
                    )
                    if idx is not None:
                        selected_indices.append(idx)

        valid_indices = [i for i, _ in candidates]
        selected_indices = [i for i in selected_indices if i in valid_indices]
        if not selected_indices:
            selected_indices = valid_indices

        cache_targets = min(self.max_targets, len(selected_indices))
        if cache_targets <= 0:
            return PowerResult(executed=False, description="No birds selected for caching")

        for idx in selected_indices[:cache_targets]:
            row.slots[idx].cache_food(self.food_type, 1)

        return PowerResult(
            food_cached={self.food_type: cache_targets},
            description=f"Cached 1 {self.food_type.value} on {cache_targets} chosen bird(s)",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        row = ctx.player.board.get_row(ctx.habitat)
        birds = sum(1 for s in row.slots if s.bird is not None)
        return min(self.max_targets, birds) * 0.9


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
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}

        # Allow keep-choice fields to be supplied in the same payload and
        # forwarded to nested DrawBonusCards execution.
        forward_bonus_choice = {}
        if isinstance(choice, dict):
            for key in ("keep_bonus_name", "keep_bonus_names"):
                if key in choice:
                    forward_bonus_choice[key] = choice[key]

        reroll_sets = choice.get("reroll_sets") if isinstance(choice, dict) else None

        def _normalized_indices(raw) -> list[int]:
            if not isinstance(raw, (list, tuple)):
                return []
            out: list[int] = []
            seen: set[int] = set()
            for v in raw:
                try:
                    idx = int(v)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(feeder.dice) or idx in seen:
                    continue
                seen.add(idx)
                out.append(idx)
            return out

        def _has_rodent(face) -> bool:
            return face == FoodType.RODENT or (isinstance(face, tuple) and FoodType.RODENT in face)

        feeder.reroll()

        for step in range(2):
            # Explicit deterministic control: reroll any chosen subset per step.
            if isinstance(reroll_sets, list):
                if step >= len(reroll_sets):
                    break
                indices = _normalized_indices(reroll_sets[step])
                if not indices:
                    # "May reroll" => empty selection means stop.
                    break
            else:
                # Default behavior for simulation: stop if already successful;
                # otherwise reroll all non-rodent dice.
                if self._rodent_count(feeder.dice) >= 3:
                    break
                indices = [i for i, die in enumerate(feeder.dice) if not _has_rodent(die)]
                if not indices:
                    break

            for idx in indices:
                feeder.dice[idx] = random.choice(feeder.dice_faces)

        out = PowerResult(description="Philippine Eagle resolved")
        if self._rodent_count(feeder.dice) >= 3:
            from backend.powers.templates.draw_cards import DrawBonusCards
            if forward_bonus_choice:
                key = f"{ctx.player.name}::{ctx.bird.name}"
                existing = list(ctx.game_state.power_choice_queues.get(key, []))
                ctx.game_state.power_choice_queues[key] = [forward_bonus_choice] + existing
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
            allow_nectar=False,
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
            allow_nectar=False,
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
