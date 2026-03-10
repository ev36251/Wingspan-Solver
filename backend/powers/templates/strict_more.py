"""Additional strict power implementations for nuanced option/conditional birds."""

from __future__ import annotations

import random

from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerContext, PowerEffect, PowerResult
from backend.powers.choices import consume_power_choice


def _draw_one_bird(ctx: PowerContext):
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


def _all_food_types_for_board(ctx: PowerContext) -> list[FoodType]:
    foods = [
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
    ]
    if ctx.game_state.board_type.value == "oceania":
        foods.append(FoodType.NECTAR)
    return foods


def _parse_food_choice(raw: str | None) -> FoodType | None:
    if not raw:
        return None
    norm = str(raw).strip().lower()
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


def _parse_habitat(raw: str | None) -> Habitat | None:
    if not raw:
        return None
    norm = str(raw).strip().lower()
    for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
        if norm in {h.value, h.name.lower()}:
            return h
    return None


def _pick_other_player(ctx: PowerContext, choice: dict | None = None):
    target_name = choice.get("target_player") if isinstance(choice, dict) else None
    others = [p for p in ctx.game_state.players if p.name != ctx.player.name]
    if not others:
        return None
    if isinstance(target_name, str) and target_name:
        selected = next((p for p in others if p.name == target_name), None)
        if selected is not None:
            return selected
    return sorted(others, key=lambda p: p.name)[0]


def _lay_one_egg_any(player) -> int:
    candidates = []
    for _, _, slot in player.board.all_slots():
        if slot.bird and slot.can_hold_more_eggs():
            candidates.append((slot.eggs_space(), slot))
    if not candidates:
        return 0
    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates[0][1].eggs += 1
    return 1


def _spend_any_food(ctx: PowerContext, preferred: FoodType | None = None) -> FoodType | None:
    options = _all_food_types_for_board(ctx)
    if preferred is not None and ctx.player.food_supply.get(preferred) > 0:
        ctx.player.food_supply.spend(preferred, 1)
        return preferred
    candidates = [ft for ft in options if ctx.player.food_supply.get(ft) > 0]
    if not candidates:
        return None
    # Deterministic: spend most abundant first.
    chosen = max(candidates, key=lambda ft: (ctx.player.food_supply.get(ft), ft.value))
    ctx.player.food_supply.spend(chosen, 1)
    return chosen


class ConditionalNeighborHasFoodTuckFromDeck(PowerEffect):
    """If neighbor has food type in supply, tuck from deck behind this bird."""

    def __init__(self, direction: str, food_type: FoodType, tuck_count: int = 1):
        self.direction = direction
        self.food_type = food_type
        self.tuck_count = int(max(0, tuck_count))

    def _neighbors(self, ctx: PowerContext):
        if self.direction == "left":
            n = ctx.game_state.player_to_left(ctx.player)
            return [n] if n else []
        if self.direction == "right":
            n = ctx.game_state.player_to_right(ctx.player)
            return [n] if n else []
        if self.direction == "left_or_right":
            out = []
            n1 = ctx.game_state.player_to_left(ctx.player)
            n2 = ctx.game_state.player_to_right(ctx.player)
            if n1 is not None:
                out.append(n1)
            if n2 is not None and (n1 is None or n2.name != n1.name):
                out.append(n2)
            return out
        return []

    def execute(self, ctx: PowerContext) -> PowerResult:
        neighbors = self._neighbors(ctx)
        if not neighbors:
            return PowerResult(executed=False, description="No neighbor")
        if not any(n.food_supply.get(self.food_type) > 0 for n in neighbors):
            return PowerResult(executed=False, description=f"Neighbor has no {self.food_type.value}")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        for _ in range(self.tuck_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
        if tucked <= 0:
            return PowerResult(executed=False, description="Deck empty")
        return PowerResult(cards_tucked=tucked, description=f"Tucked {tucked} card(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.tuck_count * 0.7


class EmuGainAllSeedKeepHalfDistributeRemainder(PowerEffect):
    """Gain all seed from feeder; keep half rounded up; distribute remainder to others."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        feeder = ctx.game_state.birdfeeder
        total_seed = 0
        while feeder.take_food(FoodType.SEED):
            total_seed += 1
        if total_seed <= 0:
            return PowerResult(executed=False, description="No seed in feeder")

        keep = (total_seed + 1) // 2
        give = total_seed - keep
        ctx.player.food_supply.add(FoodType.SEED, keep)

        others = sorted((p for p in ctx.game_state.players if p.name != ctx.player.name), key=lambda p: p.name)
        allocations: dict[str, int] = {p.name: 0 for p in others}
        if give > 0 and others:
            choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
            raw = choice.get("distribution") if isinstance(choice, dict) else None
            used = 0
            if isinstance(raw, dict):
                for p in others:
                    try:
                        requested = int(raw.get(p.name, 0))
                    except (TypeError, ValueError):
                        requested = 0
                    grant = max(0, min(requested, give - used))
                    allocations[p.name] += grant
                    used += grant
                    if used >= give:
                        break

            remaining = give - sum(allocations.values())
            idx = 0
            while remaining > 0 and others:
                p = others[idx % len(others)]
                allocations[p.name] += 1
                remaining -= 1
                idx += 1

            for p in others:
                cnt = allocations.get(p.name, 0)
                if cnt > 0:
                    p.food_supply.add(FoodType.SEED, cnt)

        return PowerResult(
            food_gained={FoodType.SEED: keep},
            description=f"Gained {keep} seed; distributed {give} seed",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        feeder = ctx.game_state.birdfeeder
        available = sum(1 for d in feeder.dice if d == FoodType.SEED or (isinstance(d, tuple) and FoodType.SEED in d))
        return 0.4 + 0.3 * float(available)


class ChooseOtherPlayerResetFeederGainFoodThenSelfTuck(PowerEffect):
    """Choose other player: they reset feeder + gain food if present; you tuck cards."""

    def __init__(self, food_type: FoodType, tuck_count: int = 2):
        self.food_type = food_type
        self.tuck_count = int(max(0, tuck_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")

        feeder = ctx.game_state.birdfeeder
        feeder.reroll()
        if feeder.take_food(self.food_type):
            target.food_supply.add(self.food_type, 1)

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        for _ in range(self.tuck_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1

        return PowerResult(cards_tucked=tucked, description=f"Target reset/gained; tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.tuck_count * 0.8


class IfEggLaidOnThisBirdThisTurnTuckFromDeck(PowerEffect):
    """If this bird had an egg laid on it this action, tuck from deck."""

    def __init__(self, count: int = 1):
        self.count = int(max(0, count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        marker = getattr(ctx.game_state, "_eggs_laid_this_action", set())
        key = (ctx.player.name, ctx.habitat.value, int(ctx.slot_index))
        if key not in marker:
            return PowerResult(executed=False, description="No egg laid on this bird this turn")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        for _ in range(self.count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
        if tucked <= 0:
            return PowerResult(executed=False, description="Deck empty")
        return PowerResult(cards_tucked=tucked, description=f"Tucked {tucked}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.count * 0.6


class LookAtDeckKeepHabitatBirdHandOrTuckDiscardRest(PowerEffect):
    """Look at N deck cards; keep one matching habitat to hand or tuck; discard rest."""

    def __init__(self, look_count: int, habitat_filter: Habitat):
        self.look_count = int(max(0, look_count))
        self.habitat_filter = habitat_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        keep_name = choice.get("keep_name") if isinstance(choice, dict) else None
        keep_mode = choice.get("keep_mode") if isinstance(choice, dict) else None

        seen = []
        for _ in range(self.look_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            seen.append(card)

        if not seen:
            return PowerResult(executed=False, description="Deck empty")

        matches = [c for c in seen if c.can_live_in(self.habitat_filter)]
        chosen = None
        if isinstance(keep_name, str) and keep_name:
            chosen = next((c for c in matches if c.name == keep_name), None)
        if chosen is None and matches:
            chosen = max(matches, key=lambda c: (c.victory_points, c.name))

        cards_drawn = 0
        cards_tucked = 0
        if chosen is not None:
            mode = str(keep_mode).lower() if keep_mode is not None else "hand"
            if mode == "tuck":
                slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
                slot.tucked_cards += 1
                cards_tucked = 1
            else:
                ctx.player.hand.append(chosen)
                cards_drawn = 1
            seen.remove(chosen)

        if seen:
            ctx.game_state.discard_pile_count += len(seen)

        return PowerResult(
            executed=True,
            cards_drawn=cards_drawn,
            cards_tucked=cards_tucked,
            description=f"Looked at {self.look_count}; kept {cards_drawn + cards_tucked}, discarded {len(seen)}",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.2


class CacheAnyFoodOrTuckFromHandThenTuckDeck(PowerEffect):
    """Cache any food or tuck 1 from hand; if done, tuck 1 from deck."""

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        mode = str(choice.get("mode")).lower() if isinstance(choice, dict) and choice.get("mode") is not None else None
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]

        did_primary = False
        cached: dict[FoodType, int] = {}
        tucked_from_hand = 0

        if mode in {None, "cache"}:
            preferred = _parse_food_choice(choice.get("food_type") if isinstance(choice, dict) else None)
            spent = _spend_any_food(ctx, preferred=preferred)
            if spent is not None:
                slot.cache_food(spent, 1)
                cached[spent] = 1
                did_primary = True

        if not did_primary and mode in {None, "tuck"} and ctx.player.hand:
            tuck_name = choice.get("tuck_name") if isinstance(choice, dict) else None
            idx = next((i for i, c in enumerate(ctx.player.hand) if c.name == tuck_name), None) if isinstance(tuck_name, str) else None
            if idx is None:
                idx = min(range(len(ctx.player.hand)), key=lambda i: (ctx.player.hand[i].victory_points, ctx.player.hand[i].name))
            ctx.player.hand.pop(idx)
            slot.tucked_cards += 1
            tucked_from_hand = 1
            did_primary = True

        if not did_primary:
            return PowerResult(executed=False, description="No food to cache and no card to tuck")

        tucked_from_deck = 0
        card = _draw_one_bird(ctx)
        if card is not None:
            slot.tucked_cards += 1
            tucked_from_deck = 1

        return PowerResult(
            cards_tucked=tucked_from_hand + tucked_from_deck,
            food_cached=cached,
            description="Resolved cache/tuck primary, then tucked from deck",
        )

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.4


class TuckFromDeckBehindEachBirdInHabitat(PowerEffect):
    """Tuck one deck card behind each bird in habitat (including this one)."""

    def __init__(self, habitat: Habitat | None = None):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        use_habitat = self.habitat or ctx.habitat
        row = ctx.player.board.get_row(use_habitat)
        tucked = 0
        for slot in row.slots:
            if slot.bird is None:
                continue
            card = _draw_one_bird(ctx)
            if card is None:
                break
            slot.tucked_cards += 1
            tucked += 1
        if tucked <= 0:
            return PowerResult(executed=False, description="No cards tucked")
        return PowerResult(cards_tucked=tucked, description=f"Tucked {tucked} across habitat")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.5


class LayEggOnSelfPerOtherBirdInHabitat(PowerEffect):
    """Lay 1 egg on this bird for each other bird in the configured habitat."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not in source slot")

        row = ctx.player.board.get_row(self.habitat)
        others = 0
        for idx, slot in enumerate(row.slots):
            if slot.bird is None:
                continue
            if self.habitat == ctx.habitat and idx == ctx.slot_index:
                continue
            others += 1

        laid = min(others, target.eggs_space())
        if laid <= 0:
            return PowerResult(executed=False, description="No eggs laid")
        target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s) on this bird")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class DiscardFoodToLayEggOnSelf(PowerEffect):
    """Discard chosen food(s), then lay that many eggs on this bird."""

    def __init__(self, food_type: FoodType, max_discard: int | None = None):
        self.food_type = food_type
        self.max_discard = max_discard

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not in source slot")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested = choice.get("discard_count") if isinstance(choice, dict) else None
        try:
            requested_count = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            requested_count = None

        egg_space = target.eggs_space()
        if egg_space <= 0:
            return PowerResult(executed=False, description="No egg space")

        if self.food_type == FoodType.WILD:
            available = sum(ctx.player.food_supply.get(ft) for ft in _all_food_types_for_board(ctx))
            upper = available
            if self.max_discard is not None:
                upper = min(upper, self.max_discard)
            upper = min(upper, egg_space)
            count = upper if requested_count is None else max(0, min(requested_count, upper))
            if count <= 0:
                return PowerResult(executed=False, description="No food discarded")
            for _ in range(count):
                if _spend_any_food(ctx) is None:
                    break
            target.eggs += count
            return PowerResult(eggs_laid=count, description=f"Discarded {count} wild food; laid {count}")

        available = ctx.player.food_supply.get(self.food_type)
        upper = available
        if self.max_discard is not None:
            upper = min(upper, self.max_discard)
        upper = min(upper, egg_space)
        count = upper if requested_count is None else max(0, min(requested_count, upper))
        if count <= 0:
            return PowerResult(executed=False, description="No food discarded")
        if not ctx.player.food_supply.spend(self.food_type, count):
            return PowerResult(executed=False, description="Failed to discard food")
        target.eggs += count
        return PowerResult(eggs_laid=count, description=f"Discarded {count} food; laid {count}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class DiscardSpecificFoodThenLayEggOnSelfUpTo(PowerEffect):
    """Discard one specific food; if you do, lay up to N eggs on this bird."""

    def __init__(self, food_type: FoodType, max_eggs: int):
        self.food_type = food_type
        self.max_eggs = int(max(0, max_eggs))

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not in source slot")
        if not ctx.player.food_supply.spend(self.food_type, 1):
            return PowerResult(executed=False, description=f"No {self.food_type.value} to discard")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested = choice.get("lay_eggs") if isinstance(choice, dict) else None
        try:
            requested_count = int(requested) if requested is not None else self.max_eggs
        except (TypeError, ValueError):
            requested_count = self.max_eggs

        laid = max(0, min(requested_count, self.max_eggs, target.eggs_space()))
        if laid > 0:
            target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Discarded 1, laid {laid}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class DiscardCardThenLayEggOnSelf(PowerEffect):
    """Discard card(s) from hand; if you do, lay eggs on this bird."""

    def __init__(self, card_cost: int = 1, egg_count: int = 1):
        self.card_cost = int(max(0, card_cost))
        self.egg_count = int(max(0, egg_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        target = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not target.bird:
            return PowerResult(executed=False, description="Bird not in source slot")
        if len(ctx.player.hand) < self.card_cost:
            return PowerResult(executed=False, description="Not enough cards to discard")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        discard_names = choice.get("discard_names") if isinstance(choice, dict) else None
        removed = 0
        if isinstance(discard_names, list):
            for n in discard_names:
                if removed >= self.card_cost:
                    break
                idx = next((i for i, c in enumerate(ctx.player.hand) if c.name == str(n)), None)
                if idx is None:
                    continue
                ctx.player.hand.pop(idx)
                removed += 1

        while removed < self.card_cost and ctx.player.hand:
            idx = min(range(len(ctx.player.hand)), key=lambda i: (ctx.player.hand[i].victory_points, ctx.player.hand[i].name))
            ctx.player.hand.pop(idx)
            removed += 1

        laid = min(self.egg_count, target.eggs_space())
        if laid > 0:
            target.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Discarded {removed} card(s), laid {laid}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class ChooseOtherPlayerLayEggThenDraw(PowerEffect):
    """Choose one other player; they lay an egg, then you draw cards."""

    def __init__(self, draw_count: int = 2):
        self.draw_count = int(max(0, draw_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")

        _lay_one_egg_any(target)

        drawn = 0
        for _ in range(self.draw_count):
            card = _draw_one_bird(ctx)
            if card is None:
                break
            ctx.player.hand.append(card)
            drawn += 1

        return PowerResult(cards_drawn=drawn, description=f"Target laid 1 egg; drew {drawn}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class ChooseOtherPlayerBothLayEgg(PowerEffect):
    """Choose one other player; both lay one egg."""

    def __init__(self, eggs_each: int = 1):
        self.eggs_each = int(max(0, eggs_each))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")

        actor_laid = 0
        for _ in range(self.eggs_each):
            actor_laid += _lay_one_egg_any(ctx.player)
            _lay_one_egg_any(target)

        return PowerResult(eggs_laid=actor_laid, description=f"Both players laid up to {self.eggs_each}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class IfHasFoodLayEggOnSelf(PowerEffect):
    """If player has required food in supply, lay egg(s) on this bird."""

    def __init__(self, required_food: FoodType, egg_count: int = 1):
        self.required_food = required_food
        self.egg_count = int(max(0, egg_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        if ctx.player.food_supply.get(self.required_food) <= 0:
            return PowerResult(executed=False, description=f"No {self.required_food.value} in supply")
        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        if not slot.bird:
            return PowerResult(executed=False, description="Bird not in source slot")
        laid = min(self.egg_count, slot.eggs_space())
        if laid <= 0:
            return PowerResult(executed=False, description="No egg space")
        slot.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Laid {laid} egg(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.7


class GiveFoodToOtherThenEggsOrDie(PowerEffect):
    """Give food to other player; then either lay eggs on self or gain dice from feeder."""

    def __init__(self, food_type: FoodType, give_count: int = 1, reward_count: int = 2):
        self.food_type = food_type
        self.give_count = int(max(0, give_count))
        self.reward_count = int(max(0, reward_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")
        if not ctx.player.food_supply.spend(self.food_type, self.give_count):
            return PowerResult(executed=False, description=f"Not enough {self.food_type.value} to give")
        target.food_supply.add(self.food_type, self.give_count)

        reward_mode = str(choice.get("reward_mode")).lower() if isinstance(choice, dict) and choice.get("reward_mode") is not None else "eggs"
        if reward_mode == "die":
            feeder = ctx.game_state.birdfeeder
            gained: dict[FoodType, int] = {}
            for _ in range(self.reward_count):
                if feeder.should_reroll():
                    feeder.reroll()
                die = feeder.take_any()
                if die is None:
                    break
                ft = die if isinstance(die, FoodType) else die[0]
                ctx.player.food_supply.add(ft, 1)
                gained[ft] = gained.get(ft, 0) + 1
            return PowerResult(food_gained=gained, description="Gave food; gained dice from feeder")

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        laid = min(self.reward_count, slot.eggs_space()) if slot.bird else 0
        if laid > 0:
            slot.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Gave food; laid {laid} egg(s)")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class GiveCardToOtherThenLayEggOnSelf(PowerEffect):
    """Give a card to another player; if you do, lay eggs on this bird."""

    def __init__(self, give_count: int = 1, egg_count: int = 2):
        self.give_count = int(max(0, give_count))
        self.egg_count = int(max(0, egg_count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")
        if len(ctx.player.hand) < self.give_count:
            return PowerResult(executed=False, description="No card to give")

        give_name = choice.get("give_name") if isinstance(choice, dict) else None
        moved = 0
        if isinstance(give_name, str) and give_name:
            idx = next((i for i, c in enumerate(ctx.player.hand) if c.name == give_name), None)
            if idx is not None:
                target.hand.append(ctx.player.hand.pop(idx))
                moved += 1

        while moved < self.give_count and ctx.player.hand:
            idx = min(range(len(ctx.player.hand)), key=lambda i: (ctx.player.hand[i].victory_points, ctx.player.hand[i].name))
            target.hand.append(ctx.player.hand.pop(idx))
            moved += 1

        slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
        laid = min(self.egg_count, slot.eggs_space()) if slot.bird else 0
        if laid > 0:
            slot.eggs += laid
        return PowerResult(eggs_laid=laid, description=f"Gave {moved} card(s); laid {laid}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.8


class ChooseOtherPlayerBothGainFood(PowerEffect):
    """Choose one other player; both gain fixed food from supply."""

    def __init__(self, food_type: FoodType, count: int = 1):
        self.food_type = food_type
        self.count = int(max(0, count))

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        target = _pick_other_player(ctx, choice)
        if target is None:
            return PowerResult(executed=False, description="No other player")
        ctx.player.food_supply.add(self.food_type, self.count)
        target.food_supply.add(self.food_type, self.count)
        return PowerResult(food_gained={self.food_type: self.count}, description="Both players gained food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class ConditionalNeighborHasFoodGainFood(PowerEffect):
    """If neighbor has food type, gain one from supply."""

    def __init__(self, direction: str, food_type: FoodType, gain_count: int = 1):
        self.direction = direction
        self.food_type = food_type
        self.gain_count = int(max(0, gain_count))

    def _neighbors(self, ctx: PowerContext):
        if self.direction == "left":
            n = ctx.game_state.player_to_left(ctx.player)
            return [n] if n else []
        if self.direction == "right":
            n = ctx.game_state.player_to_right(ctx.player)
            return [n] if n else []
        if self.direction == "left_or_right":
            out = []
            l = ctx.game_state.player_to_left(ctx.player)
            r = ctx.game_state.player_to_right(ctx.player)
            if l is not None:
                out.append(l)
            if r is not None and (l is None or r.name != l.name):
                out.append(r)
            return out
        return []

    def execute(self, ctx: PowerContext) -> PowerResult:
        neighbors = self._neighbors(ctx)
        if not neighbors:
            return PowerResult(executed=False, description="No neighbor")
        if not any(n.food_supply.get(self.food_type) > 0 for n in neighbors):
            return PowerResult(executed=False, description=f"Neighbor has no {self.food_type.value}")
        ctx.player.food_supply.add(self.food_type, self.gain_count)
        return PowerResult(food_gained={self.food_type: self.gain_count}, description="Condition met; gained food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.5


class DiscardFoodToGainOtherFood(PowerEffect):
    """Discard specific food (or any food) to gain another food 1:1."""

    def __init__(self, discard_type: FoodType, gain_type: FoodType, max_discard: int | None = None):
        self.discard_type = discard_type
        self.gain_type = gain_type
        self.max_discard = max_discard

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        requested = choice.get("discard_count") if isinstance(choice, dict) else None
        try:
            requested_count = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            requested_count = None

        if self.discard_type == FoodType.WILD:
            available = sum(ctx.player.food_supply.get(ft) for ft in _all_food_types_for_board(ctx))
            upper = available
            if self.max_discard is not None:
                upper = min(upper, self.max_discard)
            count = upper if requested_count is None else max(0, min(requested_count, upper))
            if count <= 0:
                return PowerResult(executed=False, description="No food to discard")
            for _ in range(count):
                if _spend_any_food(ctx) is None:
                    break
        else:
            available = ctx.player.food_supply.get(self.discard_type)
            upper = available
            if self.max_discard is not None:
                upper = min(upper, self.max_discard)
            count = upper if requested_count is None else max(0, min(requested_count, upper))
            if count <= 0:
                return PowerResult(executed=False, description="No food to discard")
            if not ctx.player.food_supply.spend(self.discard_type, count):
                return PowerResult(executed=False, description="Failed to discard")

        ctx.player.food_supply.add(self.gain_type, count)
        return PowerResult(food_gained={self.gain_type: count}, description=f"Discarded {count}, gained {count}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.4


class GainFoodOrDiscardSameGainOther(PowerEffect):
    """Gain one base food, or discard one base food to gain one target food."""

    def __init__(self, base_food: FoodType, alternate_gain: FoodType):
        self.base_food = base_food
        self.alternate_gain = alternate_gain

    def execute(self, ctx: PowerContext) -> PowerResult:
        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        mode = str(choice.get("mode")).lower() if isinstance(choice, dict) and choice.get("mode") is not None else "gain"

        if mode == "discard_for_other" and ctx.player.food_supply.spend(self.base_food, 1):
            ctx.player.food_supply.add(self.alternate_gain, 1)
            return PowerResult(food_gained={self.alternate_gain: 1}, description="Discarded base food for alternate gain")

        ctx.player.food_supply.add(self.base_food, 1)
        return PowerResult(food_gained={self.base_food: 1}, description="Gained base food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.6


class GainFoodPerBirdWingspanInHabitat(PowerEffect):
    """Gain food equal to number of birds in habitat under wingspan threshold."""

    def __init__(self, food_type: FoodType, habitat: Habitat, max_wingspan_exclusive: int):
        self.food_type = food_type
        self.habitat = habitat
        self.max_wingspan_exclusive = int(max_wingspan_exclusive)

    def execute(self, ctx: PowerContext) -> PowerResult:
        count = 0
        row = ctx.player.board.get_row(self.habitat)
        for slot in row.slots:
            if not slot.bird:
                continue
            ws = slot.bird.wingspan_cm
            if ws is None:
                continue
            if ws < self.max_wingspan_exclusive:
                count += 1
        if count <= 0:
            return PowerResult(executed=False, description="No matching wingspan birds")
        ctx.player.food_supply.add(self.food_type, count)
        return PowerResult(food_gained={self.food_type: count}, description=f"Gained {count} {self.food_type.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9


class PerEggedBirdInHabitatRollDieGainFood(PowerEffect):
    """For each bird with egg in habitat, roll a die and gain one rolled food."""

    def __init__(self, habitat: Habitat):
        self.habitat = habitat

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(self.habitat)
        rolls = sum(1 for s in row.slots if s.bird and s.eggs > 0)
        if rolls <= 0:
            return PowerResult(executed=False, description="No egged birds in habitat")

        gained: dict[FoodType, int] = {}
        for _ in range(rolls):
            face = random.choice(ctx.game_state.birdfeeder.dice_faces)
            if isinstance(face, tuple):
                ft = min(face, key=lambda f: (ctx.player.food_supply.get(f), f.value))
            else:
                ft = face
            ctx.player.food_supply.add(ft, 1)
            gained[ft] = gained.get(ft, 0) + 1

        return PowerResult(food_gained=gained, description=f"Rolled {rolls} die; gained {sum(gained.values())} food")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.0


class RollOneDiePerBirdInHabitatIfAnyTargetGainMayCache(PowerEffect):
    """Roll one die per bird in habitat; if any hit target food, gain one and may cache."""

    def __init__(self, habitat: Habitat, target_food: FoodType, gain_food: FoodType):
        self.habitat = habitat
        self.target_food = target_food
        self.gain_food = gain_food

    def execute(self, ctx: PowerContext) -> PowerResult:
        row = ctx.player.board.get_row(self.habitat)
        rolls = sum(1 for s in row.slots if s.bird)
        if rolls <= 0:
            return PowerResult(executed=False, description="No birds in habitat")

        hit = False
        for _ in range(rolls):
            face = random.choice(ctx.game_state.birdfeeder.dice_faces)
            if face == self.target_food or (isinstance(face, tuple) and self.target_food in face):
                hit = True

        if not hit:
            return PowerResult(executed=False, description="No target food rolled")

        choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) or {}
        cache = bool(choice.get("cache", True)) if isinstance(choice, dict) else True
        if cache:
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            slot.cache_food(self.gain_food, 1)
            return PowerResult(food_cached={self.gain_food: 1}, description="Hit target; cached 1")

        ctx.player.food_supply.add(self.gain_food, 1)
        return PowerResult(food_gained={self.gain_food: 1}, description="Hit target; gained 1")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.9
