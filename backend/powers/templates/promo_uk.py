"""Exact power implementations for Promo UK Pack birds whose auto-parsed
mappings dropped clauses. One class per bird's full rules text."""

from backend.models.enums import FoodType, Habitat
from backend.powers.base import PowerEffect, PowerContext, PowerResult
from backend.powers.templates.tuck_cards import _draw_one_bird

_HABITATS = (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND)


def _gain_one_from_feeder(game, player):
    """Take 1 die from the birdfeeder (rerolling if empty), add to supply."""
    feeder = game.birdfeeder
    if not feeder.available_food_types():
        if hasattr(feeder, "should_reroll") and feeder.should_reroll():
            feeder.reroll()
    res = feeder.take_any()
    if res is None:
        return 0
    foods = res if isinstance(res, tuple) else (res,)
    for ft in foods:
        player.food_supply.add(ft, 1)
    return 1


class TuckHandThenAllPlayersGainDie(PowerEffect):
    """Tuck 1 [card] from hand behind this bird; if you do, each player gains 1
    [die] from the birdfeeder. (European Golden Plover.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        if not pl.hand:
            return PowerResult(executed=False, description="No card to tuck")
        pl.hand.pop()
        slot = pl.board.get_row(ctx.habitat).slots[ctx.slot_index]
        slot.tucked_cards += 1
        # Active player first, then others (start with player of your choice).
        ordered = [pl] + [p for p in ctx.game_state.players if p is not pl]
        for p in ordered:
            _gain_one_from_feeder(ctx.game_state, p)
        return PowerResult(cards_tucked=1, description="Tucked 1; each player gained a die")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 1.6 if ctx.player.hand else 0.0  # tuck + your own die; opponents also gain


class TuckHandThenCacheFoodPerOwnCost(PowerEffect):
    """Tuck 1 [card] from hand behind this bird; for each [food_type] in THIS
    bird's food cost, cache 1 [food_type] from the supply on it. (Tawny Owl.)"""

    def __init__(self, food_type: FoodType = FoodType.RODENT):
        self.food_type = food_type

    def _cost_count(self, ctx: PowerContext) -> int:
        items = getattr(ctx.bird.food_cost, "items", ())
        return sum(1 for f in items if f == self.food_type)

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        slot = pl.board.get_row(ctx.habitat).slots[ctx.slot_index]
        tucked = 0
        if pl.hand:
            pl.hand.pop()
            slot.tucked_cards += 1
            tucked = 1
        n = self._cost_count(ctx)
        if n > 0:
            slot.cache_food(self.food_type, n)
        return PowerResult(cards_tucked=tucked, food_cached={self.food_type: n} if n else {},
                           description=f"Tucked {tucked}; cached {n} {self.food_type.value}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return (1.0 if ctx.player.hand else 0.0) + self._cost_count(ctx)


class ColumnTuckFromHandThenDraw(PowerEffect):
    """Choose 1-3 birds in this column; tuck 1 [card] from hand behind each. If
    you tuck at least 1, draw 1 [card]. (Whooper Swan.)"""

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        col = ctx.slot_index
        # Birds in this column across the 3 habitats (best use of limited hand).
        targets = [pl.board.get_row(h).slots[col]
                   for h in _HABITATS
                   if pl.board.get_row(h).slots[col].bird is not None]
        tucked = 0
        for slot in targets:
            if tucked >= 3 or not pl.hand:
                break
            pl.hand.pop()
            slot.tucked_cards += 1
            tucked += 1
        drawn = 0
        if tucked >= 1:
            card = _draw_one_bird(ctx)
            if card is not None:
                pl.hand.append(card)
                drawn = 1
        return PowerResult(cards_tucked=tucked, cards_drawn=drawn,
                           description=f"Tucked {tucked} in column; drew {drawn}")

    def estimate_value(self, ctx: PowerContext) -> float:
        col = ctx.slot_index
        avail = sum(1 for h in _HABITATS
                    if ctx.player.board.get_row(h).slots[col].bird is not None)
        return min(min(avail, 3), len(ctx.player.hand)) * 0.9


class CacheWildThenGainDiePerCache(PowerEffect):
    """Cache up to 3 [wild] from your supply on this bird; for each, gain 1
    [die] from the birdfeeder. (Western Jackdaw.)"""

    def __init__(self, max_cache: int = 3):
        self.max_cache = max_cache

    def _spend_one_any_food(self, pl) -> FoodType | None:
        for ft in (FoodType.RODENT, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE,
                   FoodType.SEED, FoodType.NECTAR):
            if pl.food_supply.get(ft) > 0:
                pl.food_supply.spend(ft, 1)
                return ft
        return None

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        slot = pl.board.get_row(ctx.habitat).slots[ctx.slot_index]
        cached = {}
        n = 0
        for _ in range(self.max_cache):
            ft = self._spend_one_any_food(pl)
            if ft is None:
                break
            slot.cache_food(ft, 1)
            cached[ft] = cached.get(ft, 0) + 1
            _gain_one_from_feeder(ctx.game_state, pl)
            n += 1
        return PowerResult(food_cached=cached,
                           description=f"Cached {n} wild, gained {n} dice")

    def estimate_value(self, ctx: PowerContext) -> float:
        food = sum(ctx.player.food_supply.get(f) for f in FoodType)
        return min(self.max_cache, food) * 1.0  # cache scores 1; die roughly refunds the food


class DrawDiscardThenGainIfFoodInCost(PowerEffect):
    """Draw and discard 1 [card] from the deck. If it has [food_type] in its food
    cost, gain 1 [food_type] from the supply. (Ruddy Turnstone.)"""

    def __init__(self, food_type: FoodType = FoodType.INVERTEBRATE):
        self.food_type = food_type

    def execute(self, ctx: PowerContext) -> PowerResult:
        card = _draw_one_bird(ctx)  # drawn then discarded (not kept)
        if card is None:
            return PowerResult(executed=False, description="Deck empty")
        gained = {}
        items = getattr(getattr(card, "food_cost", None), "items", ())
        if any(f == self.food_type for f in items):
            ctx.player.food_supply.add(self.food_type, 1)
            gained = {self.food_type: 1}
        return PowerResult(food_gained=gained,
                           description=f"Drew/discarded {card.name}; gained {gained}")

    def estimate_value(self, ctx: PowerContext) -> float:
        return 0.5  # ~50% of cards carry a given food in cost


class DiscardFoodToLayEggs(PowerEffect):
    """Discard up to `max_n` food (from `allowed`) in any combination; for each,
    lay 1 [egg] on this bird. (European Stonechat.)"""

    def __init__(self, allowed=(FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FRUIT),
                 max_n: int = 3):
        self.allowed = tuple(allowed)
        self.max_n = max_n

    def execute(self, ctx: PowerContext) -> PowerResult:
        pl = ctx.player
        slot = pl.board.get_row(ctx.habitat).slots[ctx.slot_index]
        laid = 0
        for _ in range(self.max_n):
            if not slot.can_hold_more_eggs():
                break
            spent = None
            for ft in self.allowed:
                if pl.food_supply.get(ft) > 0:
                    pl.food_supply.spend(ft, 1)
                    spent = ft
                    break
            if spent is None:
                break
            slot.eggs += 1
            laid += 1
        return PowerResult(eggs_laid=laid, description=f"Discarded {laid} food, laid {laid} eggs")

    def estimate_value(self, ctx: PowerContext) -> float:
        pl = ctx.player
        food = sum(pl.food_supply.get(f) for f in self.allowed)
        slot = pl.board.get_row(ctx.habitat).slots[ctx.slot_index]
        space = slot.bird.egg_limit - slot.eggs if slot.bird else 0
        return float(min(self.max_n, food, max(0, space)))
