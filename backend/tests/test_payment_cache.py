"""Correctness guards for the memoized food-payment layer in engine.rules.

can_pay_food_cost / find_food_payment_options are cached by (cost signature,
effective supply). These tests pin the invariants the cache must preserve:
identical answers to an uncached recompute, and fresh mutable dicts per call
(callers attach payment dicts to Move objects and may mutate them).
"""

import itertools

import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine import rules
from backend.engine.rules import (
    _find_food_payment_options_impl,
    _can_pay_food_cost_impl,
    _effective_supply,
    can_pay_food_cost,
    find_food_payment_options,
)
from backend.models.bird import FoodCost
from backend.models.enums import FoodType
from backend.models.player import FoodSupply, Player


def setup_module():
    load_all(EXCEL_FILE)


def _player(**food) -> Player:
    return Player(name="P", food_supply=FoodSupply(**food))


# A spread of representative costs: single, multi, OR, wild, nectar-only.
_COSTS = [
    FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
    FoodCost(items=(FoodType.INVERTEBRATE, FoodType.SEED), is_or=False, total=2),
    FoodCost(items=(FoodType.SEED, FoodType.SEED), is_or=False, total=2),
    FoodCost(items=(FoodType.FISH, FoodType.RODENT), is_or=True, total=1),
    FoodCost(items=(FoodType.WILD, FoodType.WILD), is_or=False, total=2),
    FoodCost(items=(FoodType.FRUIT, FoodType.WILD), is_or=False, total=2),
]


@pytest.fixture(autouse=True)
def _clear_caches():
    rules._can_pay_cache.clear()
    rules._pay_options_cache.clear()
    yield


def test_cached_matches_uncached_over_supply_grid():
    """Cached public API must equal a direct uncached recompute everywhere."""
    for counts in itertools.product([0, 1, 2, 3], repeat=3):
        inv, seed, nectar = counts
        player = _player(invertebrate=inv, seed=seed, nectar=nectar, fish=1, rodent=1, fruit=1)
        supply = _effective_supply(player)
        for cost in _COSTS:
            got_can = can_pay_food_cost(player, cost)
            want_can = _can_pay_food_cost_impl(supply, cost)
            assert got_can == want_can, (counts, cost)

            got_opts = find_food_payment_options(player, cost)
            want_opts = _find_food_payment_options_impl(supply, cost)
            assert got_opts == want_opts, (counts, cost)

            # A valid payment must be affordable and match the can-pay verdict.
            assert bool(got_opts) == got_can[0], (counts, cost)


def test_payment_options_are_independent_copies():
    """Two calls with the same key must return distinct, mutation-safe dicts."""
    player = _player(seed=3, invertebrate=3)
    cost = FoodCost(items=(FoodType.SEED,), is_or=False, total=1)

    first = find_food_payment_options(player, cost)
    assert first and first[0]  # non-empty payment
    first[0][FoodType.SEED] = 999  # caller mutates its copy

    second = find_food_payment_options(player, cost)
    assert second[0].get(FoodType.SEED) != 999, "cache leaked a mutated dict"


def test_cache_actually_populates():
    player = _player(seed=2)
    cost = FoodCost(items=(FoodType.SEED,), is_or=False, total=1)
    assert not rules._can_pay_cache
    can_pay_food_cost(player, cost)
    assert rules._can_pay_cache  # a hit was memoized


def test_cached_food_pool_counts_spendable_only():
    """Regression: the direct-iteration pool must equal the documented sum."""
    from backend.models.enums import Habitat

    player = _player(seed=1)
    forest = player.board.get_row(Habitat.FOREST)
    # Simulate a bird slot with spendable cached food.
    forest.slots[0].spendable_cached_food = {FoodType.FISH: 2}
    forest.slots[1].spendable_cached_food = {FoodType.FISH: 1, FoodType.SEED: 1}
    pool = rules._get_cached_food_pool(player)
    assert pool == {FoodType.FISH: 3, FoodType.SEED: 1}
