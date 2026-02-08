"""Tests for the setup draft advisor."""

import pytest
from backend.data.registries import load_all, get_bird_registry, get_bonus_registry, get_goal_registry
from backend.config import EXCEL_FILE
from backend.solver.setup_advisor import (
    analyze_setup, _can_afford_bird, _play_birds_greedily,
    _evaluate_option, STARTING_FOOD, SetupRecommendation,
)
from backend.models.enums import FoodType


@pytest.fixture(scope="module", autouse=True)
def _load_data():
    load_all(EXCEL_FILE)


@pytest.fixture
def bird_reg():
    return get_bird_registry()


@pytest.fixture
def bonus_reg():
    return get_bonus_registry()


@pytest.fixture
def goal_reg():
    return get_goal_registry()


class TestCanAffordBird:
    def test_free_bird(self, bird_reg):
        """A bird with 0 food cost is always affordable."""
        # Find a free bird
        for bird in bird_reg.all_birds:
            if bird.food_cost.total == 0:
                assert _can_afford_bird(bird, {})
                break

    def test_affordable_with_matching_food(self, bird_reg):
        """A bird whose cost matches available food should be affordable."""
        # Find a bird with non-wild food costs
        for bird in bird_reg.all_birds:
            if bird.food_cost.total > 0 and FoodType.WILD not in bird.food_cost.items:
                food = {}
                for ft in bird.food_cost.items:
                    food[ft] = food.get(ft, 0) + 1
                assert _can_afford_bird(bird, food), f"{bird.name} should be affordable"
                break

    def test_not_affordable_empty(self, bird_reg):
        """An expensive bird with no food should not be affordable."""
        for bird in bird_reg.all_birds:
            if bird.food_cost.total >= 3 and not bird.food_cost.is_or:
                assert not _can_afford_bird(bird, {})
                break

    def test_nectar_substitution(self, bird_reg):
        """Nectar should substitute for any food type."""
        for bird in bird_reg.all_birds:
            if bird.food_cost.total == 1 and not bird.food_cost.is_or:
                food = {FoodType.NECTAR: 1}
                assert _can_afford_bird(bird, food)
                break


class TestPlayBirdsGreedily:
    def test_no_birds(self):
        """Empty bird list returns empty playable list."""
        playable, remaining = _play_birds_greedily([], {FoodType.SEED: 3})
        assert playable == []
        assert remaining[FoodType.SEED] == 3

    def test_all_free_birds(self, bird_reg):
        """Free birds are all playable."""
        free_birds = [b for b in bird_reg.all_birds if b.food_cost.total == 0][:3]
        playable, _ = _play_birds_greedily(free_birds, {})
        assert len(playable) == len(free_birds)

    def test_limited_food(self, bird_reg):
        """With limited food, not all expensive birds can be played."""
        expensive = [b for b in bird_reg.all_birds
                     if b.food_cost.total >= 2 and not b.food_cost.is_or][:3]
        if len(expensive) >= 2:
            # Give only 2 food total
            food = {FoodType.SEED: 1, FoodType.INVERTEBRATE: 1}
            playable, _ = _play_birds_greedily(expensive, food)
            assert len(playable) < len(expensive)


class TestAnalyzeSetup:
    def test_returns_recommendations(self, bird_reg, bonus_reg):
        """analyze_setup returns a non-empty list of recommendations."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        assert len(results) > 0
        assert len(results) <= 10

    def test_recommendations_are_ranked(self, bird_reg, bonus_reg):
        """Recommendations should be ranked 1, 2, 3, ..."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        for i, rec in enumerate(results):
            assert rec.rank == i + 1

    def test_scores_descending(self, bird_reg, bonus_reg):
        """Recommendations should be in descending score order."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_bird_food_total_is_five(self, bird_reg, bonus_reg):
        """Each recommendation should keep birds + food = 5 total."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        for rec in results:
            num_birds = len(rec.birds_to_keep)
            num_food = sum(rec.food_to_keep.values())
            assert num_birds + num_food == 5, (
                f"Expected 5 total, got {num_birds} birds + {num_food} food"
            )

    def test_with_goals(self, bird_reg, bonus_reg, goal_reg):
        """Including goals should not cause errors."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        goals = goal_reg.all_goals[:4]
        results = analyze_setup(birds, bonus_cards, goals)
        assert len(results) > 0

    def test_single_bonus_card(self, bird_reg, bonus_reg):
        """Works with a single bonus card."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:1]
        results = analyze_setup(birds, bonus_cards, [])
        assert len(results) > 0
        assert all(r.bonus_card == bonus_cards[0].name for r in results)

    def test_fewer_than_five_birds(self, bird_reg, bonus_reg):
        """Works with fewer than 5 birds dealt."""
        birds = bird_reg.all_birds[:3]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        assert len(results) > 0
        for rec in results:
            assert len(rec.birds_to_keep) <= 3

    def test_has_reasoning(self, bird_reg, bonus_reg):
        """Each recommendation should have reasoning text."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(birds, bonus_cards, [])
        for rec in results:
            assert rec.reasoning != ""

    def test_bonus_card_is_from_dealt(self, bird_reg, bonus_reg):
        """Recommended bonus card should be one of the dealt cards."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        bc_names = {bc.name for bc in bonus_cards}
        results = analyze_setup(birds, bonus_cards, [])
        for rec in results:
            assert rec.bonus_card in bc_names
