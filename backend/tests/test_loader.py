"""Tests for Excel data loading and parsing correctness."""

import pytest
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.loader import load_birds, load_bonus_cards, load_goals
from backend.data.registries import load_all, BirdRegistry, BonusCardRegistry, GoalRegistry
from backend.models.enums import (
    FoodType, Habitat, NestType, PowerColor, BeakDirection, GameSet,
)


@pytest.fixture(scope="module")
def registries():
    """Load all data once for the test module."""
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(registries) -> BirdRegistry:
    return registries[0]


@pytest.fixture(scope="module")
def bonus_reg(registries) -> BonusCardRegistry:
    return registries[1]


@pytest.fixture(scope="module")
def goal_reg(registries) -> GoalRegistry:
    return registries[2]


# --- Bird count and set filtering ---

class TestBirdLoading:
    def test_total_bird_count(self, bird_reg):
        """Should load birds from core + european + oceania + asia only."""
        assert len(bird_reg) >= 440
        assert len(bird_reg) <= 460

    def test_no_americas_birds(self, bird_reg):
        """No americas or promo birds should be loaded."""
        for bird in bird_reg.all_birds:
            assert bird.game_set in {GameSet.CORE, GameSet.EUROPEAN, GameSet.OCEANIA, GameSet.ASIA}

    def test_birds_per_set(self, bird_reg):
        assert len(bird_reg.by_set(GameSet.CORE)) == 180
        assert len(bird_reg.by_set(GameSet.EUROPEAN)) == 81
        assert len(bird_reg.by_set(GameSet.OCEANIA)) == 95
        assert len(bird_reg.by_set(GameSet.ASIA)) == 90

    # --- Specific bird spot checks ---

    def test_abbotts_booby(self, bird_reg):
        """Check a known Oceania bird parses correctly."""
        bird = bird_reg.get("Abbott's Booby")
        assert bird is not None
        assert bird.game_set == GameSet.OCEANIA
        assert bird.color == PowerColor.WHITE
        assert bird.victory_points == 5
        assert bird.nest_type == NestType.PLATFORM
        assert bird.egg_limit == 1
        assert bird.wingspan_cm == 190
        assert Habitat.WETLAND in bird.habitats
        assert Habitat.FOREST not in bird.habitats
        assert bird.food_cost.total == 2
        assert bird.food_cost.is_or is False
        assert bird.beak_direction == BeakDirection.LEFT
        assert "bonus cards" in bird.power_text.lower() or "Draw 3 bonus cards" in bird.power_text

    def test_acorn_woodpecker(self, bird_reg):
        """Check a core brown bird."""
        bird = bird_reg.get("Acorn Woodpecker")
        assert bird is not None
        assert bird.game_set == GameSet.CORE
        assert bird.color == PowerColor.BROWN
        assert bird.nest_type == NestType.CAVITY
        assert "seed" in bird.power_text.lower() or "[seed]" in bird.power_text

    def test_american_robin_or_cost(self, bird_reg):
        """American Robin has a slash (OR) food cost."""
        bird = bird_reg.get("American Robin")
        assert bird is not None
        assert bird.food_cost.is_or is True
        # Should have invertebrate and fruit as options
        types = bird.food_cost.distinct_types
        assert FoodType.INVERTEBRATE in types
        assert FoodType.FRUIT in types

    def test_no_power_birds(self, bird_reg):
        """Birds with no power should have PowerColor.NONE."""
        bird = bird_reg.get("Trumpeter Swan")
        assert bird is not None
        assert bird.color == PowerColor.NONE
        assert bird.victory_points >= 8

    def test_flightless_bird(self, bird_reg):
        """Birds with '*' wingspan should have None."""
        # Check if any bird has None wingspan
        flightless = [b for b in bird_reg.all_birds if b.wingspan_cm is None]
        # There should be at least one (Kiwi birds, etc.)
        assert len(flightless) >= 1

    def test_nectar_food_cost(self, bird_reg):
        """Oceania birds can have nectar in their food cost."""
        bird = bird_reg.get("Rainbow Lorikeet")
        assert bird is not None
        assert bird.game_set == GameSet.OCEANIA
        assert FoodType.NECTAR in bird.food_cost.distinct_types

    def test_wild_food_cost(self, bird_reg):
        """Birds with wild food in their cost."""
        bird = bird_reg.get("American Crow")
        assert bird is not None
        assert FoodType.WILD in bird.food_cost.distinct_types

    def test_predator_bird(self, bird_reg):
        """Predator birds should be flagged."""
        predators = [b for b in bird_reg.all_birds if b.is_predator]
        assert len(predators) >= 30  # There are many predators across sets

    def test_multi_habitat_bird(self, bird_reg):
        """Some birds can live in multiple habitats."""
        multi = [b for b in bird_reg.all_birds if len(b.habitats) > 1]
        assert len(multi) > 50

    def test_search(self, bird_reg):
        """Bird name search should find partial matches."""
        results = bird_reg.search("eagle")
        assert len(results) >= 3
        assert all("eagle" in b.name.lower() for b in results)


# --- Bonus card tests ---

class TestBonusCardLoading:
    def test_bonus_count(self, bonus_reg):
        """Should load bonus cards from included sets only."""
        assert len(bonus_reg) >= 30
        assert len(bonus_reg) <= 55

    def test_no_americas_bonus(self, bonus_reg):
        """No americas-only bonus cards."""
        americas_only = {"Hummingbird Gardener", "Hummingbird Counter", "Apiarist",
                         "Brilliant Specialist", "Emerald Specialist",
                         "Mango Specialist", "Topaz Specialist", "Ecolodge Owner"}
        for card in bonus_reg.all_cards:
            assert card.name not in americas_only

    def test_per_bird_scoring(self, bonus_reg):
        """Bird Counter should be 2 per bird."""
        card = bonus_reg.get("Bird Counter")
        assert card is not None
        assert card.is_per_bird is True
        assert card.score(0) == 0
        assert card.score(3) == 6
        assert card.score(5) == 10

    def test_tiered_scoring(self, bonus_reg):
        """Backyard Birder: 5-6 birds = 3pts, 7+ = 6pts."""
        card = bonus_reg.get("Backyard Birder")
        assert card is not None
        assert card.is_per_bird is False
        assert card.score(4) == 0
        assert card.score(5) == 3
        assert card.score(6) == 3
        assert card.score(7) == 6
        assert card.score(10) == 6

    def test_anatomist_multi_set(self, bonus_reg):
        """Anatomist belongs to both core and asia."""
        card = bonus_reg.get("Anatomist")
        assert card is not None
        assert GameSet.CORE in card.game_sets
        assert GameSet.ASIA in card.game_sets

    def test_three_tier_scoring(self, bonus_reg):
        """Forester: 3-4 birds = 4pts, 5 birds = 8pts."""
        card = bonus_reg.get("Forester")
        assert card is not None
        assert card.score(2) == 0
        assert card.score(3) == 4
        assert card.score(4) == 4
        assert card.score(5) == 8


# --- Goal tests ---

class TestGoalLoading:
    def test_goal_count(self, goal_reg):
        """Should load competitive goals (not Asia duet, not 'no goal')."""
        assert len(goal_reg) >= 28
        assert len(goal_reg) <= 40

    def test_no_asia_duet_goals(self, goal_reg):
        """Asia duet goals (with None scoring) should be excluded."""
        for goal in goal_reg.all_goals:
            assert not all(v == 0.0 for v in goal.scoring) or goal.game_set != GameSet.ASIA

    def test_core_goal_scoring(self, goal_reg):
        """Core goal: birds in forest = (0, 1, 2, 3) for 4th-1st."""
        forest_goals = [g for g in goal_reg.all_goals
                        if "forest" in g.description.lower() and "[bird]" in g.description]
        assert len(forest_goals) >= 1
        goal = forest_goals[0]
        assert goal.score_for_placement(1) == 3.0  # 1st place
        assert goal.score_for_placement(4) == 0.0  # 4th place

    def test_european_negative_scoring(self, goal_reg):
        """European goals can have negative scores."""
        eu_goals = goal_reg.by_set(GameSet.EUROPEAN)
        has_negative = any(
            any(v < 0 for v in g.scoring)
            for g in eu_goals
        )
        assert has_negative

    def test_goals_by_set(self, goal_reg):
        assert len(goal_reg.by_set(GameSet.CORE)) >= 14
        assert len(goal_reg.by_set(GameSet.EUROPEAN)) >= 8
        assert len(goal_reg.by_set(GameSet.OCEANIA)) >= 6
