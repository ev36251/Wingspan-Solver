"""Tests for custom bonus card scoring (21 cards without spreadsheet eligibility)."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat, NestType, PowerColor
from backend.models.board import PlayerBoard, BirdSlot
from backend.models.player import Player, FoodSupply
from backend.models.game_state import create_new_game
from backend.engine.scoring import (
    score_bonus_cards,
    _count_breeding_manager,
    _count_ecologist,
    _count_oologist,
    _count_visionary_leader,
    _count_behaviorist,
    _count_citizen_scientist,
    _count_ethologist,
    _longest_consecutive_sequence,
    _count_data_analyst,
    _count_ranger,
    _count_population_monitor,
    _count_mechanical_engineer,
    _score_site_selection_expert,
    _count_avian_theriogenologist,
    _count_pellet_dissector,
    _count_winter_feeder,
)


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture(scope="module")
def bonus_reg(regs):
    return regs[1]


def _place_bird(player, bird, habitat, slot_idx, eggs=0, tucked=0, cached=None):
    """Helper to place a bird on a player's board."""
    row = player.board.get_row(habitat)
    row.slots[slot_idx].bird = bird
    row.slots[slot_idx].eggs = eggs
    row.slots[slot_idx].tucked_cards = tucked
    if cached:
        for ft, count in cached.items():
            row.slots[slot_idx].cache_food(ft, count)


# --- Breeding Manager ---

class TestBreedingManager:
    def test_no_birds(self):
        player = Player(name="Test")
        assert _count_breeding_manager(player) == 0

    def test_birds_without_enough_eggs(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=3)
        assert _count_breeding_manager(player) == 0

    def test_birds_with_four_eggs(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=4)
        assert _count_breeding_manager(player) == 1

    def test_multiple_qualifying(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=5)
        _place_bird(player, bird, Habitat.WETLAND, 1, eggs=4)
        _place_bird(player, bird, Habitat.WETLAND, 2, eggs=2)
        assert _count_breeding_manager(player) == 2

    def test_scoring_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Breeding Manager")
        assert bonus is not None
        player.bonus_cards.append(bonus)
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=4)
        _place_bird(player, bird, Habitat.WETLAND, 1, eggs=5)
        # 2 per-bird * 1pt = 2
        assert score_bonus_cards(player) == 2


# --- Ecologist ---

class TestEcologist:
    def test_empty_board(self):
        player = Player(name="Test")
        assert _count_ecologist(player) == 0

    def test_equal_birds_each_habitat(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0)
        _place_bird(player, bird, Habitat.GRASSLAND, 0)
        _place_bird(player, bird, Habitat.WETLAND, 0)
        assert _count_ecologist(player) == 1

    def test_unequal_habitats(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0)
        _place_bird(player, bird, Habitat.FOREST, 1)
        _place_bird(player, bird, Habitat.GRASSLAND, 0)
        # Min is 0 (wetland)
        assert _count_ecologist(player) == 0

    def test_scoring_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Ecologist")
        assert bonus is not None
        player.bonus_cards.append(bonus)
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0)
        _place_bird(player, bird, Habitat.FOREST, 1)
        _place_bird(player, bird, Habitat.GRASSLAND, 0)
        _place_bird(player, bird, Habitat.GRASSLAND, 1)
        _place_bird(player, bird, Habitat.WETLAND, 0)
        _place_bird(player, bird, Habitat.WETLAND, 1)
        # Min is 2, 2 per bird = 4
        assert score_bonus_cards(player) == 4


# --- Oologist ---

class TestOologist:
    def test_no_eggs(self, bird_reg):
        player = Player(name="Test")
        _place_bird(player, bird_reg.get("American Crow"), Habitat.FOREST, 0)
        assert _count_oologist(player) == 0

    def test_all_birds_with_eggs(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        for i in range(5):
            _place_bird(player, bird, Habitat.FOREST, i, eggs=1)
        for i in range(4):
            _place_bird(player, bird, Habitat.GRASSLAND, i, eggs=1)
        assert _count_oologist(player) == 9

    def test_scoring_tiers(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Oologist")
        assert bonus is not None
        player.bonus_cards.append(bonus)
        bird = bird_reg.get("American Crow")
        # 7 birds with eggs = 3pts
        for i in range(5):
            _place_bird(player, bird, Habitat.FOREST, i, eggs=1)
        _place_bird(player, bird, Habitat.GRASSLAND, 0, eggs=1)
        _place_bird(player, bird, Habitat.GRASSLAND, 1, eggs=1)
        assert score_bonus_cards(player) == 3


# --- Visionary Leader ---

class TestVisionaryLeader:
    def test_empty_hand(self):
        player = Player(name="Test")
        assert _count_visionary_leader(player) == 0

    def test_known_cards(self, bird_reg):
        player = Player(name="Test")
        for _ in range(5):
            player.hand.append(bird_reg.get("American Crow"))
        assert _count_visionary_leader(player) == 5

    def test_unknown_cards(self):
        player = Player(name="Test")
        player.unknown_hand_count = 6
        assert _count_visionary_leader(player) == 6

    def test_mixed(self, bird_reg):
        player = Player(name="Test")
        for _ in range(3):
            player.hand.append(bird_reg.get("American Crow"))
        player.unknown_hand_count = 5
        assert _count_visionary_leader(player) == 8

    def test_scoring_tiers(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Visionary Leader")
        assert bonus is not None
        player.bonus_cards.append(bonus)
        player.unknown_hand_count = 8
        # 8+ = 7pts
        assert score_bonus_cards(player) == 7


# --- Behaviorist ---

class TestBehaviorist:
    def test_empty_board(self):
        player = Player(name="Test")
        assert _count_behaviorist(player) == 0

    def test_column_with_three_colors(self, bird_reg):
        player = Player(name="Test")
        # We need 3 birds in the same column with different power colors
        # Find birds by color
        browns = [b for b in bird_reg.all_birds
                  if b.color == PowerColor.BROWN and Habitat.FOREST in b.habitats]
        whites = [b for b in bird_reg.all_birds
                  if b.color == PowerColor.WHITE and Habitat.GRASSLAND in b.habitats]
        pinks = [b for b in bird_reg.all_birds
                 if b.color == PowerColor.PINK and Habitat.WETLAND in b.habitats]

        if browns and whites and pinks:
            _place_bird(player, browns[0], Habitat.FOREST, 0)
            _place_bird(player, whites[0], Habitat.GRASSLAND, 0)
            _place_bird(player, pinks[0], Habitat.WETLAND, 0)
            assert _count_behaviorist(player) == 1

    def test_no_power_counts_as_white(self, bird_reg):
        player = Player(name="Test")
        # Birds with no power (color=NONE) count as white
        no_powers = [b for b in bird_reg.all_birds
                     if b.color == PowerColor.NONE and Habitat.FOREST in b.habitats]
        browns = [b for b in bird_reg.all_birds
                  if b.color == PowerColor.BROWN and Habitat.GRASSLAND in b.habitats]
        pinks = [b for b in bird_reg.all_birds
                 if b.color == PowerColor.PINK and Habitat.WETLAND in b.habitats]

        if no_powers and browns and pinks:
            # NONE counts as WHITE, so column has: white, brown, pink = 3 colors
            _place_bird(player, no_powers[0], Habitat.FOREST, 0)
            _place_bird(player, browns[0], Habitat.GRASSLAND, 0)
            _place_bird(player, pinks[0], Habitat.WETLAND, 0)
            assert _count_behaviorist(player) == 1


# --- Citizen Scientist ---

class TestCitizenScientist:
    def test_no_tucked(self, bird_reg):
        player = Player(name="Test")
        _place_bird(player, bird_reg.get("American Crow"), Habitat.FOREST, 0)
        assert _count_citizen_scientist(player) == 0

    def test_with_tucked(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0, tucked=3)
        _place_bird(player, bird, Habitat.FOREST, 1, tucked=1)
        _place_bird(player, bird, Habitat.FOREST, 2, tucked=0)
        assert _count_citizen_scientist(player) == 2


# --- Ethologist ---

class TestEthologist:
    def test_empty_board(self):
        player = Player(name="Test")
        assert _count_ethologist(player) == 0

    def test_single_habitat_multiple_colors(self, bird_reg):
        player = Player(name="Test")
        browns = [b for b in bird_reg.all_birds
                  if b.color == PowerColor.BROWN and Habitat.FOREST in b.habitats]
        whites = [b for b in bird_reg.all_birds
                  if b.color == PowerColor.WHITE and Habitat.FOREST in b.habitats]
        pinks = [b for b in bird_reg.all_birds
                 if b.color == PowerColor.PINK and Habitat.FOREST in b.habitats]

        placed = 0
        if browns:
            _place_bird(player, browns[0], Habitat.FOREST, placed)
            placed += 1
        if whites:
            _place_bird(player, whites[0], Habitat.FOREST, placed)
            placed += 1
        if pinks:
            _place_bird(player, pinks[0], Habitat.FOREST, placed)
            placed += 1

        assert _count_ethologist(player) == placed


# --- Consecutive Sequence Helper ---

class TestConsecutiveSequence:
    def test_empty(self):
        assert _longest_consecutive_sequence([]) == 0

    def test_single(self):
        assert _longest_consecutive_sequence([5]) == 1

    def test_ascending(self):
        assert _longest_consecutive_sequence([1, 3, 5, 7]) == 4

    def test_descending(self):
        assert _longest_consecutive_sequence([7, 5, 3, 1]) == 4

    def test_mixed(self):
        # 1, 3, 5 ascending = 3, then 2 breaks
        assert _longest_consecutive_sequence([1, 3, 5, 2, 4]) == 3

    def test_equal_values_break(self):
        assert _longest_consecutive_sequence([1, 3, 3, 5]) == 2

    def test_none_breaks(self):
        assert _longest_consecutive_sequence([1, None, 5]) == 1

    def test_all_none(self):
        assert _longest_consecutive_sequence([None, None]) == 1


# --- Data Analyst ---

class TestDataAnalyst:
    def test_ascending_wingspans(self, bird_reg):
        player = Player(name="Test")
        # Find birds with known wingspans for forest
        forest_birds = sorted(
            [b for b in bird_reg.all_birds
             if Habitat.FOREST in b.habitats and b.wingspan_cm is not None],
            key=lambda b: b.wingspan_cm,
        )
        if len(forest_birds) >= 3:
            _place_bird(player, forest_birds[0], Habitat.FOREST, 0)
            _place_bird(player, forest_birds[1], Habitat.FOREST, 1)
            _place_bird(player, forest_birds[2], Habitat.FOREST, 2)
            result = _count_data_analyst(player, Habitat.FOREST)
            assert result >= 3

    def test_no_birds(self):
        player = Player(name="Test")
        assert _count_data_analyst(player, Habitat.FOREST) == 0


# --- Ranger ---

class TestRanger:
    def test_ascending_vp(self, bird_reg):
        player = Player(name="Test")
        # Find forest birds sorted by VP
        forest_birds = sorted(
            [b for b in bird_reg.all_birds if Habitat.FOREST in b.habitats],
            key=lambda b: b.victory_points,
        )
        # Pick 3 with strictly different VPs
        unique_vp = []
        seen = set()
        for b in forest_birds:
            if b.victory_points not in seen:
                unique_vp.append(b)
                seen.add(b.victory_points)
            if len(unique_vp) >= 3:
                break

        if len(unique_vp) >= 3:
            for i, bird in enumerate(unique_vp[:3]):
                _place_bird(player, bird, Habitat.FOREST, i)
            result = _count_ranger(player, Habitat.FOREST)
            assert result >= 3


# --- Population Monitor ---

class TestPopulationMonitor:
    def test_empty(self):
        player = Player(name="Test")
        assert _count_population_monitor(player, Habitat.FOREST) == 0

    def test_diverse_nests(self, bird_reg):
        player = Player(name="Test")
        # Find forest birds with different nest types
        nests_seen = set()
        placed = 0
        for b in bird_reg.all_birds:
            if (Habitat.FOREST in b.habitats
                    and b.nest_type != NestType.WILD
                    and b.nest_type not in nests_seen
                    and placed < 4):
                _place_bird(player, b, Habitat.FOREST, placed)
                nests_seen.add(b.nest_type)
                placed += 1

        result = _count_population_monitor(player, Habitat.FOREST)
        assert result == placed

    def test_wild_nest_adds_diversity(self, bird_reg):
        player = Player(name="Test")
        bowl_bird = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        wild_bird = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.WILD and Habitat.FOREST in b.habitats), None)

        if bowl_bird and wild_bird:
            _place_bird(player, bowl_bird, Habitat.FOREST, 0)
            _place_bird(player, wild_bird, Habitat.FOREST, 1)
            # Bowl + wild-as-other = 2 distinct types
            assert _count_population_monitor(player, Habitat.FOREST) == 2


# --- Mechanical Engineer ---

class TestMechanicalEngineer:
    def test_empty_board(self):
        player = Player(name="Test")
        assert _count_mechanical_engineer(player) == 0

    def test_one_complete_set(self, bird_reg):
        player = Player(name="Test")
        nests_placed = {}
        slot = 0
        for b in bird_reg.all_birds:
            nt = b.nest_type
            if nt == NestType.WILD:
                continue
            if nt not in nests_placed and Habitat.FOREST in b.habitats and slot < 5:
                _place_bird(player, b, Habitat.FOREST, slot)
                nests_placed[nt] = True
                slot += 1
            if len(nests_placed) == 4:
                break

        if len(nests_placed) == 4:
            assert _count_mechanical_engineer(player) == 1

    def test_wild_fills_gap(self, bird_reg):
        player = Player(name="Test")
        # Place 3 nest types + 1 wild = 1 complete set
        nests_placed = {}
        slot = 0
        for b in bird_reg.all_birds:
            nt = b.nest_type
            if nt == NestType.WILD:
                continue
            if nt not in nests_placed and Habitat.FOREST in b.habitats and slot < 3:
                _place_bird(player, b, Habitat.FOREST, slot)
                nests_placed[nt] = True
                slot += 1
            if len(nests_placed) == 3:
                break

        wild_bird = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.WILD and Habitat.FOREST in b.habitats), None)

        if len(nests_placed) == 3 and wild_bird:
            _place_bird(player, wild_bird, Habitat.FOREST, 3)
            assert _count_mechanical_engineer(player) == 1


# --- Site Selection Expert ---

class TestSiteSelectionExpert:
    def test_empty_board(self):
        player = Player(name="Test")
        assert _score_site_selection_expert(player) == 0

    def test_matching_pair(self, bird_reg):
        player = Player(name="Test")
        # Place 2 birds with same nest type in column 0
        bowl_forest = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        bowl_grass = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.GRASSLAND in b.habitats), None)

        if bowl_forest and bowl_grass:
            _place_bird(player, bowl_forest, Habitat.FOREST, 0)
            _place_bird(player, bowl_grass, Habitat.GRASSLAND, 0)
            assert _score_site_selection_expert(player) == 1

    def test_matching_trio(self, bird_reg):
        player = Player(name="Test")
        bowl_forest = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        bowl_grass = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.GRASSLAND in b.habitats), None)
        bowl_wetland = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.WETLAND in b.habitats), None)

        if bowl_forest and bowl_grass and bowl_wetland:
            _place_bird(player, bowl_forest, Habitat.FOREST, 0)
            _place_bird(player, bowl_grass, Habitat.GRASSLAND, 0)
            _place_bird(player, bowl_wetland, Habitat.WETLAND, 0)
            assert _score_site_selection_expert(player) == 3

    def test_wild_nest_helps_match(self, bird_reg):
        player = Player(name="Test")
        bowl_bird = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        wild_bird = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.WILD and Habitat.GRASSLAND in b.habitats), None)

        if bowl_bird and wild_bird:
            _place_bird(player, bowl_bird, Habitat.FOREST, 0)
            _place_bird(player, wild_bird, Habitat.GRASSLAND, 0)
            # Wild matches bowl → pair → 1pt
            assert _score_site_selection_expert(player) == 1

    def test_multiple_columns(self, bird_reg):
        player = Player(name="Test")
        bowl_f = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        bowl_g = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.GRASSLAND in b.habitats), None)
        cavity_f = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.CAVITY and Habitat.FOREST in b.habitats), None)
        cavity_g = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.CAVITY and Habitat.GRASSLAND in b.habitats), None)

        if bowl_f and bowl_g and cavity_f and cavity_g:
            _place_bird(player, bowl_f, Habitat.FOREST, 0)
            _place_bird(player, bowl_g, Habitat.GRASSLAND, 0)
            _place_bird(player, cavity_f, Habitat.FOREST, 1)
            _place_bird(player, cavity_g, Habitat.GRASSLAND, 1)
            # 2 columns with pairs = 2 points
            assert _score_site_selection_expert(player) == 2


# --- Avian Theriogenologist ---

class TestAvianTheriogenologist:
    def test_no_full_nests(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=1)
        assert _count_avian_theriogenologist(player) == 0

    def test_full_nests(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        _place_bird(player, bird, Habitat.WETLAND, 0, eggs=bird.egg_limit)
        assert _count_avian_theriogenologist(player) == 1

    def test_zero_egg_limit_excluded(self, bird_reg):
        player = Player(name="Test")
        # Find a bird with egg_limit == 0
        zero_egg = next(
            (b for b in bird_reg.all_birds if b.egg_limit == 0), None)
        if zero_egg:
            for hab in zero_egg.habitats:
                _place_bird(player, zero_egg, hab, 0, eggs=0)
                break
            assert _count_avian_theriogenologist(player) == 0


# --- Pellet Dissector ---

class TestPelletDissector:
    def test_no_cached(self, bird_reg):
        player = Player(name="Test")
        _place_bird(player, bird_reg.get("American Crow"), Habitat.FOREST, 0)
        assert _count_pellet_dissector(player) == 0

    def test_fish_and_rodent_cached(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0,
                    cached={FoodType.FISH: 2, FoodType.RODENT: 1})
        assert _count_pellet_dissector(player) == 3

    def test_other_food_not_counted(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")
        _place_bird(player, bird, Habitat.FOREST, 0,
                    cached={FoodType.SEED: 5, FoodType.FISH: 1})
        assert _count_pellet_dissector(player) == 1


# --- Winter Feeder ---

class TestWinterFeeder:
    def test_empty_supply(self):
        player = Player(name="Test")
        assert _count_winter_feeder(player) == 0

    def test_with_food(self):
        player = Player(name="Test")
        player.food_supply.add(FoodType.SEED, 3)
        player.food_supply.add(FoodType.FISH, 2)
        player.food_supply.add(FoodType.NECTAR, 2)
        assert _count_winter_feeder(player) == 7

    def test_scoring_tiers(self, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Winter Feeder")
        assert bonus is not None
        player.bonus_cards.append(bonus)
        player.food_supply.add(FoodType.SEED, 4)
        # 4-6 food = 4pts
        assert score_bonus_cards(player) == 4


# --- Integration: score_bonus_cards with custom cards ---

class TestCustomBonusIntegration:
    def test_site_selection_expert_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Site Selection Expert")
        assert bonus is not None
        player.bonus_cards.append(bonus)

        bowl_f = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.FOREST in b.habitats), None)
        bowl_g = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.GRASSLAND in b.habitats), None)
        bowl_w = next(
            (b for b in bird_reg.all_birds
             if b.nest_type == NestType.BOWL and Habitat.WETLAND in b.habitats), None)

        if bowl_f and bowl_g and bowl_w:
            _place_bird(player, bowl_f, Habitat.FOREST, 0)
            _place_bird(player, bowl_g, Habitat.GRASSLAND, 0)
            _place_bird(player, bowl_w, Habitat.WETLAND, 0)
            # Trio in column 0 = 3pts
            assert score_bonus_cards(player) == 3

    def test_data_analyst_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Forest Data Analyst")
        assert bonus is not None
        player.bonus_cards.append(bonus)

        # Place 3 forest birds with ascending wingspans
        forest_birds = sorted(
            [b for b in bird_reg.all_birds
             if Habitat.FOREST in b.habitats and b.wingspan_cm is not None],
            key=lambda b: b.wingspan_cm,
        )
        # Deduplicate wingspans to ensure strictly ascending
        unique = []
        seen_ws = set()
        for b in forest_birds:
            if b.wingspan_cm not in seen_ws:
                unique.append(b)
                seen_ws.add(b.wingspan_cm)
            if len(unique) >= 3:
                break

        if len(unique) >= 3:
            for i, bird in enumerate(unique[:3]):
                _place_bird(player, bird, Habitat.FOREST, i)
            # 3 consecutive ascending = 3pts
            assert score_bonus_cards(player) == 3

    def test_ranger_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Forest Ranger")
        assert bonus is not None
        player.bonus_cards.append(bonus)

        # Place 3 forest birds with ascending VP
        forest_birds = sorted(
            [b for b in bird_reg.all_birds if Habitat.FOREST in b.habitats],
            key=lambda b: b.victory_points,
        )
        unique = []
        seen_vp = set()
        for b in forest_birds:
            if b.victory_points not in seen_vp:
                unique.append(b)
                seen_vp.add(b.victory_points)
            if len(unique) >= 3:
                break

        if len(unique) >= 3:
            for i, bird in enumerate(unique[:3]):
                _place_bird(player, bird, Habitat.FOREST, i)
            # 3 consecutive ascending = 3pts
            assert score_bonus_cards(player) == 3

    def test_population_monitor_integration(self, bird_reg, bonus_reg):
        player = Player(name="Test")
        bonus = bonus_reg.get("Forest Population Monitor")
        assert bonus is not None
        player.bonus_cards.append(bonus)

        # Place birds with different nest types in forest
        nests_seen = set()
        placed = 0
        for b in bird_reg.all_birds:
            if (Habitat.FOREST in b.habitats
                    and b.nest_type != NestType.WILD
                    and b.nest_type not in nests_seen
                    and placed < 4):
                _place_bird(player, b, Habitat.FOREST, placed)
                nests_seen.add(b.nest_type)
                placed += 1

        if placed >= 4:
            # 4 nest types = 5pts
            assert score_bonus_cards(player) == 5

    def test_multiple_custom_bonus_cards(self, bird_reg, bonus_reg):
        player = Player(name="Test")

        oologist = bonus_reg.get("Oologist")
        winter = bonus_reg.get("Winter Feeder")
        assert oologist and winter
        player.bonus_cards.extend([oologist, winter])

        bird = bird_reg.get("American Crow")
        for i in range(5):
            _place_bird(player, bird, Habitat.FOREST, i, eggs=1)
        for i in range(4):
            _place_bird(player, bird, Habitat.GRASSLAND, i, eggs=1)

        player.food_supply.add(FoodType.SEED, 7)

        # Oologist: 9 birds with eggs = 6pts
        # Winter Feeder: 7 food = 7pts
        assert score_bonus_cards(player) == 13
