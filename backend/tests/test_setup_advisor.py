"""Tests for the setup draft advisor."""

import pytest
from backend.data.registries import load_all, get_bird_registry, get_bonus_registry, get_goal_registry
from backend.config import EXCEL_FILE
from backend.solver.setup_advisor import (
    analyze_setup, _can_afford_bird, _play_birds_greedily,
    _evaluate_option, _generate_reasoning, STARTING_FOOD, SetupRecommendation,
    CUSTOM_DRAFT_SYNERGY,
    _draft_synergy_behaviorist,
    _draft_synergy_ethologist,
    _draft_synergy_population_monitor,
    _draft_synergy_mechanical_engineer,
    _draft_synergy_site_selection,
    _draft_synergy_data_analyst,
    _draft_synergy_ranger,
    _draft_synergy_ecologist,
    _draft_synergy_avian_therio,
    _draft_synergy_breeding_manager,
    _draft_synergy_oologist,
    _draft_synergy_citizen_scientist,
    _draft_synergy_pellet_dissector,
    _draft_synergy_no_signal,
)
from backend.models.enums import FoodType, Habitat, PowerColor, NestType


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

    def test_rollout_rerank_path(self, bird_reg, bonus_reg):
        """Rollout reranking should execute and return ranked recommendations."""
        birds = bird_reg.all_birds[:5]
        bonus_cards = bonus_reg.all_cards[:2]
        results = analyze_setup(
            birds,
            bonus_cards,
            [],
            rollout_top_k=2,
            rollout_simulations=1,
            rollout_max_turns=80,
        )
        assert len(results) > 0
        assert results[0].rank == 1

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


# --- Draft Synergy Function Tests ---


class TestDraftSynergyBehaviorist:
    def test_diverse_colors_all_count(self, bird_reg):
        """3+ distinct colors → all birds count."""
        # Find birds with 3 different colors
        white = next(b for b in bird_reg.all_birds if b.color == PowerColor.WHITE)
        brown = next(b for b in bird_reg.all_birds if b.color == PowerColor.BROWN)
        pink = next(b for b in bird_reg.all_birds if b.color == PowerColor.PINK)
        birds = [white, brown, pink]
        assert _draft_synergy_behaviorist(birds) == 3

    def test_monochrome_zero(self, bird_reg):
        """Only 1 color → 0."""
        browns = [b for b in bird_reg.all_birds if b.color == PowerColor.BROWN][:3]
        assert _draft_synergy_behaviorist(browns) == 0

    def test_two_colors_zero(self, bird_reg):
        """Only 2 colors → 0."""
        white = next(b for b in bird_reg.all_birds if b.color == PowerColor.WHITE)
        brown = next(b for b in bird_reg.all_birds if b.color == PowerColor.BROWN)
        assert _draft_synergy_behaviorist([white, brown]) == 0


class TestDraftSynergyEthologist:
    def test_diverse_in_one_habitat(self, bird_reg):
        """Multiple colors in one habitat → positive count."""
        # Get forest birds with diverse colors
        forest_white = next(
            (b for b in bird_reg.all_birds
             if b.color == PowerColor.WHITE and Habitat.FOREST in b.habitats), None)
        forest_brown = next(
            (b for b in bird_reg.all_birds
             if b.color == PowerColor.BROWN and Habitat.FOREST in b.habitats), None)
        if forest_white and forest_brown:
            result = _draft_synergy_ethologist([forest_white, forest_brown])
            assert result >= 2

    def test_single_bird_zero(self, bird_reg):
        """One bird can't have 2 colors → 0."""
        bird = bird_reg.all_birds[0]
        assert _draft_synergy_ethologist([bird]) == 0


class TestDraftSynergyPopulationMonitor:
    def test_distinct_nests_in_habitat(self, bird_reg):
        """Birds with different nests in target habitat → count."""
        forest_birds = [b for b in bird_reg.all_birds if Habitat.FOREST in b.habitats]
        nests_seen = set()
        diverse = []
        for b in forest_birds:
            if b.nest_type not in nests_seen:
                nests_seen.add(b.nest_type)
                diverse.append(b)
            if len(diverse) >= 3:
                break
        result = _draft_synergy_population_monitor(diverse, Habitat.FOREST)
        assert result >= 3

    def test_wrong_habitat_zero(self, bird_reg):
        """Birds that can't go in target habitat → 0 nest diversity."""
        # Find birds only in forest
        forest_only = [b for b in bird_reg.all_birds
                       if b.habitats == frozenset({Habitat.FOREST})][:3]
        if forest_only:
            result = _draft_synergy_population_monitor(forest_only, Habitat.WETLAND)
            assert result == 0


class TestDraftSynergyMechanicalEngineer:
    def test_four_types_covered(self, bird_reg):
        """Birds covering all 4 nest types → 4."""
        required = {NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM}
        chosen = []
        for nt in required:
            bird = next((b for b in bird_reg.all_birds if b.nest_type == nt), None)
            if bird:
                chosen.append(bird)
        if len(chosen) == 4:
            assert _draft_synergy_mechanical_engineer(chosen) == 4

    def test_wild_fills_gap(self, bird_reg):
        """Wild nest type should fill in for missing types."""
        wild_bird = next(
            (b for b in bird_reg.all_birds if b.nest_type == NestType.WILD), None)
        bowl_bird = next(
            (b for b in bird_reg.all_birds if b.nest_type == NestType.BOWL), None)
        if wild_bird and bowl_bird:
            result = _draft_synergy_mechanical_engineer([wild_bird, bowl_bird])
            assert result == 2  # 1 non-wild + 1 wild


class TestDraftSynergySiteSelection:
    def test_matching_pairs(self, bird_reg):
        """Birds sharing a nest type should contribute."""
        bowl_birds = [b for b in bird_reg.all_birds if b.nest_type == NestType.BOWL][:2]
        if len(bowl_birds) == 2:
            result = _draft_synergy_site_selection(bowl_birds)
            assert result >= 2

    def test_all_different_no_wild(self, bird_reg):
        """All different nest types with no wild → 0 matches."""
        unique = []
        seen = set()
        for b in bird_reg.all_birds:
            if b.nest_type != NestType.WILD and b.nest_type not in seen:
                seen.add(b.nest_type)
                unique.append(b)
            if len(unique) >= 4:
                break
        result = _draft_synergy_site_selection(unique)
        assert result == 0


class TestDraftSynergyDataAnalyst:
    def test_birds_with_wingspan_in_habitat(self, bird_reg):
        """Count birds with wingspan that can go in target habitat."""
        forest_with_ws = [b for b in bird_reg.all_birds
                          if Habitat.FOREST in b.habitats and b.wingspan_cm is not None][:3]
        result = _draft_synergy_data_analyst(forest_with_ws, Habitat.FOREST)
        assert result == len(forest_with_ws)

    def test_wrong_habitat_excluded(self, bird_reg):
        """Birds not eligible for habitat shouldn't count."""
        forest_only = [b for b in bird_reg.all_birds
                       if b.habitats == frozenset({Habitat.FOREST})
                       and b.wingspan_cm is not None][:2]
        if forest_only:
            result = _draft_synergy_data_analyst(forest_only, Habitat.WETLAND)
            assert result == 0


class TestDraftSynergyRanger:
    def test_eligible_birds(self, bird_reg):
        """Count birds eligible for target habitat."""
        wetland_birds = [b for b in bird_reg.all_birds
                         if Habitat.WETLAND in b.habitats][:4]
        result = _draft_synergy_ranger(wetland_birds, Habitat.WETLAND)
        assert result == len(wetland_birds)


class TestDraftSynergyEcologist:
    def test_multi_habitat_birds(self, bird_reg):
        """Birds with 2+ habitats contribute."""
        multi = [b for b in bird_reg.all_birds if len(b.habitats) >= 2][:3]
        result = _draft_synergy_ecologist(multi)
        assert result == len(multi)

    def test_single_habitat_zero(self, bird_reg):
        """Single-habitat birds → 0."""
        singles = [b for b in bird_reg.all_birds if len(b.habitats) == 1][:3]
        if singles:
            result = _draft_synergy_ecologist(singles)
            assert result == 0


class TestDraftSynergyAvianTherio:
    def test_small_egg_limit(self, bird_reg):
        """Birds with egg_limit 1-3 count."""
        small = [b for b in bird_reg.all_birds if 0 < b.egg_limit <= 3][:3]
        result = _draft_synergy_avian_therio(small)
        assert result == len(small)

    def test_large_egg_limit_excluded(self, bird_reg):
        """Birds with egg_limit >= 4 don't count."""
        large = [b for b in bird_reg.all_birds if b.egg_limit >= 4][:3]
        if large:
            result = _draft_synergy_avian_therio(large)
            assert result == 0


class TestDraftSynergyBreedingManager:
    def test_large_egg_limit(self, bird_reg):
        """Birds with egg_limit >= 4 count."""
        large = [b for b in bird_reg.all_birds if b.egg_limit >= 4][:3]
        if large:
            result = _draft_synergy_breeding_manager(large)
            assert result == len(large)


class TestDraftSynergyOologist:
    def test_birds_with_eggs(self, bird_reg):
        """Birds with egg_limit > 0 count."""
        with_eggs = [b for b in bird_reg.all_birds if b.egg_limit > 0][:4]
        result = _draft_synergy_oologist(with_eggs)
        assert result == len(with_eggs)

    def test_zero_egg_excluded(self, bird_reg):
        """Birds with egg_limit == 0 don't count."""
        no_eggs = [b for b in bird_reg.all_birds if b.egg_limit == 0][:2]
        if no_eggs:
            result = _draft_synergy_oologist(no_eggs)
            assert result == 0


class TestDraftSynergyCitizenScientist:
    def test_flocking_birds(self, bird_reg):
        """Flocking birds count."""
        flockers = [b for b in bird_reg.all_birds if b.is_flocking][:3]
        if flockers:
            result = _draft_synergy_citizen_scientist(flockers)
            assert result == len(flockers)

    def test_tuck_power_counts(self, bird_reg):
        """Birds with 'tuck' in power text count."""
        tuckers = [b for b in bird_reg.all_birds
                   if "tuck" in b.power_text.lower() and not b.is_flocking][:2]
        if tuckers:
            result = _draft_synergy_citizen_scientist(tuckers)
            assert result == len(tuckers)


class TestDraftSynergyPelletDissector:
    def test_predator_birds(self, bird_reg):
        """Predator birds count."""
        predators = [b for b in bird_reg.all_birds if b.is_predator][:3]
        if predators:
            result = _draft_synergy_pellet_dissector(predators)
            assert result == len(predators)

    def test_no_predators_zero(self, bird_reg):
        """Non-predator birds → 0."""
        non_pred = [b for b in bird_reg.all_birds if not b.is_predator][:3]
        result = _draft_synergy_pellet_dissector(non_pred)
        assert result == 0


class TestDraftSynergyNoSignal:
    def test_always_zero(self, bird_reg):
        """No-signal cards always return 0."""
        birds = bird_reg.all_birds[:5]
        assert _draft_synergy_no_signal(birds) == 0
        assert _draft_synergy_no_signal([]) == 0


class TestCustomDraftSynergyCoverage:
    def test_all_custom_cards_covered(self):
        """All 21 custom bonus cards should be in CUSTOM_DRAFT_SYNERGY."""
        from backend.engine.scoring import CUSTOM_BONUS_COUNTERS, CUSTOM_BONUS_FULL_SCORERS
        all_custom = set(CUSTOM_BONUS_COUNTERS.keys()) | set(CUSTOM_BONUS_FULL_SCORERS.keys())
        for name in all_custom:
            assert name in CUSTOM_DRAFT_SYNERGY, f"{name} missing from CUSTOM_DRAFT_SYNERGY"

    def test_dict_has_21_entries(self):
        """Should have exactly 21 entries."""
        assert len(CUSTOM_DRAFT_SYNERGY) == 21

    def test_spreadsheet_cards_use_original_path(self, bird_reg, bonus_reg):
        """Spreadsheet-mapped cards should NOT be in CUSTOM_DRAFT_SYNERGY."""
        for bc in bonus_reg.all_cards:
            has_eligibility = any(bc.name in b.bonus_eligibility for b in bird_reg.all_birds)
            if has_eligibility:
                assert bc.name not in CUSTOM_DRAFT_SYNERGY, (
                    f"{bc.name} has eligibility data but also in CUSTOM_DRAFT_SYNERGY"
                )


class TestCustomSynergyIntegration:
    def test_behaviorist_diverse_scores_higher(self, bird_reg, bonus_reg):
        """Diverse-color birds with Behaviorist should score higher than monochrome."""
        behaviorist = bonus_reg.get("Behaviorist")
        if not behaviorist:
            pytest.skip("Behaviorist not found")

        # Diverse: 3 different colors
        white = next(b for b in bird_reg.all_birds if b.color == PowerColor.WHITE)
        brown = next(b for b in bird_reg.all_birds if b.color == PowerColor.BROWN)
        pink = next(b for b in bird_reg.all_birds if b.color == PowerColor.PINK)
        diverse = [white, brown, pink]

        # Monochrome: all same color
        mono = [b for b in bird_reg.all_birds if b.color == PowerColor.BROWN][:3]

        food = (FoodType.SEED, FoodType.FISH)
        score_diverse = _evaluate_option(diverse, food, behaviorist, [])
        score_mono = _evaluate_option(mono, food, behaviorist, [])
        assert score_diverse > score_mono

    def test_custom_synergy_in_reasoning(self, bird_reg, bonus_reg):
        """Custom synergy should appear in reasoning text."""
        oologist = bonus_reg.get("Oologist")
        if not oologist:
            pytest.skip("Oologist not found")

        birds_with_eggs = [b for b in bird_reg.all_birds if b.egg_limit > 0][:3]
        food = (FoodType.SEED, FoodType.FISH)
        reasoning = _generate_reasoning(birds_with_eggs, food, oologist, [])
        assert "match bonus card" in reasoning
