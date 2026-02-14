"""Tests for the self-play evolutionary training module."""

import copy
import random
import pytest
from dataclasses import fields

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry
from backend.models.enums import BoardType
from backend.solver.heuristics import HeuristicWeights, DEFAULT_WEIGHTS, dynamic_weights
from backend.solver.self_play import (
    TrainingConfig, Individual, WEIGHT_FIELDS, WEIGHT_BOUNDS,
    random_individual, mutate, crossover, tournament_select,
    create_training_game, play_head_to_head, evaluate_population,
    evolve, _clamp,
)


@pytest.fixture(scope="module", autouse=True)
def load_data():
    load_all(EXCEL_FILE)


class TestWeightOperations:
    def test_random_individual_within_bounds(self):
        ind = random_individual()
        for f in WEIGHT_FIELDS:
            val = getattr(ind.weights, f)
            assert WEIGHT_BOUNDS[0] <= val <= WEIGHT_BOUNDS[1], \
                f"{f}={val} out of bounds"

    def test_mutate_preserves_bounds(self):
        ind = Individual(weights=copy.deepcopy(DEFAULT_WEIGHTS))
        config = TrainingConfig(mutation_rate=1.0, mutation_sigma=0.5)
        for _ in range(20):
            mutated = mutate(ind, config)
            for f in WEIGHT_FIELDS:
                val = getattr(mutated.weights, f)
                assert WEIGHT_BOUNDS[0] <= val <= WEIGHT_BOUNDS[1], \
                    f"{f}={val} out of bounds after mutation"

    def test_crossover_mixes_parents(self):
        a = Individual(weights=HeuristicWeights(
            bird_vp=0.1, egg_points=0.1, cached_food_points=0.1,
            tucked_card_points=0.1, bonus_card_points=0.1,
            round_goal_points=0.1, nectar_points=0.1,
            engine_value=0.1, food_in_supply=0.1, cards_in_hand=0.1,
            action_cubes=0.1, habitat_diversity_bonus=0.1,
            early_game_engine_bonus=0.1, predator_penalty=0.1,
            goal_alignment=0.1, nectar_early_bonus=0.1,
            grassland_egg_synergy=0.1, play_another_bird_bonus=0.1,
            food_for_birds_bonus=0.1,
        ))
        b = Individual(weights=HeuristicWeights(
            bird_vp=5.0, egg_points=5.0, cached_food_points=5.0,
            tucked_card_points=5.0, bonus_card_points=5.0,
            round_goal_points=5.0, nectar_points=5.0,
            engine_value=5.0, food_in_supply=5.0, cards_in_hand=5.0,
            action_cubes=5.0, habitat_diversity_bonus=5.0,
            early_game_engine_bonus=5.0, predator_penalty=5.0,
            goal_alignment=5.0, nectar_early_bonus=5.0,
            grassland_egg_synergy=5.0, play_another_bird_bonus=5.0,
            food_for_birds_bonus=5.0,
        ))
        config = TrainingConfig(crossover_rate=0.5)
        child = crossover(a, b, config)
        # With 19 genes and 50% crossover, extremely unlikely all from one parent
        from_a = sum(1 for f in WEIGHT_FIELDS
                     if getattr(child.weights, f) == getattr(a.weights, f))
        from_b = sum(1 for f in WEIGHT_FIELDS
                     if getattr(child.weights, f) == getattr(b.weights, f))
        assert from_a + from_b == len(WEIGHT_FIELDS)
        # With 19 genes, prob of all from one parent is 2^(-18) ≈ 0.0004%
        assert from_a > 0 and from_b > 0

    def test_tournament_select_picks_best(self):
        pop = [
            Individual(weights=DEFAULT_WEIGHTS, fitness=10),
            Individual(weights=DEFAULT_WEIGHTS, fitness=50),
            Individual(weights=DEFAULT_WEIGHTS, fitness=30),
        ]
        # With k=3, should always pick the best of all 3
        winner = tournament_select(pop, k=3)
        assert winner.fitness == 50

    def test_clamp_within_bounds(self):
        assert _clamp(0.01) == WEIGHT_BOUNDS[0]
        assert _clamp(10.0) == WEIGHT_BOUNDS[1]
        assert _clamp(1.0) == 1.0


class TestDynamicWeightsWithBase:
    def test_none_base_unchanged(self):
        """dynamic_weights with base=None should match original behavior."""
        from backend.models.game_state import create_new_game
        game = create_new_game(["P1"])
        w1 = dynamic_weights(game, base=None)
        w2 = dynamic_weights(game)
        for f in WEIGHT_FIELDS:
            assert getattr(w1, f) == pytest.approx(getattr(w2, f))

    def test_base_scaling_applies(self):
        """Doubling a base weight should double the corresponding output."""
        from backend.models.game_state import create_new_game
        game = create_new_game(["P1"])

        base = HeuristicWeights(bird_vp=DEFAULT_WEIGHTS.bird_vp * 2.0)
        w = dynamic_weights(game, base=base)
        w_default = dynamic_weights(game)
        assert w.bird_vp == pytest.approx(w_default.bird_vp * 2.0)
        # Other fields should be unchanged
        assert w.egg_points == pytest.approx(w_default.egg_points)


class TestGameCreation:
    def test_create_training_game_valid(self):
        game = create_training_game(2, BoardType.OCEANIA)
        assert len(game.players) == 2
        for p in game.players:
            assert 0 <= len(p.hand) <= 5
            assert len(p.bonus_cards) == 1
            assert p.food_supply.total_non_nectar() == (5 - len(p.hand))
            assert p.food_supply.nectar == 1
            assert p.action_cubes_remaining > 0
        assert len(game.round_goals) == 4
        assert game.birdfeeder.count > 0
        assert game.deck_remaining > 0

    def test_create_training_game_3_players(self):
        game = create_training_game(3, BoardType.BASE)
        assert len(game.players) == 3
        for p in game.players:
            assert 0 <= len(p.hand) <= 5
            assert p.food_supply.total_non_nectar() == (5 - len(p.hand))
            assert p.food_supply.nectar == 0

    def test_create_training_game_4_players(self):
        game = create_training_game(4, BoardType.OCEANIA)
        assert len(game.players) == 4

    def test_create_training_game_real5_softmax_draft_fidelity(self):
        saw_less_than_five = False
        for seed in range(20):
            random.seed(seed)
            game = create_training_game(2, BoardType.OCEANIA, setup_mode="real5_softmax")
            for p in game.players:
                assert 0 <= len(p.hand) <= 5
                assert p.food_supply.total_non_nectar() == (5 - len(p.hand))
                assert p.food_supply.nectar == 1
                if len(p.hand) < 5:
                    saw_less_than_five = True
        assert saw_less_than_five

    def test_create_training_game_legacy_mode_keeps_five(self):
        game = create_training_game(2, BoardType.BASE, setup_mode="legacy_fixed5")
        for p in game.players:
            assert len(p.hand) == 5
            assert p.food_supply.total_non_nectar() == 5
            assert p.food_supply.nectar == 0

    def test_create_training_game_invalid_knobs_are_clamped(self):
        game = create_training_game(
            2,
            BoardType.BASE,
            setup_mode="real5_softmax",
            draft_temperature=-1.0,
            draft_sample_top_k=0,
        )
        stats = getattr(game, "_setup_stats")
        assert stats["setup_mode"] == "real5_softmax"
        assert stats["draft_fallbacks"] >= 0
        for p in game.players:
            assert 0 <= len(p.hand) <= 5
            assert p.food_supply.total_non_nectar() == (5 - len(p.hand))

    def test_validation_mode_does_not_mutate_default_setup(self):
        baseline = create_training_game(2, BoardType.OCEANIA)
        seeded = create_training_game(
            2,
            BoardType.OCEANIA,
            strict_rules_mode=True,
            coverage_mode="all_5_colors_exec",
            coverage_seed_birds=True,
        )

        baseline_stats = getattr(baseline, "_setup_stats")
        seeded_stats = getattr(seeded, "_setup_stats")
        assert baseline_stats["coverage_mode"] == "off"
        assert baseline_stats["coverage_seed_birds"] is False
        assert seeded_stats["coverage_mode"] == "all_5_colors_exec"
        assert seeded_stats["coverage_seed_birds"] is True

        for p in baseline.players:
            assert 0 <= len(p.hand) <= 5
            assert p.food_supply.total_non_nectar() == (5 - len(p.hand))


class TestPlayMatch:
    def test_play_head_to_head_completes(self):
        game = create_training_game(2, BoardType.OCEANIA)
        pw = {
            "Player_1": DEFAULT_WEIGHTS,
            "Player_2": DEFAULT_WEIGHTS,
        }
        results = play_head_to_head(game, pw)
        assert "Player_1" in results
        assert "Player_2" in results
        assert results["Player_1"] >= 0
        assert results["Player_2"] >= 0

    def test_play_with_different_weights(self):
        game = create_training_game(2, BoardType.OCEANIA)
        aggressive = HeuristicWeights(bird_vp=2.0, engine_value=1.5)
        conservative = HeuristicWeights(egg_points=2.0, food_in_supply=0.8)
        pw = {"Player_1": aggressive, "Player_2": conservative}
        results = play_head_to_head(game, pw)
        assert results["Player_1"] >= 0
        assert results["Player_2"] >= 0


class TestEvolution:
    def test_evaluate_population_small(self):
        config = TrainingConfig(
            population_size=4,
            games_per_matchup=2,
            num_players=2,
        )
        pop = [Individual(weights=copy.deepcopy(DEFAULT_WEIGHTS)) for _ in range(4)]
        evaluate_population(pop, config)
        for ind in pop:
            assert ind.games_played >= 0
            assert ind.avg_score >= 0

    def test_evolve_one_generation(self):
        config = TrainingConfig(
            population_size=4,
            generations=1,
            games_per_matchup=1,
            num_players=2,
            elitism=1,
            tournament_k=2,
            seed=42,
        )
        results = evolve(config)
        assert "best_weights" in results
        assert "best_avg_score" in results
        assert "best_win_rate" in results
        assert "history" in results
        assert len(results["history"]) == 1
        assert results["best_avg_score"] >= 0
