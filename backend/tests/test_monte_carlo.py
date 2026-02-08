"""Tests for Monte Carlo solver: simulation, evaluation, and API endpoint."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat
from backend.models.game_state import create_new_game
from backend.solver.simulation import simulate_playout, deep_copy_game
from backend.solver.monte_carlo import (
    monte_carlo_evaluate, monte_carlo_best_move, MCConfig, MCResult,
)


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


# --- Simulation ---

class TestSimulation:
    def test_playout_completes(self, game):
        """A playout should finish and return scores for all players."""
        scores = simulate_playout(game)
        assert "Alice" in scores
        assert "Bob" in scores
        # Scores should be non-negative
        assert scores["Alice"] >= 0
        assert scores["Bob"] >= 0

    def test_playout_deterministic_with_seed(self, game):
        """Playouts with same seed should give same result (for debugging)."""
        import random
        random.seed(42)
        scores1 = simulate_playout(game)
        random.seed(42)
        scores2 = simulate_playout(game)
        assert scores1 == scores2

    def test_playout_varies_without_seed(self, game, bird_reg):
        """Multiple playouts should give different results (randomness works)."""
        # Give players diverse resources so playouts can meaningfully diverge
        for p in game.players:
            p.food_supply.add(FoodType.SEED, 5)
            p.food_supply.add(FoodType.FISH, 5)
            p.food_supply.add(FoodType.INVERTEBRATE, 5)
            p.food_supply.add(FoodType.FRUIT, 3)
            # Multiple birds with different costs/habitats for branching
            for name in ["Acorn Woodpecker", "American Robin", "Black Tern",
                         "Common Raven", "Coal Tit"]:
                bird = bird_reg.get(name)
                if bird:
                    p.hand.append(bird)
        game.deck_remaining = 50
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
            FoodType.INVERTEBRATE, FoodType.RODENT,
        ])

        results = [simulate_playout(game) for _ in range(20)]
        # Not all playouts should give identical scores
        unique_scores = set(r["Alice"] for r in results)
        # Allow for some collisions, but should have at least 2 different values
        assert len(unique_scores) >= 2

    def test_playout_with_birds(self, game, bird_reg):
        """Playout should handle games with birds on board."""
        player = game.current_player
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        player.food_supply.add(FoodType.SEED, 3)

        scores = simulate_playout(game)
        # Should still complete
        assert "Alice" in scores
        # With a bird on board, Alice should score at least the bird VP
        assert scores["Alice"] >= bird.victory_points

    def test_deep_copy_independence(self, game):
        """Deep copy should be independent of original."""
        copy = deep_copy_game(game)
        copy.current_round = 4
        copy.players[0].food_supply.add(FoodType.SEED, 10)

        assert game.current_round == 1
        assert game.players[0].food_supply.get(FoodType.SEED) == 0


# --- Monte Carlo evaluation ---

class TestMonteCarloEvaluation:
    def test_mc_returns_results(self, game):
        """MC evaluation should return results for each move."""
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
        ])
        game.deck_remaining = 50

        config = MCConfig(simulations_per_move=5, time_limit_seconds=10.0)
        results = monte_carlo_evaluate(game, config)

        assert len(results) > 0
        assert results[0].rank == 1
        assert results[0].simulations > 0
        assert results[0].avg_score >= 0

    def test_mc_results_sorted(self, game):
        """Results should be sorted by average score descending."""
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
        ])
        game.deck_remaining = 50

        config = MCConfig(simulations_per_move=5, time_limit_seconds=10.0)
        results = monte_carlo_evaluate(game, config)

        for i in range(1, len(results)):
            assert results[i].avg_score <= results[i - 1].avg_score

    def test_mc_best_move(self, game):
        """Best move helper should return a single result."""
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        game.deck_remaining = 50

        best = monte_carlo_best_move(game, simulations=5, time_limit=5.0)
        assert best is not None
        assert best.rank == 1
        assert best.simulations > 0

    def test_mc_no_moves(self, game):
        """MC should return empty list when no moves available."""
        game.current_player.action_cubes_remaining = 0
        results = monte_carlo_evaluate(game)
        assert results == []

    def test_mc_with_bird_in_hand(self, game, bird_reg):
        """MC should handle play-bird moves correctly."""
        player = game.current_player
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        game.deck_remaining = 50

        # Give player a bird and food
        bird = bird_reg.get("Acorn Woodpecker")
        player.hand.append(bird)
        player.food_supply.add(FoodType.SEED, 3)
        player.food_supply.add(FoodType.INVERTEBRATE, 2)

        config = MCConfig(simulations_per_move=5, time_limit_seconds=10.0)
        results = monte_carlo_evaluate(game, config)

        assert len(results) > 0
        # Should include play-bird moves
        play_moves = [r for r in results if r.move.action_type.value == "play_bird"]
        assert len(play_moves) >= 1

    def test_mc_respects_time_limit(self, game):
        """MC should stop within the time limit."""
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])
        game.deck_remaining = 50

        import time
        config = MCConfig(simulations_per_move=1000, time_limit_seconds=2.0)
        start = time.perf_counter()
        monte_carlo_evaluate(game, config)
        elapsed = time.perf_counter() - start

        # Should finish within ~3x the time limit (generous for test)
        assert elapsed < 8.0


# --- API endpoint ---

class TestMonteCarloAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_mc_endpoint(self, client):
        """Full round trip: create game, call MC solver."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(
            f"/api/games/{game_id}/solve/monte-carlo",
            json={"simulations_per_move": 5, "time_limit_seconds": 5.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert data["simulations_run"] > 0
        assert data["evaluation_time_ms"] > 0

        rec = data["recommendations"][0]
        assert rec["rank"] == 1
        assert "score" in rec

    def test_mc_endpoint_default_params(self, client):
        """MC endpoint should work with default parameters."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/monte-carlo")
        assert resp.status_code == 200

    def test_mc_game_over(self, client):
        """MC should return error for finished games."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        # Set game to over
        state = client.get(f"/api/games/{game_id}").json()
        state["current_round"] = 5
        client.put(f"/api/games/{game_id}/state", json=state)

        resp = client.post(f"/api/games/{game_id}/solve/monte-carlo")
        assert resp.status_code == 400
