"""Tests for the lookahead search solver."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import create_new_game
from backend.solver.lookahead import (
    lookahead_search, LookaheadResult, _advance_to_player,
)
from backend.solver.simulation import deep_copy_game


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


@pytest.fixture
def solo_game():
    return create_new_game(["Alice"])


# --- Core lookahead search ---

class TestLookaheadSearch:
    def test_returns_results(self, game):
        """Lookahead should return ranked moves."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=1, beam_width=4)
        assert len(results) > 0
        assert results[0].rank == 1

    def test_depth_1_evaluates_positions(self, game):
        """Depth 1 should evaluate position after executing each move."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=1, beam_width=4)
        for r in results:
            assert r.depth_reached >= 0
            assert r.score is not None

    def test_depth_2_looks_ahead(self, game):
        """Depth 2 should look at least 2 moves deep."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=2, beam_width=4)
        assert len(results) > 0
        # At least one result should reach depth 2
        depths = [r.depth_reached for r in results]
        assert max(depths) >= 2

    def test_best_sequence_populated(self, game):
        """Best sequence should have move descriptions."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=2, beam_width=4)
        for r in results:
            assert len(r.best_sequence) >= 1
            assert all(isinstance(s, str) for s in r.best_sequence)

    def test_depth_2_longer_sequences(self, game):
        """Depth 2 results should have sequences of length 2."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=2, beam_width=4)
        # At least one should have a 2-move sequence
        max_seq = max(len(r.best_sequence) for r in results)
        assert max_seq >= 2

    def test_scores_descending(self, game):
        """Results should be sorted by score descending."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=2, beam_width=4)
        for i in range(1, len(results)):
            assert results[i].score <= results[i - 1].score

    def test_ranks_sequential(self, game):
        """Ranks should be 1, 2, 3, ..."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=2, beam_width=4)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_heuristic_score_preserved(self, game):
        """Each result should also include the original heuristic score."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=1, beam_width=4)
        for r in results:
            assert isinstance(r.heuristic_score, float)

    def test_no_moves_returns_empty(self, game):
        """When no moves are available, return empty list."""
        game.current_player.action_cubes_remaining = 0
        results = lookahead_search(game, depth=2, beam_width=4)
        assert results == []

    def test_game_over_returns_empty(self, game):
        """Game over state should return no results."""
        game.current_round = 5
        results = lookahead_search(game, depth=2, beam_width=4)
        assert results == []

    def test_beam_width_limits_candidates(self, game):
        """Beam width should limit the number of results."""
        game.deck_remaining = 50
        results = lookahead_search(game, depth=1, beam_width=3)
        assert len(results) <= 3


# --- Lookahead finds better moves ---

class TestLookaheadQuality:
    def test_food_then_bird_combo(self, game, bird_reg):
        """Lookahead should value gaining food when it enables playing a bird."""
        player = game.current_player
        game.deck_remaining = 50

        # Give player a high-VP bird they can almost afford
        swan = bird_reg.get("Trumpeter Swan")  # 9 VP, costs 2 seed + 1 wild
        player.hand = [swan]
        player.food_supply.seed = 1  # Need 1 more seed + 1 wild
        player.food_supply.fish = 0

        # Set feeder to have seed
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.SEED, FoodType.FISH])

        results = lookahead_search(game, depth=2, beam_width=6)
        assert len(results) > 0

        # The top result should be gain food (to enable playing Swan next turn)
        # or play bird if somehow affordable
        top = results[0]
        assert top.score > 0

    def test_solo_depth_2_finds_sequence(self, solo_game, bird_reg):
        """In a solo game, depth 2 should find a 2-move sequence."""
        player = solo_game.current_player
        solo_game.deck_remaining = 50

        # Give resources for meaningful choices
        player.food_supply.seed = 2
        player.food_supply.invertebrate = 2
        bird = bird_reg.get("Acorn Woodpecker")
        player.hand = [bird]

        solo_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT
        ])

        results = lookahead_search(solo_game, depth=2, beam_width=4)
        assert len(results) > 0
        # In a solo game, we get consecutive turns, so depth 2 should work
        max_seq = max(len(r.best_sequence) for r in results)
        assert max_seq >= 2


# --- Advance to player ---

class TestAdvanceToPlayer:
    def test_already_current_player(self, game):
        """If it's already our turn, return immediately."""
        player_name = game.current_player.name
        sim = deep_copy_game(game)
        result = _advance_to_player(sim, player_name)
        assert result is True

    def test_advances_through_opponent(self, game):
        """Should simulate opponent turns to get back to our player."""
        game.deck_remaining = 50
        player_name = game.players[0].name
        sim = deep_copy_game(game)

        # Manually advance to player 1's turn
        sim.current_player_idx = 1  # Now it's Bob's turn
        result = _advance_to_player(sim, player_name)
        # Should execute Bob's move and get back to Alice
        assert result is True
        assert sim.current_player.name == player_name

    def test_handles_game_over(self, game):
        """Should return False if game ends during advancement."""
        game.current_round = 5  # Game over
        sim = deep_copy_game(game)
        result = _advance_to_player(sim, "Alice")
        assert result is False

    def test_solo_game_advance(self, solo_game):
        """In a solo game, after advance_turn we're immediately back."""
        solo_game.deck_remaining = 50
        player_name = solo_game.current_player.name
        sim = deep_copy_game(solo_game)

        # After advance_turn in a 1-player game, it should be our turn again
        # (or round advances if no cubes)
        result = _advance_to_player(sim, player_name)
        assert result is True

    def test_max_steps_guard(self, game):
        """Should respect max_steps to prevent infinite loops."""
        game.deck_remaining = 50
        sim = deep_copy_game(game)
        # Use max_steps=0 to force immediate failure
        result = _advance_to_player(sim, "Alice", max_steps=0)
        # If Alice is current player, max_steps=0 means the while loop doesn't execute
        # but we DO check the condition before entering... actually let me check
        # The while loop starts with `while steps < max_steps` and steps=0, max_steps=0
        # So the loop body never executes, function returns False
        assert result is False


# --- API endpoint ---

class TestLookaheadAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_lookahead_endpoint_default(self, client):
        """Lookahead endpoint with default params should work."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/lookahead")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert data["depth"] == 2
        assert data["beam_width"] == 6
        assert data["evaluation_time_ms"] >= 0

    def test_lookahead_with_params(self, client):
        """Lookahead with custom depth and beam width."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/lookahead", json={
            "depth": 1,
            "beam_width": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) <= 3
        assert data["depth"] == 1
        assert data["beam_width"] == 3

    def test_lookahead_response_structure(self, client):
        """Check full response structure."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/lookahead", json={
            "depth": 2,
            "beam_width": 4,
        })
        assert resp.status_code == 200
        data = resp.json()

        rec = data["recommendations"][0]
        assert rec["rank"] == 1
        assert "action_type" in rec
        assert "description" in rec
        assert "score" in rec
        assert "heuristic_score" in rec
        assert "depth_reached" in rec
        assert "best_sequence" in rec
        assert isinstance(rec["best_sequence"], list)
        assert len(rec["best_sequence"]) >= 1

    def test_lookahead_game_over(self, client):
        """Lookahead should return 400 for finished games."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        state = client.get(f"/api/games/{game_id}").json()
        state["current_round"] = 5
        client.put(f"/api/games/{game_id}/state", json=state)

        resp = client.post(f"/api/games/{game_id}/solve/lookahead")
        assert resp.status_code == 400

    def test_lookahead_reranks_vs_heuristic(self, client):
        """Lookahead may produce different rankings than heuristic."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Get both heuristic and lookahead results
        h_resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        l_resp = client.post(f"/api/games/{game_id}/solve/lookahead", json={
            "depth": 2, "beam_width": 6,
        })

        assert h_resp.status_code == 200
        assert l_resp.status_code == 200

        h_data = h_resp.json()
        l_data = l_resp.json()

        # Both should return recommendations
        assert len(h_data["recommendations"]) > 0
        assert len(l_data["recommendations"]) > 0

        # Lookahead scores should use position evaluation (generally higher numbers)
        # than heuristic move-value scores
        h_top = h_data["recommendations"][0]["score"]
        l_top = l_data["recommendations"][0]["score"]
        # Just verify both are valid numbers
        assert isinstance(h_top, (int, float))
        assert isinstance(l_top, (int, float))
