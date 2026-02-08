"""Tests for the API layer: data routes, game management, actions, scoring."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- Health ---

class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# --- Data routes ---

class TestBirdRoutes:
    def test_list_all_birds(self, client):
        resp = client.get("/api/birds?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 446
        assert len(data["birds"]) == 10

    def test_list_birds_pagination(self, client):
        resp = client.get("/api/birds?limit=5&offset=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["birds"]) == 5

    def test_search_birds(self, client):
        resp = client.get("/api/birds?search=swan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any("Swan" in b["name"] for b in data["birds"])

    def test_filter_by_set(self, client):
        resp = client.get("/api/birds?game_set=core&limit=500")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 180
        for b in data["birds"]:
            assert b["game_set"] == "core"

    def test_filter_by_habitat(self, client):
        resp = client.get("/api/birds?habitat=forest&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        for b in data["birds"]:
            assert "forest" in b["habitats"]

    def test_filter_by_color(self, client):
        resp = client.get("/api/birds?color=brown&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        for b in data["birds"]:
            assert b["color"] == "brown"

    def test_get_bird_by_name(self, client):
        resp = client.get("/api/birds/Trumpeter Swan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Trumpeter Swan"
        assert data["victory_points"] == 9

    def test_get_bird_not_found(self, client):
        resp = client.get("/api/birds/Nonexistent Bird")
        assert resp.status_code == 404

    def test_bird_schema_fields(self, client):
        resp = client.get("/api/birds/Acorn Woodpecker")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "food_cost" in data
        assert "items" in data["food_cost"]
        assert "habitats" in data
        assert isinstance(data["habitats"], list)


class TestBonusCardRoutes:
    def test_list_all_bonus_cards(self, client):
        resp = client.get("/api/bonus-cards")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 30
        assert len(data["bonus_cards"]) >= 30

    def test_filter_by_set(self, client):
        resp = client.get("/api/bonus-cards?game_set=core")
        assert resp.status_code == 200
        data = resp.json()
        for c in data["bonus_cards"]:
            assert "core" in c["game_sets"]

    def test_bonus_card_schema(self, client):
        resp = client.get("/api/bonus-cards")
        data = resp.json()
        card = data["bonus_cards"][0]
        assert "name" in card
        assert "scoring_tiers" in card
        assert "is_per_bird" in card


class TestGoalRoutes:
    def test_list_all_goals(self, client):
        resp = client.get("/api/goals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 20

    def test_filter_by_set(self, client):
        resp = client.get("/api/goals?game_set=core")
        assert resp.status_code == 200
        data = resp.json()
        for g in data["goals"]:
            assert g["game_set"] == "core"

    def test_goal_schema(self, client):
        resp = client.get("/api/goals")
        data = resp.json()
        goal = data["goals"][0]
        assert "description" in goal
        assert "scoring" in goal
        assert len(goal["scoring"]) == 4


# --- Game management ---

class TestGameCreation:
    def test_create_game(self, client):
        resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "game_id" in data
        assert len(data["state"]["players"]) == 2
        assert data["state"]["players"][0]["name"] == "Alice"
        assert data["state"]["current_round"] == 1

    def test_create_game_with_goals(self, client):
        # Get a valid goal description first
        goals_resp = client.get("/api/goals?game_set=core")
        goals = goals_resp.json()["goals"]
        goal_descs = [g["description"] for g in goals[:4]]

        resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob", "Charlie"],
            "round_goals": goal_descs,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["state"]["round_goals"]) == 4

    def test_create_game_invalid_goal(self, client):
        resp = client.post("/api/games", json={
            "player_names": ["Alice"],
            "round_goals": ["Nonexistent goal"],
        })
        assert resp.status_code == 400

    def test_get_game(self, client):
        # Create a game first
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.get(f"/api/games/{game_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["players"]) == 2

    def test_get_game_not_found(self, client):
        resp = client.get("/api/games/nonexistent")
        assert resp.status_code == 404


class TestLegalMoves:
    def test_legal_moves_new_game(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.get(f"/api/games/{game_id}/legal-moves")
        assert resp.status_code == 200
        data = resp.json()
        # New game: should have gain food, lay eggs, draw cards at minimum
        action_types = {m["action_type"] for m in data["moves"]}
        assert "gain_food" in action_types
        assert "draw_cards" in action_types


class TestScoring:
    def test_scores_new_game(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.get(f"/api/games/{game_id}/score")
        assert resp.status_code == 200
        data = resp.json()
        assert "Alice" in data["scores"]
        assert "Bob" in data["scores"]
        # New game: all scores should be 0
        assert data["scores"]["Alice"]["total"] == 0


class TestActions:
    def test_gain_food(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Get available food
        moves_resp = client.get(f"/api/games/{game_id}/legal-moves")
        food_move = next(
            m for m in moves_resp.json()["moves"]
            if m["action_type"] == "gain_food"
        )
        available = food_move["details"]["available_food"]

        resp = client.post(f"/api/games/{game_id}/gain-food", json={
            "food_choices": [available[0]],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["action_type"] == "gain_food"

    def test_draw_cards(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/draw-cards", json={
            "from_tray_indices": [],
            "from_deck_count": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["cards_drawn"] == 1


# --- Solver placeholder routes ---

class TestSolverPlaceholders:
    def test_heuristic_implemented(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]
        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) > 0

    def test_monte_carlo_implemented(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]
        resp = client.post(f"/api/games/{game_id}/solve/monte-carlo",
                           json={"simulations_per_move": 5, "time_limit_seconds": 5.0})
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert data["simulations_run"] > 0

    def test_max_score_returns_result(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]
        resp = client.post(f"/api/games/{game_id}/solve/max-score")
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_possible_score"] > 0

    def test_analyze_returns_result(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]
        resp = client.post(f"/api/games/{game_id}/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "actual_score" in data
        assert "deviations" in data
