"""Tests for the heuristic solver: move generation, evaluation, and ranking."""

import numpy as np
import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import create_new_game
from backend.ml.action_codec import action_signature
from backend.solver.move_generator import (
    Move, generate_all_moves, generate_play_bird_moves,
    generate_gain_food_moves, generate_lay_eggs_moves, generate_draw_cards_moves,
)
from backend.solver.lookahead import endgame_search, lookahead_search
from backend.solver.heuristics import (
    evaluate_position, rank_moves, RankedMove, HeuristicWeights,
    _estimate_engine_value, detect_strategic_phase, _should_concede_goal,
)
from backend.engine_search.is_mcts import _model_priors


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture(scope="module")
def bonus_reg(regs):
    return regs[1]


@pytest.fixture(scope="module")
def goal_reg(regs):
    return regs[2]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


# --- Move generation ---

class TestMoveGeneration:
    def test_new_game_has_moves(self, game):
        """A new game should have gain food, lay eggs, and draw cards."""
        moves = generate_all_moves(game)
        types = {m.action_type for m in moves}
        assert ActionType.GAIN_FOOD in types
        assert ActionType.DRAW_CARDS in types
        # Lay eggs is legal even with no birds (wastes action)
        assert ActionType.LAY_EGGS in types

    def test_no_play_bird_without_hand(self, game):
        """Can't play birds if hand is empty."""
        game.current_player.hand.clear()
        moves = generate_play_bird_moves(game, game.current_player)
        assert len(moves) == 0

    def test_play_bird_with_hand(self, game, bird_reg):
        """Should generate play-bird moves when birds are in hand."""
        player = game.current_player
        bird = bird_reg.get("Acorn Woodpecker")
        player.hand.append(bird)
        player.food_supply.add(FoodType.SEED, 3)
        player.food_supply.add(FoodType.INVERTEBRATE, 2)

        moves = generate_play_bird_moves(game, player)
        assert len(moves) >= 1
        assert all(m.action_type == ActionType.PLAY_BIRD for m in moves)
        assert all(m.bird_name == "Acorn Woodpecker" for m in moves)

    def test_play_bird_multi_habitat(self, game, bird_reg):
        """Birds with multiple habitats should generate moves for each."""
        player = game.current_player
        # Find a multi-habitat bird
        multi_hab = None
        for b in bird_reg.all_birds:
            if len(b.habitats) >= 2 and b.food_cost.total <= 2:
                multi_hab = b
                break
        assert multi_hab is not None, "Need a multi-habitat bird for test"

        player.hand.append(multi_hab)
        # Give enough food to play
        for ft in (FoodType.SEED, FoodType.INVERTEBRATE, FoodType.FISH):
            player.food_supply.add(ft, 3)

        moves = generate_play_bird_moves(game, player)
        habitats_in_moves = {m.habitat for m in moves}
        assert len(habitats_in_moves) >= 2

    def test_gain_food_moves(self, game):
        """Gain food should generate moves for each food type in feeder."""
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])
        moves = generate_gain_food_moves(game, game.current_player)
        assert len(moves) >= 3
        food_types = {m.food_choices[0] for m in moves if m.food_choices}
        assert FoodType.SEED in food_types
        assert FoodType.FISH in food_types

    def test_lay_eggs_no_birds(self, game):
        """Lay eggs with no birds should still be a valid move."""
        moves = generate_lay_eggs_moves(game, game.current_player)
        assert len(moves) == 1
        assert moves[0].egg_distribution == {}

    def test_lay_eggs_with_birds(self, game, bird_reg):
        """Lay eggs should distribute to birds with space."""
        player = game.current_player
        bird = bird_reg.get("Trumpeter Swan")
        player.board.forest.slots[0].bird = bird

        moves = generate_lay_eggs_moves(game, player)
        assert len(moves) == 1
        total_eggs = sum(moves[0].egg_distribution.values())
        assert total_eggs >= 1

    def test_draw_cards_moves(self, game, bird_reg):
        """Draw cards should include deck and tray options."""
        game.deck_remaining = 50
        tray_bird = bird_reg.get("Acorn Woodpecker")
        game.card_tray.face_up = [tray_bird]

        moves = generate_draw_cards_moves(game, game.current_player)
        # Should have at least: draw from deck, take from tray
        assert len(moves) >= 2

        deck_moves = [m for m in moves if m.deck_draws > 0 and not m.tray_indices]
        tray_moves = [m for m in moves if m.tray_indices]
        assert len(deck_moves) >= 1
        assert len(tray_moves) >= 1

    def test_no_moves_when_game_over(self, game):
        """No moves should be generated when the game is over."""
        game.current_round = 5  # Past round 4
        moves = generate_all_moves(game)
        assert len(moves) == 0

    def test_no_moves_when_no_cubes(self, game):
        """No moves when player has no action cubes."""
        game.current_player.action_cubes_remaining = 0
        moves = generate_all_moves(game)
        assert len(moves) == 0


# --- Position evaluation ---

class TestPositionEvaluation:
    def test_empty_board_value(self, game):
        """Empty board should have a low but non-negative value."""
        value = evaluate_position(game, game.current_player)
        assert value >= 0

    def test_birds_increase_value(self, game, bird_reg):
        """Placing birds should increase position value."""
        player = game.current_player
        base_value = evaluate_position(game, player)

        bird = bird_reg.get("Trumpeter Swan")  # 9 VP
        player.board.forest.slots[0].bird = bird

        new_value = evaluate_position(game, player)
        assert new_value > base_value

    def test_eggs_increase_value(self, game, bird_reg):
        """Eggs should increase value (1 point each)."""
        player = game.current_player
        bird = bird_reg.get("Trumpeter Swan")
        player.board.forest.slots[0].bird = bird

        base_value = evaluate_position(game, player)

        player.board.forest.slots[0].eggs = 3
        egg_value = evaluate_position(game, player)
        assert egg_value > base_value

    def test_food_has_resource_value(self, game):
        """Food in supply should have some value (enables future plays)."""
        player = game.current_player
        base_value = evaluate_position(game, player)

        player.food_supply.add(FoodType.SEED, 5)
        food_value = evaluate_position(game, player)
        assert food_value > base_value

    def test_engine_value_with_brown_power(self, game, bird_reg):
        """Brown power birds should contribute engine value."""
        player = game.current_player

        # Find a brown power bird
        brown_bird = None
        for b in bird_reg.all_birds:
            if b.color == PowerColor.BROWN:
                brown_bird = b
                break
        assert brown_bird is not None

        player.board.forest.slots[0].bird = brown_bird
        engine_val = _estimate_engine_value(game, player)
        assert engine_val > 0

    def test_no_engine_value_for_no_power(self, game, bird_reg):
        """Birds with no power shouldn't contribute engine value."""
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")  # No power
        player.board.wetland.slots[0].bird = swan

        engine_val = _estimate_engine_value(game, player)
        assert engine_val == 0.0


class TestStrategicPhase:
    def test_detect_engine_phase_on_fresh_board(self, game):
        assert detect_strategic_phase(game, game.current_player) == "engine"

    def test_detect_scoring_phase_round_four(self, game):
        game.current_round = 4
        assert detect_strategic_phase(game, game.current_player) == "scoring"

    def test_detect_scoring_phase_point_generators(self, game, bird_reg):
        player = game.current_player
        game.current_round = 3
        point_gens = [
            b for b in bird_reg.all_birds
            if b.color == PowerColor.BROWN and any(k in b.power_text.lower() for k in ("tuck", "cache", "flock"))
        ][:3]
        assert len(point_gens) == 3
        player.board.forest.slots[0].bird = point_gens[0]
        player.board.grassland.slots[0].bird = point_gens[1]
        player.board.wetland.slots[0].bird = point_gens[2]
        assert detect_strategic_phase(game, player) == "scoring"

    def test_goal_concession_when_gap_unreachable(self, game, goal_reg, bird_reg):
        player = game.current_player
        opp = game.players[1]
        goal = next(g for g in goal_reg.all_goals if "[bird] in [forest]" in g.description.lower())
        game.round_goals = [goal]
        game.current_round = 1
        player.action_cubes_remaining = 1
        opp.action_cubes_remaining = 1
        # Opponent already far ahead on this goal.
        opp.board.forest.slots[0].bird = bird_reg.get("Trumpeter Swan")
        opp.board.forest.slots[1].bird = bird_reg.get("American Woodcock")
        opp.board.forest.slots[2].bird = bird_reg.get("Great Hornbill")
        opp.board.forest.slots[3].bird = bird_reg.get("Wedge-tailed Eagle")
        opp.board.forest.slots[4].bird = bird_reg.get("Australian Raven")

        concede_first, pursue_second = _should_concede_goal(game, player, 0)
        assert concede_first is True
        assert pursue_second is False


class TestEndgameSearch:
    def test_endgame_search_returns_ranked_moves(self, game):
        game.current_round = 4
        for p in game.players:
            p.action_cubes_remaining = 1
        game.deck_remaining = 20
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])

        results = endgame_search(game, game.current_player, max_total_actions=10)
        assert len(results) > 0
        assert results[0].rank == 1
        assert all(r.depth_reached >= 1 for r in results)


class TestNNHooks:
    def test_lookahead_leaf_evaluator_override(self, game):
        results = lookahead_search(
            game,
            player=game.current_player,
            depth=1,
            beam_width=4,
            leaf_evaluator=lambda _g, _p, _w: 42.0,
        )
        assert len(results) > 0
        assert all(abs(r.score - 42.0) < 1e-6 for r in results)

    def test_model_priors_override_heuristic_priors(self, game, bird_reg):
        class DummyEncoder:
            def encode(self, _game, _player_idx):
                return np.zeros(8, dtype=np.float32)

        class DummyModel:
            def forward(self, _state):
                return {}, None

            def score_move(self, _state, move, _player, logits=None):
                if move.action_type == ActionType.PLAY_BIRD:
                    return 5.0
                if move.action_type == ActionType.GAIN_FOOD:
                    return 2.0
                if move.action_type == ActionType.DRAW_CARDS:
                    return 1.0
                return 0.5

        player = game.current_player
        bird = bird_reg.get("Acorn Woodpecker")
        player.hand.append(bird)
        player.food_supply.add(FoodType.SEED, 3)
        player.food_supply.add(FoodType.INVERTEBRATE, 2)

        moves = generate_all_moves(game, player)
        priors = _model_priors(game, 0, moves, DummyModel(), DummyEncoder())
        assert priors is not None
        assert abs(sum(priors.values()) - 1.0) < 1e-6

        best = max(moves, key=lambda m: priors[action_signature(m)])
        assert best.action_type == ActionType.PLAY_BIRD


# --- Move ranking ---

class TestMoveRanking:
    def test_rank_moves_basic(self, game):
        """Ranking should return sorted moves."""
        game.deck_remaining = 50
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT
        ])
        ranked = rank_moves(game)
        assert len(ranked) > 0
        assert ranked[0].rank == 1
        # Scores should be in descending order
        for i in range(1, len(ranked)):
            assert ranked[i].score <= ranked[i - 1].score

    def test_play_high_vp_ranked_well(self, game, bird_reg):
        """High VP birds should rank well when affordable."""
        player = game.current_player
        game.deck_remaining = 50
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT
        ])

        # Trumpeter Swan: 9 VP, costs 2 seed + 1 wild, wetland only
        swan = bird_reg.get("Trumpeter Swan")
        player.hand.append(swan)
        player.food_supply.add(FoodType.SEED, 3)
        player.food_supply.add(FoodType.FISH, 2)

        ranked = rank_moves(game)
        play_bird_moves = [r for r in ranked if r.move.action_type == ActionType.PLAY_BIRD]
        assert len(play_bird_moves) > 0
        # Trumpeter Swan (9 VP) should rank near the top
        best_play = play_bird_moves[0]
        assert best_play.score > 5.0  # Should be quite valuable

    def test_gain_food_when_needed(self, game, bird_reg):
        """Gain food should rank higher when player needs food to play birds."""
        player = game.current_player
        game.deck_remaining = 50
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT
        ])

        # Give player an expensive bird but no food
        expensive = None
        for b in bird_reg.all_birds:
            if b.food_cost.total >= 2 and Habitat.FOREST in b.habitats:
                expensive = b
                break
        if expensive:
            player.hand.append(expensive)

        ranked = rank_moves(game)
        food_moves = [r for r in ranked if r.move.action_type == ActionType.GAIN_FOOD]
        assert len(food_moves) > 0
        # With birds in hand but no food, gain food should be reasonably ranked
        assert food_moves[0].score > 0

    def test_empty_hand_draw_cards_valued(self, game):
        """Draw cards should be more valuable when hand is empty."""
        player = game.current_player
        player.hand.clear()
        game.deck_remaining = 50
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])

        ranked = rank_moves(game)
        draw_moves = [r for r in ranked if r.move.action_type == ActionType.DRAW_CARDS]
        assert len(draw_moves) > 0
        # Empty hand urgency should boost draw cards
        assert draw_moves[0].score >= 2.0

    def test_late_game_eggs_valued(self, game, bird_reg):
        """In late game, laying eggs (guaranteed points) should score well."""
        player = game.current_player
        game.current_round = 4
        game.deck_remaining = 10
        game.birdfeeder.set_dice([FoodType.SEED])

        # Place birds that can hold eggs
        bird = bird_reg.get("Trumpeter Swan")
        player.board.grassland.slots[0].bird = bird
        player.board.forest.slots[0].bird = bird_reg.get("Acorn Woodpecker")

        ranked = rank_moves(game)
        egg_moves = [r for r in ranked if r.move.action_type == ActionType.LAY_EGGS]
        assert len(egg_moves) > 0
        assert egg_moves[0].score > 1.0

    def test_no_moves_returns_empty(self, game):
        """When no moves available, ranking returns empty list."""
        game.current_player.action_cubes_remaining = 0
        ranked = rank_moves(game)
        assert ranked == []

    def test_all_moves_have_ranks(self, game):
        """Every returned move should have a valid rank."""
        game.deck_remaining = 50
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])
        ranked = rank_moves(game)
        for rm in ranked:
            assert rm.rank >= 1
            assert rm.score is not None


# --- API integration ---

class TestSolverAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_heuristic_solver_endpoint(self, client):
        """Full round trip: create game, call solver, get recommendations."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert data["evaluation_time_ms"] >= 0

        # Check recommendation structure
        rec = data["recommendations"][0]
        assert rec["rank"] == 1
        assert "action_type" in rec
        assert "description" in rec
        assert "score" in rec

    def test_solver_after_action(self, client):
        """Solver should work after an action has been taken."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Take a draw cards action
        client.post(f"/api/games/{game_id}/draw-cards", json={
            "from_tray_indices": [],
            "from_deck_count": 1,
        })

        # Now Bob's turn — solver should work
        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) > 0

    def test_solver_uses_exact_endgame_mode(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]
        state = client.get(f"/api/games/{game_id}").json()
        state["current_round"] = 4
        state["turn_in_round"] = 1
        for p in state["players"]:
            p["action_cubes_remaining"] = 1
        upd = client.put(f"/api/games/{game_id}/state", json=state)
        assert upd.status_code == 200

        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) > 0
        assert recs[0]["details"]["search_mode"] == "exact_endgame"

    def test_solver_uses_iterative_lookahead_mode(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]
        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) > 0
        assert recs[0]["details"]["search_mode"] == "lookahead_iterative"
        assert recs[0]["details"]["score_mode"] == "iterative_lookahead"

    def test_solver_game_over(self, client):
        """Solver should return error for finished games."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        # Manually set game to over via state update
        state = client.get(f"/api/games/{game_id}").json()
        state["current_round"] = 5
        client.put(f"/api/games/{game_id}/state", json=state)

        resp = client.post(f"/api/games/{game_id}/solve/heuristic")
        assert resp.status_code == 400

    def test_after_reset_feeder(self, client):
        """After-reset endpoint should recommend food from new feeder dice."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "feeder",
            "new_feeder_dice": ["seed", "invertebrate", "fish", "fruit", "rodent"],
            "total_to_gain": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reset_type"] == "feeder"
        assert data["total_to_gain"] == 2
        assert len(data["recommendations"]) > 0
        # Top recommendation should have a description
        rec = data["recommendations"][0]
        assert rec["rank"] == 1
        assert rec["description"]
        assert rec["score"] > 0

    def test_after_reset_feeder_choice_dice(self, client):
        """After-reset with choice dice (e.g., nectar/fruit)."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"],
            "board_type": "oceania",
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "feeder",
            "new_feeder_dice": [
                "seed", "invertebrate", ["nectar", "fruit"], "fish", "rodent"
            ],
            "total_to_gain": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) > 0

    def test_after_reset_tray(self, client, bird_reg):
        """After-reset endpoint should recommend cards from new tray."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Pick 3 real bird names for the new tray
        birds = list(bird_reg.all_birds)[:3]
        bird_names = [b.name for b in birds]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "tray",
            "new_tray_cards": bird_names,
            "total_to_gain": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reset_type"] == "tray"
        assert data["total_to_gain"] == 2
        assert len(data["recommendations"]) > 0

    def test_engine_solver_endpoint(self, client):
        """Engine endpoint should return best move and debug stats."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"],
            "board_type": "oceania",
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/engine", json={
            "time_budget_ms": 1500,
            "num_determinizations": 8,
            "max_rollout_depth": 20,
            "top_k": 3,
            "return_debug": True,
            "seed": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "top_k_moves" in data
        assert len(data["top_k_moves"]) > 0
        assert data["best_move"] is not None
        assert data["best_move"]["rank"] == 1
        assert "search_stats" in data
        assert data["search_stats"]["simulations"] >= 0
        assert data["search_stats"]["elapsed_ms"] >= 0
        rec = data["top_k_moves"][0]
        assert rec["rank"] == 1
        assert rec["description"]
        assert "details" in rec

    def test_hybrid_solver_endpoint(self, client):
        """Hybrid endpoint should return shortlist-constrained engine results."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"],
            "board_type": "oceania",
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/hybrid", json={
            "total_time_budget_ms": 1500,
            "lookahead_time_budget_ms": 600,
            "top_candidates": 3,
            "num_determinizations": 4,
            "max_rollout_depth": 16,
            "return_debug": True,
            "seed": 7,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_move"] is not None
        assert len(data["top_k_moves"]) > 0
        assert "search_stats" in data
        assert data["search_stats"]["mode"] == "hybrid"
        assert data["search_stats"]["shortlist_size"] >= 1

    def test_engine_solver_auto_determinizations(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"],
            "board_type": "oceania",
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/engine", json={
            "time_budget_ms": 2000,
            "num_determinizations": 0,
            "max_rollout_depth": 16,
            "top_k": 2,
            "return_debug": True,
            "seed": 9,
        })
        assert resp.status_code == 200
        data = resp.json()
        stats = data["search_stats"]
        assert stats["determinizations_requested"] == 0
        assert stats["determinizations"] > 0

    def test_after_reset_auto_count(self, client):
        """When total_to_gain is 0, should auto-calculate from column."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "feeder",
            "new_feeder_dice": ["seed", "invertebrate", "fish", "fruit", "rodent"],
            "total_to_gain": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should auto-calculate from the forest column
        assert data["total_to_gain"] > 0

    def test_after_reset_invalid_type(self, client):
        """Should reject invalid reset_type."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "invalid",
        })
        assert resp.status_code == 400

    def test_after_reset_feeder_missing_dice(self, client):
        """Should reject feeder reset without dice."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "feeder",
            "new_feeder_dice": [],
        })
        assert resp.status_code == 400

    def test_after_reset_tray_invalid_bird(self, client):
        """Should reject unknown bird names in tray cards."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/after-reset", json={
            "reset_type": "tray",
            "new_tray_cards": ["Not A Real Bird"],
        })
        assert resp.status_code == 400
