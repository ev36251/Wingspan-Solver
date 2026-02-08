"""Tests for the heuristic solver: move generation, evaluation, and ranking."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import create_new_game
from backend.solver.move_generator import (
    Move, generate_all_moves, generate_play_bird_moves,
    generate_gain_food_moves, generate_lay_eggs_moves, generate_draw_cards_moves,
)
from backend.solver.heuristics import (
    evaluate_position, rank_moves, RankedMove, HeuristicWeights,
    _estimate_engine_value,
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
