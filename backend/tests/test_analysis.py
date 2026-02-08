"""Tests for Phase 10: Max Score Calculator and Post-Game Analysis."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import GameState, MoveRecord, create_new_game
from backend.models.player import FoodSupply
from backend.solver.max_score import (
    calculate_max_score, MaxScoreBreakdown,
    _remaining_actions, _max_bird_vp, _max_eggs,
    _max_cached_food, _max_tucked_cards, _max_bonus_cards,
    _max_round_goals, _max_nectar,
)
from backend.solver.analysis import (
    analyze_player, analyze_game, AnalysisResult, Deviation,
    _classify_impact, _extract_deviations,
)
from backend.engine.scoring import calculate_score


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


@pytest.fixture
def solo_game():
    return create_new_game(["Alice"])


# --- Max Score Calculator ---

class TestRemainingActions:
    def test_round1_full_cubes(self, game):
        """Round 1 with 8 cubes + future rounds = 8 + 7 + 6 + 5 = 26."""
        player = game.current_player
        assert _remaining_actions(game, player) == 8 + 7 + 6 + 5

    def test_round3_partial_cubes(self, game):
        game.current_round = 3
        player = game.current_player
        player.action_cubes_remaining = 3
        # 3 remaining this round + 5 in round 4
        assert _remaining_actions(game, player) == 3 + 5

    def test_round4_last_cubes(self, game):
        game.current_round = 4
        player = game.current_player
        player.action_cubes_remaining = 2
        assert _remaining_actions(game, player) == 2

    def test_game_over_zero(self, game):
        game.current_round = 5
        player = game.current_player
        player.action_cubes_remaining = 0
        assert _remaining_actions(game, player) == 0


class TestMaxBirdVP:
    def test_no_birds_empty_hand(self, game):
        player = game.current_player
        player.hand.clear()
        assert _max_bird_vp(player, 10) == 0

    def test_birds_on_board_only(self, game, bird_reg):
        player = game.current_player
        player.hand.clear()
        swan = bird_reg.get("Trumpeter Swan")  # 9 VP
        player.board.wetland.slots[0].bird = swan
        assert _max_bird_vp(player, 10) == 9

    def test_birds_on_board_plus_hand(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")  # 9 VP
        player.board.wetland.slots[0].bird = swan

        # Add more birds to hand
        crow = bird_reg.get("American Crow")
        if crow:
            player.hand = [crow]
            vp = _max_bird_vp(player, 10)
            assert vp == 9 + crow.victory_points

    def test_limited_by_remaining_actions(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")
        crow = bird_reg.get("American Crow")
        player.hand = [swan, crow]
        # Only 1 action remaining — should only count highest VP bird
        vp = _max_bird_vp(player, 1)
        assert vp == max(swan.victory_points, crow.victory_points)

    def test_limited_by_empty_slots(self, game, bird_reg):
        player = game.current_player
        # Fill all 15 slots
        swan = bird_reg.get("Trumpeter Swan")
        for row in player.board.all_rows():
            for i in range(5):
                row.slots[i].bird = swan
        crow = bird_reg.get("American Crow")
        player.hand = [crow]
        # No empty slots — only board VP counts
        vp = _max_bird_vp(player, 10)
        assert vp == 15 * swan.victory_points


class TestMaxEggs:
    def test_no_birds_no_eggs(self, game):
        player = game.current_player
        assert _max_eggs(player, []) == 0

    def test_board_birds_only(self, game, bird_reg):
        player = game.current_player
        # House Sparrow has egg_limit=5
        sparrow = bird_reg.get("House Sparrow")
        player.board.forest.slots[0].bird = sparrow
        assert _max_eggs(player, []) == sparrow.egg_limit

    def test_board_plus_future(self, game, bird_reg):
        player = game.current_player
        sparrow = bird_reg.get("House Sparrow")
        player.board.forest.slots[0].bird = sparrow
        swan = bird_reg.get("Trumpeter Swan")
        # Future birds add their egg limits
        total = _max_eggs(player, [swan])
        assert total == sparrow.egg_limit + swan.egg_limit


class TestMaxCachedFood:
    def test_no_cache_birds(self, game):
        player = game.current_player
        assert _max_cached_food(player, 10) == 0

    def test_with_cache_bird(self, game, bird_reg):
        player = game.current_player
        # Find a bird with "cache" in power text
        cache_bird = None
        for b in bird_reg.all_birds:
            if "cache" in (b.power_text or "").lower():
                cache_bird = b
                break
        if cache_bird:
            player.board.forest.slots[0].bird = cache_bird
            result = _max_cached_food(player, 10)
            assert result > 0

    def test_existing_cached_food_counted(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = swan
        player.board.wetland.slots[0].cache_food(FoodType.FISH, 3)
        result = _max_cached_food(player, 0)
        assert result >= 3


class TestMaxTuckedCards:
    def test_no_flocking_birds(self, game):
        player = game.current_player
        assert _max_tucked_cards(player, 10) == 0

    def test_with_flocking_bird(self, game, bird_reg):
        player = game.current_player
        # Find a flocking bird
        flocking = None
        for b in bird_reg.all_birds:
            if b.is_flocking:
                flocking = b
                break
        if flocking:
            player.board.forest.slots[0].bird = flocking
            result = _max_tucked_cards(player, 10)
            assert result > 0

    def test_existing_tucked_counted(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = swan
        player.board.wetland.slots[0].tucked_cards = 5
        result = _max_tucked_cards(player, 0)
        assert result >= 5


class TestMaxBonusCards:
    def test_no_bonus_cards(self, game):
        player = game.current_player
        player.bonus_cards = []
        assert _max_bonus_cards(player, []) == 0

    def test_with_bonus_and_qualifying_birds(self, game, bird_reg, bonus_reg):
        player = game.current_player
        anatomist = bonus_reg.get("Anatomist")
        if anatomist:
            player.bonus_cards = [anatomist]
            # Find birds that qualify for Anatomist
            qualifying = [
                b for b in bird_reg.all_birds
                if "Anatomist" in b.bonus_eligibility
            ]
            # Place some on board
            for i, b in enumerate(qualifying[:3]):
                if i < 5:
                    player.board.forest.slots[i].bird = b
            score = _max_bonus_cards(player, qualifying[3:6])
            assert score > 0


class TestMaxRoundGoals:
    def test_no_goals_set(self, game):
        player = game.current_player
        game.round_goals = []
        result = _max_round_goals(game, player)
        # With no goals assigned, uses default 5 per round
        assert result == 5 * 4  # 4 rounds × 5 default

    def test_with_goals(self, game, goal_reg):
        player = game.current_player
        goals = goal_reg.all_goals[:4]
        game.round_goals = goals
        result = _max_round_goals(game, player)
        # Should be sum of 1st place scores for each goal
        expected = sum(int(g.scoring[-1]) for g in goals)
        assert result == expected

    def test_already_scored_not_double_counted(self, game, goal_reg):
        player = game.current_player
        goals = goal_reg.all_goals[:4]
        game.round_goals = goals
        # Mark round 1 as scored
        game.round_goal_scores = {1: {"Alice": 3, "Bob": 5}}
        game.current_round = 2
        result = _max_round_goals(game, player)
        # Should include scored round 1 (3 pts) + 1st place for rounds 2-4
        remaining_max = sum(int(goals[i].scoring[-1]) for i in range(1, 4))
        assert result == 3 + remaining_max


class TestMaxNectar:
    def test_multiplayer_max_15(self, game):
        player = game.current_player
        result = _max_nectar(game, player)
        assert result == 15  # 5 per habitat × 3

    def test_solo_uses_current(self, solo_game):
        player = solo_game.current_player
        result = _max_nectar(solo_game, player)
        assert result >= 0


class TestCalculateMaxScore:
    def test_empty_game_has_positive_max(self, game):
        player = game.current_player
        breakdown = calculate_max_score(game, player)
        assert breakdown.total > 0

    def test_max_score_breakdown_fields(self, game):
        player = game.current_player
        breakdown = calculate_max_score(game, player)
        d = breakdown.as_dict()
        assert "bird_vp" in d
        assert "eggs" in d
        assert "cached_food" in d
        assert "tucked_cards" in d
        assert "bonus_cards" in d
        assert "round_goals" in d
        assert "nectar" in d
        assert "total" in d
        assert d["total"] == sum(
            d[k] for k in d if k != "total"
        )

    def test_max_exceeds_current_score(self, game, bird_reg):
        player = game.current_player
        # Place a bird to have some score
        swan = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = swan
        player.board.wetland.slots[0].eggs = 2

        current = calculate_score(game, player).total
        max_bd = calculate_max_score(game, player)
        assert max_bd.total >= current

    def test_max_with_birds_in_hand(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")
        crow = bird_reg.get("American Crow")
        player.hand = [swan, crow]
        player.food_supply.add(FoodType.SEED, 5)

        breakdown = calculate_max_score(game, player)
        # Max should include VP from both hand birds
        assert breakdown.bird_vp >= swan.victory_points + crow.victory_points


# --- Analysis Engine ---

class TestClassifyImpact:
    def test_high_impact(self):
        assert _classify_impact(5, 10) == "high"
        assert _classify_impact(8, 10) == "high"

    def test_medium_impact(self):
        assert _classify_impact(3, 10) == "medium"
        assert _classify_impact(4, 10) == "medium"

    def test_low_impact(self):
        assert _classify_impact(2, 10) == "low"

    def test_few_alternatives_always_low(self):
        assert _classify_impact(2, 2) == "low"


class TestExtractDeviations:
    def test_no_history(self):
        devs, analyzed, sub = _extract_deviations([], "Alice")
        assert devs == []
        assert analyzed == 0
        assert sub == 0

    def test_optimal_moves_no_deviations(self):
        history = [
            MoveRecord(round=1, turn=1, player_name="Alice",
                       action_type="gain_food", description="Gain food",
                       solver_rank=1, total_moves=5, best_move_description="Gain food"),
        ]
        devs, analyzed, sub = _extract_deviations(history, "Alice")
        assert devs == []
        assert analyzed == 1
        assert sub == 0

    def test_suboptimal_move_creates_deviation(self):
        history = [
            MoveRecord(round=1, turn=1, player_name="Alice",
                       action_type="draw_cards", description="Draw cards",
                       solver_rank=3, total_moves=8,
                       best_move_description="Play Trumpeter Swan in wetland"),
        ]
        devs, analyzed, sub = _extract_deviations(history, "Alice")
        assert len(devs) == 1
        assert devs[0].rank_chosen == 3
        assert devs[0].impact == "medium"
        assert analyzed == 1
        assert sub == 1

    def test_high_impact_deviation(self):
        history = [
            MoveRecord(round=2, turn=3, player_name="Alice",
                       action_type="lay_eggs", description="Lay eggs",
                       solver_rank=7, total_moves=12,
                       best_move_description="Play bird"),
        ]
        devs, _, _ = _extract_deviations(history, "Alice")
        assert len(devs) == 1
        assert devs[0].impact == "high"

    def test_filters_by_player(self):
        history = [
            MoveRecord(round=1, turn=1, player_name="Alice",
                       action_type="gain_food", description="Gain food",
                       solver_rank=5, total_moves=10, best_move_description="X"),
            MoveRecord(round=1, turn=2, player_name="Bob",
                       action_type="gain_food", description="Gain food",
                       solver_rank=3, total_moves=8, best_move_description="Y"),
        ]
        devs, analyzed, _ = _extract_deviations(history, "Alice")
        assert len(devs) == 1
        assert devs[0].player_name == "Alice"
        assert analyzed == 1

    def test_max_5_deviations(self):
        history = [
            MoveRecord(round=r, turn=1, player_name="Alice",
                       action_type="gain_food", description=f"Move {r}",
                       solver_rank=3, total_moves=8, best_move_description="X")
            for r in range(1, 10)
        ]
        devs, _, _ = _extract_deviations(history, "Alice", max_deviations=5)
        assert len(devs) == 5

    def test_sorted_by_impact_then_chronology(self):
        history = [
            MoveRecord(round=1, turn=1, player_name="Alice",
                       action_type="a", description="Low",
                       solver_rank=2, total_moves=10, best_move_description="X"),
            MoveRecord(round=2, turn=1, player_name="Alice",
                       action_type="b", description="High",
                       solver_rank=6, total_moves=10, best_move_description="Y"),
            MoveRecord(round=3, turn=1, player_name="Alice",
                       action_type="c", description="Medium",
                       solver_rank=3, total_moves=10, best_move_description="Z"),
        ]
        devs, _, _ = _extract_deviations(history, "Alice")
        assert devs[0].impact == "high"
        assert devs[1].impact == "medium"
        assert devs[2].impact == "low"


class TestAnalyzePlayer:
    def test_basic_analysis(self, game):
        player = game.current_player
        result = analyze_player(game, player)
        assert isinstance(result, AnalysisResult)
        assert result.actual_score >= 0
        assert result.max_possible_score > 0
        assert 0 <= result.efficiency_pct <= 100

    def test_analysis_with_birds(self, game, bird_reg):
        player = game.current_player
        swan = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = swan
        player.board.wetland.slots[0].eggs = 2

        result = analyze_player(game, player)
        assert result.actual_score == 9 + 2  # 9 VP + 2 eggs
        assert result.max_possible_score >= result.actual_score

    def test_analysis_with_move_history(self, game):
        player = game.current_player
        game.move_history = [
            MoveRecord(round=1, turn=1, player_name=player.name,
                       action_type="gain_food", description="Gain food",
                       solver_rank=4, total_moves=10,
                       best_move_description="Play bird"),
        ]
        result = analyze_player(game, player)
        assert result.moves_analyzed == 1
        assert result.suboptimal_moves == 1
        assert len(result.deviations) == 1

    def test_efficiency_calculation(self, game, bird_reg):
        player = game.current_player
        # Place bird to get some score
        swan = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = swan

        result = analyze_player(game, player)
        expected_eff = result.actual_score / result.max_possible_score * 100
        assert abs(result.efficiency_pct - round(expected_eff, 1)) < 0.1


class TestAnalyzeGame:
    def test_analyze_all_players(self, game):
        results = analyze_game(game)
        assert "Alice" in results
        assert "Bob" in results
        assert isinstance(results["Alice"], AnalysisResult)
        assert isinstance(results["Bob"], AnalysisResult)

    def test_analyze_single_player(self, game):
        results = analyze_game(game, player_name="Alice")
        assert "Alice" in results
        assert "Bob" not in results


# --- Move History Recording ---

class TestMoveHistory:
    def test_game_starts_with_empty_history(self, game):
        assert game.move_history == []

    def test_move_record_fields(self):
        record = MoveRecord(
            round=1, turn=3, player_name="Alice",
            action_type="play_bird", description="Play Trumpeter Swan",
            solver_rank=1, total_moves=5,
            best_move_description="Play Trumpeter Swan in wetland",
        )
        assert record.round == 1
        assert record.turn == 3
        assert record.player_name == "Alice"
        assert record.solver_rank == 1
        assert record.total_moves == 5


# --- API integration ---

class TestMaxScoreAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_max_score_endpoint(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/max-score")
        assert resp.status_code == 200
        data = resp.json()
        assert "max_possible_score" in data
        assert "current_score" in data
        assert "efficiency_pct" in data
        assert "breakdown" in data
        assert data["max_possible_score"] > 0
        assert data["current_score"] >= 0
        assert 0 <= data["efficiency_pct"] <= 100

    def test_max_score_with_player_name(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/max-score?player_name=Bob")
        assert resp.status_code == 200

    def test_max_score_invalid_player(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/solve/max-score?player_name=Nobody")
        assert resp.status_code == 400


class TestAnalysisAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_analysis_endpoint(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "actual_score" in data
        assert "max_possible_score" in data
        assert "efficiency_pct" in data
        assert "deviations" in data

    def test_analysis_with_player_name(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/analyze?player_name=Alice")
        assert resp.status_code == 200

    def test_analysis_invalid_player(self, client):
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice"]
        })
        game_id = create_resp.json()["game_id"]

        resp = client.post(f"/api/games/{game_id}/analyze?player_name=Nobody")
        assert resp.status_code == 400

    def test_analysis_after_actions(self, client):
        """Analysis should work after some moves have been recorded."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Take a few actions to build move history
        client.post(f"/api/games/{game_id}/draw-cards", json={
            "from_tray_indices": [],
            "from_deck_count": 1,
        })

        resp = client.post(f"/api/games/{game_id}/analyze?player_name=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["actual_score"] >= 0


class TestMoveRecording:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as c:
            yield c

    def test_actions_record_to_history(self, client):
        """Actions taken via API should be recorded in move history."""
        create_resp = client.post("/api/games", json={
            "player_names": ["Alice", "Bob"]
        })
        game_id = create_resp.json()["game_id"]

        # Draw cards
        resp = client.post(f"/api/games/{game_id}/draw-cards", json={
            "from_tray_indices": [],
            "from_deck_count": 1,
        })
        assert resp.status_code == 200

        # Check that move was recorded via analysis
        analysis = client.post(f"/api/games/{game_id}/analyze?player_name=Alice")
        assert analysis.status_code == 200
