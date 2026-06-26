"""Tests for the percentage-play advisor (solver.advisor + /solve/advisor)."""

import random

import pytest
from fastapi.testclient import TestClient

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry
from backend.models.enums import GameSet, BoardType
from backend.models.game_state import create_new_game
from backend.solver.self_play import create_training_game
from backend.solver import advisor as adv


@pytest.fixture(scope="module", autouse=True)
def _regs():
    return load_all(EXCEL_FILE)


# --------------------------- unseen-pool model ----------------------------- #
def test_unseen_pool_excludes_seen_birds():
    game = create_new_game(["Alice", "Bob"])
    reg = get_bird_registry()
    total = len(reg.all_birds)
    pool0 = adv.unseen_pool(game)
    assert len(pool0) == total  # empty boards/hands -> nothing seen

    bird = reg.all_birds[0]
    game.players[0].hand.append(bird)
    names = {b.name for b in adv.unseen_pool(game)}
    assert bird.name not in names
    assert len(names) == total - 1


def test_unseen_pool_respects_active_sets():
    game = create_new_game(["Alice", "Bob"])
    reg = get_bird_registry()
    core_only = adv.unseen_pool(game, active_sets={GameSet.CORE})
    assert core_only, "expected some core birds"
    assert all(b.game_set == GameSet.CORE for b in core_only)
    assert len(core_only) == len(reg.by_set(GameSet.CORE))


def test_draw_probability_monotonic_and_clamped():
    # more draws -> higher odds; bigger pool -> lower odds
    assert adv.draw_probability(100, 1) < adv.draw_probability(100, 5)
    assert adv.draw_probability(200, 5) < adv.draw_probability(50, 5)
    # clamps
    assert adv.draw_probability(0, 5) == 0.0
    assert adv.draw_probability(100, 0) == 0.0
    assert adv.draw_probability(1, 100) <= 0.99


def test_estimate_future_draws_drops_with_round():
    game = create_new_game(["Alice", "Bob"])
    p = game.players[0]
    p.action_cubes_remaining = 5
    game.deck_remaining = 200
    game.current_round = 1
    early = adv.estimate_future_draws(game, p)
    game.current_round = 4
    late = adv.estimate_future_draws(game, p)
    assert early > late >= 0


def test_estimate_future_draws_capped_by_deck():
    game = create_new_game(["Alice", "Bob"])
    p = game.players[0]
    p.action_cubes_remaining = 8
    game.current_round = 1
    game.deck_remaining = 3
    assert adv.estimate_future_draws(game, p) <= 3


# --------------------------- main-line framing ----------------------------- #
def test_quantile_interpolates():
    vals = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert adv.quantile(vals, 0.0) == 0.0
    assert adv.quantile(vals, 1.0) == 40.0
    assert adv.quantile(vals, 0.5) == 20.0


def test_main_line_percentiles_and_sentence():
    scores = list(range(60, 110))  # 60..109
    ml = adv.main_line("Play X in wetland", "play_bird", scores, floor_prob=0.70)
    # 70% of games >= the 30th percentile; that floor must be below the median
    assert ml.floor_score <= ml.typical <= ml.stretch_score
    assert ml.samples == len(scores)
    s = ml.sentence()
    assert "70%" in s and "points" in s
    assert str(ml.floor_score) in s


# --------------------------- endpoint smoke -------------------------------- #
def test_advisor_endpoint_returns_lines(monkeypatch):
    from backend.main import app
    import backend.api.routes_game as routes_game

    random.seed(7)
    game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
    gid = "advtest"
    routes_game._games[gid] = game

    with TestClient(app) as client:
        resp = client.post(f"/api/games/{gid}/solve/advisor", json={
            "player_idx": game.current_player_idx,
            "active_sets": ["oceania", "core", "asia", "european", "promo_uk"],
            "main_sims": 20,
            "upside_shortlist": 4,
            "upside_sims": 6,
        })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["main_line"] is not None
    ml = data["main_line"]
    assert ml["sentence"] and "%" in ml["sentence"]
    assert ml["floor_score"] <= ml["typical"] <= ml["stretch_score"]
    assert data["pool_size"] > 0
    # upside lines are optional (depend on draws) but must be well-formed if present
    for u in data["upside_lines"]:
        assert 0.0 <= u["draw_prob"] <= 0.99
        assert u["lift_vs_main"] >= adv.UPSIDE_MIN_LIFT
        assert u["bird_name"] in u["sentence"]
