"""Integration tests for full-game power color execution coverage."""

import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.tests.helpers.power_coverage import run_power_coverage_validation_game


@pytest.fixture(scope="module", autouse=True)
def load_data():
    load_all(EXCEL_FILE)


def test_validation_game_executes_all_5_colors_per_game():
    for seed in range(5):
        result = run_power_coverage_validation_game(seed=9000 + seed, board_type=BoardType.OCEANIA)
        assert result["game_over"], f"seed={seed} did not complete"
        assert result["all_colors_executed"], (
            f"seed={seed} missing colors; executed={result['executed_colors']}"
        )


def test_validation_games_are_full_not_5_bird_toys():
    for seed in range(3):
        result = run_power_coverage_validation_game(seed=9100 + seed, board_type=BoardType.OCEANIA)
        assert result["game_over"], f"seed={seed} did not complete"
        assert result["total_birds_played"] >= 8, (
            f"seed={seed} birds_played={result['total_birds_played']}, scores={result['scores']}"
        )
        assert result["winner_score"] >= 0
