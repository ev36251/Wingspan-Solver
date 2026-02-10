import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType, Habitat, FoodType
from backend.models.game_state import create_new_game
from backend.solver.simulation import simulate_playout


def test_simulation_rejects_non_strict_power_mapping_in_strict_mode():
    birds, _, _ = load_all(EXCEL_FILE)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA, strict_rules_mode=True)
    a = game.players[0]
    # Abbott's Booby is regex-parsed (non-strict).
    a.board.forest.slots[0].bird = birds.get("Abbott's Booby")
    with pytest.raises(RuntimeError, match="non-strict power mapping"):
        simulate_playout(game, max_turns=5)


def test_simulation_allows_strict_mapped_birds_in_strict_mode():
    birds, _, _ = load_all(EXCEL_FILE)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA, strict_rules_mode=True)
    a = game.players[0]
    a.board.forest.slots[0].bird = birds.get("Wood Duck")
    a.food_supply.invertebrate = 1
    a.food_supply.seed = 1
    a.food_supply.fruit = 1
    a.food_supply.nectar = 1
    game.deck_remaining = 20
    # Should run without strict-mode rejection.
    scores = simulate_playout(game, max_turns=1)
    assert "A" in scores
    assert "B" in scores
