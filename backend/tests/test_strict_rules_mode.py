import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.bird import Bird, FoodCost
from backend.models.enums import (
    BeakDirection,
    BoardType,
    FoodType,
    GameSet,
    Habitat,
    NestType,
    PowerColor,
)
from backend.models.game_state import create_new_game
from backend.solver.simulation import simulate_playout


def test_simulation_rejects_non_strict_power_mapping_in_strict_mode():
    load_all(EXCEL_FILE)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA, strict_rules_mode=True)
    a = game.players[0]
    # Synthetic fallback power should be rejected in strict mode.
    a.board.forest.slots[0].bird = Bird(
        name="Strict Rejection Dummy",
        scientific_name="Dummy",
        game_set=GameSet.CORE,
        color=PowerColor.BROWN,
        power_text="Unrecognized power text for strict-mode rejection.",
        victory_points=2,
        nest_type=NestType.BOWL,
        egg_limit=3,
        wingspan_cm=20,
        habitats=frozenset({Habitat.FOREST}),
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.LEFT,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )
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
