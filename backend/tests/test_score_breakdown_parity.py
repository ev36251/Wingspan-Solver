import json
from pathlib import Path

from backend.engine.scoring import calculate_score
from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard, BonusScoringTier
from backend.models.enums import BeakDirection, BoardType, FoodType, GameSet, Habitat, NestType, PowerColor
from backend.models.game_state import create_new_game
from backend.solver.replay_video_game import run_scripted_replay


def _load_fixture() -> dict:
    path = Path(__file__).parent / "fixtures" / "score_breakdown_parity.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _make_bird(name: str, vp: int, *, eligible_bonus: bool = False) -> Bird:
    bonus = frozenset({"Fixture Bonus"}) if eligible_bonus else frozenset()
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.CORE,
        color=PowerColor.NONE,
        power_text="",
        victory_points=vp,
        nest_type=NestType.BOWL,
        egg_limit=5,
        wingspan_cm=30,
        habitats=frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.LEFT,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=eligible_bonus,
        bonus_eligibility=bonus,
    )


def test_score_breakdown_parity_replay_fixture() -> None:
    fx = _load_fixture()["replay_seed_20260210"]
    result = run_scripted_replay(seed=fx["seed"], max_turns=260)

    assert result["divergence_count"] == 0
    assert result["you_score"] == fx["you_score"]
    assert result["opp_score"] == fx["opp_score"]
    assert result["you_score"]["total"] == sum(v for k, v in result["you_score"].items() if k != "total")
    assert result["opp_score"]["total"] == sum(v for k, v in result["opp_score"].items() if k != "total")


def test_score_breakdown_parity_constructed_oceania_fixture() -> None:
    fx = _load_fixture()["constructed_oceania_fixture"]
    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players

    bird_a = _make_bird("Fixture Bird A", 5, eligible_bonus=True)
    bird_b = _make_bird("Fixture Bird B", 3, eligible_bonus=False)
    alice.board.forest.slots[0].bird = bird_a
    alice.board.grassland.slots[0].bird = bird_b
    alice.board.forest.slots[0].eggs = 2
    alice.board.grassland.slots[0].eggs = 1
    alice.board.forest.slots[0].cache_food(FoodType.FISH, 1)
    alice.board.grassland.slots[0].cache_food(FoodType.FRUIT, 1)
    alice.board.forest.slots[0].tucked_cards = 2
    alice.board.grassland.slots[0].tucked_cards = 1

    fixture_bonus = BonusCard(
        name="Fixture Bonus",
        game_sets=frozenset({GameSet.CORE}),
        condition_text="Fixture parity bonus",
        explanation_text=None,
        scoring_tiers=(BonusScoringTier(min_count=1, max_count=None, points=2),),
        is_per_bird=True,
        is_automa=False,
        draft_value_pct=None,
    )
    alice.bonus_cards.append(fixture_bonus)

    game.round_goal_scores = {
        1: {"Alice": 4, "Bob": 1},
        2: {"Alice": 2, "Bob": 5},
    }

    # Oceania nectar majority setup.
    alice.board.forest.nectar_spent = 3
    alice.board.grassland.nectar_spent = 1
    bob.board.forest.nectar_spent = 1
    bob.board.grassland.nectar_spent = 1
    bob.board.wetland.nectar_spent = 2

    alice_score = calculate_score(game, alice).as_dict()
    bob_score = calculate_score(game, bob).as_dict()

    assert alice_score == fx["alice_score"]
    assert bob_score == fx["bob_score"]
    assert alice_score["total"] == sum(v for k, v in alice_score.items() if k != "total")
    assert bob_score["total"] == sum(v for k, v in bob_score.items() if k != "total")
