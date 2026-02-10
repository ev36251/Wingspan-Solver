from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.actions import execute_gain_food
from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard, BonusScoringTier
from backend.models.enums import (
    ActionType,
    BeakDirection,
    BoardType,
    FoodType,
    GameSet,
    Habitat,
    NestType,
    PowerColor,
)
from backend.models.game_state import create_new_game
from backend.engine.timed_powers import trigger_end_of_game_powers


def _make_test_bird(
    *,
    name: str,
    color: PowerColor,
    power_text: str,
    nest_type: NestType = NestType.BOWL,
    egg_limit: int = 5,
) -> Bird:
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.EUROPEAN,
        color=color,
        power_text=power_text,
        victory_points=2,
        nest_type=nest_type,
        egg_limit=egg_limit,
        wingspan_cm=30,
        habitats=frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.LEFT,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )


def test_pink_between_turn_trigger_gain_food() -> None:
    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    bob = game.players[1]

    pink = _make_test_bird(
        name="Pink Seed Collector",
        color=PowerColor.PINK,
        power_text='When another player takes the "gain food" action, gain 1 [seed] from the supply.',
    )
    bob.board.forest.slots[0].bird = pink
    bob.food_supply.seed = 0

    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])
    result = execute_gain_food(game, alice, [FoodType.SEED], bonus_count=0, reset_bonus=False)

    assert result.success
    assert bob.food_supply.seed >= 1


def test_end_of_round_lay_eggs_power_triggers_on_advance_round() -> None:
    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]

    teal = _make_test_bird(
        name="Teal Round Layer",
        color=PowerColor.TEAL,
        power_text="At end of round, lay 1 [egg] on each of your [ground] nesting birds.",
    )
    ground_target = _make_test_bird(
        name="Ground Target",
        color=PowerColor.NONE,
        power_text="",
        nest_type=NestType.GROUND,
        egg_limit=4,
    )
    alice.board.forest.slots[0].bird = teal
    alice.board.grassland.slots[0].bird = ground_target
    alice.board.grassland.slots[0].eggs = 0

    game.current_round = 1
    game.advance_round()

    assert alice.board.grassland.slots[0].eggs >= 1


def test_end_of_game_yellow_bonus_copy_triggers_once() -> None:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    yellow = bird_reg.get("Greater Adjutant")
    assert yellow is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    bob = game.players[1]
    alice.board.forest.slots[0].bird = yellow

    bonus = BonusCard(
        name="Test Bonus",
        game_sets=frozenset({GameSet.CORE}),
        condition_text="Test",
        explanation_text=None,
        scoring_tiers=(BonusScoringTier(min_count=0, max_count=None, points=4),),
        is_per_bird=False,
        is_automa=False,
        draft_value_pct=0.5,
    )
    bob.bonus_cards.append(bonus)

    game.current_round = 4
    game.advance_round()  # round 5 -> game over, should trigger end-of-game powers
    assert game.is_game_over
    assert any(bc.name == "Test Bonus" for bc in alice.bonus_cards)

    before = len(alice.bonus_cards)
    n = trigger_end_of_game_powers(game)
    assert n == 0
    assert len(alice.bonus_cards) == before

