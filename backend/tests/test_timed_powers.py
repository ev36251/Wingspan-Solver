from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.actions import ActionResult, PowerActivation, execute_gain_food
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
from backend.engine.timed_powers import (
    trigger_between_turn_powers,
    trigger_end_of_game_powers,
    trigger_end_of_round_powers,
)
from backend.powers.base import PowerResult
from backend.powers.choices import queue_power_choice


def _make_test_bird(
    *,
    name: str,
    color: PowerColor,
    power_text: str,
    nest_type: NestType = NestType.BOWL,
    egg_limit: int = 5,
    wingspan_cm: int = 30,
    is_predator: bool = False,
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
        wingspan_cm=wingspan_cm,
        habitats=frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.LEFT,
        is_predator=is_predator,
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


def test_yellowhammer_only_plays_bonus_bird_if_all_four_action_types_used() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    yellowhammer = birds.get("Yellowhammer")
    assert yellowhammer is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.grassland.slots[0].bird = yellowhammer
    alice.hand = [
        _make_test_bird(
            name="Bonus Play Candidate",
            color=PowerColor.NONE,
            power_text="",
        )
    ]
    alice.food_supply.seed = 2

    alice.action_types_used_this_round = {
        ActionType.GAIN_FOOD,
        ActionType.LAY_EGGS,
        ActionType.DRAW_CARDS,
    }
    before_birds = alice.total_birds
    trigger_end_of_round_powers(game, 1)
    assert alice.total_birds == before_birds
    assert len(alice.hand) == 1

    alice.action_types_used_this_round.add(ActionType.PLAY_BIRD)
    trigger_end_of_round_powers(game, 1)
    assert alice.total_birds == before_birds + 1
    assert len(alice.hand) == 0


def test_pacific_black_duck_game_end_eggs_scale_with_wetland_eggs() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    duck = birds.get("Pacific Black Duck")
    assert duck is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.wetland.slots[0].bird = duck
    helper = _make_test_bird(name="Wetland Helper", color=PowerColor.NONE, power_text="")
    alice.board.wetland.slots[1].bird = helper
    alice.board.wetland.slots[1].eggs = 4  # floor(4 / 2) = 2 eggs on duck

    n = trigger_end_of_game_powers(game)
    assert n >= 1
    assert alice.board.wetland.slots[0].eggs == 2


def test_australian_magpie_discards_row_column_eggs_and_caches_seed() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    magpie = birds.get("Australian Magpie")
    assert magpie is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.grassland.slots[1].bird = magpie
    alice.board.grassland.slots[1].eggs = 2  # excluded
    # Same row, excluding self.
    for idx in (0, 2):
        alice.board.grassland.slots[idx].bird = _make_test_bird(
            name=f"Row Bird {idx}", color=PowerColor.NONE, power_text=""
        )
        alice.board.grassland.slots[idx].eggs = 1
    # Same column in other habitats.
    for hab in (Habitat.FOREST, Habitat.WETLAND):
        slot = alice.board.get_row(hab).slots[1]
        slot.bird = _make_test_bird(name=f"{hab.value.title()} Col Bird", color=PowerColor.NONE, power_text="")
        slot.eggs = 1

    n = trigger_end_of_game_powers(game)
    assert n >= 1
    assert alice.board.grassland.slots[0].eggs == 0
    assert alice.board.grassland.slots[2].eggs == 0
    assert alice.board.forest.slots[1].eggs == 0
    assert alice.board.wetland.slots[1].eggs == 0
    # 4 discarded eggs -> 8 cached seed on Australian Magpie.
    assert alice.board.grassland.slots[1].cached_food.get(FoodType.SEED, 0) == 8


def test_black_swan_only_lays_on_large_wingspan_birds() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    swan = birds.get("Black Swan")
    assert swan is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.wetland.slots[0].bird = swan
    big = _make_test_bird(
        name="Big Wingspan Bird", color=PowerColor.NONE, power_text="", wingspan_cm=120
    )
    small = _make_test_bird(
        name="Small Wingspan Bird", color=PowerColor.NONE, power_text="", wingspan_cm=80
    )
    alice.board.forest.slots[0].bird = big
    alice.board.forest.slots[1].bird = small

    n = trigger_end_of_game_powers(game)
    assert n >= 1
    assert alice.board.forest.slots[0].eggs == 1
    assert alice.board.forest.slots[1].eggs == 0
    assert alice.board.wetland.slots[0].eggs >= 1


def test_kakapo_draws_bonus_cards_keep_one() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    kakapo = birds.get("Kākāpō")
    assert kakapo is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.forest.slots[0].bird = kakapo
    before = len(alice.bonus_cards)
    n = trigger_end_of_game_powers(game)
    assert n >= 1
    assert len(alice.bonus_cards) == before + 1


def test_magpie_lark_game_end_discards_forest_eggs_and_plays_grassland_bird_ignore_egg_cost() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    magpie_lark = birds.get("Magpie-Lark")
    assert magpie_lark is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.grassland.slots[0].bird = magpie_lark
    # Add two eggs in forest to pay Magpie-Lark condition.
    forest_a = _make_test_bird(name="Forest Egg A", color=PowerColor.NONE, power_text="")
    forest_b = _make_test_bird(name="Forest Egg B", color=PowerColor.NONE, power_text="")
    alice.board.forest.slots[0].bird = forest_a
    alice.board.forest.slots[0].eggs = 1
    alice.board.forest.slots[1].bird = forest_b
    alice.board.forest.slots[1].eggs = 1

    # Candidate grassland bird should land in slot 1 (normally costs 1 egg).
    candidate = _make_test_bird(name="Late Grass Bird", color=PowerColor.NONE, power_text="")
    alice.hand = [candidate]
    alice.food_supply.seed = 1

    n = trigger_end_of_game_powers(game)
    assert n >= 1
    assert alice.board.forest.slots[0].eggs == 0
    assert alice.board.forest.slots[1].eggs == 0
    assert alice.board.grassland.slots[1].bird is not None
    assert alice.board.grassland.slots[1].bird.name == "Late Grass Bird"
    assert len(alice.hand) == 0


def test_common_cuckoo_lays_on_another_bowl_or_ground_bird() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    cuckoo = birds.get("Common Cuckoo")
    assert cuckoo is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.grassland.slots[0].bird = cuckoo
    target = _make_test_bird(
        name="Ground Nest Target",
        color=PowerColor.NONE,
        power_text="",
        nest_type=NestType.GROUND,
    )
    bob.board.grassland.slots[1].bird = target
    before = bob.board.grassland.slots[1].eggs

    trigger_result = ActionResult(success=True, action_type=ActionType.LAY_EGGS, habitat=Habitat.GRASSLAND)
    n = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.LAY_EGGS,
        trigger_result=trigger_result,
    )
    assert n >= 1
    assert bob.board.grassland.slots[1].eggs == before + 1


def test_belted_kingfisher_triggers_on_opponent_wetland_play() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    kingfisher = birds.get("Belted Kingfisher")
    assert kingfisher is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.wetland.slots[0].bird = kingfisher
    before = bob.food_supply.fish

    trigger_result = ActionResult(success=True, action_type=ActionType.PLAY_BIRD, habitat=Habitat.WETLAND)
    n = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.PLAY_BIRD,
        trigger_result=trigger_result,
    )
    assert n >= 1
    assert bob.food_supply.fish == before + 1


def test_european_goldfinch_tucks_on_opponent_tuck_event() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    goldfinch = birds.get("European Goldfinch")
    assert goldfinch is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.grassland.slots[0].bird = goldfinch
    game._deck_cards = [_make_test_bird(name="Deck Tuck Bird", color=PowerColor.NONE, power_text="")]
    game.deck_remaining = 1

    trigger_result = ActionResult(
        success=True,
        action_type=ActionType.DRAW_CARDS,
        power_activations=[
            PowerActivation(
                bird_name="Any Source",
                slot_index=0,
                result=PowerResult(cards_tucked=1),
            )
        ],
    )
    n = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.DRAW_CARDS,
        trigger_result=trigger_result,
    )
    assert n >= 1
    assert bob.board.grassland.slots[0].tucked_cards == 1


def test_snow_bunting_tucks_from_hand_then_draws_on_opponent_tuck() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    bunting = birds.get("Snow Bunting")
    assert bunting is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.grassland.slots[0].bird = bunting
    bob.hand = [
        _make_test_bird(name="Hand Card To Tuck", color=PowerColor.NONE, power_text=""),
    ]
    game._deck_cards = [_make_test_bird(name="Drawn After Tuck", color=PowerColor.NONE, power_text="")]
    game.deck_remaining = 1

    trigger_result = ActionResult(
        success=True,
        action_type=ActionType.PLAY_BIRD,
        power_activations=[
            PowerActivation(
                bird_name="Any Source",
                slot_index=0,
                result=PowerResult(cards_tucked=1),
            )
        ],
    )
    n = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.PLAY_BIRD,
        trigger_result=trigger_result,
    )
    assert n >= 1
    assert bob.board.grassland.slots[0].tucked_cards == 1
    assert len(bob.hand) == 1
    assert bob.hand[0].name == "Drawn After Tuck"


def test_black_vulture_triggers_only_when_predator_succeeds() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    vulture = birds.get("Black Vulture")
    assert vulture is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.forest.slots[0].bird = vulture
    predator = _make_test_bird(
        name="Predator Source",
        color=PowerColor.BROWN,
        power_text="",
        is_predator=True,
    )
    alice.board.forest.slots[0].bird = predator
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])

    miss_result = ActionResult(
        success=True,
        action_type=ActionType.GAIN_FOOD,
        power_activations=[
            PowerActivation(
                bird_name="Predator Source",
                slot_index=0,
                result=PowerResult(cards_tucked=0, food_cached={}),
            )
        ],
    )
    n_miss = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.GAIN_FOOD,
        trigger_result=miss_result,
    )
    assert n_miss == 0
    before_total = bob.food_supply.total()

    hit_result = ActionResult(
        success=True,
        action_type=ActionType.GAIN_FOOD,
        power_activations=[
            PowerActivation(
                bird_name="Predator Source",
                slot_index=0,
                result=PowerResult(food_cached={FoodType.RODENT: 1}),
            )
        ],
    )
    n_hit = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.GAIN_FOOD,
        trigger_result=hit_result,
    )
    assert n_hit >= 1
    assert bob.food_supply.total() == before_total + 1


def test_spangled_drongo_gains_nectar_when_opponent_gains_nectar() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    drongo = birds.get("Spangled Drongo")
    assert drongo is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice, bob = game.players
    bob.board.forest.slots[0].bird = drongo
    before = bob.food_supply.nectar

    trigger_result = ActionResult(
        success=True,
        action_type=ActionType.GAIN_FOOD,
        food_gained={FoodType.NECTAR: 1},
    )
    n = trigger_between_turn_powers(
        game,
        trigger_player=alice,
        trigger_action=ActionType.GAIN_FOOD,
        trigger_result=trigger_result,
    )
    assert n >= 1
    assert bob.food_supply.nectar == before + 1


def test_brown_power_can_be_explicitly_skipped_via_activation_choice() -> None:
    birds, _, _ = load_all(EXCEL_FILE)
    hornbill = birds.get("Great Hornbill")
    assert hornbill is not None

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    alice.board.forest.slots[0].bird = hornbill
    alice.hand = [_make_test_bird(name="Card To Tuck", color=PowerColor.NONE, power_text="")]
    alice.food_supply.add(FoodType.FRUIT, 1)
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT])

    queue_power_choice(game, "Alice", "Great Hornbill", {"activate": False})
    res = execute_gain_food(game, alice, [FoodType.SEED], bonus_count=0, reset_bonus=False)
    assert res.success
    assert alice.board.forest.slots[0].tucked_cards == 0
    assert alice.board.forest.slots[0].cached_food.get(FoodType.FRUIT, 0) == 0
