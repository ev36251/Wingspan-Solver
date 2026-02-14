from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType, GameSet, Habitat
from backend.models.game_state import create_new_game
from backend.models.goal import Goal


def _goal_by_desc(goal_reg, desc: str):
    return next((g for g in goal_reg.all_goals if g.description.lower() == desc.lower()), None)


def test_base_round_order_end_round_powers_then_goal_then_cleanup() -> None:
    bird_reg, _, goal_reg = load_all(EXCEL_FILE)
    goal = _goal_by_desc(goal_reg, "[egg] in [grassland]")
    assert goal is not None

    game = create_new_game(["Alice", "Bob"], round_goals=[goal], board_type=BoardType.BASE)
    alice, bob = game.players

    whitethroat = bird_reg.get("Lesser Whitethroat")
    twite = bird_reg.get("Twite")
    assert whitethroat is not None
    assert twite is not None

    # Alice has a round-end egg layer in grasslands and another grassland bird to receive eggs.
    alice.board.grassland.slots[0].bird = whitethroat
    alice.board.grassland.slots[1].bird = twite
    bob.board.grassland.slots[0].bird = twite
    alice.board.grassland.slots[0].eggs = 0
    alice.board.grassland.slots[1].eggs = 0
    bob.board.grassland.slots[0].eggs = 0

    # Ensure cleanup state is reset after scoring.
    alice.play_bird_actions_this_round = 2
    bob.play_bird_actions_this_round = 1

    game.advance_round()

    # Round-end power should have run before scoring, giving Alice the goal lead.
    assert game.round_goal_scores[1]["Alice"] > game.round_goal_scores[1]["Bob"]
    assert alice.board.grassland.total_eggs() >= 2

    # Per-round counters should be cleaned up after round scoring.
    assert alice.play_bird_actions_this_round == 0
    assert bob.play_bird_actions_this_round == 0


def test_oceania_round_order_discards_nectar_before_goal_scoring() -> None:
    # Use a custom scoring tuple with clear 1st/2nd separation.
    goal = Goal(
        description="[wild] in personal supply",
        game_set=GameSet.CORE,
        scoring=(0, 0, 1, 4),
        reverse_description="[card] in hand",
    )

    game = create_new_game(["Alice", "Bob"], round_goals=[goal], board_type=BoardType.OCEANIA)
    alice, bob = game.players

    # Alice starts with more nectar, but nectar should be discarded before scoring.
    alice.food_supply.nectar = 3
    bob.food_supply.nectar = 0
    bob.food_supply.seed = 1
    alice.play_bird_actions_this_round = 1
    bob.play_bird_actions_this_round = 2

    game.advance_round()

    assert game.round_goal_scores[1]["Bob"] > game.round_goal_scores[1]["Alice"]
    # Cleanup should still happen by end of transition.
    assert alice.food_supply.nectar == 0
    assert bob.food_supply.nectar == 0
    assert alice.play_bird_actions_this_round == 0
    assert bob.play_bird_actions_this_round == 0
