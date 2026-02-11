from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.actions import execute_play_bird
from backend.engine.scoring import goal_progress_for_round
from backend.models.enums import FoodType, Habitat
from backend.models.game_state import create_new_game
from backend.rules.rules_gap_audit import run_rules_gap_audit


def _goal_by_desc(goal_reg, desc: str):
    return next((g for g in goal_reg.all_goals if g.description.lower() == desc.lower()), None)


def test_goal_nest_bird_with_egg_counts_only_birds_with_eggs() -> None:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    goal_reg = regs[2]

    goal = _goal_by_desc(goal_reg, "[bowl] [bird] with [egg]")
    assert goal is not None

    game = create_new_game(["Alice", "Bob"])
    p = game.players[0]

    bowl_birds = [b for b in bird_reg.all_birds if b.nest_type.value == "bowl"]
    assert len(bowl_birds) >= 2
    p.board.forest.slots[0].bird = bowl_birds[0]
    p.board.grassland.slots[0].bird = bowl_birds[1]
    p.board.forest.slots[0].eggs = 1
    p.board.grassland.slots[0].eggs = 0

    assert goal_progress_for_round(p, goal) == 1


def test_goal_card_in_hand_and_food_supply_progress() -> None:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    goal_reg = regs[2]

    g_cards = _goal_by_desc(goal_reg, "[card] in hand")
    g_food = _goal_by_desc(goal_reg, "[wild] in personal supply")
    assert g_cards is not None and g_food is not None

    game = create_new_game(["Alice", "Bob"])
    p = game.players[0]
    p.hand.extend(bird_reg.all_birds[:3])
    p.unknown_hand_count = 2
    p.food_supply.add(FoodType.SEED, 2)
    p.food_supply.add(FoodType.FISH, 1)

    assert goal_progress_for_round(p, g_cards) == 5
    assert goal_progress_for_round(p, g_food) == 3


def test_goal_sets_of_eggs_across_habitats_progress() -> None:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    goal_reg = regs[2]

    goal = _goal_by_desc(goal_reg, "sets of [egg][egg][egg] in [wetland][grassland][forest]")
    assert goal is not None

    game = create_new_game(["Alice", "Bob"])
    p = game.players[0]
    birds = bird_reg.all_birds[:3]
    p.board.forest.slots[0].bird = birds[0]
    p.board.grassland.slots[0].bird = birds[1]
    p.board.wetland.slots[0].bird = birds[2]
    p.board.forest.slots[0].eggs = 3
    p.board.grassland.slots[0].eggs = 2
    p.board.wetland.slots[0].eggs = 5

    assert goal_progress_for_round(p, goal) == 2


def test_rules_gap_audit_round_goal_support_is_near_complete() -> None:
    report = run_rules_gap_audit()
    assert report["round_goals_unsupported"] == 0
    assert report["unsupported_goal_descriptions"] == []


def test_goal_action_cube_gray_counts_play_bird_actions_and_resets() -> None:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    goal_reg = regs[2]
    goal = _goal_by_desc(goal_reg, '[action_cube_gray] cubes on "play a bird"')
    assert goal is not None

    game = create_new_game(["Alice", "Bob"], round_goals=[goal])
    alice = game.players[0]
    bob = game.players[1]

    birds_forest = [b for b in bird_reg.all_birds if b.can_live_in(Habitat.FOREST)]
    birds_grass = [b for b in bird_reg.all_birds if b.can_live_in(Habitat.GRASSLAND)]
    assert birds_forest and birds_grass
    bird1, bird2 = birds_forest[0], birds_grass[0]
    alice.hand.extend([bird1, bird2])

    for ft in (FoodType.SEED, FoodType.INVERTEBRATE, FoodType.FRUIT, FoodType.FISH, FoodType.RODENT, FoodType.NECTAR):
        alice.food_supply.add(ft, 10)

    def payment_for(bird):
        if bird.food_cost.is_or:
            ft = next(iter(bird.food_cost.distinct_types))
            if ft == FoodType.WILD:
                ft = FoodType.SEED
            return {ft: 1}
        pay: dict[FoodType, int] = {}
        for ft in bird.food_cost.items:
            if ft == FoodType.WILD:
                ft = FoodType.SEED
            pay[ft] = pay.get(ft, 0) + 1
        return pay

    r1 = execute_play_bird(game, alice, bird1, Habitat.FOREST, payment_for(bird1))
    r2 = execute_play_bird(game, alice, bird2, Habitat.GRASSLAND, payment_for(bird2))
    assert r1.success and r2.success

    alice_progress = goal_progress_for_round(alice, goal)
    bob_progress = goal_progress_for_round(bob, goal)
    assert alice_progress == 2
    assert bob_progress == 0

    game.advance_round()
    assert 1 in game.round_goal_scores
    assert "Alice" in game.round_goal_scores[1]
    assert "Bob" in game.round_goal_scores[1]
    assert alice.play_bird_actions_this_round == 0
