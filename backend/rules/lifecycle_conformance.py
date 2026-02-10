"""Deterministic lifecycle conformance checks for official timing windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.actions import execute_gain_food
from backend.engine.scoring import compute_round_goal_scores
from backend.engine.timed_powers import trigger_end_of_game_powers
from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard, BonusScoringTier
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
from backend.models.goal import Goal


@dataclass
class LifecycleCheck:
    check_id: str
    passed: bool
    detail: str


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


def run_lifecycle_conformance_checks() -> dict:
    """Run deterministic conformance checks for lifecycle timing/order."""
    checks: list[LifecycleCheck] = []

    checks.append(_check_between_turn_pink_trigger())
    checks.append(_check_round_end_before_round_goal_scoring())
    checks.extend(_check_game_end_trigger_and_guard())
    checks.append(_check_round_transition_action_cubes())

    passed = sum(1 for c in checks if c.passed)
    return {
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": len(checks) - passed,
        "all_passed": passed == len(checks),
        "checks": [asdict(c) for c in checks],
    }


def _check_between_turn_pink_trigger() -> LifecycleCheck:
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
    ok = bool(result.success and bob.food_supply.seed >= 1)
    detail = f"success={result.success}, bob_seed={bob.food_supply.seed}"
    return LifecycleCheck(
        check_id="between_turn_pink_trigger_after_action",
        passed=ok,
        detail=detail,
    )


def _check_round_end_before_round_goal_scoring() -> LifecycleCheck:
    game = create_new_game(
        ["Alice", "Bob"],
        round_goals=[
            Goal(
                description="[Egg] in [Ground]",
                game_set=GameSet.CORE,
                scoring=(0, 1, 3, 5),
                reverse_description="[Egg] in [Platform]",
            )
        ],
        board_type=BoardType.OCEANIA,
    )
    alice = game.players[0]
    bob = game.players[1]

    teal = _make_test_bird(
        name="Teal Round Layer",
        color=PowerColor.TEAL,
        power_text="At end of round, lay 1 [egg] on each of your [ground] nesting birds.",
    )
    ground_a = _make_test_bird(
        name="Ground A",
        color=PowerColor.NONE,
        power_text="",
        nest_type=NestType.GROUND,
        egg_limit=4,
    )
    ground_b = _make_test_bird(
        name="Ground B",
        color=PowerColor.NONE,
        power_text="",
        nest_type=NestType.GROUND,
        egg_limit=4,
    )

    alice.board.forest.slots[0].bird = teal
    alice.board.grassland.slots[0].bird = ground_a
    bob.board.grassland.slots[0].bird = ground_b
    alice.board.grassland.slots[0].eggs = 0
    bob.board.grassland.slots[0].eggs = 0

    pre = compute_round_goal_scores(game, 1)
    game.advance_round()
    post = game.round_goal_scores.get(1, {})
    ok = (
        pre.get("Alice") == pre.get("Bob")
        and alice.board.grassland.slots[0].eggs == 1
        and bob.board.grassland.slots[0].eggs == 0
        and post.get("Alice", -1) > post.get("Bob", -1)
    )
    detail = (
        f"pre={pre}, post={post}, eggs_a={alice.board.grassland.slots[0].eggs}, "
        f"eggs_b={bob.board.grassland.slots[0].eggs}"
    )
    return LifecycleCheck(
        check_id="round_end_powers_before_round_goal_scoring",
        passed=ok,
        detail=detail,
    )


def _check_game_end_trigger_and_guard() -> list[LifecycleCheck]:
    regs = load_all(EXCEL_FILE)
    bird_reg = regs[0]
    yellow = bird_reg.get("Greater Adjutant")
    bonus = BonusCard(
        name="Lifecycle Bonus",
        game_sets=frozenset({GameSet.CORE}),
        condition_text="Lifecycle check fixture",
        explanation_text=None,
        scoring_tiers=(BonusScoringTier(min_count=0, max_count=None, points=4),),
        is_per_bird=False,
        is_automa=False,
        draft_value_pct=0.5,
    )

    if yellow is None:
        return [
            LifecycleCheck(
                check_id="game_end_power_triggers_on_round_4_to_5",
                passed=False,
                detail="required fixture data not found",
            ),
            LifecycleCheck(
                check_id="game_end_power_executes_once_guard",
                passed=False,
                detail="required fixture data not found",
            ),
        ]

    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    alice = game.players[0]
    bob = game.players[1]
    alice.board.forest.slots[0].bird = yellow
    bob.bonus_cards.append(bonus)
    game.current_round = 4

    game.advance_round()
    first_ok = game.is_game_over and any(bc.name == bonus.name for bc in alice.bonus_cards)
    first = LifecycleCheck(
        check_id="game_end_power_triggers_on_round_4_to_5",
        passed=first_ok,
        detail=f"is_game_over={game.is_game_over}, alice_bonus_count={len(alice.bonus_cards)}",
    )

    before = len(alice.bonus_cards)
    n = trigger_end_of_game_powers(game)
    second_ok = n == 0 and len(alice.bonus_cards) == before
    second = LifecycleCheck(
        check_id="game_end_power_executes_once_guard",
        passed=second_ok,
        detail=f"manual_trigger_count={n}, bonus_before={before}, bonus_after={len(alice.bonus_cards)}",
    )
    return [first, second]


def _check_round_transition_action_cubes() -> LifecycleCheck:
    game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
    for p in game.players:
        p.action_cubes_remaining = 0
    game.advance_round()
    expected = 7
    ok = game.current_round == 2 and all(p.action_cubes_remaining == expected for p in game.players)
    detail = f"round={game.current_round}, cubes={[p.action_cubes_remaining for p in game.players]}"
    return LifecycleCheck(
        check_id="round_transition_resets_action_cubes",
        passed=ok,
        detail=detail,
    )
