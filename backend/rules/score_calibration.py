"""Score-calibration helpers for deterministic constructed full-game states."""

from __future__ import annotations

import random
from typing import Iterable

from backend.engine.scoring import calculate_score
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


def run_constructed_score_case(
    *,
    seed: int,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
) -> dict:
    """Build a deterministic end-of-game state and return score breakdowns."""
    rng = random.Random(int(seed))
    game = create_new_game(
        [f"Player_{i+1}" for i in range(players)],
        board_type=board_type,
    )

    birds = [_fixture_bird(idx=i) for i in range(1, 81)]
    bird_idx = 0
    for player in game.players:
        for row in player.board.all_rows():
            filled_slots = rng.randint(2, len(row.slots))
            for slot_i in range(filled_slots):
                bird = birds[bird_idx % len(birds)]
                bird_idx += 1
                slot = row.slots[slot_i]
                slot.bird = bird
                slot.eggs = rng.randint(0, max(0, bird.egg_limit))
                slot.tucked_cards = rng.randint(0, 3)
                for _ in range(rng.randint(0, 2)):
                    slot.cache_food(_fixture_food(rng))
            row.nectar_spent = rng.randint(0, 4)

        bonus_pool = _fixture_bonus_cards()
        bonus_count = rng.randint(1, min(3, len(bonus_pool)))
        player.bonus_cards = rng.sample(bonus_pool, bonus_count)

    game.round_goal_scores = {}
    for round_num in range(1, 5):
        game.round_goal_scores[round_num] = {
            p.name: rng.randint(0, 7) for p in game.players
        }

    scores = {p.name: calculate_score(game, p).as_dict() for p in game.players}
    return {
        "seed": int(seed),
        "scores": scores,
    }


def build_score_calibration_goldens(
    seeds: Iterable[int],
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    strict_rules_mode: bool = True,
) -> dict:
    """Generate a portable golden payload for score calibration."""
    cases = [
        run_constructed_score_case(
            seed=int(seed),
            players=players,
            board_type=board_type,
        )
        for seed in seeds
    ]
    return {
        "version": 2,
        "policy": "constructed_full_game_states",
        "players": int(players),
        "board_type": board_type.value,
        "strict_rules_mode": bool(strict_rules_mode),
        "max_turns": int(max_turns),
        "cases": cases,
    }


def _fixture_food(rng: random.Random) -> FoodType:
    foods = (
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
    )
    return foods[rng.randrange(len(foods))]


def _fixture_bird(*, idx: int) -> Bird:
    vp = 2 + (idx % 8)
    egg_limit = 2 + (idx % 5)
    nest_types = (
        NestType.BOWL,
        NestType.CAVITY,
        NestType.GROUND,
        NestType.PLATFORM,
        NestType.WILD,
    )
    habitat_triplets = (
        frozenset({Habitat.FOREST}),
        frozenset({Habitat.GRASSLAND}),
        frozenset({Habitat.WETLAND}),
        frozenset({Habitat.FOREST, Habitat.GRASSLAND}),
        frozenset({Habitat.FOREST, Habitat.WETLAND}),
        frozenset({Habitat.GRASSLAND, Habitat.WETLAND}),
        frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
    )
    bonus_tags: set[str] = set()
    if idx % 2 == 0:
        bonus_tags.add("Fixture Bonus A")
    if idx % 3 == 0:
        bonus_tags.add("Fixture Bonus B")
    if idx % 5 == 0:
        bonus_tags.add("Fixture Bonus C")
    return Bird(
        name=f"Fixture Bird {idx}",
        scientific_name=f"Fixture Bird {idx}",
        game_set=GameSet.CORE,
        color=PowerColor.NONE,
        power_text="",
        victory_points=vp,
        nest_type=nest_types[idx % len(nest_types)],
        egg_limit=egg_limit,
        wingspan_cm=20 + (idx % 200),
        habitats=habitat_triplets[idx % len(habitat_triplets)],
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.LEFT if idx % 2 == 0 else BeakDirection.RIGHT,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=bool(bonus_tags),
        bonus_eligibility=frozenset(bonus_tags),
    )


def _fixture_bonus_cards() -> list[BonusCard]:
    return [
        BonusCard(
            name="Fixture Bonus A",
            game_sets=frozenset({GameSet.CORE}),
            condition_text="Fixture bonus A",
            explanation_text=None,
            scoring_tiers=(BonusScoringTier(min_count=1, max_count=None, points=1),),
            is_per_bird=True,
            is_automa=False,
            draft_value_pct=None,
        ),
        BonusCard(
            name="Fixture Bonus B",
            game_sets=frozenset({GameSet.CORE}),
            condition_text="Fixture bonus B",
            explanation_text=None,
            scoring_tiers=(BonusScoringTier(min_count=1, max_count=None, points=2),),
            is_per_bird=True,
            is_automa=False,
            draft_value_pct=None,
        ),
        BonusCard(
            name="Fixture Bonus C",
            game_sets=frozenset({GameSet.CORE}),
            condition_text="Fixture bonus C",
            explanation_text=None,
            scoring_tiers=(
                BonusScoringTier(min_count=0, max_count=1, points=0),
                BonusScoringTier(min_count=2, max_count=4, points=3),
                BonusScoringTier(min_count=5, max_count=None, points=6),
            ),
            is_per_bird=False,
            is_automa=False,
            draft_value_pct=None,
        ),
    ]
