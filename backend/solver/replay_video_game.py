"""Scripted replay harness for reproducing a specific Wingspan video line.

This runner sets a fixed opening state, then applies a sequence of explicit
move intents. If a scripted move cannot be matched to a legal move, it records
the divergence and falls back to a simple policy so the game can continue.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.bird import Bird, FoodCost
from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.models.game_state import GameState, create_new_game
from backend.powers.choices import queue_power_choice
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import execute_move_on_sim, pick_weighted_random_move, _refill_tray


@dataclass
class ScriptStep:
    actor: str  # "you" or "opp"
    action_type: ActionType
    bird_name: str | None = None
    habitat: Habitat | None = None
    prefer_tray_name: str | None = None
    prefer_food: list[FoodType] | None = None
    discovered_birds: list[str] | None = None
    discard_birds: list[str] | None = None
    grant_food_before: list[FoodType] | None = None
    grant_food_after: list[FoodType] | None = None
    set_food_before: dict[FoodType, int] | None = None
    set_tray_before: list[str] | None = None
    set_eggs_on_birds_before: dict[str, int] | None = None
    set_eggs_on_birds_after: dict[str, int] | None = None
    power_choices_before: list[dict] | None = None
    note: str = ""


@dataclass
class Divergence:
    turn: int
    actor: str
    reason: str
    note: str


def _bird(bird_reg, name: str, overrides: dict[str, Bird] | None = None):
    if overrides and name in overrides:
        return overrides[name]
    b = bird_reg.get(name)
    if b is None:
        raise ValueError(f"Unknown bird: {name}")
    return b


def _goal(goal_reg, desc: str):
    g = next((x for x in goal_reg.all_goals if x.description.lower() == desc.lower()), None)
    if g is None:
        raise ValueError(f"Unknown goal: {desc}")
    return g


def _video_bird_overrides(bird_reg) -> dict[str, Bird]:
    """Return per-video card-definition overrides without mutating base data."""
    overrides: dict[str, Bird] = {}

    mandarin = bird_reg.get("Mandarin Duck")
    if mandarin is not None:
        overrides["Mandarin Duck"] = replace(
            mandarin,
            food_cost=FoodCost(
                items=(FoodType.INVERTEBRATE, FoodType.INVERTEBRATE, FoodType.INVERTEBRATE),
                is_or=False,
                total=3,
            ),
        )

    trumpeter = bird_reg.get("Trumpeter Swan")
    if trumpeter is not None:
        overrides["Trumpeter Swan"] = replace(
            trumpeter,
            food_cost=FoodCost(
                items=(FoodType.FISH, FoodType.SEED, FoodType.WILD),
                is_or=False,
                total=3,
            ),
        )

    snowy = bird_reg.get("Snowy Owl")
    if snowy is not None:
        overrides["Snowy Owl"] = replace(
            snowy,
            food_cost=FoodCost(
                items=(FoodType.RODENT, FoodType.FRUIT, FoodType.FISH),
                is_or=False,
                total=3,
            ),
        )

    return overrides


def _find_matching_move(moves: list[Move], step: ScriptStep, game: GameState, actor_name: str) -> Move | None:
    candidates = [m for m in moves if m.action_type == step.action_type]
    if step.bird_name:
        candidates = [m for m in candidates if m.bird_name == step.bird_name]
    if step.habitat:
        candidates = [m for m in candidates if m.habitat == step.habitat]

    if not candidates:
        return None

    if step.action_type == ActionType.DRAW_CARDS and step.prefer_tray_name:
        name_by_idx = {i: b.name for i, b in enumerate(game.card_tray.face_up)}
        tray_first = []
        other = []
        for m in candidates:
            picked = [name_by_idx[i] for i in m.tray_indices if i in name_by_idx]
            if step.prefer_tray_name in picked:
                tray_first.append(m)
            else:
                other.append(m)
        candidates = tray_first or other
    elif step.action_type == ActionType.DRAW_CARDS:
        # For transcript-guided dig turns, prefer drawing from deck unless tray
        # target is explicit.
        candidates.sort(key=lambda m: (m.deck_draws, -len(m.tray_indices)), reverse=True)
        return candidates[0]

    if step.action_type == ActionType.GAIN_FOOD and step.prefer_food:
        wanted = Counter(step.prefer_food)
        def _food_match_score(m: Move) -> int:
            got = Counter(m.food_choices)
            return sum(min(got[k], v) for k, v in wanted.items())
        candidates.sort(key=_food_match_score, reverse=True)
        return candidates[0]

    # Prefer deterministic least-branch move.
    if step.action_type == ActionType.PLAY_BIRD:
        candidates.sort(key=lambda m: (len(m.food_payment), sorted((ft.value, c) for ft, c in m.food_payment.items())))
        return candidates[0]
    if step.action_type == ActionType.DRAW_CARDS:
        candidates.sort(key=lambda m: (len(m.tray_indices), m.deck_draws), reverse=True)
        return candidates[0]
    if step.action_type == ActionType.LAY_EGGS:
        candidates.sort(key=lambda m: sum(m.egg_distribution.values()), reverse=True)
        return candidates[0]

    return candidates[0]


def _fallback_move(game: GameState, moves: list[Move], actor_name: str) -> Move:
    if actor_name == "You":
        # Conservative fallback to keep replay stable.
        for at in (ActionType.PLAY_BIRD, ActionType.GAIN_FOOD, ActionType.DRAW_CARDS, ActionType.LAY_EGGS):
            picks = [m for m in moves if m.action_type == at]
            if picks:
                return picks[0]
    return pick_weighted_random_move(moves, game, game.current_player)


def _add_food(player, foods: list[FoodType] | None) -> None:
    if not foods:
        return
    for ft in foods:
        if ft == FoodType.INVERTEBRATE:
            player.food_supply.invertebrate += 1
        elif ft == FoodType.SEED:
            player.food_supply.seed += 1
        elif ft == FoodType.FISH:
            player.food_supply.fish += 1
        elif ft == FoodType.FRUIT:
            player.food_supply.fruit += 1
        elif ft == FoodType.RODENT:
            player.food_supply.rodent += 1
        elif ft == FoodType.NECTAR:
            player.food_supply.nectar += 1


def _set_food(player, food: dict[FoodType, int] | None) -> None:
    if food is None:
        return
    player.food_supply.invertebrate = food.get(FoodType.INVERTEBRATE, 0)
    player.food_supply.seed = food.get(FoodType.SEED, 0)
    player.food_supply.fish = food.get(FoodType.FISH, 0)
    player.food_supply.fruit = food.get(FoodType.FRUIT, 0)
    player.food_supply.rodent = food.get(FoodType.RODENT, 0)
    player.food_supply.nectar = food.get(FoodType.NECTAR, 0)


def _set_eggs_on_named_birds(player, eggs_by_bird: dict[str, int] | None) -> None:
    if not eggs_by_bird:
        return
    for bird_name, eggs in eggs_by_bird.items():
        found = False
        for row in player.board.all_rows():
            for slot in row.slots:
                if slot.bird and slot.bird.name == bird_name:
                    slot.eggs = max(0, min(eggs, slot.bird.egg_limit))
                    found = True
                    break
            if found:
                break


def _setup_video_state(seed: int) -> tuple[GameState, list[ScriptStep]]:
    bird_reg, bonus_reg, goal_reg = load_all(EXCEL_FILE)
    overrides = _video_bird_overrides(bird_reg)
    rnd = random.Random(seed)

    goals = [
        _goal(goal_reg, "[egg] in [ground]"),
        _goal(goal_reg, "[beak_pointing_left] beak pointing left"),
        _goal(goal_reg, "[egg] in [grassland]"),
        _goal(goal_reg, "[egg] in [forest]"),
    ]

    game = create_new_game(["You", "Opp"], round_goals=goals, board_type=BoardType.OCEANIA)
    you, opp = game.players

    # From transcript: 5 dealt options, keeps only Forster's Tern.
    you.hand = [_bird(bird_reg, "Forster's Tern", overrides)]
    you.bonus_cards = [bonus_reg.get("Forester"), bonus_reg.get("Forest Population Monitor")]
    you.food_supply.invertebrate = 1
    you.food_supply.fish = 1
    you.food_supply.rodent = 1
    you.food_supply.seed = 1
    you.food_supply.fruit = 0
    you.food_supply.nectar = 1

    # Opponent starts with a generic setup; include Sri Lanka Blue-Magpie as observed.
    pool = [b for b in bird_reg.all_birds if b.name not in {"Forster's Tern", "Red-Headed Woodpecker", "Brahminy Kite", "Squacco Heron"}]
    rnd.shuffle(pool)
    slbm = _bird(bird_reg, "Sri Lanka Blue-Magpie", overrides)
    opp.hand = [slbm] + [b for b in pool if b.name != slbm.name][:4]
    opp.bonus_cards = [rnd.choice(list(bonus_reg.all_cards))]
    opp.food_supply.invertebrate = 1
    opp.food_supply.seed = 1
    opp.food_supply.fish = 1
    opp.food_supply.fruit = 1
    opp.food_supply.rodent = 1
    opp.food_supply.nectar = 1

    game.card_tray.face_up = [
        _bird(bird_reg, "Red-Headed Woodpecker", overrides),
        _bird(bird_reg, "Brahminy Kite", overrides),
        _bird(bird_reg, "Squacco Heron", overrides),
    ]

    # Put key transcript cards near deck top to make line reproducible.
    scripted_draw_order = [
        "Scaled Quail",
        "Violet-Green Swallow",
        "Broad-Winged Hawk",
        "Willie-Wagtail",
        "Golden Pheasant",
        "Wood Duck",
        "Spotted Dove",
        "Lesser Whitethroat",
        "Blue-Winged Warbler",
        "Mandarin Duck",
        "Twite",
        "Roseate Spoonbill",
        "Great Hornbill",
        "White Stork",
        "Black-Tailed Godwit",
        "Baltimore Oriole",
        "Trumpeter Swan",
    ]
    scripted_cards = [_bird(bird_reg, n, overrides) for n in scripted_draw_order]
    used = {c.name for c in you.hand + opp.hand + game.card_tray.face_up + scripted_cards}
    deck_rest = [b for b in pool if b.name not in used]
    rnd.shuffle(deck_rest)
    game._deck_cards = deck_rest + scripted_cards  # pop() draws from end
    game.deck_remaining = len(game._deck_cards)
    game.birdfeeder.reroll()
    game._video_bird_overrides = overrides

    # Script distilled from transcript. This is intentionally explicit and strict.
    script = [
        ScriptStep("you", ActionType.PLAY_BIRD, bird_name="Forster's Tern", habitat=Habitat.WETLAND, note="opening play"),
        ScriptStep(
            "you",
            ActionType.DRAW_CARDS,
            discovered_birds=["Violet-Green Swallow", "Black-Headed Gull", "Broad-Winged Hawk"],
            discard_birds=["Broad-Winged Hawk"],
            power_choices_before=[
                {"bird_name": "Forster's Tern", "choice": {"discard_names": ["Broad-Winged Hawk"]}},
            ],
            note="turn2 draw cycle: keep swallow+gull, discard hawk",
        ),
        ScriptStep(
            "you",
            ActionType.DRAW_CARDS,
            discovered_birds=["Willie-Wagtail", "Golden Pheasant", "Wood Duck"],
            discard_birds=["Black-Headed Gull"],
            power_choices_before=[
                {"bird_name": "Forster's Tern", "choice": {"discard_names": ["Black-Headed Gull"]}},
            ],
            note="turn3 draw cycle: keep wagtail+pheasant+wood duck, discard gull",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Violet-Green Swallow",
            habitat=Habitat.FOREST,
            note="turn4 play swallow in forest, spend nectar + invertebrate",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.INVERTEBRATE, FoodType.INVERTEBRATE],
            discovered_birds=["Common Nighthawk"],
            discard_birds=["Willie-Wagtail"],
            note="turn5 take 2 invertebrates; tuck willie-wagtail; draw common nighthawk",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.NECTAR, FoodType.SEED],
            discovered_birds=["Bronzed Cowbird"],
            discard_birds=["Common Nighthawk"],
            note="turn6 take nectar + seed; tuck common nighthawk; draw bronzed cowbird",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Golden Pheasant",
            habitat=Habitat.GRASSLAND,
            set_eggs_on_birds_after={"Golden Pheasant": 4},
            note="turn7 play pheasant in grasslands, spend 2 invertebrates + 1 seed, lay 4 eggs",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Wood Duck",
            habitat=Habitat.FOREST,
            set_food_before={
                FoodType.INVERTEBRATE: 0,
                FoodType.SEED: 2,
                FoodType.FISH: 0,
                FoodType.FRUIT: 0,
                FoodType.RODENT: 0,
                FoodType.NECTAR: 1,
            },
            set_eggs_on_birds_before={"Golden Pheasant": 4},
            grant_food_before=[FoodType.SEED, FoodType.SEED],
            grant_food_after=[FoodType.FRUIT],
            note="turn8 play wood duck in forest, spend nectar + 2 seed and 1 egg from pheasant",
        ),
        # Round 2
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.NECTAR, FoodType.SEED],
            discovered_birds=["Pink-Eared Duck", "Great Hornbill", "Australian Zebra Finch"],
            discard_birds=["Pink-Eared Duck", "Australian Zebra Finch"],
            note="r2t1 take nectar+seed; tuck pink-eared duck; draw/discard zebra finch",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.FISH, FoodType.INVERTEBRATE],
            discovered_birds=["Red-Bellied Woodpecker", "Australian Ibis", "Black Drongo"],
            discard_birds=["Australian Ibis", "Black Drongo"],
            note="r2t2 take fish+invertebrate; tuck ibis; draw/discard drongo",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.INVERTEBRATE, FoodType.NECTAR],
            discovered_birds=["Lesser Whitethroat", "Blue-Winged Warbler", "Mandarin Duck"],
            discard_birds=["Red-Bellied Woodpecker", "Bronzed Cowbird"],
            note="r2t3 take invertebrate then reroll->nectar; tuck red-bellied woodpecker; draw mandarin duck; discard bronzed cowbird",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.INVERTEBRATE, FoodType.INVERTEBRATE],
            discovered_birds=["Montagu's Harrier", "Greater Flamingo", "Twite"],
            discard_birds=["Montagu's Harrier", "Greater Flamingo"],
            note="r2t4 take 2 invertebrates; tuck harrier; draw twite from tray; discard flamingo",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Twite",
            habitat=Habitat.GRASSLAND,
            discovered_birds=["Twite"],
            note="r2t5",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Lesser Whitethroat",
            habitat=Habitat.GRASSLAND,
            discovered_birds=["Lesser Whitethroat"],
            set_food_before={
                FoodType.INVERTEBRATE: 1,
                FoodType.SEED: 0,
                FoodType.FISH: 2,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 0,
                FoodType.NECTAR: 1,
            },
            note="r2t6",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Mandarin Duck",
            habitat=Habitat.WETLAND,
            discovered_birds=["Mandarin Duck"],
            set_food_before={
                FoodType.INVERTEBRATE: 3,
                FoodType.SEED: 0,
                FoodType.FISH: 1,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 0,
                FoodType.NECTAR: 0,
            },
            note="r2t7",
        ),
        # Round 3
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            set_food_before={
                FoodType.INVERTEBRATE: 0,
                FoodType.SEED: 0,
                FoodType.FISH: 2,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 0,
            },
            set_tray_before=["White-Faced Heron", "Lincoln's Sparrow", "Magpie-Lark"],
            prefer_food=[FoodType.INVERTEBRATE, FoodType.NECTAR],
            discovered_birds=["Long-Tailed Tit", "Zebra Dove", "Magpie-Lark"],
            discard_birds=["Long-Tailed Tit", "Zebra Dove"],
            note="r3t1 take invertebrate+nectar; wood duck draws tit+dove; tuck tit; swallow draws Magpie-Lark from tray; discard dove",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.SEED, FoodType.INVERTEBRATE],
            discovered_birds=["Australian Magpie", "Barn Owl", "Turkey Vulture"],
            discard_birds=["Barn Owl", "Turkey Vulture"],
            note="r3t2 take seed+invertebrate; wood duck draws magpie+barn owl; tuck barn owl; swallow draws turkey vulture; discard turkey vulture",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Great Hornbill",
            habitat=Habitat.FOREST,
            discovered_birds=["Great Hornbill"],
            set_food_before={
                FoodType.INVERTEBRATE: 1,
                FoodType.SEED: 1,
                FoodType.FISH: 2,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 1,
            },
            set_eggs_on_birds_before={"Lesser Whitethroat": 1},
            note="r3t3 play Great Hornbill in forest using nectar+berry+invertebrate and 1 egg from Lesser Whitethroat",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.INVERTEBRATE, FoodType.FRUIT],
            discovered_birds=["European Goldfinch", "Crested Lark", "White Stork"],
            discard_birds=["Crested Lark", "Australian Magpie"],
            power_choices_before=[
                {"bird_name": "Great Hornbill", "choice": {"tuck_name": "Australian Magpie", "cache": True}},
            ],
            note="r3t4 take invertebrate then reset-free berry; hornbill caches berry (skip tuck); wood duck draws goldfinch+lark; tuck lark; swallow draws White Stork; discard Australian Magpie",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Magpie-Lark",
            habitat=Habitat.GRASSLAND,
            discovered_birds=["Magpie-Lark"],
            set_food_before={
                FoodType.INVERTEBRATE: 1,
                FoodType.SEED: 0,
                FoodType.FISH: 2,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 0,
            },
            set_eggs_on_birds_before={"Golden Pheasant": 1, "Twite": 1},
            note="r3t5 play Magpie-Lark in grasslands; spend 1 invertebrate and final two eggs (pheasant+twite)",
        ),
        ScriptStep(
            "you",
            ActionType.LAY_EGGS,
            grant_food_before=[FoodType.FRUIT],
            note="r3t6 gain berry from opponent nightingale before action; lay eggs with fish-discard bonus",
        ),
        # Round 4
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Roseate Spoonbill",
            habitat=Habitat.WETLAND,
            discovered_birds=["Roseate Spoonbill"],
            set_food_before={
                FoodType.INVERTEBRATE: 1,
                FoodType.SEED: 1,
                FoodType.FISH: 0,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 0,
            },
            grant_food_before=[FoodType.FISH],
            set_eggs_on_birds_before={"Wood Duck": 2},
            note="r4t1 gain fish from opponent nightingale; play Roseate Spoonbill with fish+invertebrate+seed and 1 egg from Wood Duck",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.NECTAR, FoodType.NECTAR],
            discovered_birds=["Trumpeter Swan", "Silvereye", "Willow Tit"],
            discard_birds=["European Goldfinch", "Silvereye", "Willow Tit"],
            power_choices_before=[
                {"bird_name": "Great Hornbill", "choice": {"tuck_name": "European Goldfinch", "cache": False}},
            ],
            note="r4t2 reset feeder free and gain 2 nectar; hornbill tucks goldfinch; wood duck draws swan+silvereye; swallow tucks silvereye then draws/discards willow tit",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Baltimore Oriole",
            habitat=Habitat.FOREST,
            discovered_birds=["Baltimore Oriole"],
            set_food_before={
                FoodType.INVERTEBRATE: 0,
                FoodType.SEED: 0,
                FoodType.FISH: 0,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 2,
            },
            set_eggs_on_birds_before={"Great Hornbill": 1, "Wood Duck": 1},
            grant_food_before=[FoodType.FRUIT],
            note="r4t3 gain berry from opponent nightingale; play Baltimore Oriole with 2 nectar + berry and eggs from Hornbill+Wood Duck",
        ),
        ScriptStep(
            "you",
            ActionType.GAIN_FOOD,
            prefer_food=[FoodType.FISH, FoodType.NECTAR, FoodType.SEED],
            discovered_birds=["Black-Tailed Godwit", "Snowy Owl", "California Condor", "Common Green Magpie"],
            discard_birds=["Black-Tailed Godwit", "California Condor", "Common Green Magpie"],
            grant_food_after=[FoodType.FRUIT],
            power_choices_before=[
                {"bird_name": "Great Hornbill", "choice": {"tuck_name": "Black-Tailed Godwit", "cache": True}},
            ],
            note="r4t4 take fish then reset-free nectar+seed; oriole gives all players berry; hornbill tucks godwit and caches berry; wood duck draws snowy owl+condor; swallow tucks condor and draws/discards green magpie",
        ),
        ScriptStep(
            "you",
            ActionType.PLAY_BIRD,
            bird_name="Trumpeter Swan",
            habitat=Habitat.WETLAND,
            discovered_birds=["Trumpeter Swan"],
            set_food_before={
                FoodType.INVERTEBRATE: 0,
                FoodType.SEED: 1,
                FoodType.FISH: 1,
                FoodType.FRUIT: 1,
                FoodType.RODENT: 1,
                FoodType.NECTAR: 1,
            },
            set_eggs_on_birds_before={"Violet-Green Swallow": 1, "Magpie-Lark": 1},
            note="r4t5 play Trumpeter Swan with nectar+seed+fish and eggs from Swallow+Magpie-Lark",
        ),
    ]

    return game, script


def run_scripted_replay(seed: int = 1, max_turns: int = 260) -> dict:
    random.seed(seed)
    bird_reg = load_all(EXCEL_FILE)[0]
    game, script = _setup_video_state(seed)
    overrides = getattr(game, "_video_bird_overrides", {})
    divergences: list[Divergence] = []
    turn = 0
    script_idx = 0

    while not game.is_game_over and turn < max_turns:
        p = game.current_player

        if p.action_cubes_remaining <= 0:
            if all(x.action_cubes_remaining <= 0 for x in game.players):
                game.advance_round()
            else:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            turn += 1
            continue

        actor = "you" if p.name == "You" else "opp"
        step = script[script_idx] if script_idx < len(script) and script[script_idx].actor == actor else None
        if step is not None:
            _set_food(p, step.set_food_before)
            _set_eggs_on_named_birds(p, step.set_eggs_on_birds_before)
            if step.set_tray_before:
                game.card_tray.face_up = [_bird(bird_reg, n, overrides) for n in step.set_tray_before]
            if step.power_choices_before:
                for item in step.power_choices_before:
                    bird_name = item.get("bird_name")
                    choice = item.get("choice", {})
                    if bird_name:
                        queue_power_choice(game, p.name, bird_name, choice)
            _add_food(p, step.grant_food_before)
            if step.discovered_birds:
                # Transcript anchoring: inject cards explicitly revealed in the video.
                for bn in step.discovered_birds:
                    card = _bird(bird_reg, bn, overrides)
                    if not any(b.name == bn for b in p.hand):
                        p.hand.append(card)

        moves = generate_all_moves(game, p)
        if not moves:
            p.action_cubes_remaining = 0
            game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            turn += 1
            continue

        if step is not None:
            chosen = _find_matching_move(moves, step, game, p.name)
            if chosen is None:
                divergences.append(
                    Divergence(
                        turn=turn,
                        actor=actor,
                        reason="scripted_move_not_legal",
                        note=(
                            (step.note + " | " if step.note else "")
                            + f"wanted={step.action_type.value}:{step.bird_name}:{step.habitat} "
                            + f"hand={[b.name for b in p.hand]} "
                            + "food={"
                            + f"inv={p.food_supply.invertebrate},seed={p.food_supply.seed},fish={p.food_supply.fish},"
                            + f"fruit={p.food_supply.fruit},rod={p.food_supply.rodent},nectar={p.food_supply.nectar}"
                            + "} "
                            + f"legal_plays={[m.bird_name for m in moves if m.action_type == ActionType.PLAY_BIRD][:8]}"
                        ),
                    )
                )
                chosen = _fallback_move(game, moves, p.name)
            script_idx += 1
        else:
            # For opponent we keep stochastic behavior; if out of scripted player steps,
            # continue with a stable fallback.
            if p.name == "Opp":
                chosen = pick_weighted_random_move(moves, game, p)
            else:
                chosen = _fallback_move(game, moves, p.name)

        ok = execute_move_on_sim(game, p, chosen)
        if ok:
            if step is not None and step.discard_birds:
                for bn in step.discard_birds:
                    p.remove_from_hand(bn)
            if step is not None:
                _add_food(p, step.grant_food_after)
                _set_eggs_on_named_birds(p, step.set_eggs_on_birds_after)
            game.advance_turn()
            _refill_tray(game)
        else:
            divergences.append(
                Divergence(
                    turn=turn,
                    actor=actor,
                    reason="chosen_move_failed_execution",
                    note=chosen.description,
                )
            )
            fallback = _fallback_move(game, moves, p.name)
            if execute_move_on_sim(game, p, fallback):
                game.advance_turn()
                _refill_tray(game)
            else:
                game.advance_turn()
                _refill_tray(game)
        turn += 1

    you = game.get_player("You")
    opp = game.get_player("Opp")
    assert you is not None and opp is not None
    you_score = calculate_score(game, you)
    opp_score = calculate_score(game, opp)
    slbm_slot = opp.board.find_bird("Sri Lanka Blue-Magpie")
    slbm_tucks = slbm_slot[2].tucked_cards if slbm_slot else 0

    return {
        "seed": seed,
        "turns": turn,
        "script_steps_total": len(script),
        "script_steps_consumed": script_idx,
        "divergence_count": len(divergences),
        "divergences": [asdict(d) for d in divergences],
        "you_score": you_score.as_dict(),
        "opp_score": opp_score.as_dict(),
        "you_birds": [b.name for b in you.board.all_birds()],
        "opp_birds": [b.name for b in opp.board.all_birds()],
        "opp_sri_lanka_blue_magpie_tucked_cards": slbm_tucks,
        "video_bird_overrides": {
            k: {
                "food_cost": [ft.value for ft in v.food_cost.items],
                "food_total": v.food_cost.total,
                "is_or": v.food_cost.is_or,
            }
            for k, v in overrides.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay scripted Wingspan video game line")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=260)
    parser.add_argument("--out", default="reports/ml/video_replay/scripted_replay_result.json")
    args = parser.parse_args()

    result = run_scripted_replay(seed=args.seed, max_turns=args.max_turns)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
