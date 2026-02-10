"""Rules conformance certification for Wingspan engine correctness.

This module provides a deterministic test harness that runs random legal play
and validates strict invariants after each move.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import get_bird_registry, load_all
from backend.engine.scoring import ScoreBreakdown, calculate_score
from backend.models.enums import BoardType, FoodType
from backend.models.game_state import GameState
from backend.powers.registry import clear_cache, get_registry_stats
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import _refill_tray, execute_move_on_sim


@dataclass
class ConformanceIssue:
    game_index: int
    step: int
    player: str
    rule_id: str
    detail: str


def _validate_score_breakdown(game: GameState) -> list[str]:
    errs: list[str] = []
    for p in game.players:
        sc: ScoreBreakdown = calculate_score(game, p)
        if sc.total != (sc.bird_vp + sc.eggs + sc.cached_food + sc.tucked_cards + sc.bonus_cards + sc.round_goals + sc.nectar):
            errs.append(f"score_total_mismatch:{p.name}")
    return errs


def _validate_state_invariants(game: GameState) -> list[str]:
    errs: list[str] = []

    if game.deck_remaining < 0:
        errs.append("deck_negative")
    if len(game.card_tray.face_up) > 3:
        errs.append("tray_overflow")
    if game.current_round < 1:
        errs.append("round_below_1")
    if game.current_round > 5:
        errs.append("round_above_terminal")

    for p in game.players:
        if p.action_cubes_remaining < 0:
            errs.append(f"negative_action_cubes:{p.name}")

        fs = p.food_supply
        for ft, val in (
            (FoodType.INVERTEBRATE, fs.invertebrate),
            (FoodType.SEED, fs.seed),
            (FoodType.FISH, fs.fish),
            (FoodType.FRUIT, fs.fruit),
            (FoodType.RODENT, fs.rodent),
            (FoodType.NECTAR, fs.nectar),
        ):
            if val < 0:
                errs.append(f"negative_food:{p.name}:{ft.value}")

        for row in p.board.all_rows():
            if row.bird_count > 5:
                errs.append(f"too_many_birds_in_row:{p.name}:{row.habitat.value}")
            if row.nectar_spent < 0:
                errs.append(f"negative_nectar_spent:{p.name}:{row.habitat.value}")

            for idx, slot in enumerate(row.slots):
                if slot.eggs < 0:
                    errs.append(f"negative_eggs:{p.name}:{row.habitat.value}:{idx}")
                if slot.tucked_cards < 0:
                    errs.append(f"negative_tucked:{p.name}:{row.habitat.value}:{idx}")
                if any(v < 0 for v in slot.cached_food.values()):
                    errs.append(f"negative_cached_food:{p.name}:{row.habitat.value}:{idx}")
                if slot.bird is None and (slot.eggs > 0 or slot.tucked_cards > 0 or slot.total_cached_food > 0):
                    errs.append(f"tokens_on_empty_slot:{p.name}:{row.habitat.value}:{idx}")
                if slot.bird is not None and slot.eggs > slot.bird.egg_limit:
                    errs.append(f"eggs_over_limit:{p.name}:{row.habitat.value}:{idx}:{slot.bird.name}")

    errs.extend(_validate_score_breakdown(game))
    return errs


def _run_random_game(game_idx: int, num_players: int, board_type: BoardType, max_steps: int, seed: int) -> tuple[list[ConformanceIssue], dict]:
    rnd = random.Random(seed)
    game = create_training_game(num_players=num_players, board_type=board_type)

    issues: list[ConformanceIssue] = []
    steps = 0

    # Initial sanity
    for e in _validate_state_invariants(game):
        issues.append(ConformanceIssue(game_idx, steps, game.current_player.name, e.split(":")[0], e))

    while not game.is_game_over and steps < max_steps:
        p = game.current_player

        if p.action_cubes_remaining <= 0:
            if all(x.action_cubes_remaining <= 0 for x in game.players):
                game.advance_round()
            else:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            steps += 1
            continue

        moves = generate_all_moves(game, p)
        if not moves:
            p.action_cubes_remaining = 0
            game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            steps += 1
            continue

        move = rnd.choice(moves)
        success = execute_move_on_sim(game, p, move)

        if success:
            game.advance_turn()
            _refill_tray(game)
        else:
            issues.append(
                ConformanceIssue(
                    game_index=game_idx,
                    step=steps,
                    player=p.name,
                    rule_id="legal_move_failed",
                    detail=f"Move failed despite being generated as legal: {move.description}",
                )
            )
            p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1)
            game.advance_turn()
            _refill_tray(game)

        steps += 1

        for e in _validate_state_invariants(game):
            issues.append(ConformanceIssue(game_idx, steps, game.current_player.name, e.split(":")[0], e))

    final_scores = {pl.name: calculate_score(game, pl).total for pl in game.players}
    summary = {
        "steps": steps,
        "is_game_over": game.is_game_over,
        "final_round": game.current_round,
        "scores": final_scores,
        "winner_score": max(final_scores.values()) if final_scores else 0,
    }
    return issues, summary


def power_mapping_report() -> dict:
    """Return power-mapping coverage stats for all registered birds."""
    load_all(EXCEL_FILE)
    bird_reg = get_bird_registry()
    clear_cache()
    stats = get_registry_stats(bird_reg.all_birds)
    total = sum(stats.values())
    fallback = stats.get("FallbackPower", 0)
    coverage = (total - fallback) / total if total else 0.0
    return {
        "total_birds": total,
        "fallback_count": fallback,
        "coverage_non_fallback": round(coverage, 6),
        "stats_by_power": dict(sorted(stats.items(), key=lambda kv: kv[0])),
    }


def run_conformance_suite(
    games: int = 100,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    max_steps: int = 300,
    seed: int = 1,
) -> dict:
    load_all(EXCEL_FILE)

    started = time.time()
    all_issues: list[ConformanceIssue] = []
    winner_scores: list[int] = []

    for i in range(1, games + 1):
        issues, summary = _run_random_game(
            game_idx=i,
            num_players=players,
            board_type=board_type,
            max_steps=max_steps,
            seed=seed + i,
        )
        all_issues.extend(issues)
        winner_scores.append(summary["winner_score"])

    by_rule: dict[str, int] = {}
    for issue in all_issues:
        by_rule[issue.rule_id] = by_rule.get(issue.rule_id, 0) + 1

    result = {
        "meta": {
            "games": games,
            "players": players,
            "board_type": board_type.value,
            "max_steps": max_steps,
            "seed": seed,
            "elapsed_sec": round(time.time() - started, 2),
        },
        "power_mapping": power_mapping_report(),
        "issues_total": len(all_issues),
        "issues_by_rule": dict(sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0]))),
        "issues": [asdict(i) for i in all_issues[:5000]],
        "winner_score_mean": round(sum(winner_scores) / max(1, len(winner_scores)), 3),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wingspan rules conformance suite")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="reports/rules_conformance.json")
    args = parser.parse_args()

    out = run_conformance_suite(
        games=args.games,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_steps=args.max_steps,
        seed=args.seed,
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(
        f"conformance complete | games={args.games} | issues={out['issues_total']} | "
        f"power_coverage={out['power_mapping']['coverage_non_fallback']:.3f}"
    )


if __name__ == "__main__":
    main()
