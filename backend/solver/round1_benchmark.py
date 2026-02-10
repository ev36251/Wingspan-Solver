"""Round-1 benchmark harness for strict opening diagnostics.

Runs games through round 1 only (8 actions/player in standard rules) and
reports score distribution + component breakdown stats.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType, BoardType
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import execute_move_on_sim, pick_weighted_random_move, _refill_tray


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round((len(s) - 1) * pct))
    return float(s[max(0, min(len(s) - 1, i))])


def run_round1_benchmark(
    games: int = 200,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)

    load_all(EXCEL_FILE)
    totals: list[int] = []
    in_band = 0
    rejected = 0
    rows: list[dict] = []

    for g in range(games):
        game = create_training_game(players, board_type, strict_rules_mode=strict_rules_only)
        turns = 0
        strict_violation = False

        while not game.is_game_over and game.current_round == 1 and turns < 400:
            if reject_non_strict_powers:
                from backend.powers.registry import get_power_source, is_strict_power_source_allowed

                for p in game.players:
                    for b in p.board.all_birds():
                        src = get_power_source(b)
                        if not is_strict_power_source_allowed(src):
                            strict_violation = True
                            break
                    if strict_violation:
                        break
                if strict_violation:
                    break

            p = game.current_player
            if p.action_cubes_remaining <= 0:
                if all(x.action_cubes_remaining <= 0 for x in game.players):
                    game.advance_round()
                    continue
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            moves = generate_all_moves(game, p)
            if not moves:
                p.action_cubes_remaining = 0
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            move = pick_weighted_random_move(moves, game, p)
            try:
                ok = execute_move_on_sim(game, p, move)
            except RuntimeError as e:
                if "Strict rules mode rejected" in str(e):
                    strict_violation = True
                    break
                raise
            if ok:
                game.advance_turn()
                _refill_tray(game)
            else:
                # deterministic fallback on common actions
                fallback = next((m for m in moves if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS)), moves[0])
                if execute_move_on_sim(game, p, fallback):
                    game.advance_turn()
                    _refill_tray(game)
                else:
                    game.advance_turn()
                    _refill_tray(game)
            turns += 1

        if strict_violation:
            rejected += 1
            continue

        for pi, p in enumerate(game.players):
            sb = calculate_score(game, p).as_dict()
            total = int(sb["total"])
            totals.append(total)
            if 15 <= total <= 30:
                in_band += 1
            rows.append(
                {
                    "game_index": g + 1,
                    "player_index": pi,
                    "round": game.current_round,
                    "score": sb,
                }
            )

    n = len(totals)
    mean = (sum(totals) / n) if n else 0.0
    report = {
        "games_requested": games,
        "players": players,
        "board_type": board_type.value,
        "strict_rules_only": strict_rules_only,
        "reject_non_strict_powers": reject_non_strict_powers,
        "strict_rejected_games": rejected,
        "samples": n,
        "totals": {
            "mean": round(mean, 3),
            "median": round(_percentile([float(x) for x in totals], 0.5), 3),
            "p10": round(_percentile([float(x) for x in totals], 0.1), 3),
            "p90": round(_percentile([float(x) for x in totals], 0.9), 3),
            "min": min(totals) if totals else 0,
            "max": max(totals) if totals else 0,
            "pct_15_30": round((in_band / n) if n else 0.0, 4),
        },
        "rows": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-1 benchmark harness")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--strict-rules-only", action="store_true")
    parser.add_argument("--reject-non-strict-powers", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="reports/round1_benchmark.json")
    args = parser.parse_args()

    rep = run_round1_benchmark(
        games=args.games,
        players=args.players,
        board_type=BoardType(args.board_type),
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
