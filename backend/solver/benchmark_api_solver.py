"""Benchmark API top-recommendation loop using solver endpoint outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from backend.api.routes_game import _games
from backend.api.routes_solver import solve_heuristic
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.solver.move_generator import Move
from backend.solver.self_play import create_training_game
from backend.solver.simulation import _refill_tray, execute_move_on_sim


def _pct(sorted_vals: list[int], pct: int) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((pct / 100) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def run_benchmark(
    games_per_count: int = 3,
    max_turns: int = 220,
    board_type: BoardType = BoardType.OCEANIA,
    player_counts: tuple[int, ...] = (2, 3, 4),
) -> dict:
    load_all(EXCEL_FILE)
    out: dict = {}
    start_all = time.time()

    for num_players in player_counts:
        t0 = time.time()
        scores: list[int] = []
        winners: list[int] = []
        failed_games = 0

        for gi in range(games_per_count):
            game_failed = False
            game = create_training_game(num_players, board_type)
            gid = f"api_bench_{num_players}_{gi}_{int(time.time() * 1000) % 100000}"
            _games[gid] = game
            turns = 0
            while turns < max_turns and not game.is_game_over:
                try:
                    resp = asyncio.run(solve_heuristic(gid, None))
                    recs = getattr(resp, "recommendations", [])
                    if not recs:
                        game_failed = True
                        break
                    top = recs[0]
                    d = top.details or {}
                    move = Move(
                        action_type=ActionType(top.action_type),
                        description=top.description,
                        bird_name=d.get("bird_name"),
                        habitat=Habitat(d["habitat"]) if d.get("habitat") else None,
                        food_payment={
                            FoodType(k): int(v)
                            for k, v in (d.get("food_payment") or {}).items()
                        },
                        food_choices=[FoodType(f) for f in (d.get("food_choices") or [])],
                        egg_distribution={
                            (Habitat(h), int(si)): int(c)
                            for h, slots in (d.get("egg_distribution") or {}).items()
                            for si, c in slots.items()
                        },
                        tray_indices=list(d.get("tray_indices") or []),
                        deck_draws=int(d.get("deck_draws", 0) or 0),
                        bonus_count=int(d.get("bonus_count", 0) or 0),
                        reset_bonus=bool(d.get("reset_bonus", False)),
                    )
                    if not execute_move_on_sim(game, game.current_player, move):
                        game_failed = True
                        break
                    game.advance_turn()
                    _refill_tray(game)
                    turns += 1
                except Exception:
                    game_failed = True
                    break

            if game_failed:
                failed_games += 1

            game_scores = [int(calculate_score(game, p).total) for p in game.players]
            scores.extend(game_scores)
            winners.append(max(game_scores) if game_scores else 0)
            _games.pop(gid, None)

        s = sorted(scores)
        out[num_players] = {
            "games": games_per_count,
            "failed_games": failed_games,
            "sample_size_player_scores": len(scores),
            "mean_player_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "player_p50": _pct(s, 50),
            "player_p90": _pct(s, 90),
            "player_max": max(s) if s else 0,
            "rate_ge_100": round(sum(1 for x in scores if x >= 100) / len(scores), 4) if scores else 0,
            "rate_ge_110": round(sum(1 for x in scores if x >= 110) / len(scores), 4) if scores else 0,
            "rate_ge_120": round(sum(1 for x in scores if x >= 120) / len(scores), 4) if scores else 0,
            "rate_ge_135": round(sum(1 for x in scores if x >= 135) / len(scores), 4) if scores else 0,
            "rate_ge_150": round(sum(1 for x in scores if x >= 150) / len(scores), 4) if scores else 0,
            "mean_table_winner_score": round(sum(winners) / len(winners), 2) if winners else 0,
            "max_table_winner_score": max(winners) if winners else 0,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    out["meta"] = {
        "mode": "api_top_recommendation_loop",
        "board_type": board_type.value,
        "games_per_count": games_per_count,
        "max_turns": max_turns,
        "total_elapsed_sec": round(time.time() - start_all, 1),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark API top recommendation loop")
    parser.add_argument("--games", type=int, default=3, help="Games per player count (2,3,4)")
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--players", default="2,3,4")
    parser.add_argument("--out", default="reports/score_distribution_api_solver.json")
    args = parser.parse_args()

    player_counts = tuple(int(p.strip()) for p in args.players.split(",") if p.strip())
    for p in player_counts:
        if p not in (2, 3, 4):
            raise ValueError(f"Unsupported player count: {p}")

    result = run_benchmark(
        games_per_count=args.games,
        max_turns=args.max_turns,
        player_counts=player_counts,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
