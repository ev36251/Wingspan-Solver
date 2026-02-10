"""Iterative 2-player optimization loop for heuristic weights.

Workflow per cycle:
1) Run self-play evolution for 2P.
2) Benchmark with solver-in-the-loop simulation (2P only).
3) Stop when threshold targets are met or max cycles reached.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.solver.self_play import TrainingConfig, evolve
from backend.solver.benchmark_thresholds import run_benchmark


def optimize_2p(
    cycles: int,
    train_games: int,
    generations: int,
    pop_size: int,
    benchmark_games: int,
    max_turns: int,
    target_ge_100: float,
    target_ge_110: float,
    output_path: Path,
) -> dict:
    load_all(EXCEL_FILE)
    run_started = time.time()
    history: list[dict] = []
    best_cycle: dict | None = None

    for cycle in range(1, cycles + 1):
        train_out = Path(f"reports/opt2p_cycle_{cycle}_train.json")
        cfg = TrainingConfig(
            population_size=pop_size,
            generations=generations,
            games_per_matchup=train_games,
            num_players=2,
            board_type=BoardType.OCEANIA.value,
            output_path=str(train_out),
        )
        train_result = evolve(cfg)

        bench = run_benchmark(
            games_per_count=benchmark_games,
            max_turns=max_turns,
            player_counts=(2,),
        )
        bench_2p = bench[2]

        row = {
            "cycle": cycle,
            "train": {
                "best_avg_score": train_result.get("best_avg_score", 0),
                "best_win_rate": train_result.get("best_win_rate", 0),
                "best_fitness": train_result.get("best_fitness", 0),
                "best_weights": train_result.get("best_weights", {}),
            },
            "benchmark_2p": bench_2p,
            "meets_target": (
                bench_2p.get("rate_ge_100", 0) >= target_ge_100
                and bench_2p.get("rate_ge_110", 0) >= target_ge_110
            ),
        }
        history.append(row)

        if best_cycle is None:
            best_cycle = row
        else:
            b = best_cycle["benchmark_2p"]
            c = row["benchmark_2p"]
            if (
                c.get("mean_winner_score", 0) > b.get("mean_winner_score", 0)
                or (
                    c.get("mean_winner_score", 0) == b.get("mean_winner_score", 0)
                    and c.get("mean_player_score", 0) > b.get("mean_player_score", 0)
                )
            ):
                best_cycle = row

        if row["meets_target"]:
            break

    result = {
        "meta": {
            "mode": "optimize_2p",
            "cycles_requested": cycles,
            "cycles_completed": len(history),
            "train_games_per_matchup": train_games,
            "generations": generations,
            "population_size": pop_size,
            "benchmark_games": benchmark_games,
            "benchmark_max_turns": max_turns,
            "target_ge_100": target_ge_100,
            "target_ge_110": target_ge_110,
            "elapsed_sec": round(time.time() - run_started, 1),
        },
        "best_cycle": best_cycle,
        "history": history,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively optimize 2P heuristic strategy")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--train-games", type=int, default=200)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--benchmark-games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--target-ge-100", type=float, default=0.8)
    parser.add_argument("--target-ge-110", type=float, default=0.4)
    parser.add_argument("--out", default="reports/optimize_2p.json")
    args = parser.parse_args()

    result = optimize_2p(
        cycles=args.cycles,
        train_games=args.train_games,
        generations=args.generations,
        pop_size=args.pop_size,
        benchmark_games=args.benchmark_games,
        max_turns=args.max_turns,
        target_ge_100=args.target_ge_100,
        target_ge_110=args.target_ge_110,
        output_path=Path(args.out),
    )
    print(
        "Finished optimize_2p: cycles="
        f"{result['meta']['cycles_completed']}, elapsed={result['meta']['elapsed_sec']}s"
    )


if __name__ == "__main__":
    main()
