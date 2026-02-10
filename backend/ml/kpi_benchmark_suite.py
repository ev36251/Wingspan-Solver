"""Standardized KPI benchmark suite for round-1 and full-game diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.models.enums import BoardType
from backend.solver.benchmark_thresholds import run_benchmark
from backend.solver.round1_benchmark import run_round1_benchmark


def run_kpi_benchmark_suite(
    out_dir: str,
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    round1_games: int = 50,
    full_games: int = 10,
    max_turns: int = 220,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    seed: int | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    round1 = run_round1_benchmark(
        games=round1_games,
        players=players,
        board_type=board_type,
        strict_rules_only=strict_rules_only,
        reject_non_strict_powers=reject_non_strict_powers,
        seed=seed,
    )
    round1_path = out / "round1_benchmark.json"
    round1_path.write_text(json.dumps(round1, indent=2), encoding="utf-8")

    full = run_benchmark(
        games_per_count=full_games,
        max_turns=max_turns,
        board_type=board_type,
        player_counts=(players,),
    )
    full_path = out / "full_game_benchmark.json"
    full_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    full_key = int(players)
    full_summary = full.get(full_key, {})
    summary = {
        "board_type": board_type.value,
        "players": players,
        "round1_games": round1_games,
        "full_games": full_games,
        "max_turns": max_turns,
        "strict_rules_only": strict_rules_only,
        "reject_non_strict_powers": reject_non_strict_powers,
        "round1": {
            "samples": round1.get("samples", 0),
            "strict_rejected_games": round1.get("strict_rejected_games", 0),
            "mean_total": round1.get("totals", {}).get("mean", 0.0),
            "pct_15_30": round1.get("totals", {}).get("pct_15_30", 0.0),
        },
        "full_game": {
            "sample_size_player_scores": full_summary.get("sample_size_player_scores", 0),
            "mean_player_score": full_summary.get("mean_player_score", 0.0),
            "mean_winner_score": full_summary.get("mean_winner_score", 0.0),
            "rate_ge_100": full_summary.get("rate_ge_100", 0.0),
            "rate_ge_120": full_summary.get("rate_ge_120", 0.0),
            "max_winner_score": full_summary.get("max_winner_score", 0),
            "failed_games": full_summary.get("failed_games", 0),
        },
        "artifacts": {
            "round1_benchmark": str(round1_path),
            "full_game_benchmark": str(full_path),
        },
    }

    summary_path = out / "kpi_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized KPI benchmark suite")
    parser.add_argument("--out-dir", default="reports/ml/benchmarks")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--round1-games", type=int, default=50)
    parser.add_argument("--full-games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--strict-rules-only", action="store_true")
    parser.add_argument("--reject-non-strict-powers", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    summary = run_kpi_benchmark_suite(
        out_dir=args.out_dir,
        players=args.players,
        board_type=BoardType(args.board_type),
        round1_games=args.round1_games,
        full_games=args.full_games,
        max_turns=args.max_turns,
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
