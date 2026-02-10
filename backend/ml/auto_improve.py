"""Direct auto-improve entrypoint for the strict factorized pipeline only."""

from __future__ import annotations

import argparse

from backend.models.enums import BoardType
from backend.ml.auto_improve_factorized import run_auto_improve_factorized


def run_auto_improve(
    out_dir: str,
    iterations: int,
    players: int,
    board_type: BoardType,
    max_turns: int,
    games_per_iter: int,
    train_epochs: int,
    train_batch: int,
    train_hidden: int,
    train_lr: float,
    value_weight: float,
    val_split: float,
    eval_games: int,
    seed: int,
    clean_out_dir: bool = True,
    proposal_top_k: int = 6,
    lookahead_depth: int = 2,
    strict_kpi_gate_enabled: bool = True,
) -> dict:
    """Retained API name, but runs only the factorized strict pipeline."""
    return run_auto_improve_factorized(
        out_dir=out_dir,
        iterations=iterations,
        players=players,
        board_type=board_type,
        max_turns=max_turns,
        games_per_iter=games_per_iter,
        proposal_top_k=proposal_top_k,
        lookahead_depth=lookahead_depth,
        n_step=2,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=2,
        train_epochs=train_epochs,
        train_batch=train_batch,
        train_hidden=train_hidden,
        train_lr=train_lr,
        train_value_weight=value_weight,
        val_split=val_split,
        eval_games=eval_games,
        promotion_games=max(20, min(200, eval_games * 10)),
        pool_games_per_opponent=max(10, min(60, eval_games * 5)),
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=0.0,
        min_gate_mean_score=0.0,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        engine_teacher_prob=0.15,
        engine_time_budget_ms=25,
        engine_num_determinizations=0,
        engine_max_rollout_depth=24,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        strict_kpi_gate_enabled=strict_kpi_gate_enabled,
        seed=seed,
        clean_out_dir=clean_out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-improve factorized strict training")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve_factorized")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=120)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden", type=int, default=192)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.set_defaults(strict_kpi_gate_enabled=True)
    parser.add_argument("--strict-kpi-gate-enabled", dest="strict_kpi_gate_enabled", action="store_true")
    parser.add_argument("--disable-strict-kpi-gate", dest="strict_kpi_gate_enabled", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = run_auto_improve(
        out_dir=args.out_dir,
        iterations=args.iterations,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        games_per_iter=args.games_per_iter,
        train_epochs=args.train_epochs,
        train_batch=args.train_batch,
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
        value_weight=args.value_weight,
        val_split=args.val_split,
        eval_games=args.eval_games,
        seed=args.seed,
        clean_out_dir=not args.resume,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
        strict_kpi_gate_enabled=args.strict_kpi_gate_enabled,
    )

    best = manifest.get("best")
    print(f"auto-improve complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
