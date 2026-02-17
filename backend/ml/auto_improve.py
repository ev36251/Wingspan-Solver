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
    train_hidden1: int = 384,
    train_hidden2: int = 192,
    train_dropout: float = 0.2,
    train_lr_init: float = 1e-4,
    train_lr_peak: float = 5e-4,
    train_lr_warmup_epochs: int = 3,
    train_lr_decay_every: int = 5,
    train_lr_decay_factor: float = 0.7,
    train_early_stop_enabled: bool = True,
    train_early_stop_patience: int = 5,
    train_early_stop_min_delta: float = 1e-4,
    train_early_stop_restore_best: bool = True,
    value_weight: float = 0.5,
    val_split: float = 0.1,
    eval_games: int = 40,
    champion_self_play_enabled: bool = True,
    champion_switch_after_first_promotion: bool = True,
    champion_switch_min_stable_iters: int = 2,
    champion_switch_min_eval_win_rate: float = 0.5,
    champion_switch_min_iteration: int = 3,
    champion_teacher_source: str = "engine_only",
    champion_engine_time_budget_ms: int = 50,
    champion_engine_num_determinizations: int = 8,
    champion_engine_max_rollout_depth: int = 24,
    promotion_primary_opponent: str = "champion",
    best_selection_eval_games: int = 0,
    min_gate_win_rate: float = 0.50,
    min_gate_mean_score: float = 52.0,
    require_non_negative_champion_margin: bool = True,
    strict_curriculum_enabled: bool = True,
    strict_fraction_start: float = 1.0,
    strict_fraction_end: float = 0.0,
    strict_fraction_warmup_iters: int = 5,
    data_accumulation_enabled: bool = True,
    data_accumulation_decay: float = 0.5,
    max_accumulated_samples: int = 200_000,
    dataset_workers: int = 4,
    seed: int = 0,
    clean_out_dir: bool = True,
    proposal_top_k: int = 6,
    lookahead_depth: int = 2,
    strict_kpi_gate_enabled: bool = True,
    train_hidden: int | None = None,
    train_lr: float | None = None,
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
        train_hidden1=train_hidden1,
        train_hidden2=train_hidden2,
        train_dropout=train_dropout,
        train_lr_init=train_lr_init,
        train_lr_peak=train_lr_peak,
        train_lr_warmup_epochs=train_lr_warmup_epochs,
        train_lr_decay_every=train_lr_decay_every,
        train_lr_decay_factor=train_lr_decay_factor,
        train_early_stop_enabled=train_early_stop_enabled,
        train_early_stop_patience=train_early_stop_patience,
        train_early_stop_min_delta=train_early_stop_min_delta,
        train_early_stop_restore_best=train_early_stop_restore_best,
        train_value_weight=value_weight,
        val_split=val_split,
        eval_games=eval_games,
        promotion_games=max(40, eval_games * 2),
        pool_games_per_opponent=max(20, eval_games),
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=min_gate_win_rate,
        min_gate_mean_score=min_gate_mean_score,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        require_non_negative_champion_margin=require_non_negative_champion_margin,
        data_accumulation_enabled=data_accumulation_enabled,
        data_accumulation_decay=data_accumulation_decay,
        max_accumulated_samples=max_accumulated_samples,
        dataset_workers=dataset_workers,
        champion_self_play_enabled=champion_self_play_enabled,
        champion_switch_after_first_promotion=champion_switch_after_first_promotion,
        champion_switch_min_stable_iters=champion_switch_min_stable_iters,
        champion_switch_min_eval_win_rate=champion_switch_min_eval_win_rate,
        champion_switch_min_iteration=champion_switch_min_iteration,
        champion_teacher_source=champion_teacher_source,
        champion_engine_time_budget_ms=champion_engine_time_budget_ms,
        champion_engine_num_determinizations=champion_engine_num_determinizations,
        champion_engine_max_rollout_depth=champion_engine_max_rollout_depth,
        promotion_primary_opponent=promotion_primary_opponent,
        best_selection_eval_games=best_selection_eval_games,
        engine_teacher_prob=0.15,
        engine_time_budget_ms=25,
        engine_num_determinizations=0,
        engine_max_rollout_depth=24,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        strict_curriculum_enabled=strict_curriculum_enabled,
        strict_fraction_start=strict_fraction_start,
        strict_fraction_end=strict_fraction_end,
        strict_fraction_warmup_iters=strict_fraction_warmup_iters,
        strict_kpi_gate_enabled=strict_kpi_gate_enabled,
        seed=seed,
        clean_out_dir=clean_out_dir,
        train_hidden=train_hidden,
        train_lr=train_lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-improve factorized strict training")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve_factorized")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=400)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden1", type=int, default=384)
    parser.add_argument("--train-hidden2", type=int, default=192)
    parser.add_argument("--train-dropout", type=float, default=0.2)
    parser.add_argument("--train-lr-init", type=float, default=1e-4)
    parser.add_argument("--train-lr-peak", type=float, default=5e-4)
    parser.add_argument("--train-lr-warmup-epochs", type=int, default=3)
    parser.add_argument("--train-lr-decay-every", type=int, default=5)
    parser.add_argument("--train-lr-decay-factor", type=float, default=0.7)
    parser.set_defaults(train_early_stop_enabled=True, train_early_stop_restore_best=True)
    parser.add_argument("--train-early-stop-enabled", dest="train_early_stop_enabled", action="store_true")
    parser.add_argument("--disable-train-early-stop", dest="train_early_stop_enabled", action="store_false")
    parser.add_argument("--train-early-stop-patience", type=int, default=5)
    parser.add_argument("--train-early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--train-early-stop-restore-best", dest="train_early_stop_restore_best", action="store_true")
    parser.add_argument("--no-train-early-stop-restore-best", dest="train_early_stop_restore_best", action="store_false")
    parser.add_argument("--train-hidden", type=int, default=None, help="Deprecated: maps hidden1=hidden2=train-hidden")
    parser.add_argument("--train-lr", type=float, default=None, help="Deprecated: maps train-lr-peak=train-lr")
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.set_defaults(champion_self_play_enabled=True, champion_switch_after_first_promotion=True)
    parser.add_argument("--champion-self-play-enabled", dest="champion_self_play_enabled", action="store_true")
    parser.add_argument("--disable-champion-self-play", dest="champion_self_play_enabled", action="store_false")
    parser.add_argument("--champion-switch-after-first-promotion", dest="champion_switch_after_first_promotion", action="store_true")
    parser.add_argument("--disable-champion-switch", dest="champion_switch_after_first_promotion", action="store_false")
    parser.add_argument("--champion-teacher-source", default="engine_only", choices=["probabilistic_engine", "engine_only"])
    parser.add_argument("--champion-engine-time-budget-ms", type=int, default=50)
    parser.add_argument("--champion-engine-num-determinizations", type=int, default=8)
    parser.add_argument("--champion-engine-max-rollout-depth", type=int, default=24)
    parser.add_argument("--champion-switch-min-stable-iters", type=int, default=2)
    parser.add_argument("--champion-switch-min-eval-win-rate", type=float, default=0.5)
    parser.add_argument("--champion-switch-min-iteration", type=int, default=3)
    parser.add_argument("--promotion-primary-opponent", default="champion", choices=["heuristic", "champion"])
    parser.add_argument(
        "--best-selection-eval-games",
        type=int,
        default=0,
        help="0 => auto (max(promotion_games, eval_games))",
    )
    parser.add_argument("--min-gate-win-rate", type=float, default=0.50)
    parser.add_argument("--min-gate-mean-score", type=float, default=52.0)
    parser.set_defaults(require_non_negative_champion_margin=True)
    parser.add_argument("--require-non-negative-champion-margin", dest="require_non_negative_champion_margin", action="store_true")
    parser.add_argument("--allow-negative-champion-margin", dest="require_non_negative_champion_margin", action="store_false")
    parser.set_defaults(strict_curriculum_enabled=True)
    parser.add_argument("--strict-curriculum-enabled", dest="strict_curriculum_enabled", action="store_true")
    parser.add_argument("--disable-strict-curriculum", dest="strict_curriculum_enabled", action="store_false")
    parser.add_argument("--strict-fraction-start", type=float, default=1.0)
    parser.add_argument("--strict-fraction-end", type=float, default=0.0)
    parser.add_argument("--strict-fraction-warmup-iters", type=int, default=5)
    parser.set_defaults(data_accumulation_enabled=True)
    parser.add_argument("--data-accumulation-enabled", dest="data_accumulation_enabled", action="store_true")
    parser.add_argument("--disable-data-accumulation", dest="data_accumulation_enabled", action="store_false")
    parser.add_argument("--data-accumulation-decay", type=float, default=0.5)
    parser.add_argument("--max-accumulated-samples", type=int, default=200000)
    parser.add_argument("--dataset-workers", type=int, default=4)
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
        train_hidden1=args.train_hidden1,
        train_hidden2=args.train_hidden2,
        train_dropout=args.train_dropout,
        train_lr_init=args.train_lr_init,
        train_lr_peak=args.train_lr_peak,
        train_lr_warmup_epochs=args.train_lr_warmup_epochs,
        train_lr_decay_every=args.train_lr_decay_every,
        train_lr_decay_factor=args.train_lr_decay_factor,
        train_early_stop_enabled=args.train_early_stop_enabled,
        train_early_stop_patience=args.train_early_stop_patience,
        train_early_stop_min_delta=args.train_early_stop_min_delta,
        train_early_stop_restore_best=args.train_early_stop_restore_best,
        value_weight=args.value_weight,
        val_split=args.val_split,
        eval_games=args.eval_games,
        champion_self_play_enabled=args.champion_self_play_enabled,
        champion_switch_after_first_promotion=args.champion_switch_after_first_promotion,
        champion_switch_min_stable_iters=args.champion_switch_min_stable_iters,
        champion_switch_min_eval_win_rate=args.champion_switch_min_eval_win_rate,
        champion_switch_min_iteration=args.champion_switch_min_iteration,
        champion_teacher_source=args.champion_teacher_source,
        champion_engine_time_budget_ms=args.champion_engine_time_budget_ms,
        champion_engine_num_determinizations=args.champion_engine_num_determinizations,
        champion_engine_max_rollout_depth=args.champion_engine_max_rollout_depth,
        promotion_primary_opponent=args.promotion_primary_opponent,
        best_selection_eval_games=args.best_selection_eval_games,
        min_gate_win_rate=args.min_gate_win_rate,
        min_gate_mean_score=args.min_gate_mean_score,
        require_non_negative_champion_margin=args.require_non_negative_champion_margin,
        strict_curriculum_enabled=args.strict_curriculum_enabled,
        strict_fraction_start=args.strict_fraction_start,
        strict_fraction_end=args.strict_fraction_end,
        strict_fraction_warmup_iters=args.strict_fraction_warmup_iters,
        data_accumulation_enabled=args.data_accumulation_enabled,
        data_accumulation_decay=args.data_accumulation_decay,
        max_accumulated_samples=args.max_accumulated_samples,
        dataset_workers=args.dataset_workers,
        seed=args.seed,
        clean_out_dir=not args.resume,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
        strict_kpi_gate_enabled=args.strict_kpi_gate_enabled,
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
    )

    best = manifest.get("best")
    print(f"auto-improve complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
