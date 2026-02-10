"""Legacy auto-improve entrypoint bridged to factorized training.

This module keeps the historical `run_auto_improve` API stable while routing
all training to the factorized BC improvement loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.auto_improve_factorized import run_auto_improve_factorized


def _best_history_row(history: list[dict]) -> dict | None:
    if not history:
        return None
    return max(history, key=lambda r: float((r.get("eval_summary") or {}).get("nn_mean_margin", -1e30)))


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
    teacher_policy: str = "lookahead",
    proposal_top_k: int = 6,
    lookahead_depth: int = 2,
) -> dict:
    """Compatibility wrapper.

    Legacy knobs that do not apply to factorized training (`teacher_policy`) are
    accepted but ignored.
    """
    del teacher_policy

    fac_manifest = run_auto_improve_factorized(
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
        seed=seed,
        clean_out_dir=clean_out_dir,
    )

    out_base = Path(out_dir)
    best_row = fac_manifest.get("best") or _best_history_row(fac_manifest.get("history", []))
    if best_row is None:
        raise ValueError("Factorized auto-improve produced no history rows")

    src_model = Path(best_row["model"])
    src_meta = Path(best_row["dataset_meta"])
    best_model = out_base / "best_model.npz"
    best_meta = out_base / "best_dataset.meta.json"
    if src_model.exists():
        shutil.copy2(src_model, best_model)
    if src_meta.exists():
        shutil.copy2(src_meta, best_meta)

    compat_best = {
        "iteration": best_row.get("iteration"),
        "primary_score": round(float((best_row.get("eval_summary") or {}).get("nn_mean_margin", 0.0)), 6),
        "eval_summary": best_row.get("eval_summary"),
        "model": str(best_model),
        "dataset_meta": str(best_meta),
        "source": "factorized_promoted" if fac_manifest.get("best") else "factorized_best_eval",
    }
    compat_manifest = {
        "version": 2,
        "mode": "factorized_bridge",
        "legacy_entrypoint": "backend.ml.auto_improve",
        "factorized_manifest_path": str(out_base / "auto_improve_factorized_manifest.json"),
        "config": fac_manifest.get("config"),
        "best": compat_best,
        "history": fac_manifest.get("history", []),
    }

    compat_path = out_base / "auto_improve_manifest.json"
    compat_path.write_text(json.dumps(compat_manifest, indent=2), encoding="utf-8")
    return compat_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy auto-improve CLI (now bridged to factorized training)"
    )
    parser.add_argument("--out-dir", default="reports/ml/auto_improve")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=120)
    parser.add_argument("--teacher-policy", default="lookahead")
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden", type=int, default=192)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        "NOTICE: backend.ml.auto_improve now routes to factorized training. "
        "Use backend.ml.auto_improve_factorized directly for full control."
    )

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
        teacher_policy=args.teacher_policy,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
    )

    best = manifest.get("best") or {}
    print(
        "auto-improve (factorized bridge) complete | "
        f"best_iter={best.get('iteration')} | "
        f"best_margin={(best.get('eval_summary') or {}).get('nn_mean_margin')}"
    )


if __name__ == "__main__":
    main()
