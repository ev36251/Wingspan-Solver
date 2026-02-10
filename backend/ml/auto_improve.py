"""Hands-free improvement loop for Wingspan NN policy/value model.

Per iteration:
1) Generate fresh self-play dataset.
2) Train policy-value model.
3) Evaluate vs heuristic baseline.
4) Keep best checkpoint by primary score.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.evaluate_policy_value import evaluate_nn_vs_heuristic
from backend.ml.self_play_dataset import generate_dataset
from backend.ml.train_policy_value import train


def _primary_score(eval_result: dict) -> float:
    """Primary selection score: maximize mean margin, tie-break by wins."""
    return float(eval_result["nn_mean_margin"]) + 0.01 * float(eval_result["nn_wins"])


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
) -> dict:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / "auto_improve_manifest.json"
    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_dataset.meta.json"

    history: list[dict] = []
    best: dict | None = None
    started = time.time()

    for i in range(1, iterations + 1):
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        dataset_jsonl = iter_dir / "dataset.jsonl"
        dataset_meta = iter_dir / "dataset.meta.json"
        model_path = iter_dir / "model.npz"
        eval_path = iter_dir / "eval.json"

        ds_meta = generate_dataset(
            output_jsonl=str(dataset_jsonl),
            metadata_path=str(dataset_meta),
            games=games_per_iter,
            players=players,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed,
        )

        train_meta = train(
            dataset_jsonl=str(dataset_jsonl),
            metadata_json=str(dataset_meta),
            out_model=str(model_path),
            epochs=train_epochs,
            batch_size=train_batch,
            hidden_dim=train_hidden,
            learning_rate=train_lr,
            value_loss_weight=value_weight,
            val_split=val_split,
            seed=iter_seed,
        )

        ev = evaluate_nn_vs_heuristic(
            model_path=str(model_path),
            dataset_meta_path=str(dataset_meta),
            games=eval_games,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed,
        )
        ev_dict = ev.__dict__
        eval_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

        row = {
            "iteration": i,
            "seed": iter_seed,
            "dataset": str(dataset_jsonl),
            "dataset_meta": str(dataset_meta),
            "model": str(model_path),
            "eval": str(eval_path),
            "dataset_summary": {
                "samples": ds_meta["samples"],
                "mean_player_score": ds_meta["mean_player_score"],
                "mean_winner_score": ds_meta["mean_winner_score"],
                "actions": ds_meta["action_space"]["num_actions"],
            },
            "train_summary": {
                "train_samples": train_meta["train_samples"],
                "val_samples": train_meta["val_samples"],
                "last_epoch": train_meta["history"][-1] if train_meta["history"] else {},
            },
            "eval_summary": ev_dict,
            "primary_score": round(_primary_score(ev_dict), 6),
        }
        history.append(row)

        if best is None or row["primary_score"] > best["primary_score"]:
            best = {
                "iteration": i,
                "primary_score": row["primary_score"],
                "eval_summary": ev_dict,
                "model": str(best_model_path),
                "dataset_meta": str(best_meta_path),
            }
            shutil.copy2(model_path, best_model_path)
            shutil.copy2(dataset_meta, best_meta_path)

        manifest = {
            "version": 1,
            "started_at_epoch": int(started),
            "updated_at_epoch": int(time.time()),
            "elapsed_sec": round(time.time() - started, 2),
            "config": {
                "iterations": iterations,
                "players": players,
                "board_type": board_type.value,
                "max_turns": max_turns,
                "games_per_iter": games_per_iter,
                "train_epochs": train_epochs,
                "train_batch": train_batch,
                "train_hidden": train_hidden,
                "train_lr": train_lr,
                "value_weight": value_weight,
                "val_split": val_split,
                "eval_games": eval_games,
                "seed": seed,
            },
            "best": best,
            "history": history,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(
            f"iter {i}/{iterations} | "
            f"samples={row['dataset_summary']['samples']} actions={row['dataset_summary']['actions']} | "
            f"nn_wins={ev_dict['nn_wins']}/{ev_dict['games']} margin={ev_dict['nn_mean_margin']:.2f} | "
            f"best_iter={best['iteration']} best_margin={best['eval_summary']['nn_mean_margin']:.2f}"
        )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hands-free NN improve loop")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden", type=int, default=192)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=100)
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
    )

    best = manifest.get("best") or {}
    print(
        "auto-improve complete | "
        f"best_iter={best.get('iteration')} | "
        f"best_margin={(best.get('eval_summary') or {}).get('nn_mean_margin')}"
    )


if __name__ == "__main__":
    main()
