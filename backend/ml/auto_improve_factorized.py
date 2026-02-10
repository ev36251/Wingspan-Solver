"""Auto-improve loop for factorized BC with strict promotion gating.

Implements:
- Policy improvement data generation (proposal+lookahead)
- BC training with value targets
- 200-game strict promotion gate: promote only if nn_wins > heuristic_wins
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.train_factorized_bc import train_bc


def run_auto_improve_factorized(
    out_dir: str,
    iterations: int,
    players: int,
    board_type: BoardType,
    max_turns: int,
    games_per_iter: int,
    proposal_top_k: int,
    lookahead_depth: int,
    n_step: int,
    gamma: float,
    bootstrap_mix: float,
    train_epochs: int,
    train_batch: int,
    train_hidden: int,
    train_lr: float,
    train_value_weight: float,
    val_split: float,
    eval_games: int,
    promotion_games: int,
    seed: int,
) -> dict:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / "auto_improve_factorized_manifest.json"
    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_dataset.meta.json"

    best: dict | None = None
    history: list[dict] = []
    started = time.time()

    proposal_model_path: str | None = None

    for i in range(1, iterations + 1):
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        dataset_jsonl = iter_dir / "bc_dataset.jsonl"
        dataset_meta = iter_dir / "bc_dataset.meta.json"
        model_path = iter_dir / "factorized_model.npz"
        eval_path = iter_dir / "eval.json"
        gate_path = iter_dir / "promotion_gate_eval.json"

        ds_meta = generate_bc_dataset(
            out_jsonl=str(dataset_jsonl),
            out_meta=str(dataset_meta),
            games=games_per_iter,
            players=players,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed,
            proposal_model_path=proposal_model_path,
            proposal_top_k=proposal_top_k,
            lookahead_depth=lookahead_depth,
            n_step=n_step,
            gamma=gamma,
            bootstrap_mix=bootstrap_mix,
        )

        tr = train_bc(
            dataset_jsonl=str(dataset_jsonl),
            meta_json=str(dataset_meta),
            out_model=str(model_path),
            epochs=train_epochs,
            batch_size=train_batch,
            hidden=train_hidden,
            lr=train_lr,
            val_split=val_split,
            seed=iter_seed,
            value_loss_weight=train_value_weight,
        )

        ev = evaluate_factorized_vs_heuristic(
            model_path=str(model_path),
            games=eval_games,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed,
        )
        ev_dict = ev.__dict__
        eval_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

        gate = evaluate_factorized_vs_heuristic(
            model_path=str(model_path),
            games=promotion_games,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed + 777,
        )
        gate_dict = gate.__dict__
        gate_path.write_text(json.dumps(gate_dict, indent=2), encoding="utf-8")

        promoted = gate_dict["nn_wins"] > gate_dict["heuristic_wins"]

        row = {
            "iteration": i,
            "seed": iter_seed,
            "dataset": str(dataset_jsonl),
            "dataset_meta": str(dataset_meta),
            "model": str(model_path),
            "dataset_summary": {
                "samples": ds_meta["samples"],
                "mean_player_score": ds_meta["mean_player_score"],
            },
            "train_summary": {
                "train_samples": tr["train_samples"],
                "val_samples": tr["val_samples"],
                "last_epoch": tr["history"][-1] if tr["history"] else {},
            },
            "eval_summary": ev_dict,
            "promotion_gate": {
                "games": promotion_games,
                "result": gate_dict,
                "promoted": promoted,
            },
        }
        history.append(row)

        if promoted:
            if best is None or gate_dict["nn_mean_margin"] > best["promotion_gate"]["result"]["nn_mean_margin"]:
                best = row
                shutil.copy2(model_path, best_model_path)
                shutil.copy2(dataset_meta, best_meta_path)
                proposal_model_path = str(best_model_path)

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
                "proposal_top_k": proposal_top_k,
                "lookahead_depth": lookahead_depth,
                "n_step": n_step,
                "gamma": gamma,
                "bootstrap_mix": bootstrap_mix,
                "train_epochs": train_epochs,
                "train_batch": train_batch,
                "train_hidden": train_hidden,
                "train_lr": train_lr,
                "train_value_weight": train_value_weight,
                "val_split": val_split,
                "eval_games": eval_games,
                "promotion_games": promotion_games,
                "seed": seed,
            },
            "best": best,
            "history": history,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(
            f"iter {i}/{iterations} | samples={ds_meta['samples']} | "
            f"eval_wins={ev_dict['nn_wins']}/{ev_dict['games']} margin={ev_dict['nn_mean_margin']:.2f} | "
            f"gate_wins={gate_dict['nn_wins']}/{promotion_games} promoted={promoted}"
        )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-improve factorized BC with strict gating")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve_factorized")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--bootstrap-mix", type=float, default=0.35)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden", type=int, default=192)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=80)
    parser.add_argument("--promotion-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = run_auto_improve_factorized(
        out_dir=args.out_dir,
        iterations=args.iterations,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        games_per_iter=args.games_per_iter,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
        n_step=args.n_step,
        gamma=args.gamma,
        bootstrap_mix=args.bootstrap_mix,
        train_epochs=args.train_epochs,
        train_batch=args.train_batch,
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
        train_value_weight=args.train_value_weight,
        val_split=args.val_split,
        eval_games=args.eval_games,
        promotion_games=args.promotion_games,
        seed=args.seed,
    )

    best = manifest.get("best")
    print(f"auto-improve-factorized complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
