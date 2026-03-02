"""AlphaZero self-play training pipeline for Wingspan.

Mirrors the structure of auto_improve_factorized.py but replaces the heuristic
teacher + lookahead data generation with MCTS self-play.

Loop per iteration:
  1. generate_self_play_dataset  — N games of MCTS self-play (parallel workers)
  2. train_bc                    — train NN with delta value targets
  3. eval_vs_heuristic           — quick sanity check
  4. promotion_gate              — 80-game match vs champion; promote if majority wins
  5. if promoted → copy to best_model.npz; champion used for next iter data gen

Value target: delta = my_final_score − best_opponent_final_score.
  Normalized by /80 (stored in meta["value_target_config"]["score_scale"]).
  No code changes to train_factorized_bc.py required — the scale comes from meta.

CLI (smoke test — ~25 min on MPS):
    python -m backend.ml.auto_improve_alphazero \\
      --out-dir reports/ml/alphazero_smoke \\
      --iterations 1 --games-per-iter 5 --mcts-sims 20 \\
      --train-epochs 2 --eval-games 5 --promotion-games 10 \\
      --dataset-workers 1

CLI (pilot — 10 iters, 50 sims, ~25-30 min/iter):
    python -m backend.ml.auto_improve_alphazero \\
      --out-dir reports/ml/alphazero_pilot_v1 \\
      --iterations 10 --games-per-iter 200 --mcts-sims 50 \\
      --train-init-model-path reports/ml/phase4_slot_v3/best_model.npz \\
      --use-per-slot-encoding --dataset-workers 4

CLI (full — 20 iters, 150 sims, overnight):
    python -m backend.ml.auto_improve_alphazero \\
      --out-dir reports/ml/alphazero_v1 \\
      --iterations 20 --games-per-iter 200 --mcts-sims 150 \\
      --train-init-model-path reports/ml/alphazero_pilot_v1/best_model.npz \\
      --use-per-slot-encoding --dataset-workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.alphazero_self_play import generate_self_play_dataset, DELTA_SCALE, DELTA_BIAS
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic, evaluate_nn_vs_nn
from backend.ml.train_factorized_bc import train_bc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _win_rate_from_eval(eval_dict: dict) -> float:
    games = max(1, int(eval_dict.get("games", 0)))
    return float(eval_dict.get("nn_wins", 0)) / games


def _eval_to_summary(eval_result) -> dict:
    """Convert EvalResult dataclass or dict to a JSON-safe summary dict."""
    if dataclasses.is_dataclass(eval_result) and not isinstance(eval_result, type):
        d = dataclasses.asdict(eval_result)
    elif isinstance(eval_result, dict):
        d = eval_result
    else:
        d = {}
    return {
        "games": int(d.get("games", 0)),
        "nn_wins": int(d.get("nn_wins", 0)),
        "heuristic_wins": int(d.get("heuristic_wins", 0)),
        "ties": int(d.get("ties", 0)),
        "nn_mean_score": float(d.get("nn_mean_score", 0.0)),
        "heuristic_mean_score": float(d.get("heuristic_mean_score", 0.0)),
        "nn_mean_margin": float(d.get("nn_mean_margin", 0.0)),
        "nn_win_rate": round(float(d.get("nn_wins", 0)) / max(1, int(d.get("games", 1))), 4),
        "nn_rate_ge_100": float(d.get("nn_rate_ge_100", 0.0)),
        "nn_rate_ge_120": float(d.get("nn_rate_ge_120", 0.0)),
    }


def _run_az_shard(task: dict) -> dict:
    """Worker entrypoint for parallel self-play shard generation."""
    # Limit BLAS to 1 thread per worker. Without this, each spawned process tries to
    # use all CPU cores for numpy BLAS, inflating forward-pass time from 0.1ms to
    # ~150ms due to thread management overhead on tiny GEMV operations.
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("BLAS_NUM_THREADS", "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    meta = generate_self_play_dataset(
        model_path=task["model_path"],
        out_jsonl=task["out_jsonl"],
        out_meta=task["out_meta"],
        games=task["games"],
        players=task["players"],
        board_type=BoardType(task["board_type"]),
        mcts_sims=task["mcts_sims"],
        c_puct=task["c_puct"],
        value_blend=task["value_blend"],
        rollout_policy=task["rollout_policy"],
        temperature_cutoff=task["temperature_cutoff"],
        seed=task["seed"],
        max_turns=task["max_turns"],
        strict_rules_mode=task["strict_rules_mode"],
        value_target_score_scale=task["value_target_score_scale"],
        value_target_score_bias=task["value_target_score_bias"],
        enable_identity_features=task.get("enable_identity_features"),
        identity_hash_dim=task.get("identity_hash_dim"),
        use_per_slot_encoding=task.get("use_per_slot_encoding"),
        use_hand_habitat_features=task.get("use_hand_habitat_features"),
        use_tray_per_slot_encoding=task.get("use_tray_per_slot_encoding"),
        use_opponent_board_encoding=task.get("use_opponent_board_encoding"),
        use_power_features=task.get("use_power_features"),
    )
    return {
        "jsonl": task["out_jsonl"],
        "meta_path": task["out_meta"],
        "meta": meta,
    }


def _merge_az_shards(
    shard_results: list[dict],
    *,
    out_jsonl: Path,
    out_meta: Path,
) -> dict:
    """Concatenate shard JSONL files and merge meta dicts."""
    if not shard_results:
        raise ValueError("No shard results to merge")

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for shard in shard_results:
            with Path(shard["jsonl"]).open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)

    metas = [shard["meta"] for shard in shard_results]
    base = dict(metas[0])
    base["games"] = sum(int(m.get("games", 0)) for m in metas)
    base["samples"] = sum(int(m.get("samples", 0)) for m in metas)
    base["elapsed_sec"] = round(sum(float(m.get("elapsed_sec", 0.0)) for m in metas), 2)

    import numpy as np
    all_deltas: list[float] = []
    for m in metas:
        n = int(m.get("samples", 0))
        mean_d = float(m.get("mean_delta", 0.0))
        all_deltas.extend([mean_d] * max(1, n))
    if all_deltas:
        base["mean_delta"] = round(float(np.mean(all_deltas)), 3)
        base["std_delta"] = round(float(np.std(all_deltas)), 3)

    out_meta.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_auto_improve_alphazero(
    out_dir: str,
    iterations: int,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    games_per_iter: int = 200,
    mcts_sims: int = 150,
    c_puct: float = 1.5,
    value_blend: float = 0.5,
    rollout_policy: str = "fast",
    temperature_cutoff: int = 8,
    strict_rules_mode: bool = False,
    # Training
    train_epochs: int = 30,
    train_batch: int = 128,
    train_hidden1: int = 384,
    train_hidden2: int = 192,
    train_dropout: float = 0.2,
    train_lr_init: float = 1e-4,
    train_lr_peak: float = 5e-4,
    train_lr_warmup_epochs: int = 3,
    train_lr_decay_every: int = 5,
    train_lr_decay_factor: float = 0.7,
    train_momentum: float = 0.9,
    train_value_weight: float = 0.5,
    train_early_stop_enabled: bool = True,
    train_early_stop_patience: int = 5,
    train_early_stop_min_delta: float = 1e-4,
    train_init_model_path: str | None = None,
    train_init_first_iter_only: bool = False,
    # Eval / promotion
    eval_games: int = 40,
    promotion_games: int = 80,
    min_promotion_win_rate: float = 0.5,
    gate_mode: str = "heuristic",   # "heuristic" | "champion"
    gate_mcts_sims: int = 20,       # sims per player in NN vs NN gate
    # State encoder
    state_encoder_use_per_slot: bool = False,
    state_encoder_enable_identity: bool = False,
    state_encoder_identity_hash_dim: int = 128,
    state_encoder_max_hand_slots: int = 8,
    state_encoder_use_hand_habitat_features: bool = False,
    state_encoder_use_tray_per_slot: bool = False,
    state_encoder_use_opponent_board: bool = False,
    state_encoder_use_power_features: bool = False,
    # Delta value target normalization
    value_target_score_scale: float = DELTA_SCALE,
    value_target_score_bias: float = DELTA_BIAS,
    # Data accumulation (simple replay buffer)
    data_accumulation_enabled: bool = True,
    data_accumulation_decay: float = 0.5,
    max_accumulated_samples: int = 200_000,
    # Parallelism
    dataset_workers: int = 4,
    # Misc
    seed: int = 0,
    clean_out_dir: bool = True,
    train_hidden: int | None = None,
) -> dict:
    """Run the AlphaZero self-play training pipeline.

    Returns a manifest dict summarising all iterations.
    """
    import numpy as np

    # Deprecated alias
    if train_hidden is not None:
        train_hidden1 = int(train_hidden)
        train_hidden2 = int(train_hidden)
        print("warning: train_hidden is deprecated; prefer train_hidden1/train_hidden2")

    # Auto-scale hidden sizes for per-slot encoding
    if state_encoder_use_per_slot and train_hidden1 == 384 and train_hidden2 == 192:
        train_hidden1, train_hidden2 = 768, 384
        print("auto_improve_alphazero: per-slot encoding → auto-scaled hidden=768/384")

    # CI / test safety: force single worker
    if os.getenv("PYTEST_CURRENT_TEST"):
        dataset_workers = 1
    main_mod = sys.modules.get("__main__")
    if not getattr(main_mod, "__file__", None):
        dataset_workers = 1

    dataset_workers = max(1, int(dataset_workers))

    if train_init_model_path is not None:
        init_path = Path(train_init_model_path)
        if not init_path.exists():
            raise FileNotFoundError(
                f"train_init_model_path does not exist: {train_init_model_path}"
            )

    base = Path(out_dir)
    if clean_out_dir and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_model.meta.json"
    manifest_path = base / "auto_improve_alphazero_manifest.json"

    history: list[dict] = []
    started = time.time()
    champion_eval_summary: dict | None = None   # last promotion-gate eval vs heuristic

    # Start with user-supplied init model as the "champion"
    if train_init_model_path and not best_model_path.exists():
        shutil.copy2(train_init_model_path, best_model_path)
        print(f"  Seeded best_model.npz from {train_init_model_path}")

    accumulation_sources: list[dict] = []   # for replay buffer

    for i in range(1, iterations + 1):
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        dataset_jsonl = iter_dir / "az_dataset.jsonl"
        dataset_meta_path = iter_dir / "az_dataset.meta.json"
        model_path = iter_dir / "model.npz"
        eval_path = iter_dir / "eval.json"
        gate_path = iter_dir / "promotion_gate.json"

        print(f"\n{'='*60}")
        print(f"  AlphaZero iter {i}/{iterations} | mcts_sims={mcts_sims}")
        print(f"  out: {iter_dir}")
        print(f"{'='*60}")

        # Determine which model to use for data generation (champion or init)
        if best_model_path.exists():
            data_gen_model = str(best_model_path)
        elif train_init_model_path:
            data_gen_model = train_init_model_path
        else:
            raise RuntimeError(
                "No model available for self-play data generation. "
                "Provide --train-init-model-path for the first iteration."
            )

        # ----------------------------------------------------------------
        # Step 1: Generate self-play dataset
        # ----------------------------------------------------------------
        print(f"\n  [1/4] Generating self-play data ({games_per_iter} games, "
              f"{mcts_sims} sims, {dataset_workers} workers)...")

        shard_common = dict(
            model_path=data_gen_model,
            players=players,
            board_type=board_type.value,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
            value_blend=value_blend,
            rollout_policy=rollout_policy,
            temperature_cutoff=temperature_cutoff,
            max_turns=max_turns,
            strict_rules_mode=strict_rules_mode,
            value_target_score_scale=value_target_score_scale,
            value_target_score_bias=value_target_score_bias,
            # State encoder overrides: passed to generate_self_play_dataset so
            # training data is encoded with the requested feature set even when
            # the current champion model was trained without those features.
            enable_identity_features=state_encoder_enable_identity if state_encoder_enable_identity else None,
            identity_hash_dim=state_encoder_identity_hash_dim if state_encoder_enable_identity else None,
            use_per_slot_encoding=state_encoder_use_per_slot if state_encoder_use_per_slot else None,
            use_hand_habitat_features=state_encoder_use_hand_habitat_features if state_encoder_use_hand_habitat_features else None,
            use_tray_per_slot_encoding=state_encoder_use_tray_per_slot if state_encoder_use_tray_per_slot else None,
            use_opponent_board_encoding=state_encoder_use_opponent_board if state_encoder_use_opponent_board else None,
            use_power_features=state_encoder_use_power_features if state_encoder_use_power_features else None,
        )

        if dataset_workers <= 1 or games_per_iter <= 1:
            ds_meta = generate_self_play_dataset(
                out_jsonl=str(dataset_jsonl),
                out_meta=str(dataset_meta_path),
                games=games_per_iter,
                seed=iter_seed,
                **{k: v for k, v in shard_common.items() if k not in ("board_type",)},
                board_type=board_type,
            )
        else:
            shard_dir = iter_dir / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            worker_count = min(dataset_workers, games_per_iter)
            base_games = games_per_iter // worker_count
            extras = games_per_iter % worker_count

            shard_tasks: list[dict] = []
            for shard_idx in range(worker_count):
                shard_games = base_games + (1 if shard_idx < extras else 0)
                if shard_games <= 0:
                    continue
                shard_tasks.append(
                    {
                        **shard_common,
                        "out_jsonl": str(shard_dir / f"az_{shard_idx:02d}.jsonl"),
                        "out_meta": str(shard_dir / f"az_{shard_idx:02d}.meta.json"),
                        "games": shard_games,
                        "seed": iter_seed + shard_idx * 100_003,
                    }
                )

            if len(shard_tasks) <= 1:
                t = shard_tasks[0]
                ds_meta = generate_self_play_dataset(
                    model_path=t["model_path"],
                    out_jsonl=t["out_jsonl"],
                    out_meta=t["out_meta"],
                    games=t["games"],
                    seed=t["seed"],
                    board_type=board_type,
                    **{k: v for k, v in t.items()
                       if k not in ("model_path", "out_jsonl", "out_meta", "games", "seed", "board_type")},
                )
                shutil.copy2(t["out_jsonl"], dataset_jsonl)
                shutil.copy2(t["out_meta"], dataset_meta_path)
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=len(shard_tasks)) as pool:
                    shard_results = pool.map(_run_az_shard, shard_tasks)
                ds_meta = _merge_az_shards(
                    shard_results,
                    out_jsonl=dataset_jsonl,
                    out_meta=dataset_meta_path,
                )

        n_samples = int(ds_meta.get("samples", 0))
        print(f"  Data gen complete: {n_samples} samples | "
              f"mean_delta={ds_meta.get('mean_delta', 0.0):.1f}")

        # ----------------------------------------------------------------
        # Optional data accumulation (replay buffer)
        # ----------------------------------------------------------------
        train_jsonl = dataset_jsonl
        train_meta = dataset_meta_path

        if data_accumulation_enabled and accumulation_sources:
            combined_jsonl = iter_dir / "az_dataset_combined.jsonl"
            combined_meta_path = iter_dir / "az_dataset_combined.meta.json"

            # Count how many old samples to include (decayed)
            total_new = n_samples
            budget_old = max_accumulated_samples - total_new
            old_rows_to_add: list[tuple[Path, int]] = []
            remaining = budget_old
            for src in reversed(accumulation_sources):
                src_n = int(src.get("rows", 0))
                if src_n <= 0 or remaining <= 0:
                    continue
                take = min(src_n, remaining)
                old_rows_to_add.append((Path(src["jsonl"]), take, src_n))
                remaining -= take

            written = 0
            import random as _rng
            rng = _rng.Random(iter_seed + 777)
            with combined_jsonl.open("w", encoding="utf-8") as out_f:
                # Write new data first
                with dataset_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)
                            written += 1
                # Append sampled old data
                for src_path, take, total in old_rows_to_add:
                    if not src_path.exists():
                        continue
                    keep_idx = set(rng.sample(range(total), min(take, total)))
                    row_idx = 0
                    with src_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            if row_idx in keep_idx:
                                out_f.write(line)
                                written += 1
                            row_idx += 1

            combined_meta = dict(ds_meta)
            combined_meta["samples"] = written
            combined_meta_path.write_text(json.dumps(combined_meta, indent=2), encoding="utf-8")
            train_jsonl = combined_jsonl
            train_meta = combined_meta_path
            print(f"  Combined dataset: {written} samples (new={n_samples} + replay)")

        # Save current iteration as accumulation source
        accumulation_sources.append(
            {"jsonl": str(dataset_jsonl), "rows": n_samples, "iter": i}
        )
        # Apply decay: drop oldest if over budget
        while sum(int(s.get("rows", 0)) for s in accumulation_sources) > max_accumulated_samples:
            accumulation_sources.pop(0)

        # ----------------------------------------------------------------
        # Step 2: Train
        # ----------------------------------------------------------------
        print(f"\n  [2/4] Training ({train_epochs} epochs, "
              f"hidden={train_hidden1}/{train_hidden2})...")

        # Only warm-start from user-supplied init on first iter (if configured)
        use_init = (
            train_init_model_path is not None
            and i == 1
            and train_init_first_iter_only
        ) or (
            train_init_model_path is not None
            and not train_init_first_iter_only
        )
        # Always warm-start from previous champion if available
        warmstart_path: str | None = None
        if best_model_path.exists():
            warmstart_path = str(best_model_path)
        elif use_init:
            warmstart_path = train_init_model_path

        train_result = train_bc(
            dataset_jsonl=str(train_jsonl),
            meta_json=str(train_meta),
            out_model=str(model_path),
            epochs=train_epochs,
            batch_size=train_batch,
            hidden1=train_hidden1,
            hidden2=train_hidden2,
            dropout=train_dropout,
            lr_init=train_lr_init,
            lr_peak=train_lr_peak,
            lr_warmup_epochs=train_lr_warmup_epochs,
            lr_decay_every=train_lr_decay_every,
            lr_decay_factor=train_lr_decay_factor,
            momentum=train_momentum,
            value_loss_weight=train_value_weight,
            early_stop_enabled=train_early_stop_enabled,
            early_stop_patience=train_early_stop_patience,
            early_stop_min_delta=train_early_stop_min_delta,
            init_model_path=warmstart_path,
            seed=iter_seed,
        )
        print(
            f"  Training done: epochs={train_result.get('epochs_completed')} | "
            f"val_loss={train_result.get('best_val_loss', 0.0):.4f}"
        )

        # ----------------------------------------------------------------
        # Step 3: Eval vs heuristic
        # ----------------------------------------------------------------
        print(f"\n  [3/4] Evaluating vs heuristic ({eval_games} games)...")
        try:
            eval_result = evaluate_factorized_vs_heuristic(
                model_path=str(model_path),
                games=eval_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=iter_seed + 1,
            )
            eval_summary = _eval_to_summary(eval_result)
        except Exception as exc:
            print(f"  [warn] eval failed: {exc}")
            eval_summary = {}

        eval_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        print(
            f"  Eval: nn_wins={eval_summary.get('nn_wins')} / "
            f"{eval_summary.get('games')} | "
            f"mean_score={eval_summary.get('nn_mean_score', 0.0):.1f} | "
            f"win_rate={eval_summary.get('nn_win_rate', 0.0):.3f}"
        )

        # ----------------------------------------------------------------
        # Step 4: Promotion gate (candidate vs champion)
        # ----------------------------------------------------------------
        print(f"\n  [4/4] Promotion gate ({promotion_games} games, mode={gate_mode})...")
        promoted = False
        gate_summary: dict = {}

        try:
            if gate_mode == "champion" and best_model_path.exists():
                # NN vs NN: candidate vs current champion, both using MCTS.
                # best_model_path still points to the previous champion at this
                # point — it is only overwritten after a successful promotion.
                gate_summary = evaluate_nn_vs_nn(
                    model_a_path=str(model_path),       # candidate
                    model_b_path=str(best_model_path),  # champion
                    games=promotion_games,
                    board_type=board_type,
                    mcts_sims=gate_mcts_sims,
                    seed=iter_seed + 2,
                )
            else:
                gate_result = evaluate_factorized_vs_heuristic(
                    model_path=str(model_path),
                    games=promotion_games,
                    board_type=board_type,
                    max_turns=max_turns,
                    seed=iter_seed + 2,
                )
                gate_summary = _eval_to_summary(gate_result)
        except Exception as exc:
            print(f"  [warn] promotion gate eval failed: {exc}")
            gate_summary = {}

        candidate_win_rate = float(gate_summary.get("nn_win_rate", 0.0))
        if candidate_win_rate >= min_promotion_win_rate:
            promoted = True

        # Also promote if first iteration with no existing champion
        if not best_model_path.exists():
            promoted = True

        gate_summary["promoted"] = promoted
        gate_summary["min_promotion_win_rate"] = min_promotion_win_rate
        gate_path.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")

        print(
            f"  Gate: nn_wins={gate_summary.get('nn_wins')} / "
            f"{gate_summary.get('games')} | "
            f"win_rate={candidate_win_rate:.3f} | "
            f"promoted={promoted}"
        )

        if promoted:
            shutil.copy2(model_path, best_model_path)
            best_meta_path.write_text(
                json.dumps(
                    {
                        "promoted_at_iter": i,
                        "eval_summary": eval_summary,
                        "gate_summary": gate_summary,
                        "train_result": {
                            k: v
                            for k, v in train_result.items()
                            if k not in ("history",)
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  ✓ Promoted! best_model.npz updated.")
            champion_eval_summary = gate_summary

        # ----------------------------------------------------------------
        # Record iteration history
        # ----------------------------------------------------------------
        iter_record = {
            "iter": i,
            "samples": n_samples,
            "mean_delta": float(ds_meta.get("mean_delta", 0.0)),
            "train_val_loss": float(train_result.get("best_val_loss", 0.0)),
            "train_val_action_acc": float(
                (train_result.get("history") or [{}])[-1].get("val_action_acc", 0.0)
            ),
            "eval_summary": eval_summary,
            "gate_summary": gate_summary,
            "promoted": promoted,
            "elapsed_sec": round(time.time() - started, 1),
        }
        history.append(iter_record)

        manifest = {
            "pipeline": "auto_improve_alphazero",
            "out_dir": out_dir,
            "iterations_completed": i,
            "iterations_total": iterations,
            "mcts_sims": mcts_sims,
            "gate_mode": gate_mode,
            "gate_mcts_sims": gate_mcts_sims,
            "total_elapsed_sec": round(time.time() - started, 1),
            "history": history,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  AlphaZero training complete: {iterations} iterations")
    print(f"  Total elapsed: {(time.time()-started)/3600:.2f} hours")
    print(f"  Best model: {best_model_path}")
    print(f"{'='*60}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="AlphaZero self-play training pipeline for Wingspan"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--board-type", default="oceania")
    p.add_argument("--max-turns", type=int, default=240)
    p.add_argument("--games-per-iter", type=int, default=200)
    p.add_argument("--mcts-sims", type=int, default=150)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--value-blend", type=float, default=0.5)
    p.add_argument("--rollout-policy", default="fast")
    p.add_argument("--temperature-cutoff", type=int, default=8)
    p.add_argument("--strict-rules-mode", action="store_true", default=False)
    # Training
    p.add_argument("--train-epochs", type=int, default=30)
    p.add_argument("--train-batch", type=int, default=128)
    p.add_argument("--train-hidden1", type=int, default=384)
    p.add_argument("--train-hidden2", type=int, default=192)
    p.add_argument("--train-dropout", type=float, default=0.2)
    p.add_argument("--train-lr-init", type=float, default=1e-4)
    p.add_argument("--train-lr-peak", type=float, default=5e-4)
    p.add_argument("--train-lr-warmup-epochs", type=int, default=3)
    p.add_argument("--train-early-stop-patience", type=int, default=5)
    p.add_argument("--train-value-weight", type=float, default=0.5)
    p.add_argument("--train-init-model-path", default=None)
    p.add_argument(
        "--train-init-first-iter-only",
        action="store_true",
        default=False,
        help="Use init model only for warm-start on iter 1; champion after that",
    )
    # Eval / promotion
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--promotion-games", type=int, default=80)
    p.add_argument("--min-promotion-win-rate", type=float, default=0.5)
    p.add_argument(
        "--gate-mode",
        default="heuristic",
        choices=["heuristic", "champion"],
        help="Promotion gate opponent: 'heuristic' (default) or 'champion' (NN vs NN with MCTS)",
    )
    p.add_argument(
        "--gate-mcts-sims",
        type=int,
        default=20,
        help="MCTS simulations per player turn in champion gate (default: 20)",
    )
    # State encoder
    p.add_argument("--use-per-slot-encoding", action="store_true", default=False)
    p.add_argument("--enable-identity-features", action="store_true", default=False)
    p.add_argument("--identity-hash-dim", type=int, default=128)
    p.add_argument("--use-hand-habitat-features", action="store_true", default=False,
                   help="Add 15 hand-board synergy features (5/habitat) to the state vector")
    p.add_argument("--use-tray-per-slot-encoding", action="store_true", default=False,
                   help="Add 114 tray per-bird features (3 slots × 38) so the NN sees each tray card in full")
    p.add_argument("--use-opponent-board-encoding", action="store_true", default=False,
                   help="Add 750 opponent board features (15 slots × 50) so the NN sees every bird the opponent has played")
    p.add_argument("--use-power-features", action="store_true", default=False,
                   help="Add 322 power-effect features for own board+hand (15+8 slots × 14) covering what each power does and who benefits")
    # Data accumulation
    p.add_argument("--data-accumulation-enabled", action="store_true", default=True)
    p.add_argument(
        "--no-data-accumulation", dest="data_accumulation_enabled", action="store_false"
    )
    p.add_argument("--max-accumulated-samples", type=int, default=200_000)
    # Parallelism
    p.add_argument("--dataset-workers", type=int, default=4)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-clean", dest="clean_out_dir", action="store_false", default=True)

    args = p.parse_args()

    bt = BoardType.OCEANIA if args.board_type.lower() == "oceania" else BoardType.BASE

    result = run_auto_improve_alphazero(
        out_dir=args.out_dir,
        iterations=args.iterations,
        players=args.players,
        board_type=bt,
        max_turns=args.max_turns,
        games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        value_blend=args.value_blend,
        rollout_policy=args.rollout_policy,
        temperature_cutoff=args.temperature_cutoff,
        strict_rules_mode=args.strict_rules_mode,
        train_epochs=args.train_epochs,
        train_batch=args.train_batch,
        train_hidden1=args.train_hidden1,
        train_hidden2=args.train_hidden2,
        train_dropout=args.train_dropout,
        train_lr_init=args.train_lr_init,
        train_lr_peak=args.train_lr_peak,
        train_lr_warmup_epochs=args.train_lr_warmup_epochs,
        train_early_stop_patience=args.train_early_stop_patience,
        train_value_weight=args.train_value_weight,
        train_init_model_path=args.train_init_model_path,
        train_init_first_iter_only=args.train_init_first_iter_only,
        eval_games=args.eval_games,
        promotion_games=args.promotion_games,
        min_promotion_win_rate=args.min_promotion_win_rate,
        gate_mode=args.gate_mode,
        gate_mcts_sims=args.gate_mcts_sims,
        state_encoder_use_per_slot=args.use_per_slot_encoding,
        state_encoder_enable_identity=args.enable_identity_features,
        state_encoder_identity_hash_dim=args.identity_hash_dim,
        state_encoder_use_hand_habitat_features=args.use_hand_habitat_features,
        state_encoder_use_tray_per_slot=args.use_tray_per_slot_encoding,
        state_encoder_use_opponent_board=args.use_opponent_board_encoding,
        state_encoder_use_power_features=args.use_power_features,
        data_accumulation_enabled=args.data_accumulation_enabled,
        max_accumulated_samples=args.max_accumulated_samples,
        dataset_workers=args.dataset_workers,
        seed=args.seed,
        clean_out_dir=args.clean_out_dir,
    )

    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "history"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
