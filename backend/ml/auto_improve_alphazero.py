"""AlphaZero self-play training pipeline for Wingspan.

Mirrors the structure of auto_improve_factorized.py but replaces the heuristic
teacher + lookahead data generation with MCTS self-play.

Loop per iteration:
  1. generate_self_play_dataset  — N games of MCTS self-play (parallel workers)
  2. train_bc                    — train NN with absolute-score value targets
  3. eval_vs_heuristic           — quick sanity check
  4. promotion_gate              — 80-game match vs champion; promote if majority wins
  5. if promoted → copy to best_model.npz; champion used for next iter data gen

Value target: absolute final score for the acting player.
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
import math
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

RULES_BASELINE_ID_DEFAULT = "rules_2026_03_07_semantic446"


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
        "nn_max_score": int(d.get("nn_max_score", 0)),
        "heuristic_max_score": int(d.get("heuristic_max_score", 0)),
    }


def _wilson_ci95(wins: int, games: int) -> tuple[float, float]:
    n = max(1, int(games))
    p = float(wins) / float(n)
    z = 1.96
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _combine_gate_summaries(parts: list[dict]) -> dict:
    if not parts:
        return {}
    games = sum(int(p.get("games", 0)) for p in parts)
    if games <= 0:
        return {}
    nn_wins = sum(int(p.get("nn_wins", 0)) for p in parts)
    h_wins = sum(int(p.get("heuristic_wins", 0)) for p in parts)
    ties = sum(int(p.get("ties", 0)) for p in parts)
    mean_score = sum(float(p.get("nn_mean_score", 0.0)) * int(p.get("games", 0)) for p in parts) / games
    mean_h = sum(float(p.get("heuristic_mean_score", 0.0)) * int(p.get("games", 0)) for p in parts) / games
    mean_margin = sum(float(p.get("nn_mean_margin", 0.0)) * int(p.get("games", 0)) for p in parts) / games
    ge100 = sum(float(p.get("nn_rate_ge_100", 0.0)) * int(p.get("games", 0)) for p in parts) / games
    ge120 = sum(float(p.get("nn_rate_ge_120", 0.0)) * int(p.get("games", 0)) for p in parts) / games
    out = {
        "games": int(games),
        "nn_wins": int(nn_wins),
        "heuristic_wins": int(h_wins),
        "ties": int(ties),
        "nn_mean_score": round(float(mean_score), 3),
        "heuristic_mean_score": round(float(mean_h), 3),
        "nn_mean_margin": round(float(mean_margin), 3),
        "nn_win_rate": round(float(nn_wins) / max(1, games), 4),
        "nn_rate_ge_100": round(float(ge100), 4),
        "nn_rate_ge_120": round(float(ge120), 4),
    }
    lo, hi = _wilson_ci95(nn_wins, games)
    out["nn_win_rate_ci95"] = [round(lo, 4), round(hi, 4)]
    return out


def _run_bird_audit_gate(expected_semantic_untested: int = 0) -> dict:
    """Run bird semantic coverage audit and enforce expected untested count."""
    from backend.scripts.audit_bird_power_tests import build_audit

    report = build_audit()
    summary = dict(report.get("summary", {}))
    actual = int(summary.get("semantic_untested_count", -1))
    expected = int(expected_semantic_untested)
    if actual != expected:
        raise RuntimeError(
            "Bird semantic audit gate failed: "
            f"semantic_untested_count={actual} (expected {expected}). "
            "Run: python -m backend.scripts.audit_bird_power_tests"
        )
    return summary


def _load_existing_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _assert_fresh_lineage_or_raise(
    *,
    out_dir: Path,
    clean_out_dir: bool,
    start_iter: int,
    rules_baseline_id: str,
    allow_stale_lineage: bool,
) -> None:
    """Treat prior baseline runs as stale unless explicitly allowed."""
    if allow_stale_lineage:
        return
    if clean_out_dir:
        return
    manifest = _load_existing_manifest(out_dir / "auto_improve_alphazero_manifest.json")
    if manifest is None:
        return
    existing_id = (
        (manifest.get("rules_baseline") or {}).get("id")
        if isinstance(manifest, dict)
        else None
    )
    if existing_id != rules_baseline_id:
        raise RuntimeError(
            "Existing run is stale or missing rules baseline metadata. "
            f"expected rules_baseline.id={rules_baseline_id!r}, "
            f"found {existing_id!r}. "
            "Start a fresh lineage (default) or pass --allow-stale-lineage to override."
        )
    if start_iter > 1 and int(manifest.get("iterations_completed", 0)) < (start_iter - 1):
        raise RuntimeError(
            f"Cannot resume from start_iter={start_iter}; "
            f"existing manifest only completed {manifest.get('iterations_completed', 0)} iterations."
        )


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
        root_dirichlet_epsilon=task.get("root_dirichlet_epsilon", 0.25),
        root_dirichlet_alpha=task.get("root_dirichlet_alpha", 0.3),
        rollout_policy=task["rollout_policy"],
        temperature_cutoff=task["temperature_cutoff"],
        seed=task["seed"],
        max_turns=task["max_turns"],
        strict_rules_mode=task["strict_rules_mode"],
        value_target_score_scale=task["value_target_score_scale"],
        value_target_score_bias=task["value_target_score_bias"],
        tie_value_target=task.get("tie_value_target", 0.5),
        enable_identity_features=task.get("enable_identity_features"),
        identity_hash_dim=task.get("identity_hash_dim"),
        use_per_slot_encoding=task.get("use_per_slot_encoding"),
        use_hand_habitat_features=task.get("use_hand_habitat_features"),
        use_tray_per_slot_encoding=task.get("use_tray_per_slot_encoding"),
        use_opponent_board_encoding=task.get("use_opponent_board_encoding"),
        use_power_features=task.get("use_power_features"),
        fail_on_game_exception=task.get("fail_on_game_exception", False),
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
    base["game_exceptions"] = sum(int(m.get("game_exceptions", 0)) for m in metas)
    base["games_completed"] = sum(
        int(
            m.get(
                "games_completed",
                max(0, int(m.get("games", 0)) - int(m.get("game_exceptions", 0))),
            )
        )
        for m in metas
    )
    base["samples"] = sum(int(m.get("samples", 0)) for m in metas)
    base["elapsed_sec"] = round(sum(float(m.get("elapsed_sec", 0.0)) for m in metas), 2)

    def _weighted_mean_std(
        mean_key: str,
        std_key: str,
    ) -> tuple[float, float] | None:
        total_n = 0
        sum_mean = 0.0
        sum_second_moment = 0.0

        for m in metas:
            n = int(m.get("samples", 0))
            if n <= 0 or mean_key not in m:
                continue
            mean_v = float(m.get(mean_key, 0.0))
            std_v = max(0.0, float(m.get(std_key, 0.0)))
            total_n += n
            sum_mean += n * mean_v
            sum_second_moment += n * (std_v * std_v + mean_v * mean_v)

        if total_n <= 0:
            return None
        mean = sum_mean / total_n
        var = max(0.0, (sum_second_moment / total_n) - (mean * mean))
        return mean, var**0.5

    score_stats = _weighted_mean_std("mean_score", "std_score")
    delta_stats = _weighted_mean_std("mean_delta", "std_delta")

    if score_stats is None:
        score_stats = delta_stats
    if delta_stats is None:
        delta_stats = score_stats

    if score_stats is not None:
        base["mean_score"] = round(float(score_stats[0]), 3)
        base["std_score"] = round(float(score_stats[1]), 3)
    if delta_stats is not None:
        base["mean_delta"] = round(float(delta_stats[0]), 3)
        base["std_delta"] = round(float(delta_stats[1]), 3)

    for k in (
        "mean_win_target",
        "policy_entropy_action_type_mean",
        "policy_peak_action_type_mean",
    ):
        total_n = 0
        num = 0.0
        for m in metas:
            n = int(m.get("samples", 0))
            if n <= 0 or k not in m:
                continue
            total_n += n
            num += n * float(m.get(k, 0.0))
        if total_n > 0:
            base[k] = round(float(num / total_n), 6)

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
    value_blend: float | None = None,  # legacy alias
    selfplay_value_blend: float | None = None,
    gate_value_blend: float | None = None,
    rollout_policy: str = "fast",
    selfplay_root_dirichlet_epsilon: float = 0.25,
    selfplay_root_dirichlet_alpha: float = 0.3,
    selfplay_fail_on_exception: bool = True,
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
    train_weight_decay: float = 0.0,
    train_value_weight: float | None = None,  # legacy alias for score head weight
    train_score_value_weight: float = 0.25,
    train_win_value_weight: float = 1.0,
    train_early_stop_enabled: bool = True,
    train_early_stop_patience: int = 5,
    train_early_stop_min_delta: float = 1e-4,
    train_init_model_path: str | None = None,
    train_init_first_iter_only: bool = False,
    train_reinit_value_head: bool = False,
    seed_champion_from_init: bool = True,
    # Eval / promotion
    eval_games: int = 40,
    eval_use_mcts: bool = False,
    eval_mcts_sims: int = 40,
    eval_mcts_c_puct: float = 1.5,
    eval_mcts_value_blend: float | None = None,
    eval_mcts_rollout_policy: str | None = None,
    eval_heuristic_policy: str = "greedy",
    promotion_games: int = 80,
    min_promotion_win_rate: float = 0.5,
    gate_mode: str = "heuristic",   # "heuristic" | "champion"
    gate_mcts_sims: int = 40,       # sims per player in NN vs NN gate
    gate_stage_games: int = 20,
    pilot_early_stop_iter: int = 3,
    pilot_min_gate_win_rate: float = 0.20,
    # State encoder
    state_encoder_use_per_slot: bool = False,
    state_encoder_enable_identity: bool = False,
    state_encoder_identity_hash_dim: int = 128,
    state_encoder_max_hand_slots: int = 8,
    state_encoder_use_hand_habitat_features: bool = False,
    state_encoder_use_tray_per_slot: bool = False,
    state_encoder_use_opponent_board: bool = False,
    state_encoder_use_power_features: bool = False,
    # Value target normalization (absolute score mode)
    value_target_score_scale: float = DELTA_SCALE,
    value_target_score_bias: float = DELTA_BIAS,
    tie_value_target: float = 0.5,
    # Data accumulation (simple replay buffer)
    data_accumulation_enabled: bool = True,
    data_accumulation_decay: float = 0.5,
    max_accumulated_samples: int = 200_000,
    # Optional higher-sim teacher curriculum data mixed into each iteration.
    teacher_games_per_iter: int = 0,
    teacher_mcts_sims: int | None = None,
    teacher_value_blend: float | None = None,
    teacher_root_dirichlet_epsilon: float = 0.1,
    teacher_root_dirichlet_alpha: float = 0.25,
    teacher_rollout_policy: str | None = None,
    # Parallelism
    dataset_workers: int = 4,
    use_modal: bool = False,
    modal_cpu_per_worker: int = 2,
    # Misc
    seed: int = 0,
    clean_out_dir: bool = True,
    start_iter: int = 1,
    require_bird_audit_gate: bool = True,
    expected_semantic_untested: int = 0,
    rules_baseline_id: str = RULES_BASELINE_ID_DEFAULT,
    allow_stale_lineage: bool = False,
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
    if value_blend is None:
        value_blend = 0.5
    if selfplay_value_blend is None:
        selfplay_value_blend = float(value_blend)
    if gate_value_blend is None:
        gate_value_blend = float(value_blend)
    if eval_mcts_value_blend is None:
        eval_mcts_value_blend = float(gate_value_blend)
    if eval_mcts_rollout_policy is None:
        eval_mcts_rollout_policy = str(rollout_policy)
    if teacher_mcts_sims is None:
        teacher_mcts_sims = max(int(mcts_sims) + 16, int(mcts_sims * 2))
    if teacher_value_blend is None:
        teacher_value_blend = float(selfplay_value_blend)
    if teacher_rollout_policy is None:
        teacher_rollout_policy = str(rollout_policy)
    if train_value_weight is not None:
        train_score_value_weight = float(train_value_weight)
    eval_heuristic_policy = str(eval_heuristic_policy).strip().lower()
    if eval_heuristic_policy not in {"greedy", "weighted_random"}:
        raise ValueError(
            f"Unsupported eval_heuristic_policy={eval_heuristic_policy!r}; "
            "expected 'greedy' or 'weighted_random'."
        )

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
    data_accumulation_decay = max(0.0, min(1.0, float(data_accumulation_decay)))

    bird_audit_summary: dict | None = None
    if require_bird_audit_gate:
        bird_audit_summary = _run_bird_audit_gate(
            expected_semantic_untested=expected_semantic_untested
        )
        print(
            "  Preflight bird audit gate passed: "
            f"semantic_untested_count={bird_audit_summary.get('semantic_untested_count')}"
        )

    if train_init_model_path is not None:
        init_path = Path(train_init_model_path)
        if not init_path.exists():
            raise FileNotFoundError(
                f"train_init_model_path does not exist: {train_init_model_path}"
            )

    base = Path(out_dir)
    _assert_fresh_lineage_or_raise(
        out_dir=base,
        clean_out_dir=bool(clean_out_dir),
        start_iter=int(start_iter),
        rules_baseline_id=str(rules_baseline_id),
        allow_stale_lineage=bool(allow_stale_lineage),
    )
    if clean_out_dir and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_model.meta.json"
    manifest_path = base / "auto_improve_alphazero_manifest.json"

    history: list[dict] = []
    started = time.time()
    champion_eval_summary: dict | None = None   # last promotion-gate eval vs heuristic

    # Optionally seed the champion from the init model.
    # Set seed_champion_from_init=False to use the init model only for data
    # generation on iter 1, letting iter 1 auto-promote (replicates v10 behaviour).
    if train_init_model_path and not best_model_path.exists() and seed_champion_from_init:
        shutil.copy2(train_init_model_path, best_model_path)
        print(f"  Seeded best_model.npz from {train_init_model_path}")

    accumulation_sources: list[dict] = []   # for replay buffer

    # ----------------------------------------------------------------
    # Resume: restore history + replay buffer from completed iterations
    # ----------------------------------------------------------------
    if start_iter > 1:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            history = [h for h in existing.get("history", []) if h["iter"] < start_iter]
            print(f"  [resume] Loaded {len(history)} iter records from existing manifest.")
        for past_i in range(1, start_iter):
            past_jsonl = base / f"iter_{past_i:03d}" / "az_dataset.jsonl"
            if past_jsonl.exists():
                import subprocess as _sp
                try:
                    wc = _sp.check_output(["wc", "-l", str(past_jsonl)], text=True)
                    rows = int(wc.strip().split()[0])
                except Exception:
                    rows = 0
                accumulation_sources.append({"jsonl": str(past_jsonl), "rows": rows, "iter": past_i})
        # Apply decay budget (same logic as main loop)
        while sum(int(s.get("rows", 0)) for s in accumulation_sources) > max_accumulated_samples:
            accumulation_sources.pop(0)
        print(f"  [resume] Replay buffer: {len(accumulation_sources)} iter sources restored.")
        print(f"  [resume] Starting from iteration {start_iter} / {iterations}.")

    for i in range(start_iter, iterations + 1):
        iter_started = time.time()
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        dataset_jsonl = iter_dir / "az_dataset.jsonl"
        dataset_meta_path = iter_dir / "az_dataset.meta.json"
        model_path = iter_dir / "model.npz"
        eval_path = iter_dir / "eval.json"
        gate_path = iter_dir / "promotion_gate.json"

        # Disk space check: warn at <20 GB free, abort at <5 GB free
        _stat = os.statvfs(str(base))
        _free_gb = (_stat.f_bavail * _stat.f_frsize) / 1e9
        if _free_gb < 5.0:
            raise RuntimeError(
                f"DISK FULL: only {_free_gb:.1f} GB free on {base}. "
                "Aborting to prevent data corruption. "
                "Free up space and resume with --start-iter {i}."
            )
        elif _free_gb < 20.0:
            print(f"  [WARN] Low disk space: {_free_gb:.1f} GB free. "
                  "Consider cleaning up old training runs.")

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
        step_started = time.time()
        print(f"\n  [1/4] Generating self-play data ({games_per_iter} games, "
              f"{mcts_sims} sims, {dataset_workers} workers)...")

        shard_common = dict(
            model_path=data_gen_model,
            players=players,
            board_type=board_type.value,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
            value_blend=selfplay_value_blend,
            root_dirichlet_epsilon=selfplay_root_dirichlet_epsilon,
            root_dirichlet_alpha=selfplay_root_dirichlet_alpha,
            rollout_policy=rollout_policy,
            temperature_cutoff=temperature_cutoff,
            max_turns=max_turns,
            strict_rules_mode=strict_rules_mode,
            value_target_score_scale=value_target_score_scale,
            value_target_score_bias=value_target_score_bias,
            tie_value_target=tie_value_target,
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
            fail_on_game_exception=bool(selfplay_fail_on_exception),
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
            elif use_modal:
                from backend.ml.modal_selfplay import dispatch_shards_modal
                shard_results = dispatch_shards_modal(
                    shard_tasks,
                    model_path=data_gen_model,
                    cpu_per_worker=modal_cpu_per_worker,
                )
                ds_meta = _merge_az_shards(
                    shard_results,
                    out_jsonl=dataset_jsonl,
                    out_meta=dataset_meta_path,
                )
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
        mean_score = float(ds_meta.get("mean_score", ds_meta.get("mean_delta", 0.0)))
        print(f"  Data gen complete: {n_samples} samples | "
              f"mean_score={mean_score:.1f}")

        # Optional curriculum data: add a smaller teacher set generated with
        # higher MCTS sims and lower root noise.
        teacher_meta: dict | None = None
        if int(teacher_games_per_iter) > 0:
            teacher_games = int(teacher_games_per_iter)
            teacher_jsonl = iter_dir / "az_teacher_dataset.jsonl"
            teacher_meta_path = iter_dir / "az_teacher_dataset.meta.json"
            teacher_seed = iter_seed + 700_001
            print(
                "  Curriculum teacher data: "
                f"{teacher_games} games @ {int(teacher_mcts_sims)} sims"
            )

            teacher_task = {
                **shard_common,
                "mcts_sims": int(teacher_mcts_sims),
                "value_blend": float(teacher_value_blend),
                "root_dirichlet_epsilon": float(teacher_root_dirichlet_epsilon),
                "root_dirichlet_alpha": float(teacher_root_dirichlet_alpha),
                "rollout_policy": str(teacher_rollout_policy),
                "out_jsonl": str(teacher_jsonl),
                "out_meta": str(teacher_meta_path),
                "games": teacher_games,
                "seed": teacher_seed,
            }

            if use_modal:
                from backend.ml.modal_selfplay import dispatch_shards_modal
                teacher_results = dispatch_shards_modal(
                    [teacher_task],
                    model_path=data_gen_model,
                    cpu_per_worker=modal_cpu_per_worker,
                )
                teacher_meta = _merge_az_shards(
                    teacher_results,
                    out_jsonl=teacher_jsonl,
                    out_meta=teacher_meta_path,
                )
            else:
                teacher_meta = generate_self_play_dataset(
                    model_path=teacher_task["model_path"],
                    out_jsonl=teacher_task["out_jsonl"],
                    out_meta=teacher_task["out_meta"],
                    games=teacher_task["games"],
                    players=teacher_task["players"],
                    board_type=board_type,
                    mcts_sims=teacher_task["mcts_sims"],
                    c_puct=teacher_task["c_puct"],
                    value_blend=teacher_task["value_blend"],
                    root_dirichlet_epsilon=teacher_task["root_dirichlet_epsilon"],
                    root_dirichlet_alpha=teacher_task["root_dirichlet_alpha"],
                    rollout_policy=teacher_task["rollout_policy"],
                    temperature_cutoff=teacher_task["temperature_cutoff"],
                    seed=teacher_task["seed"],
                    max_turns=teacher_task["max_turns"],
                    strict_rules_mode=teacher_task["strict_rules_mode"],
                    value_target_score_scale=teacher_task["value_target_score_scale"],
                    value_target_score_bias=teacher_task["value_target_score_bias"],
                    tie_value_target=teacher_task["tie_value_target"],
                    enable_identity_features=teacher_task["enable_identity_features"],
                    identity_hash_dim=teacher_task["identity_hash_dim"],
                    use_per_slot_encoding=teacher_task["use_per_slot_encoding"],
                    use_hand_habitat_features=teacher_task["use_hand_habitat_features"],
                    use_tray_per_slot_encoding=teacher_task["use_tray_per_slot_encoding"],
                    use_opponent_board_encoding=teacher_task["use_opponent_board_encoding"],
                    use_power_features=teacher_task["use_power_features"],
                    fail_on_game_exception=teacher_task["fail_on_game_exception"],
                )

            merged_jsonl = iter_dir / "az_dataset_curriculum_merged.jsonl"
            merged_meta = iter_dir / "az_dataset_curriculum_merged.meta.json"
            ds_meta = _merge_az_shards(
                [
                    {"jsonl": str(dataset_jsonl), "meta": ds_meta},
                    {"jsonl": str(teacher_jsonl), "meta": teacher_meta},
                ],
                out_jsonl=merged_jsonl,
                out_meta=merged_meta,
            )
            ds_meta["teacher_data"] = {
                "games": teacher_games,
                "mcts_sims": int(teacher_mcts_sims),
                "value_blend": float(teacher_value_blend),
                "root_dirichlet_epsilon": float(teacher_root_dirichlet_epsilon),
                "root_dirichlet_alpha": float(teacher_root_dirichlet_alpha),
                "rollout_policy": str(teacher_rollout_policy),
                "samples": int((teacher_meta or {}).get("samples", 0)),
                "mean_score": float((teacher_meta or {}).get("mean_score", 0.0)),
                "game_exceptions": int((teacher_meta or {}).get("game_exceptions", 0)),
            }
            shutil.move(str(merged_jsonl), str(dataset_jsonl))
            shutil.move(str(merged_meta), str(dataset_meta_path))
            # Keep dataset directory small once merged.
            if teacher_jsonl.exists():
                teacher_jsonl.unlink()
            if teacher_meta_path.exists():
                teacher_meta_path.unlink()

            n_samples = int(ds_meta.get("samples", 0))
            mean_score = float(ds_meta.get("mean_score", ds_meta.get("mean_delta", 0.0)))
            print(
                "  Curriculum merge complete: "
                f"samples={n_samples} | mean_score={mean_score:.1f}"
            )

        # Clean up shard directory now that data is merged — shards are large
        # and redundant once az_dataset.jsonl exists.
        shard_dir_cleanup = iter_dir / "shards"
        if shard_dir_cleanup.exists():
            shutil.rmtree(shard_dir_cleanup)
            print(f"  Cleaned up shard directory.")

        # ----------------------------------------------------------------
        # Optional data accumulation (replay buffer)
        # ----------------------------------------------------------------
        train_jsonl = dataset_jsonl
        train_meta = dataset_meta_path

        if data_accumulation_enabled and accumulation_sources:
            combined_jsonl = iter_dir / "az_dataset_combined.jsonl"
            combined_meta_path = iter_dir / "az_dataset_combined.meta.json"

            total_new = n_samples
            budget_old = max_accumulated_samples - total_new
            valid_sources = []
            for src in accumulation_sources:
                src_n = int(src.get("rows", 0))
                src_path = Path(str(src.get("jsonl", "")))
                if src_n <= 0 or not src_path.exists():
                    continue
                valid_sources.append(
                    {
                        "iter": int(src.get("iter", 0)),
                        "rows": src_n,
                        "path": src_path,
                    }
                )

            target_old = min(max(0, budget_old), sum(s["rows"] for s in valid_sources))
            quotas: list[int] = []
            if target_old > 0 and valid_sources:
                if data_accumulation_decay <= 0.0:
                    latest_iter = max(s["iter"] for s in valid_sources)
                    weights = [1.0 if s["iter"] == latest_iter else 0.0 for s in valid_sources]
                else:
                    weights = [
                        float(data_accumulation_decay ** max(0, i - int(s["iter"])))
                        for s in valid_sources
                    ]
                weight_sum = max(1e-9, float(sum(weights)))
                raw = [
                    min(float(s["rows"]), target_old * w / weight_sum)
                    for s, w in zip(valid_sources, weights)
                ]
                quotas = [int(x) for x in raw]
                remaining = target_old - sum(quotas)
                order = sorted(
                    range(len(valid_sources)),
                    key=lambda idx: (raw[idx] - quotas[idx], valid_sources[idx]["iter"]),
                    reverse=True,
                )
                while remaining > 0:
                    progressed = False
                    for idx in order:
                        if quotas[idx] < valid_sources[idx]["rows"]:
                            quotas[idx] += 1
                            remaining -= 1
                            progressed = True
                            if remaining <= 0:
                                break
                    if not progressed:
                        break

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
                for src, take in zip(valid_sources, quotas):
                    if take <= 0:
                        continue
                    src_path = src["path"]
                    total = src["rows"]
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
            print(
                f"  Combined dataset: {written} samples "
                f"(new={n_samples} + replay, decay={data_accumulation_decay:.3f})"
            )
        data_gen_sec = round(time.time() - step_started, 2)

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
        step_started = time.time()
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
            weight_decay=train_weight_decay,
            value_loss_weight=train_value_weight,
            score_value_loss_weight=train_score_value_weight,
            win_value_loss_weight=train_win_value_weight,
            early_stop_enabled=train_early_stop_enabled,
            early_stop_patience=train_early_stop_patience,
            early_stop_min_delta=train_early_stop_min_delta,
            init_model_path=warmstart_path,
            reinit_value_head=train_reinit_value_head and not any(h.get("promoted", False) for h in history),
            seed=iter_seed,
        )
        print(
            f"  Training done: epochs={train_result.get('epochs_completed')} | "
            f"val_loss={train_result.get('best_val_loss', 0.0):.4f}"
        )

        # Clean up combined JSONL now that training is done — it's 2-3 GB and
        # not needed again (replay buffer uses the raw az_dataset.jsonl).
        if data_accumulation_enabled:
            combined_jsonl_cleanup = iter_dir / "az_dataset_combined.jsonl"
            combined_meta_cleanup = iter_dir / "az_dataset_combined.meta.json"
            for _f in (combined_jsonl_cleanup, combined_meta_cleanup):
                if _f.exists():
                    _f.unlink()
            print(f"  Cleaned up combined JSONL (freed ~2-3 GB).")
        train_sec = round(time.time() - step_started, 2)

        # ----------------------------------------------------------------
        # Step 3: Eval vs heuristic
        # ----------------------------------------------------------------
        step_started = time.time()
        eval_mode = "mcts" if eval_use_mcts else "policy"
        print(
            f"\n  [3/4] Evaluating vs heuristic ({eval_games} games, mode={eval_mode}"
            + (f", sims={eval_mcts_sims})..." if eval_use_mcts else ")...")
        )
        eval_exception_count = 0
        try:
            eval_result = evaluate_factorized_vs_heuristic(
                model_path=str(model_path),
                games=eval_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=iter_seed + 1,
                nn_use_mcts=bool(eval_use_mcts),
                nn_mcts_sims=int(eval_mcts_sims),
                nn_c_puct=float(eval_mcts_c_puct),
                nn_value_blend=float(eval_mcts_value_blend),
                nn_rollout_policy=str(eval_mcts_rollout_policy),
                heuristic_policy=str(eval_heuristic_policy),
            )
            eval_summary = _eval_to_summary(eval_result)
            eval_summary["eval_mode"] = eval_mode
            eval_summary["eval_heuristic_policy"] = str(eval_heuristic_policy)
            if eval_use_mcts:
                eval_summary["eval_mcts_sims"] = int(eval_mcts_sims)
                eval_summary["eval_mcts_c_puct"] = float(eval_mcts_c_puct)
                eval_summary["eval_mcts_value_blend"] = float(eval_mcts_value_blend)
                eval_summary["eval_mcts_rollout_policy"] = str(eval_mcts_rollout_policy)
        except Exception as exc:
            print(f"  [warn] eval failed: {exc}")
            eval_exception_count = 1
            eval_summary = {}

        eval_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        print(
            f"  Eval: nn_wins={eval_summary.get('nn_wins')} / "
            f"{eval_summary.get('games')} | "
            f"mean_score={eval_summary.get('nn_mean_score', 0.0):.1f} | "
            f"win_rate={eval_summary.get('nn_win_rate', 0.0):.3f}"
        )
        eval_sec = round(time.time() - step_started, 2)

        # ----------------------------------------------------------------
        # Step 4: Promotion gate (candidate vs champion)
        # ----------------------------------------------------------------
        step_started = time.time()
        print(f"\n  [4/4] Promotion gate ({promotion_games} games, mode={gate_mode})...")
        promoted = False
        gate_summary: dict = {}
        gate_exception_count = 0

        try:
            if gate_mode == "champion" and best_model_path.exists():
                # Stage 1: quick gate; require >=50% to continue.
                stage1_games = max(1, int(gate_stage_games))
                stage2_games = max(0, int(gate_stage_games))
                stage1 = evaluate_nn_vs_nn(
                    model_a_path=str(model_path),       # candidate
                    model_b_path=str(best_model_path),  # champion
                    games=stage1_games,
                    board_type=board_type,
                    mcts_sims=gate_mcts_sims,
                    value_blend=gate_value_blend,
                    rollout_policy=rollout_policy,
                    seed=iter_seed + 2,
                )
                stage1_win_rate = float(stage1.get("nn_win_rate", 0.0))
                stage_parts = [stage1]
                stage2 = None
                if stage1_win_rate >= 0.5 and stage2_games > 0:
                    stage2 = evaluate_nn_vs_nn(
                        model_a_path=str(model_path),
                        model_b_path=str(best_model_path),
                        games=stage2_games,
                        board_type=board_type,
                        mcts_sims=gate_mcts_sims,
                        value_blend=gate_value_blend,
                        rollout_policy=rollout_policy,
                        seed=iter_seed + 3,
                    )
                    stage_parts.append(stage2)
                gate_summary = _combine_gate_summaries(stage_parts)
                gate_summary["stage1"] = stage1
                gate_summary["stage1_continue_threshold"] = 0.5
                gate_summary["stage2_enabled"] = bool(stage2 is not None)
                if stage2 is not None:
                    gate_summary["stage2"] = stage2
            else:
                gate_result = evaluate_factorized_vs_heuristic(
                    model_path=str(model_path),
                    games=promotion_games,
                    board_type=board_type,
                    max_turns=max_turns,
                    seed=iter_seed + 2,
                    heuristic_policy=str(eval_heuristic_policy),
                )
                gate_summary = _eval_to_summary(gate_result)
        except Exception as exc:
            print(f"  [warn] promotion gate eval failed: {exc}")
            gate_exception_count = 1
            gate_summary = {}

        candidate_win_rate = float(gate_summary.get("nn_win_rate", 0.0))
        if int(gate_summary.get("games", 0)) > 0 and "nn_win_rate_ci95" not in gate_summary:
            lo, hi = _wilson_ci95(
                int(gate_summary.get("nn_wins", 0)), int(gate_summary.get("games", 0))
            )
            gate_summary["nn_win_rate_ci95"] = [round(lo, 4), round(hi, 4)]
        if candidate_win_rate >= min_promotion_win_rate:
            promoted = True

        # Also promote if first iteration with no existing champion
        if not best_model_path.exists():
            promoted = True

        gate_summary["promoted"] = promoted
        gate_summary["min_promotion_win_rate"] = min_promotion_win_rate
        gate_summary["gate_value_blend"] = gate_value_blend
        gate_summary["gate_mcts_sims"] = gate_mcts_sims
        gate_path.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")

        print(
            f"  Gate: nn_wins={gate_summary.get('nn_wins')} / "
            f"{gate_summary.get('games')} | "
            f"win_rate={candidate_win_rate:.3f} | "
            f"ci95={gate_summary.get('nn_win_rate_ci95')} | "
            f"promoted={promoted}"
        )
        gate_sec = round(time.time() - step_started, 2)

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

        iter_total_sec = round(time.time() - iter_started, 2)
        timing = {
            "data_gen": data_gen_sec,
            "train": train_sec,
            "eval": eval_sec,
            "gate": gate_sec,
            "iter_total": iter_total_sec,
        }
        exception_counts = {
            "selfplay": int(ds_meta.get("game_exceptions", 0)),
            "eval": int(eval_exception_count),
            "gate": int(gate_exception_count),
        }
        print(
            "  Timing (sec): "
            f"data={timing['data_gen']:.1f} "
            f"train={timing['train']:.1f} "
            f"eval={timing['eval']:.1f} "
            f"gate={timing['gate']:.1f} "
            f"iter={timing['iter_total']:.1f}"
        )

        # ----------------------------------------------------------------
        # Record iteration history
        # ----------------------------------------------------------------
        iter_record = {
            "iter": i,
            "samples": n_samples,
            "mean_delta": float(ds_meta.get("mean_delta", ds_meta.get("mean_score", 0.0))),
            "mean_score": float(ds_meta.get("mean_score", ds_meta.get("mean_delta", 0.0))),
            "teacher_data": dict(ds_meta.get("teacher_data", {}))
            if isinstance(ds_meta.get("teacher_data"), dict)
            else {},
            "train_val_loss": float(train_result.get("best_val_loss", 0.0)),
            "train_val_action_acc": float(
                (train_result.get("history") or [{}])[-1].get("val_action_acc", 0.0)
            ),
            "eval_summary": eval_summary,
            "gate_summary": gate_summary,
            "diagnostics": {
                "policy_entropy_action_type_mean": float(
                    ds_meta.get("policy_entropy_action_type_mean", 0.0)
                ),
                "policy_peak_action_type_mean": float(
                    ds_meta.get("policy_peak_action_type_mean", 0.0)
                ),
                "val_win_value_brier": float(
                    (train_result.get("history") or [{}])[-1].get("val_win_value_brier", 0.0)
                ),
                "val_win_value_log_loss": float(
                    (train_result.get("history") or [{}])[-1].get("val_win_value_log_loss", 0.0)
                ),
                "gate_win_rate_ci95": gate_summary.get("nn_win_rate_ci95"),
            },
            "exception_counts": exception_counts,
            "promoted": promoted,
            "timing_sec": timing,
            "elapsed_sec": round(time.time() - started, 1),
        }
        history.append(iter_record)

        manifest = {
            "pipeline": "auto_improve_alphazero",
            "out_dir": out_dir,
            "iterations_completed": i,
            "iterations_total": iterations,
            "rules_baseline": {
                "id": str(rules_baseline_id),
            },
            "preflight": {
                "bird_audit_gate_enabled": bool(require_bird_audit_gate),
                "expected_semantic_untested": int(expected_semantic_untested),
                "bird_audit_summary": bird_audit_summary or {},
            },
            "mcts_sims": mcts_sims,
            "eval_use_mcts": bool(eval_use_mcts),
            "eval_mcts_sims": int(eval_mcts_sims),
            "eval_mcts_c_puct": float(eval_mcts_c_puct),
            "eval_mcts_value_blend": float(eval_mcts_value_blend),
            "eval_mcts_rollout_policy": str(eval_mcts_rollout_policy),
            "eval_heuristic_policy": str(eval_heuristic_policy),
            "selfplay_value_blend": selfplay_value_blend,
            "selfplay_fail_on_exception": bool(selfplay_fail_on_exception),
            "gate_value_blend": gate_value_blend,
            "gate_mode": gate_mode,
            "gate_mcts_sims": gate_mcts_sims,
            "teacher_games_per_iter": int(teacher_games_per_iter),
            "teacher_mcts_sims": int(teacher_mcts_sims),
            "teacher_value_blend": float(teacher_value_blend),
            "pilot_early_stop_iter": pilot_early_stop_iter,
            "pilot_min_gate_win_rate": pilot_min_gate_win_rate,
            "total_elapsed_sec": round(time.time() - started, 1),
            "history": history,
        }
        if (
            i >= int(pilot_early_stop_iter)
            and not any(h.get("promoted", False) for h in history)
            and candidate_win_rate < float(pilot_min_gate_win_rate)
        ):
            reason = (
                f"pilot_early_stop: iter={i}, no_promotions, "
                f"gate_win_rate={candidate_win_rate:.3f} < {float(pilot_min_gate_win_rate):.3f}"
            )
            manifest["stopped_early"] = True
            manifest["stopped_early_reason"] = reason
            print(f"  [pilot-stop] {reason}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if manifest.get("stopped_early"):
            break

    completed = int(manifest.get("iterations_completed", 0)) if "manifest" in locals() else 0
    print(f"\n{'='*60}")
    print(f"  AlphaZero training complete: {completed}/{iterations} iterations")
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
    p.add_argument(
        "--value-blend",
        type=float,
        default=None,
        help="Legacy alias; applied to both self-play and gate blends when split args are unset.",
    )
    p.add_argument("--selfplay-value-blend", type=float, default=None)
    p.add_argument("--gate-value-blend", type=float, default=None)
    p.add_argument("--selfplay-root-dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--selfplay-root-dirichlet-alpha", type=float, default=0.3)
    p.set_defaults(selfplay_fail_on_exception=True)
    p.add_argument(
        "--selfplay-fail-on-exception",
        dest="selfplay_fail_on_exception",
        action="store_true",
        help="Abort iteration when any self-play game throws (default: enabled).",
    )
    p.add_argument(
        "--selfplay-skip-game-exceptions",
        dest="selfplay_fail_on_exception",
        action="store_false",
        help="Legacy behavior: log per-game self-play exceptions and continue.",
    )
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
    p.add_argument("--train-weight-decay", type=float, default=0.0,
                   help="L2 weight decay for SGD optimizer (default 0)")
    p.add_argument(
        "--train-value-weight",
        type=float,
        default=None,
        help="Deprecated alias for --score-value-weight",
    )
    p.add_argument("--score-value-weight", type=float, default=0.25)
    p.add_argument("--win-value-weight", type=float, default=1.0)
    p.add_argument("--train-init-model-path", default=None)
    p.add_argument(
        "--train-init-first-iter-only",
        action="store_true",
        default=False,
        help="Use init model only for warm-start on iter 1; champion after that",
    )
    p.add_argument(
        "--train-reinit-value-head",
        action="store_true",
        default=False,
        help="On iter 1, reinitialize value head weights after warm-start (clears delta-target confusion)",
    )
    p.add_argument(
        "--no-seed-champion",
        action="store_true",
        default=False,
        help="Do not copy init model to best_model.npz; iter 1 auto-promotes. "
             "Use when you want the init model only for data generation, not as the starting champion.",
    )
    # Eval / promotion
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument(
        "--eval-use-mcts",
        action="store_true",
        default=False,
        help="Evaluate NN vs heuristic using MCTS move selection (aligned with search-time strength).",
    )
    p.add_argument("--eval-mcts-sims", type=int, default=40)
    p.add_argument("--eval-mcts-c-puct", type=float, default=1.5)
    p.add_argument("--eval-mcts-value-blend", type=float, default=None)
    p.add_argument("--eval-mcts-rollout-policy", default=None)
    p.add_argument(
        "--eval-heuristic-policy",
        default="greedy",
        choices=["greedy", "weighted_random"],
        help="Opponent policy for eval/gate heuristic mode. Default is deterministic greedy.",
    )
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
        default=40,
        help="MCTS simulations per player turn in champion gate (default: 40)",
    )
    p.add_argument("--gate-stage-games", type=int, default=20)
    p.add_argument("--pilot-early-stop-iter", type=int, default=3)
    p.add_argument("--pilot-min-gate-win-rate", type=float, default=0.20)
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
    p.add_argument(
        "--teacher-games-per-iter",
        type=int,
        default=0,
        help="Optional high-sim curriculum games to append per iteration (default: off).",
    )
    p.add_argument(
        "--teacher-mcts-sims",
        type=int,
        default=None,
        help="MCTS sims for curriculum teacher games (default: max(2x base, base+16)).",
    )
    p.add_argument(
        "--teacher-value-blend",
        type=float,
        default=None,
        help="Value blend for curriculum teacher games (default: selfplay-value-blend).",
    )
    p.add_argument("--teacher-root-dirichlet-epsilon", type=float, default=0.1)
    p.add_argument("--teacher-root-dirichlet-alpha", type=float, default=0.25)
    p.add_argument("--teacher-rollout-policy", default=None)
    p.add_argument(
        "--data-accumulation-decay",
        type=float,
        default=0.5,
        help="Replay weighting by age (0=latest only, 1=uniform by available rows).",
    )
    p.add_argument(
        "--value-target-score-scale",
        type=float,
        default=DELTA_SCALE,
        help="Normalization scale for value target (default 120 for absolute mode).",
    )
    p.add_argument("--tie-value-target", type=float, default=0.5)
    # Parallelism
    p.add_argument("--dataset-workers", type=int, default=4)
    p.add_argument(
        "--use-modal",
        action="store_true",
        default=False,
        help="Dispatch self-play shard workers to Modal.com instead of local processes. "
             "Requires: pip install modal && modal setup",
    )
    p.add_argument(
        "--modal-cpu-per-worker",
        type=int,
        default=2,
        help="vCPUs to request per Modal container (default: 2)",
    )
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-clean", dest="clean_out_dir", action="store_false", default=True)
    p.add_argument("--start-iter", type=int, default=1,
                   help="Resume from this iteration (skips earlier iters, restores history/replay buffer)")
    p.set_defaults(require_bird_audit_gate=True)
    p.add_argument(
        "--require-bird-audit-gate",
        dest="require_bird_audit_gate",
        action="store_true",
        help="Require semantic bird audit gate before training starts (default: enabled).",
    )
    p.add_argument(
        "--skip-bird-audit-gate",
        dest="require_bird_audit_gate",
        action="store_false",
        help="Skip preflight semantic bird audit gate (not recommended).",
    )
    p.add_argument(
        "--expected-semantic-untested",
        type=int,
        default=0,
        help="Expected semantic_untested_count from bird audit gate.",
    )
    p.add_argument(
        "--rules-baseline-id",
        default=RULES_BASELINE_ID_DEFAULT,
        help="Lineage marker for rules-correct baseline; used to block stale resume.",
    )
    p.add_argument(
        "--allow-stale-lineage",
        action="store_true",
        default=False,
        help="Allow resuming runs whose manifest baseline id does not match rules-baseline-id.",
    )

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
        selfplay_value_blend=args.selfplay_value_blend,
        gate_value_blend=args.gate_value_blend,
        rollout_policy=args.rollout_policy,
        selfplay_root_dirichlet_epsilon=args.selfplay_root_dirichlet_epsilon,
        selfplay_root_dirichlet_alpha=args.selfplay_root_dirichlet_alpha,
        selfplay_fail_on_exception=args.selfplay_fail_on_exception,
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
        train_weight_decay=args.train_weight_decay,
        train_value_weight=args.train_value_weight,
        train_score_value_weight=args.score_value_weight,
        train_win_value_weight=args.win_value_weight,
        train_init_model_path=args.train_init_model_path,
        train_init_first_iter_only=args.train_init_first_iter_only,
        train_reinit_value_head=args.train_reinit_value_head,
        seed_champion_from_init=not args.no_seed_champion,
        eval_games=args.eval_games,
        eval_use_mcts=args.eval_use_mcts,
        eval_mcts_sims=args.eval_mcts_sims,
        eval_mcts_c_puct=args.eval_mcts_c_puct,
        eval_mcts_value_blend=args.eval_mcts_value_blend,
        eval_mcts_rollout_policy=args.eval_mcts_rollout_policy,
        eval_heuristic_policy=args.eval_heuristic_policy,
        promotion_games=args.promotion_games,
        min_promotion_win_rate=args.min_promotion_win_rate,
        gate_mode=args.gate_mode,
        gate_mcts_sims=args.gate_mcts_sims,
        gate_stage_games=args.gate_stage_games,
        pilot_early_stop_iter=args.pilot_early_stop_iter,
        pilot_min_gate_win_rate=args.pilot_min_gate_win_rate,
        state_encoder_use_per_slot=args.use_per_slot_encoding,
        state_encoder_enable_identity=args.enable_identity_features,
        state_encoder_identity_hash_dim=args.identity_hash_dim,
        state_encoder_use_hand_habitat_features=args.use_hand_habitat_features,
        state_encoder_use_tray_per_slot=args.use_tray_per_slot_encoding,
        state_encoder_use_opponent_board=args.use_opponent_board_encoding,
        state_encoder_use_power_features=args.use_power_features,
        data_accumulation_enabled=args.data_accumulation_enabled,
        data_accumulation_decay=args.data_accumulation_decay,
        max_accumulated_samples=args.max_accumulated_samples,
        teacher_games_per_iter=args.teacher_games_per_iter,
        teacher_mcts_sims=args.teacher_mcts_sims,
        teacher_value_blend=args.teacher_value_blend,
        teacher_root_dirichlet_epsilon=args.teacher_root_dirichlet_epsilon,
        teacher_root_dirichlet_alpha=args.teacher_root_dirichlet_alpha,
        teacher_rollout_policy=args.teacher_rollout_policy,
        value_target_score_scale=args.value_target_score_scale,
        tie_value_target=args.tie_value_target,
        dataset_workers=args.dataset_workers,
        use_modal=args.use_modal,
        modal_cpu_per_worker=args.modal_cpu_per_worker,
        seed=args.seed,
        clean_out_dir=args.clean_out_dir,
        start_iter=args.start_iter,
        require_bird_audit_gate=args.require_bird_audit_gate,
        expected_semantic_untested=args.expected_semantic_untested,
        rules_baseline_id=args.rules_baseline_id,
        allow_stale_lineage=args.allow_stale_lineage,
    )

    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "history"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
