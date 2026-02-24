"""Auto-improve loop for factorized BC with strict promotion gating.

Implements:
- Policy improvement data generation (proposal+lookahead)
- BC training with value targets
- 200-game strict promotion gate: promote only if nn_wins > heuristic_wins
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.collect_rl_episodes import collect_rl_episodes
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.evaluate_factorized_pool import evaluate_against_pool
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.kpi_gate import run_gate
from backend.ml.strict_kpi_compare import run_compare as run_strict_kpi_compare
from backend.ml.strict_kpi_runner import run_strict_kpi
from backend.ml.train_factorized_bc import train_bc


def _win_rate_from_eval(eval_dict: dict) -> float:
    games = max(1, int(eval_dict.get("games", 0)))
    return float(eval_dict.get("nn_wins", 0)) / games


def _champion_switch_ready(
    history: list[dict],
    *,
    min_stable_iters: int,
    min_eval_win_rate: float,
) -> bool:
    required = max(0, int(min_stable_iters))
    if required == 0:
        return True
    if len(history) < required:
        return False
    for row in history[-required:]:
        eval_summary = row.get("eval_summary", {})
        if _win_rate_from_eval(eval_summary) < float(min_eval_win_rate):
            return False
    return True


def _robust_selection_metrics(eval_dict: dict) -> dict:
    win_rate = _win_rate_from_eval(eval_dict)
    return {
        "games": int(eval_dict.get("games", 0)),
        "nn_wins": int(eval_dict.get("nn_wins", 0)),
        "nn_win_rate": round(win_rate, 4),
        "nn_mean_score": float(eval_dict.get("nn_mean_score", 0.0)),
        "nn_mean_margin": float(eval_dict.get("nn_mean_margin", 0.0)),
    }


def _robust_is_better(
    candidate: dict,
    incumbent: dict | None,
    *,
    min_win_rate_delta: float = 0.0,
    min_mean_score_delta: float = 0.0,
    max_margin_regression: float = 1.0,
) -> bool:
    if incumbent is None:
        return True
    cand_win = float(candidate.get("nn_win_rate", 0.0))
    cand_score = float(candidate.get("nn_mean_score", 0.0))
    cand_margin = float(candidate.get("nn_mean_margin", 0.0))
    inc_win = float(incumbent.get("nn_win_rate", 0.0))
    inc_score = float(incumbent.get("nn_mean_score", 0.0))
    inc_margin = float(incumbent.get("nn_mean_margin", 0.0))
    has_meaningful_improvement = (
        cand_win >= (inc_win + float(min_win_rate_delta))
        or cand_score >= (inc_score + float(min_mean_score_delta))
    )
    margin_guard_pass = cand_margin >= (inc_margin - float(max_margin_regression))
    if not has_meaningful_improvement or not margin_guard_pass:
        return False
    cand_key = (cand_win, cand_score, cand_margin)
    inc_key = (inc_win, inc_score, inc_margin)
    return cand_key > inc_key


def _sample_jsonl_to_handle(
    src_path: Path,
    dst_handle,
    *,
    sample_count: int,
    total_rows: int,
    rng,
) -> int:
    if sample_count <= 0 or total_rows <= 0:
        return 0

    written = 0
    if sample_count >= total_rows:
        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dst_handle.write(line)
                    written += 1
        return written

    keep_idx = set(rng.sample(range(total_rows), sample_count))
    with src_path.open("r", encoding="utf-8") as f:
        row_idx = 0
        for line in f:
            if not line.strip():
                continue
            if row_idx in keep_idx:
                dst_handle.write(line)
                written += 1
            row_idx += 1
    return written


def _run_dataset_shard(task: dict) -> dict:
    """Worker entrypoint for parallel dataset shard generation."""
    kwargs = dict(task["kwargs"])
    kwargs["out_jsonl"] = task["out_jsonl"]
    kwargs["out_meta"] = task["out_meta"]
    kwargs["games"] = int(task["games"])
    kwargs["seed"] = int(task["seed"])
    meta = generate_bc_dataset(**kwargs)
    return {
        "jsonl": task["out_jsonl"],
        "meta_path": task["out_meta"],
        "meta": meta,
    }


def _merge_dataset_shards(
    shard_results: list[dict],
    *,
    out_jsonl: Path,
    out_meta: Path,
) -> dict:
    """Concatenate shard JSONL files and merge key dataset metadata fields."""
    if not shard_results:
        raise ValueError("No shard results to merge")

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for shard in shard_results:
            with Path(shard["jsonl"]).open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)

    metas = [dict(shard["meta"]) for shard in shard_results]
    base = dict(metas[0])
    total_games = sum(int(m.get("games", 0)) for m in metas)
    total_samples = sum(int(m.get("samples", 0)) for m in metas)
    total_elapsed = sum(float(m.get("elapsed_sec", 0.0)) for m in metas)
    total_strict = sum(int(m.get("strict_games", 0)) for m in metas)
    total_relaxed = sum(int(m.get("relaxed_games", 0)) for m in metas)
    total_rejected = sum(int(m.get("strict_rejected_games", 0)) for m in metas)

    weighted_score_num = sum(float(m.get("mean_player_score", 0.0)) * max(1, int(m.get("games", 0))) for m in metas)
    mean_player_score = weighted_score_num / max(1, total_games)

    base["generated_at_epoch"] = int(time.time())
    base["games"] = int(total_games)
    base["samples"] = int(total_samples)
    base["elapsed_sec"] = round(total_elapsed, 2)
    base["strict_games"] = int(total_strict)
    base["relaxed_games"] = int(total_relaxed)
    base["strict_rejected_games"] = int(total_rejected)
    base["mean_player_score"] = round(mean_player_score, 3)

    policy_keys_to_sum = [
        "engine_teacher_calls",
        "engine_teacher_applied",
        "engine_teacher_miss_fallback_used",
        "move_execute_attempts",
        "move_execute_successes",
        "move_execute_fallback_used",
        "move_execute_dropped",
        "adaptive_depth2_triggered",
        "adaptive_depth2_candidates",
        "hard_replay_games",
        "hard_replay_loss_games",
        "hard_replay_rows_kept",
        "hard_replay_rows_written",
    ]
    pi = dict(base.get("policy_improvement", {}))
    for key in policy_keys_to_sum:
        pi[key] = int(sum(int(m.get("policy_improvement", {}).get(key, 0)) for m in metas))
    base["policy_improvement"] = pi

    out_meta.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


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
    value_target_score_scale: float,
    value_target_score_bias: float,
    late_round_oversample_factor: int,
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
    train_momentum: float = 0.9,
    train_init_model_path: str | None = None,
    train_init_first_iter_only: bool = True,
    train_early_stop_enabled: bool = True,
    train_early_stop_patience: int = 5,
    train_early_stop_min_delta: float = 1e-4,
    train_early_stop_restore_best: bool = True,
    train_value_weight: float = 0.5,
    val_split: float = 0.1,
    eval_games: int = 40,
    promotion_games: int = 80,
    pool_games_per_opponent: int = 40,
    min_pool_win_rate: float = 0.30,
    min_pool_mean_score: float = 0.0,
    min_pool_rate_ge_100: float = 0.0,
    min_pool_rate_ge_120: float = 0.0,
    require_pool_non_regression: bool = False,
    min_gate_win_rate: float = 0.0,
    min_gate_mean_score: float = 0.0,
    min_gate_rate_ge_100: float = 0.0,
    min_gate_rate_ge_120: float = 0.0,
    require_non_negative_champion_margin: bool = True,
    champion_self_play_enabled: bool = False,
    champion_switch_after_first_promotion: bool = True,
    champion_switch_min_stable_iters: int = 2,
    champion_switch_min_eval_win_rate: float = 0.5,
    champion_switch_min_iteration: int = 3,
    champion_teacher_source: str = "engine_only",
    champion_engine_time_budget_ms: int = 50,
    champion_engine_num_determinizations: int = 8,
    champion_engine_max_rollout_depth: int = 24,
    promotion_primary_opponent: str = "heuristic",
    frozen_opponent_model_path: str | None = None,
    best_selection_eval_games: int = 0,
    proposal_update_policy: str = "best_only",
    selection_requires_gate_pass: bool = True,
    best_min_win_rate_delta: float = 0.02,
    best_min_mean_score_delta: float = 0.5,
    best_max_margin_regression: float = 1.0,
    fixed_benchmark_seeds: bool = True,
    benchmark_seed: int = 13_371_337,
    engine_teacher_prob: float = 0.30,
    engine_time_budget_ms: int = 25,
    engine_num_determinizations: int = 0,
    engine_max_rollout_depth: int = 24,
    adaptive_teacher_depth2_enabled: bool = True,
    adaptive_teacher_uncertainty_gap: float = 2.0,
    adaptive_teacher_top_m: int = 2,
    hard_replay_games: int = 0,
    hard_replay_player: str = "proposal",
    hard_replay_target_player: int = 0,
    hard_replay_loss_oversample_factor: int = 3,
    hard_replay_only_losses: bool = True,
    rollback_on_fixed_gate_regression: bool = True,
    fixed_gate_regression_max_drop: float = 0.10,
    stop_after_consecutive_non_eligible_iters: bool = True,
    max_consecutive_non_eligible_iters: int = 4,
    seed: int = 0,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    strict_curriculum_enabled: bool = True,
    strict_fraction_start: float = 1.0,
    strict_fraction_end: float = 0.0,
    strict_fraction_warmup_iters: int = 5,
    strict_kpi_gate_enabled: bool = True,
    strict_kpi_baseline_path: str = "reports/ml/baselines/strict_kpi_baseline.json",
    strict_kpi_round1_games: int = 20,
    strict_kpi_smoke_games: int = 20,
    strict_kpi_max_turns: int = 120,
    strict_kpi_min_round1_pct_15_30: float = 0.2,
    strict_kpi_max_round1_strict_rejected_games: int = 0,
    strict_kpi_max_smoke_strict_rejected_games: int = 0,
    strict_kpi_min_strict_certified_birds: int = 50,
    strict_kpi_require_non_regression: bool = True,
    strict_kpi_mean_score_tolerance: float = 3.0,
    state_encoder_enable_identity: bool = False,
    state_encoder_identity_hash_dim: int = 128,
    state_encoder_use_per_slot: bool = False,
    state_encoder_max_hand_slots: int = 8,
    require_fixed_seed_eval_gate_pass: bool = False,
    fixed_seed_eval_gate_games: int = 40,
    fixed_seed_eval_gate_min_win_rate: float = 0.43,
    data_accumulation_enabled: bool = True,
    data_accumulation_decay: float = 0.5,
    max_accumulated_samples: int = 200_000,
    dataset_workers: int = 4,
    clean_out_dir: bool = True,
    train_hidden: int | None = None,
    train_lr: float | None = None,
    rl_games_per_iter: int = 0,
    rl_temperature: float = 1.5,
    rl_use_value_baseline: bool = True,
    rl_warm_start_path: str = "",
) -> dict:
    if promotion_primary_opponent not in ("heuristic", "champion", "frozen"):
        raise ValueError("promotion_primary_opponent must be 'heuristic', 'champion', or 'frozen'")
    if champion_teacher_source not in ("probabilistic_engine", "engine_only"):
        raise ValueError("champion_teacher_source must be 'probabilistic_engine' or 'engine_only'")
    if proposal_update_policy not in {"best_only", "latest_candidate"}:
        raise ValueError("proposal_update_policy must be 'best_only' or 'latest_candidate'")
    if hard_replay_player not in {"proposal", "best", "frozen"}:
        raise ValueError("hard_replay_player must be 'proposal', 'best', or 'frozen'")

    deprecated_train_args_used: list[str] = []
    if train_hidden is not None:
        train_hidden1 = int(train_hidden)
        train_hidden2 = int(train_hidden)
        deprecated_train_args_used.append("train_hidden")
    if train_lr is not None:
        train_lr_peak = float(train_lr)
        deprecated_train_args_used.append("train_lr")
    if deprecated_train_args_used:
        print(
            f"warning: deprecated auto-improve train args used: {', '.join(deprecated_train_args_used)}"
        )
    if train_init_model_path is not None:
        init_model_cand = Path(train_init_model_path)
        if not init_model_cand.exists():
            raise FileNotFoundError(f"train_init_model_path does not exist: {train_init_model_path}")
        train_init_model_path = str(init_model_cand)

    champion_switch_min_stable_iters = max(0, int(champion_switch_min_stable_iters))
    champion_switch_min_iteration = max(1, int(champion_switch_min_iteration))
    champion_switch_min_eval_win_rate = max(0.0, min(1.0, float(champion_switch_min_eval_win_rate)))
    best_selection_eval_games = int(best_selection_eval_games)
    if best_selection_eval_games <= 0:
        best_selection_eval_games = max(1, int(promotion_games), int(eval_games))
    else:
        best_selection_eval_games = max(1, best_selection_eval_games)

    data_accumulation_decay = max(0.0, min(1.0, float(data_accumulation_decay)))
    max_accumulated_samples = max(1, int(max_accumulated_samples))
    dataset_workers = max(1, int(dataset_workers))
    best_min_win_rate_delta = max(0.0, float(best_min_win_rate_delta))
    best_min_mean_score_delta = max(0.0, float(best_min_mean_score_delta))
    best_max_margin_regression = max(0.0, float(best_max_margin_regression))
    benchmark_seed = int(benchmark_seed)
    adaptive_teacher_uncertainty_gap = max(0.0, float(adaptive_teacher_uncertainty_gap))
    adaptive_teacher_top_m = max(2, int(adaptive_teacher_top_m))
    hard_replay_games = max(0, int(hard_replay_games))
    hard_replay_target_player = max(0, int(hard_replay_target_player))
    hard_replay_loss_oversample_factor = max(1, int(hard_replay_loss_oversample_factor))
    fixed_gate_regression_max_drop = max(0.0, float(fixed_gate_regression_max_drop))
    state_encoder_identity_hash_dim = max(1, int(state_encoder_identity_hash_dim))
    state_encoder_max_hand_slots = max(1, int(state_encoder_max_hand_slots))
    # Auto-scale hidden sizes when per-slot encoding is used and still at defaults
    if state_encoder_use_per_slot and train_hidden1 == 384 and train_hidden2 == 192:
        train_hidden1, train_hidden2 = 768, 384
        print("auto_improve: per-slot encoding → auto-scaled hidden1=768 hidden2=384")
    fixed_seed_eval_gate_games = max(1, int(fixed_seed_eval_gate_games))
    fixed_seed_eval_gate_min_win_rate = max(0.0, min(1.0, float(fixed_seed_eval_gate_min_win_rate)))
    max_consecutive_non_eligible_iters = max(1, int(max_consecutive_non_eligible_iters))
    if frozen_opponent_model_path is not None:
        frozen_path = Path(frozen_opponent_model_path)
        if not frozen_path.exists():
            raise FileNotFoundError(f"frozen_opponent_model_path does not exist: {frozen_opponent_model_path}")
        frozen_opponent_model_path = str(frozen_path)
    if promotion_primary_opponent == "frozen" and not frozen_opponent_model_path:
        raise ValueError("promotion_primary_opponent='frozen' requires --frozen-opponent-model-path")
    rl_games_per_iter = max(0, int(rl_games_per_iter))
    rl_temperature = max(0.01, float(rl_temperature))
    if rl_warm_start_path:
        ws_path = Path(rl_warm_start_path)
        if not ws_path.exists():
            raise FileNotFoundError(f"rl_warm_start_path does not exist: {rl_warm_start_path}")
        rl_warm_start_path = str(ws_path)
    else:
        rl_warm_start_path = ""
    auto_train_init_from_frozen = (
        train_init_model_path is None
        and bool(frozen_opponent_model_path)
        and bool(state_encoder_enable_identity)
        and not state_encoder_use_per_slot
    )
    if auto_train_init_from_frozen:
        train_init_model_path = str(frozen_opponent_model_path)
    # Multiprocessing can deadlock in some pytest harnesses; force single-worker
    # mode under tests for deterministic CI behavior.
    if os.getenv("PYTEST_CURRENT_TEST"):
        dataset_workers = 1
    main_mod = sys.modules.get("__main__")
    if not getattr(main_mod, "__file__", None):
        # Interactive stdin/REPL execution is not spawn-safe.
        dataset_workers = 1

    base = Path(out_dir)
    if clean_out_dir and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / "auto_improve_factorized_manifest.json"
    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_dataset.meta.json"

    best: dict | None = None
    best_selection_metrics: dict | None = None
    history: list[dict] = []
    started = time.time()

    proposal_model_path: str | None = None
    if best_model_path.exists():
        proposal_model_path = str(best_model_path)
    if proposal_model_path is None and frozen_opponent_model_path and not state_encoder_enable_identity and not state_encoder_use_per_slot:
        # Optional warm start from a frozen teacher checkpoint when feature dims match.
        proposal_model_path = str(frozen_opponent_model_path)
    # RL warm-start: if a seed model is given and no proposal is available yet,
    # use it so RL episode collection has a reasonable policy from iteration 1.
    if proposal_model_path is None and rl_warm_start_path:
        proposal_model_path = rl_warm_start_path
    accumulation_sources: list[dict] = []
    consecutive_non_eligible_iters = 0
    stopped_early = False
    stop_reason: str | None = None
    prev_fixed_gate_win_rate: float | None = None

    for i in range(1, iterations + 1):
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        if strict_curriculum_enabled and strict_rules_only:
            warm = max(1, int(strict_fraction_warmup_iters))
            start = max(0.0, min(1.0, float(strict_fraction_start)))
            end = max(0.0, min(1.0, float(strict_fraction_end)))
            if i <= warm:
                strict_game_fraction_iter = start
            else:
                rem = max(1, iterations - warm)
                t = min(1.0, max(0.0, float(i - warm) / float(rem)))
                strict_game_fraction_iter = start + t * (end - start)
        else:
            strict_game_fraction_iter = 1.0 if strict_rules_only else 0.0

        dataset_jsonl = iter_dir / "bc_dataset.jsonl"
        dataset_meta = iter_dir / "bc_dataset.meta.json"
        model_path = iter_dir / "factorized_model.npz"
        eval_path = iter_dir / "eval.json"
        gate_path = iter_dir / "promotion_gate_eval.json"
        pool_eval_path = iter_dir / "pool_eval.json"
        kpi_gate_path = iter_dir / "kpi_gate.json"
        strict_kpi_candidate_path = iter_dir / "strict_kpi_candidate.json"
        strict_kpi_compare_path = iter_dir / "strict_kpi_compare.json"
        combined_dataset_jsonl = iter_dir / "bc_dataset_combined.jsonl"
        combined_dataset_meta = iter_dir / "bc_dataset_combined.meta.json"

        champion_switch_is_ready = _champion_switch_ready(
            history,
            min_stable_iters=champion_switch_min_stable_iters,
            min_eval_win_rate=champion_switch_min_eval_win_rate,
        )
        # Keep data generation anchored to heuristic/mixed play.
        # Champion NN-vs-NN self-play is intentionally disabled to prevent
        # teacher drift and collapse after promotion.
        use_champion_mode = False
        generation_mode = "champion_nn_vs_nn" if use_champion_mode else "bootstrap_mixed"
        iter_proposal_model_path = str(best_model_path) if use_champion_mode else proposal_model_path
        iter_opponent_model_path = str(best_model_path) if use_champion_mode else None

        dataset_common_kwargs = dict(
            players=players,
            board_type=board_type,
            max_turns=max_turns,
            proposal_model_path=iter_proposal_model_path,
            opponent_model_path=iter_opponent_model_path,
            self_play_policy="champion_nn_vs_nn" if use_champion_mode else "mixed",
            proposal_top_k=proposal_top_k,
            lookahead_depth=lookahead_depth,
            n_step=n_step,
            gamma=gamma,
            bootstrap_mix=bootstrap_mix,
            value_target_score_scale=value_target_score_scale,
            value_target_score_bias=value_target_score_bias,
            late_round_oversample_factor=late_round_oversample_factor,
            engine_teacher_prob=engine_teacher_prob if not use_champion_mode else 1.0,
            teacher_source=champion_teacher_source if use_champion_mode else "probabilistic_engine",
            engine_time_budget_ms=champion_engine_time_budget_ms if use_champion_mode else engine_time_budget_ms,
            engine_num_determinizations=champion_engine_num_determinizations if use_champion_mode else engine_num_determinizations,
            engine_max_rollout_depth=champion_engine_max_rollout_depth if use_champion_mode else engine_max_rollout_depth,
            adaptive_teacher_depth2_enabled=adaptive_teacher_depth2_enabled,
            adaptive_teacher_uncertainty_gap=adaptive_teacher_uncertainty_gap,
            adaptive_teacher_top_m=adaptive_teacher_top_m,
            strict_rules_only=strict_rules_only,
            reject_non_strict_powers=reject_non_strict_powers,
            strict_game_fraction=strict_game_fraction_iter,
            state_encoder_enable_identity=state_encoder_enable_identity,
            state_encoder_identity_hash_dim=state_encoder_identity_hash_dim,
            state_encoder_use_per_slot=state_encoder_use_per_slot,
            state_encoder_max_hand_slots=state_encoder_max_hand_slots,
        )

        if dataset_workers <= 1 or games_per_iter <= 1:
            ds_meta = generate_bc_dataset(
                out_jsonl=str(dataset_jsonl),
                out_meta=str(dataset_meta),
                games=games_per_iter,
                seed=iter_seed,
                **dataset_common_kwargs,
            )
        else:
            shard_dir = iter_dir / "dataset_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            worker_count = min(dataset_workers, games_per_iter)
            base_games = games_per_iter // worker_count
            extras = games_per_iter % worker_count
            shard_tasks: list[dict] = []
            for shard_idx in range(worker_count):
                shard_games = base_games + (1 if shard_idx < extras else 0)
                if shard_games <= 0:
                    continue
                shard_jsonl = shard_dir / f"bc_dataset_{shard_idx:02d}.jsonl"
                shard_meta = shard_dir / f"bc_dataset_{shard_idx:02d}.meta.json"
                shard_tasks.append(
                    {
                        "out_jsonl": str(shard_jsonl),
                        "out_meta": str(shard_meta),
                        "games": int(shard_games),
                        "seed": int(iter_seed + shard_idx * 100_003),
                        "kwargs": dataset_common_kwargs,
                    }
                )

            if len(shard_tasks) <= 1:
                task = shard_tasks[0]
                ds_meta = generate_bc_dataset(
                    out_jsonl=task["out_jsonl"],
                    out_meta=task["out_meta"],
                    games=task["games"],
                    seed=task["seed"],
                    **task["kwargs"],
                )
                shutil.copy2(task["out_jsonl"], dataset_jsonl)
                shutil.copy2(task["out_meta"], dataset_meta)
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=len(shard_tasks)) as pool:
                    shard_results = pool.map(_run_dataset_shard, shard_tasks)
                ds_meta = _merge_dataset_shards(
                    shard_results,
                    out_jsonl=dataset_jsonl,
                    out_meta=dataset_meta,
                )

        hard_replay_player_model_path: str | None = None
        if hard_replay_player == "proposal":
            hard_replay_player_model_path = iter_proposal_model_path
        elif hard_replay_player == "best":
            hard_replay_player_model_path = str(best_model_path) if best_model_path.exists() else iter_proposal_model_path
        elif hard_replay_player == "frozen":
            hard_replay_player_model_path = str(frozen_opponent_model_path) if frozen_opponent_model_path else iter_proposal_model_path

        hard_replay_summary = {
            "enabled": bool(hard_replay_games > 0 and bool(hard_replay_player_model_path)),
            "games": 0,
            "samples_added": 0,
            "player": hard_replay_player,
            "player_model_path": hard_replay_player_model_path,
            "target_player": int(hard_replay_target_player),
            "loss_oversample_factor": int(hard_replay_loss_oversample_factor),
            "only_losses": bool(hard_replay_only_losses),
        }
        if hard_replay_games > 0 and hard_replay_player_model_path:
            hard_jsonl = iter_dir / "bc_dataset_hard.jsonl"
            hard_meta_path = iter_dir / "bc_dataset_hard.meta.json"
            hard_kwargs = dict(dataset_common_kwargs)
            hard_kwargs.update(
                {
                    "proposal_model_path": hard_replay_player_model_path,
                    "opponent_model_path": None,
                    "self_play_policy": "proposal_vs_heuristic",
                    "hard_replay_enabled": True,
                    "hard_replay_player": hard_replay_player,
                    "hard_replay_target_player": hard_replay_target_player,
                    "hard_replay_loss_oversample_factor": hard_replay_loss_oversample_factor,
                    "hard_replay_only_losses": hard_replay_only_losses,
                }
            )
            hard_meta = generate_bc_dataset(
                out_jsonl=str(hard_jsonl),
                out_meta=str(hard_meta_path),
                games=hard_replay_games,
                seed=iter_seed + 888_881,
                **hard_kwargs,
            )
            hard_pi = dict(hard_meta.get("policy_improvement", {}))

            appended = 0
            with dataset_jsonl.open("a", encoding="utf-8") as out_f:
                with hard_jsonl.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if not line.strip():
                            continue
                        out_f.write(line)
                        appended += 1

            main_games = int(ds_meta.get("games", 0))
            hard_games_count = int(hard_meta.get("games", 0))
            total_games = max(1, main_games + hard_games_count)
            ds_meta["samples"] = int(ds_meta.get("samples", 0)) + int(appended)
            ds_meta["games"] = int(main_games + hard_games_count)
            ds_meta["mean_player_score"] = round(
                (
                    float(ds_meta.get("mean_player_score", 0.0)) * main_games
                    + float(hard_meta.get("mean_player_score", 0.0)) * hard_games_count
                )
                / total_games,
                3,
            )
            ds_meta["hard_replay"] = {
                "enabled": True,
                "dataset": str(hard_jsonl),
                "dataset_meta": str(hard_meta_path),
                "games": int(hard_games_count),
                "samples_added": int(appended),
                "player": hard_replay_player,
                "player_model_path": hard_replay_player_model_path,
                "target_player": int(hard_replay_target_player),
                "loss_oversample_factor": int(hard_replay_loss_oversample_factor),
                "only_losses": bool(hard_replay_only_losses),
                "loss_games": int(hard_pi.get("hard_replay_loss_games", 0)),
                "rows_kept": int(hard_pi.get("hard_replay_rows_kept", 0)),
                "rows_written": int(hard_pi.get("hard_replay_rows_written", 0)),
            }
            dataset_meta.write_text(json.dumps(ds_meta, indent=2), encoding="utf-8")
            hard_replay_summary = dict(ds_meta["hard_replay"])

        # ------------------------------------------------------------------ #
        # RL episode generation (REINFORCE)                                   #
        # ------------------------------------------------------------------ #
        rl_summary: dict = {
            "enabled": bool(rl_games_per_iter > 0),
            "games": 0,
            "samples_added": 0,
            "temperature": float(rl_temperature),
            "use_value_baseline": bool(rl_use_value_baseline),
            "warm_start_path": rl_warm_start_path,
        }
        rl_proposal = proposal_model_path
        if rl_games_per_iter > 0:
            if rl_proposal is None:
                print(
                    f"  RL: skipping episode collection (no proposal model available at iter {i})"
                )
            else:
                rl_out = iter_dir / "rl_episodes.jsonl"
                rl_meta = collect_rl_episodes(
                    model_path=rl_proposal,
                    n_games=rl_games_per_iter,
                    out_path=str(rl_out),
                    board_type=board_type.value,
                    players=players,
                    temperature=rl_temperature,
                    use_value_baseline=rl_use_value_baseline,
                    seed=iter_seed + 9999,
                    max_turns=max_turns,
                )
                # Append RL rows to the current iteration's BC dataset so they
                # participate in data accumulation and training together.
                appended_rl = 0
                with dataset_jsonl.open("a", encoding="utf-8") as f_bc:
                    with rl_out.open("r", encoding="utf-8") as f_rl:
                        for line in f_rl:
                            if line.strip():
                                f_bc.write(line)
                                appended_rl += 1
                ds_meta["samples"] = int(ds_meta.get("samples", 0)) + appended_rl
                rl_summary = {
                    "enabled": True,
                    "games": int(rl_meta.get("games", 0)),
                    "samples_added": appended_rl,
                    "temperature": float(rl_temperature),
                    "use_value_baseline": bool(rl_use_value_baseline),
                    "warm_start_path": rl_warm_start_path,
                    "mean_score": float(rl_meta.get("mean_score", 0.0)),
                    "adv_std_raw": float(rl_meta.get("adv_std_raw", 0.0)),
                    "dataset": str(rl_out),
                }
                dataset_meta.write_text(json.dumps(ds_meta, indent=2), encoding="utf-8")
                print(
                    f"  RL: {appended_rl} samples from {rl_games_per_iter} games, "
                    f"mean_score={rl_meta.get('mean_score', 0.0):.1f} "
                    f"adv_std={rl_meta.get('adv_std_raw', 0.0):.2f}"
                )

        accumulation_sources.append(
            {
                "iteration": i,
                "jsonl": str(dataset_jsonl),
                "meta": str(dataset_meta),
                "samples": int(ds_meta.get("samples", 0)),
            }
        )

        train_dataset_jsonl = dataset_jsonl
        train_dataset_meta = dataset_meta
        accumulation_summary = {
            "enabled": bool(data_accumulation_enabled),
            "decay": float(data_accumulation_decay),
            "max_accumulated_samples": int(max_accumulated_samples),
            "sources": [],
            "target_total": int(ds_meta.get("samples", 0)),
            "combined_samples": int(ds_meta.get("samples", 0)),
        }

        if data_accumulation_enabled and accumulation_sources:
            valid_sources = [s for s in accumulation_sources if int(s.get("samples", 0)) > 0]
            total_available = sum(int(s["samples"]) for s in valid_sources)
            target_total = min(int(max_accumulated_samples), total_available)

            if target_total > 0 and valid_sources:
                weights = []
                for src in valid_sources:
                    age = max(0, i - int(src["iteration"]))
                    weights.append(float(data_accumulation_decay ** age))
                weight_sum = max(1e-9, float(sum(weights)))
                raw = [
                    min(float(src["samples"]), (target_total * w / weight_sum))
                    for src, w in zip(valid_sources, weights)
                ]
                quotas = [int(x) for x in raw]
                remaining = target_total - sum(quotas)
                # Fill remaining quota by fractional share, then by younger source.
                order = sorted(
                    range(len(valid_sources)),
                    key=lambda idx: (raw[idx] - quotas[idx], -int(valid_sources[idx]["iteration"])),
                    reverse=True,
                )
                while remaining > 0:
                    progressed = False
                    for idx in order:
                        cap = int(valid_sources[idx]["samples"])
                        if quotas[idx] < cap:
                            quotas[idx] += 1
                            remaining -= 1
                            progressed = True
                            if remaining <= 0:
                                break
                    if not progressed:
                        break

                rng = random.Random(iter_seed + 24680)
                written_total = 0
                src_rows = []
                with combined_dataset_jsonl.open("w", encoding="utf-8") as out_f:
                    for src, w, q in zip(valid_sources, weights, quotas):
                        if q <= 0:
                            continue
                        wrote = _sample_jsonl_to_handle(
                            Path(str(src["jsonl"])),
                            out_f,
                            sample_count=int(q),
                            total_rows=int(src["samples"]),
                            rng=rng,
                        )
                        written_total += wrote
                        src_rows.append(
                            {
                                "iteration": int(src["iteration"]),
                                "dataset": str(src["jsonl"]),
                                "samples_available": int(src["samples"]),
                                "weight": round(float(w), 6),
                                "quota": int(q),
                                "written": int(wrote),
                            }
                        )

                combined_meta_obj = dict(ds_meta)
                combined_meta_obj["samples"] = int(written_total)
                combined_meta_obj["accumulation"] = {
                    "enabled": True,
                    "decay": float(data_accumulation_decay),
                    "max_accumulated_samples": int(max_accumulated_samples),
                    "target_total": int(target_total),
                    "combined_samples": int(written_total),
                    "total_available": int(total_available),
                    "sources": src_rows,
                }
                combined_dataset_meta.write_text(
                    json.dumps(combined_meta_obj, indent=2),
                    encoding="utf-8",
                )
                train_dataset_jsonl = combined_dataset_jsonl
                train_dataset_meta = combined_dataset_meta
                accumulation_summary = dict(combined_meta_obj["accumulation"])

        train_init_model_path_iter = None
        if (not train_init_first_iter_only) or i == 1:
            if best_model_path.exists():
                train_init_model_path_iter = str(best_model_path)
            elif train_init_model_path:
                train_init_model_path_iter = str(train_init_model_path)
            elif frozen_opponent_model_path:
                train_init_model_path_iter = str(frozen_opponent_model_path)

        tr = train_bc(
            dataset_jsonl=str(train_dataset_jsonl),
            meta_json=str(train_dataset_meta),
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
            init_model_path=train_init_model_path_iter,
            early_stop_enabled=train_early_stop_enabled,
            early_stop_patience=train_early_stop_patience,
            early_stop_min_delta=train_early_stop_min_delta,
            early_stop_restore_best=train_early_stop_restore_best,
            val_split=val_split,
            seed=iter_seed,
            value_loss_weight=train_value_weight,
        )

        eval_seed = benchmark_seed + 11 if fixed_benchmark_seeds else iter_seed
        gate_seed = benchmark_seed + 22 if fixed_benchmark_seeds else (iter_seed + 777)
        pool_seed = benchmark_seed + 33 if fixed_benchmark_seeds else (iter_seed + 999)
        robust_seed = benchmark_seed + 44 if fixed_benchmark_seeds else (iter_seed + 31337)

        ev = evaluate_factorized_vs_heuristic(
            model_path=str(model_path),
            games=eval_games,
            board_type=board_type,
            max_turns=max_turns,
            seed=eval_seed,
            proposal_top_k=proposal_top_k,
        )
        ev_dict = ev.__dict__
        eval_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

        use_champion_primary_gate = (
            promotion_primary_opponent == "champion"
            and best_model_path.exists()
        )
        use_frozen_primary_gate = (
            promotion_primary_opponent == "frozen"
            and bool(frozen_opponent_model_path)
        )
        if use_champion_primary_gate or use_frozen_primary_gate:
            primary_opp_spec = str(best_model_path) if use_champion_primary_gate else str(frozen_opponent_model_path)
            primary_label = "champion" if use_champion_primary_gate else "frozen"
            primary_gate_eval = evaluate_against_pool(
                model_path=str(model_path),
                opponents=[primary_opp_spec],
                games_per_opponent=promotion_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=gate_seed,
                proposal_top_k=proposal_top_k,
            )
            summary = primary_gate_eval["summary"]
            by_opp = primary_gate_eval.get("by_opponent", [])
            opp_mean = float(by_opp[0]["opponent_mean_score"]) if by_opp else 0.0
            gate_dict = {
                "games": int(summary.get("games", 0)),
                "nn_wins": int(summary.get("nn_wins", 0)),
                "opponent_wins": int(summary.get("opponent_wins", 0)),
                "ties": int(summary.get("ties", 0)),
                "nn_mean_score": float(summary.get("nn_mean_score", 0.0)),
                "opponent_mean_score": opp_mean,
                "nn_mean_margin": float(summary.get("nn_mean_margin", 0.0)),
                "nn_rate_ge_100": float(summary.get("nn_rate_ge_100", 0.0)),
                "nn_rate_ge_120": float(summary.get("nn_rate_ge_120", 0.0)),
                "primary_opponent": primary_label,
                "opponent_spec": primary_opp_spec,
            }
        else:
            gate = evaluate_factorized_vs_heuristic(
                model_path=str(model_path),
                games=promotion_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=gate_seed,
                proposal_top_k=proposal_top_k,
            )
            gate_dict = gate.__dict__
            gate_dict["opponent_wins"] = int(gate_dict.get("heuristic_wins", 0))
            gate_dict["opponent_mean_score"] = float(gate_dict.get("heuristic_mean_score", 0.0))
            gate_dict["primary_opponent"] = "heuristic"
            gate_dict["opponent_spec"] = "heuristic"
        gate_path.write_text(json.dumps(gate_dict, indent=2), encoding="utf-8")

        # Opponent-pool evaluation: heuristic + frozen teacher or current best.
        opponents = ["heuristic"]
        if frozen_opponent_model_path:
            opponents.append(str(frozen_opponent_model_path))
        elif best_model_path.exists():
            opponents.append(str(best_model_path))
        pool_eval = evaluate_against_pool(
            model_path=str(model_path),
            opponents=opponents,
            games_per_opponent=pool_games_per_opponent,
            board_type=board_type,
            max_turns=max_turns,
            seed=pool_seed,
            proposal_top_k=proposal_top_k,
        )
        pool_eval_path.write_text(json.dumps(pool_eval, indent=2), encoding="utf-8")

        baseline_for_gate = None
        if best is not None:
            # Use last accepted best pool eval, if available.
            best_iter = int(best["iteration"])
            cand = base / f"iter_{best_iter:03d}" / "pool_eval.json"
            if cand.exists():
                baseline_for_gate = str(cand)

        kpi_gate = run_gate(
            candidate_path=str(pool_eval_path),
            baseline_path=baseline_for_gate,
            min_win_rate=min_pool_win_rate,
            min_mean_score=min_pool_mean_score,
            min_rate_ge_100=min_pool_rate_ge_100,
            min_rate_ge_120=min_pool_rate_ge_120,
            require_non_regression=require_pool_non_regression and baseline_for_gate is not None,
        )
        kpi_gate_path.write_text(json.dumps(kpi_gate, indent=2), encoding="utf-8")

        strict_kpi_compare: dict
        if strict_kpi_gate_enabled:
            run_strict_kpi(
                out_path=str(strict_kpi_candidate_path),
                players=players,
                board_type=board_type,
                round1_games=strict_kpi_round1_games,
                smoke_games=strict_kpi_smoke_games,
                max_turns=strict_kpi_max_turns,
                seed=iter_seed + 4242,
                min_round1_pct_15_30=strict_kpi_min_round1_pct_15_30,
                max_round1_strict_rejected_games=strict_kpi_max_round1_strict_rejected_games,
                max_smoke_strict_rejected_games=strict_kpi_max_smoke_strict_rejected_games,
            )
            strict_kpi_compare = run_strict_kpi_compare(
                candidate_path=str(strict_kpi_candidate_path),
                baseline_path=str(strict_kpi_baseline_path),
                min_round1_pct_15_30=strict_kpi_min_round1_pct_15_30,
                max_round1_strict_rejected_games=strict_kpi_max_round1_strict_rejected_games,
                max_smoke_strict_rejected_games=strict_kpi_max_smoke_strict_rejected_games,
                min_strict_certified_birds=strict_kpi_min_strict_certified_birds,
                require_non_regression=strict_kpi_require_non_regression,
                mean_score_tolerance=strict_kpi_mean_score_tolerance,
            )
            strict_kpi_compare_path.write_text(json.dumps(strict_kpi_compare, indent=2), encoding="utf-8")
        else:
            strict_kpi_compare = {
                "passed": True,
                "skipped": True,
                "reason": "strict_kpi_gate_disabled",
            }

        gate_games = max(1, int(gate_dict.get("games", 0)))
        gate_win_rate = float(gate_dict.get("nn_wins", 0)) / gate_games
        gate_mean_score = float(gate_dict.get("nn_mean_score", 0.0))
        gate_ge100 = float(gate_dict.get("nn_rate_ge_100", 0.0))
        gate_ge120 = float(gate_dict.get("nn_rate_ge_120", 0.0))

        gate_primary_pass = (
            gate_mean_score >= min_gate_mean_score
            and gate_ge100 >= min_gate_rate_ge_100
            and gate_ge120 >= min_gate_rate_ge_120
        )
        gate_secondary_pass = gate_win_rate >= min_gate_win_rate
        champion_margin_pass = True
        if (use_champion_primary_gate or use_frozen_primary_gate) and require_non_negative_champion_margin:
            champion_margin_pass = float(gate_dict.get("nn_mean_margin", 0.0)) >= 0.0
        promoted = (
            gate_primary_pass
            and gate_secondary_pass
            and champion_margin_pass
            and bool(kpi_gate["passed"])
            and bool(strict_kpi_compare.get("passed", False))
        )

        # Robust best-model selector: use a larger eval to choose best, not
        # promotion status alone.
        robust_eval = evaluate_factorized_vs_heuristic(
            model_path=str(model_path),
            games=max(1, int(best_selection_eval_games)),
            board_type=board_type,
            max_turns=max_turns,
            seed=robust_seed,
            proposal_top_k=proposal_top_k,
        )
        robust_metrics = _robust_selection_metrics(robust_eval.__dict__)
        gate_pass_for_selection = gate_primary_pass and gate_secondary_pass and champion_margin_pass
        fixed_seed_eval_gate: dict = {
            "enabled": bool(require_fixed_seed_eval_gate_pass),
            "seed": int(benchmark_seed + 55 if fixed_benchmark_seeds else (iter_seed + 55555)),
            "games_per_opponent": int(fixed_seed_eval_gate_games),
            "min_win_rate": float(fixed_seed_eval_gate_min_win_rate),
            "opponents": [str(frozen_opponent_model_path)] if frozen_opponent_model_path else ["heuristic"],
            "win_rate": 0.0,
            "passed": True,
            "summary": None,
        }
        if require_fixed_seed_eval_gate_pass:
            fixed_gate_eval = evaluate_against_pool(
                model_path=str(model_path),
                opponents=fixed_seed_eval_gate["opponents"],
                games_per_opponent=fixed_seed_eval_gate_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=fixed_seed_eval_gate["seed"],
                proposal_top_k=proposal_top_k,
            )
            fixed_summary = fixed_gate_eval.get("summary", {})
            fixed_wr = float(fixed_summary.get("nn_win_rate", 0.0))
            fixed_seed_eval_gate["summary"] = fixed_summary
            fixed_seed_eval_gate["win_rate"] = round(fixed_wr, 4)
            fixed_seed_eval_gate["passed"] = bool(fixed_wr >= fixed_seed_eval_gate_min_win_rate)
        fixed_gate_regression = {
            "enabled": bool(require_fixed_seed_eval_gate_pass and rollback_on_fixed_gate_regression),
            "max_drop": float(fixed_gate_regression_max_drop),
            "prev_win_rate": None,
            "current_win_rate": None,
            "drop": 0.0,
            "triggered": False,
        }
        if require_fixed_seed_eval_gate_pass:
            fixed_wr = float(fixed_seed_eval_gate.get("win_rate", 0.0))
            fixed_gate_regression["current_win_rate"] = round(fixed_wr, 4)
            if prev_fixed_gate_win_rate is not None:
                drop = float(prev_fixed_gate_win_rate) - fixed_wr
                fixed_gate_regression["prev_win_rate"] = round(float(prev_fixed_gate_win_rate), 4)
                fixed_gate_regression["drop"] = round(drop, 4)
                fixed_gate_regression["triggered"] = bool(
                    rollback_on_fixed_gate_regression and drop > fixed_gate_regression_max_drop
                )
            prev_fixed_gate_win_rate = fixed_wr

        eligible_for_best = (
            bool(kpi_gate["passed"])
            and bool(strict_kpi_compare.get("passed", False))
            and (not selection_requires_gate_pass or gate_pass_for_selection)
            and bool(fixed_seed_eval_gate.get("passed", True))
        )
        incumbent_metrics_before = dict(best_selection_metrics) if best_selection_metrics is not None else None
        selected_as_best = eligible_for_best and _robust_is_better(
            robust_metrics,
            best_selection_metrics,
            min_win_rate_delta=best_min_win_rate_delta,
            min_mean_score_delta=best_min_mean_score_delta,
            max_margin_regression=best_max_margin_regression,
        )

        row = {
            "iteration": i,
            "seed": iter_seed,
            "generation_mode": generation_mode,
            "champion_switch_ready": bool(champion_switch_is_ready),
            "strict_game_fraction": round(float(strict_game_fraction_iter), 4),
            "dataset": str(dataset_jsonl),
            "dataset_meta": str(dataset_meta),
            "train_dataset": str(train_dataset_jsonl),
            "train_dataset_meta": str(train_dataset_meta),
            "model": str(model_path),
            "dataset_summary": {
                "samples": ds_meta["samples"],
                "mean_player_score": ds_meta["mean_player_score"],
            },
            "hard_replay": hard_replay_summary,
            "rl_summary": rl_summary,
            "combined_dataset_summary": accumulation_summary,
            "train_summary": {
                "train_samples": tr["train_samples"],
                "val_samples": tr["val_samples"],
                "warm_start": tr.get("warm_start", {}),
                "last_epoch": tr["history"][-1] if tr["history"] else {},
            },
            "eval_summary": ev_dict,
            "promotion_gate": {
                "games": promotion_games,
                "result": gate_dict,
                "score_primary_pass": gate_primary_pass,
                "win_secondary_pass": gate_secondary_pass,
                "champion_margin_pass": champion_margin_pass,
                "win_rate": round(gate_win_rate, 4),
                "promoted": promoted,
            },
            "promotion_primary_eval": {
                "primary_opponent": gate_dict.get("primary_opponent", "heuristic"),
                "result": gate_dict,
            },
            "pool_eval": pool_eval,
            "kpi_gate": kpi_gate,
            "strict_kpi_gate": strict_kpi_compare,
            "fixed_seed_eval_gate": fixed_seed_eval_gate,
            "fixed_gate_regression": fixed_gate_regression,
            "best_selection_eval": {
                "games": int(best_selection_eval_games),
                "candidate": robust_metrics,
                "incumbent_before": incumbent_metrics_before,
                "eligible": eligible_for_best,
                "selected": selected_as_best,
            },
        }
        history.append(row)

        if selected_as_best:
            best = row
            best_selection_metrics = robust_metrics
            shutil.copy2(model_path, best_model_path)
            shutil.copy2(train_dataset_meta, best_meta_path)
            proposal_model_path = str(best_model_path)
            consecutive_non_eligible_iters = 0
        elif best_selection_metrics is None and best is not None:
            # Keep selector state aligned when resuming from an existing best row.
            candidate_sel = best.get("best_selection_eval", {}).get("candidate")
            if isinstance(candidate_sel, dict):
                best_selection_metrics = dict(candidate_sel)
        elif (not use_champion_mode) and proposal_update_policy == "latest_candidate":
            # Keep bootstrapping with the latest candidate even when strict
            # promotion fails, so proposal quality can still improve.
            proposal_model_path = str(model_path)
        elif (not use_champion_mode) and proposal_update_policy == "best_only":
            if best_model_path.exists():
                proposal_model_path = str(best_model_path)
            elif frozen_opponent_model_path and not state_encoder_enable_identity and not state_encoder_use_per_slot:
                proposal_model_path = str(frozen_opponent_model_path)

        if best is None:
            consecutive_non_eligible_iters = 0
        elif not eligible_for_best:
            consecutive_non_eligible_iters += 1
        else:
            consecutive_non_eligible_iters = 0
        early_stop_triggered = (
            bool(stop_after_consecutive_non_eligible_iters)
            and best is not None
            and consecutive_non_eligible_iters >= max_consecutive_non_eligible_iters
        )
        fixed_gate_rollback_triggered = bool(fixed_gate_regression.get("triggered", False))
        row["stability"] = {
            "consecutive_non_eligible_iters": int(consecutive_non_eligible_iters),
            "max_consecutive_non_eligible_iters": int(max_consecutive_non_eligible_iters),
            "early_stop_triggered": bool(early_stop_triggered or fixed_gate_rollback_triggered),
            "fixed_gate_rollback_triggered": bool(fixed_gate_rollback_triggered),
        }

        manifest = {
            "version": 1,
            "started_at_epoch": int(started),
            "updated_at_epoch": int(time.time()),
            "elapsed_sec": round(time.time() - started, 2),
            "stopped_early": bool(stopped_early),
            "stop_reason": stop_reason,
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
                "value_target_score_scale": value_target_score_scale,
                "value_target_score_bias": value_target_score_bias,
                "late_round_oversample_factor": late_round_oversample_factor,
                "train_epochs": train_epochs,
                "train_batch": train_batch,
                "train_hidden1": train_hidden1,
                "train_hidden2": train_hidden2,
                "train_dropout": train_dropout,
                "train_lr_init": train_lr_init,
                "train_lr_peak": train_lr_peak,
                "train_lr_warmup_epochs": train_lr_warmup_epochs,
                "train_lr_decay_every": train_lr_decay_every,
                "train_lr_decay_factor": train_lr_decay_factor,
                "train_momentum": train_momentum,
                "train_init_model_path": train_init_model_path,
                "train_init_first_iter_only": train_init_first_iter_only,
                "auto_train_init_from_frozen": bool(auto_train_init_from_frozen),
                "train_early_stop_enabled": train_early_stop_enabled,
                "train_early_stop_patience": train_early_stop_patience,
                "train_early_stop_min_delta": train_early_stop_min_delta,
                "train_early_stop_restore_best": train_early_stop_restore_best,
                "deprecated_train_args_used": deprecated_train_args_used,
                # Temporary compatibility keys.
                "train_hidden": train_hidden1,
                "train_lr": train_lr_peak,
                "train_value_weight": train_value_weight,
                "val_split": val_split,
                "eval_games": eval_games,
                "promotion_games": promotion_games,
                "pool_games_per_opponent": pool_games_per_opponent,
                "min_pool_win_rate": min_pool_win_rate,
                "min_pool_mean_score": min_pool_mean_score,
                "min_pool_rate_ge_100": min_pool_rate_ge_100,
                "min_pool_rate_ge_120": min_pool_rate_ge_120,
                "require_pool_non_regression": require_pool_non_regression,
                "min_gate_win_rate": min_gate_win_rate,
                "min_gate_mean_score": min_gate_mean_score,
                "min_gate_rate_ge_100": min_gate_rate_ge_100,
                "min_gate_rate_ge_120": min_gate_rate_ge_120,
                "require_non_negative_champion_margin": require_non_negative_champion_margin,
                "champion_self_play_enabled": champion_self_play_enabled,
                "champion_switch_after_first_promotion": champion_switch_after_first_promotion,
                "champion_switch_min_stable_iters": champion_switch_min_stable_iters,
                "champion_switch_min_eval_win_rate": champion_switch_min_eval_win_rate,
                "champion_switch_min_iteration": champion_switch_min_iteration,
                "champion_teacher_source": champion_teacher_source,
                "champion_engine_time_budget_ms": champion_engine_time_budget_ms,
                "champion_engine_num_determinizations": champion_engine_num_determinizations,
                "champion_engine_max_rollout_depth": champion_engine_max_rollout_depth,
                "promotion_primary_opponent": promotion_primary_opponent,
                "frozen_opponent_model_path": frozen_opponent_model_path,
                "best_selection_eval_games": best_selection_eval_games,
                "proposal_update_policy": proposal_update_policy,
                "selection_requires_gate_pass": selection_requires_gate_pass,
                "best_min_win_rate_delta": best_min_win_rate_delta,
                "best_min_mean_score_delta": best_min_mean_score_delta,
                "best_max_margin_regression": best_max_margin_regression,
                "fixed_benchmark_seeds": fixed_benchmark_seeds,
                "benchmark_seed": benchmark_seed,
                "engine_teacher_prob": engine_teacher_prob,
                "engine_time_budget_ms": engine_time_budget_ms,
                "engine_num_determinizations": engine_num_determinizations,
                "engine_max_rollout_depth": engine_max_rollout_depth,
                "adaptive_teacher_depth2_enabled": adaptive_teacher_depth2_enabled,
                "adaptive_teacher_uncertainty_gap": adaptive_teacher_uncertainty_gap,
                "adaptive_teacher_top_m": adaptive_teacher_top_m,
                "hard_replay_games": hard_replay_games,
                "hard_replay_player": hard_replay_player,
                "hard_replay_target_player": hard_replay_target_player,
                "hard_replay_loss_oversample_factor": hard_replay_loss_oversample_factor,
                "hard_replay_only_losses": hard_replay_only_losses,
                "rollback_on_fixed_gate_regression": rollback_on_fixed_gate_regression,
                "fixed_gate_regression_max_drop": fixed_gate_regression_max_drop,
                "stop_after_consecutive_non_eligible_iters": stop_after_consecutive_non_eligible_iters,
                "max_consecutive_non_eligible_iters": max_consecutive_non_eligible_iters,
                "strict_rules_only": strict_rules_only,
                "reject_non_strict_powers": reject_non_strict_powers,
                "strict_curriculum_enabled": strict_curriculum_enabled,
                "strict_fraction_start": strict_fraction_start,
                "strict_fraction_end": strict_fraction_end,
                "strict_fraction_warmup_iters": strict_fraction_warmup_iters,
                "strict_kpi_gate_enabled": strict_kpi_gate_enabled,
                "strict_kpi_baseline_path": strict_kpi_baseline_path,
                "strict_kpi_round1_games": strict_kpi_round1_games,
                "strict_kpi_smoke_games": strict_kpi_smoke_games,
                "strict_kpi_max_turns": strict_kpi_max_turns,
                "strict_kpi_min_round1_pct_15_30": strict_kpi_min_round1_pct_15_30,
                "strict_kpi_max_round1_strict_rejected_games": strict_kpi_max_round1_strict_rejected_games,
                "strict_kpi_max_smoke_strict_rejected_games": strict_kpi_max_smoke_strict_rejected_games,
                "strict_kpi_min_strict_certified_birds": strict_kpi_min_strict_certified_birds,
                "strict_kpi_require_non_regression": strict_kpi_require_non_regression,
                "strict_kpi_mean_score_tolerance": strict_kpi_mean_score_tolerance,
                "state_encoder_enable_identity": state_encoder_enable_identity,
                "state_encoder_identity_hash_dim": state_encoder_identity_hash_dim,
                "state_encoder_use_per_slot": state_encoder_use_per_slot,
                "state_encoder_max_hand_slots": state_encoder_max_hand_slots,
                "require_fixed_seed_eval_gate_pass": require_fixed_seed_eval_gate_pass,
                "fixed_seed_eval_gate_games": fixed_seed_eval_gate_games,
                "fixed_seed_eval_gate_min_win_rate": fixed_seed_eval_gate_min_win_rate,
                "data_accumulation_enabled": data_accumulation_enabled,
                "data_accumulation_decay": data_accumulation_decay,
                "max_accumulated_samples": max_accumulated_samples,
                "dataset_workers": dataset_workers,
                "rl_games_per_iter": rl_games_per_iter,
                "rl_temperature": rl_temperature,
                "rl_use_value_baseline": rl_use_value_baseline,
                "rl_warm_start_path": rl_warm_start_path,
                "seed": seed,
            },
            "best": best,
            "history": history,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(
            f"iter {i}/{iterations} | samples={ds_meta['samples']} | "
            f"eval_wins={ev_dict['nn_wins']}/{ev_dict['games']} margin={ev_dict['nn_mean_margin']:.2f} | "
            f"gate_wins={gate_dict['nn_wins']}/{promotion_games} | "
            f"pool_win={pool_eval['summary']['nn_win_rate']:.3f} ge100={pool_eval['summary']['nn_rate_ge_100']:.3f} | "
            f"kpi_pass={kpi_gate['passed']} strict_kpi_pass={strict_kpi_compare.get('passed', False)} "
            f"fixed_seed_gate_pass={fixed_seed_eval_gate.get('passed', True)} "
            f"promoted={promoted} selected_best={selected_as_best}"
        )
        if early_stop_triggered:
            stopped_early = True
            stop_reason = (
                f"non_eligible_streak_reached_{consecutive_non_eligible_iters}"
            )
            manifest["stopped_early"] = True
            manifest["stop_reason"] = stop_reason
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"early stop triggered: {stop_reason}")
            break
        if fixed_gate_rollback_triggered:
            if best_model_path.exists():
                proposal_model_path = str(best_model_path)
            elif frozen_opponent_model_path and not state_encoder_enable_identity and not state_encoder_use_per_slot:
                proposal_model_path = str(frozen_opponent_model_path)
            stopped_early = True
            drop = float(fixed_gate_regression.get("drop", 0.0))
            stop_reason = f"fixed_gate_regression_drop_{drop:.4f}"
            manifest["stopped_early"] = True
            manifest["stop_reason"] = stop_reason
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"early stop triggered: {stop_reason}")
            break

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-improve factorized BC with strict gating")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve_factorized")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=400)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--bootstrap-mix", type=float, default=0.35)
    parser.add_argument("--value-target-score-scale", type=float, default=160.0)
    parser.add_argument("--value-target-score-bias", type=float, default=0.0)
    parser.add_argument("--late-round-oversample-factor", type=int, default=2)
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
    parser.add_argument("--train-momentum", type=float, default=0.9)
    parser.add_argument("--train-init-model-path", default=None)
    parser.set_defaults(train_init_first_iter_only=True)
    parser.add_argument("--train-init-first-iter-only", dest="train_init_first_iter_only", action="store_true")
    parser.add_argument("--train-init-every-iter", dest="train_init_first_iter_only", action="store_false")
    parser.set_defaults(train_early_stop_enabled=True, train_early_stop_restore_best=True)
    parser.add_argument("--train-early-stop-enabled", dest="train_early_stop_enabled", action="store_true")
    parser.add_argument("--disable-train-early-stop", dest="train_early_stop_enabled", action="store_false")
    parser.add_argument("--train-early-stop-patience", type=int, default=5)
    parser.add_argument("--train-early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--train-early-stop-restore-best", dest="train_early_stop_restore_best", action="store_true")
    parser.add_argument("--no-train-early-stop-restore-best", dest="train_early_stop_restore_best", action="store_false")
    parser.add_argument("--train-hidden", type=int, default=None, help="Deprecated: maps hidden1=hidden2=train-hidden")
    parser.add_argument("--train-lr", type=float, default=None, help="Deprecated: maps train-lr-peak=train-lr")
    parser.add_argument("--train-value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--promotion-games", type=int, default=80)
    parser.add_argument("--pool-games-per-opponent", type=int, default=40)
    parser.add_argument("--min-pool-win-rate", type=float, default=0.30)
    parser.add_argument("--min-pool-mean-score", type=float, default=0.0)
    parser.add_argument("--min-pool-rate-ge-100", type=float, default=0.0)
    parser.add_argument("--min-pool-rate-ge-120", type=float, default=0.0)
    parser.set_defaults(require_pool_non_regression=False)
    parser.add_argument("--require-pool-non-regression", dest="require_pool_non_regression", action="store_true")
    parser.add_argument("--disable-pool-non-regression", dest="require_pool_non_regression", action="store_false")
    parser.add_argument("--min-gate-win-rate", type=float, default=0.0)
    parser.add_argument("--min-gate-mean-score", type=float, default=0.0)
    parser.add_argument("--min-gate-rate-ge-100", type=float, default=0.0)
    parser.add_argument("--min-gate-rate-ge-120", type=float, default=0.0)
    parser.set_defaults(require_non_negative_champion_margin=True)
    parser.add_argument(
        "--require-non-negative-champion-margin",
        dest="require_non_negative_champion_margin",
        action="store_true",
    )
    parser.add_argument(
        "--allow-negative-champion-margin",
        dest="require_non_negative_champion_margin",
        action="store_false",
    )
    parser.set_defaults(data_accumulation_enabled=True)
    parser.add_argument("--data-accumulation-enabled", dest="data_accumulation_enabled", action="store_true")
    parser.add_argument("--disable-data-accumulation", dest="data_accumulation_enabled", action="store_false")
    parser.add_argument("--data-accumulation-decay", type=float, default=0.5)
    parser.add_argument("--max-accumulated-samples", type=int, default=200000)
    parser.add_argument("--dataset-workers", type=int, default=4)
    parser.set_defaults(champion_self_play_enabled=False, champion_switch_after_first_promotion=True)
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
    parser.add_argument("--promotion-primary-opponent", default="heuristic", choices=["heuristic", "champion", "frozen"])
    parser.add_argument("--frozen-opponent-model-path", default=None)
    parser.add_argument(
        "--best-selection-eval-games",
        type=int,
        default=0,
        help="0 => auto (max(promotion_games, eval_games))",
    )
    parser.add_argument("--proposal-update-policy", default="best_only", choices=["best_only", "latest_candidate"])
    parser.set_defaults(selection_requires_gate_pass=True)
    parser.add_argument("--selection-requires-gate-pass", dest="selection_requires_gate_pass", action="store_true")
    parser.add_argument(
        "--allow-selection-without-gate-pass",
        dest="selection_requires_gate_pass",
        action="store_false",
    )
    parser.add_argument("--best-min-win-rate-delta", type=float, default=0.02)
    parser.add_argument("--best-min-mean-score-delta", type=float, default=0.5)
    parser.add_argument("--best-max-margin-regression", type=float, default=1.0)
    parser.set_defaults(fixed_benchmark_seeds=True)
    parser.add_argument("--fixed-benchmark-seeds", dest="fixed_benchmark_seeds", action="store_true")
    parser.add_argument("--rotating-benchmark-seeds", dest="fixed_benchmark_seeds", action="store_false")
    parser.add_argument("--benchmark-seed", type=int, default=13_371_337)
    parser.add_argument("--engine-teacher-prob", type=float, default=0.30)
    parser.add_argument("--engine-time-budget-ms", type=int, default=25)
    parser.add_argument("--engine-num-determinizations", type=int, default=0)
    parser.add_argument("--engine-max-rollout-depth", type=int, default=24)
    parser.set_defaults(adaptive_teacher_depth2_enabled=True)
    parser.add_argument("--adaptive-teacher-depth2-enabled", dest="adaptive_teacher_depth2_enabled", action="store_true")
    parser.add_argument("--disable-adaptive-teacher-depth2", dest="adaptive_teacher_depth2_enabled", action="store_false")
    parser.add_argument("--adaptive-teacher-uncertainty-gap", type=float, default=2.0)
    parser.add_argument("--adaptive-teacher-top-m", type=int, default=2)
    parser.add_argument("--hard-replay-games", type=int, default=0)
    parser.add_argument("--hard-replay-player", default="proposal", choices=["proposal", "best", "frozen"])
    parser.add_argument("--hard-replay-target-player", type=int, default=0)
    parser.add_argument("--hard-replay-loss-oversample-factor", type=int, default=3)
    parser.set_defaults(hard_replay_only_losses=True)
    parser.add_argument("--hard-replay-only-losses", dest="hard_replay_only_losses", action="store_true")
    parser.add_argument("--hard-replay-include-wins", dest="hard_replay_only_losses", action="store_false")
    parser.set_defaults(rollback_on_fixed_gate_regression=True)
    parser.add_argument(
        "--rollback-on-fixed-gate-regression",
        dest="rollback_on_fixed_gate_regression",
        action="store_true",
    )
    parser.add_argument(
        "--disable-rollback-on-fixed-gate-regression",
        dest="rollback_on_fixed_gate_regression",
        action="store_false",
    )
    parser.add_argument("--fixed-gate-regression-max-drop", type=float, default=0.10)
    parser.set_defaults(stop_after_consecutive_non_eligible_iters=True)
    parser.add_argument(
        "--stop-after-consecutive-non-eligible-iters",
        dest="stop_after_consecutive_non_eligible_iters",
        action="store_true",
    )
    parser.add_argument(
        "--disable-stop-after-consecutive-non-eligible-iters",
        dest="stop_after_consecutive_non_eligible_iters",
        action="store_false",
    )
    parser.add_argument("--max-consecutive-non-eligible-iters", type=int, default=4)
    parser.set_defaults(strict_rules_only=True, reject_non_strict_powers=True)
    parser.add_argument("--strict-rules-only", dest="strict_rules_only", action="store_true")
    parser.add_argument("--allow-non-strict-rules", dest="strict_rules_only", action="store_false")
    parser.add_argument("--reject-non-strict-powers", dest="reject_non_strict_powers", action="store_true")
    parser.add_argument("--allow-non-strict-powers", dest="reject_non_strict_powers", action="store_false")
    parser.set_defaults(strict_curriculum_enabled=True)
    parser.add_argument("--strict-curriculum-enabled", dest="strict_curriculum_enabled", action="store_true")
    parser.add_argument("--disable-strict-curriculum", dest="strict_curriculum_enabled", action="store_false")
    parser.add_argument("--strict-fraction-start", type=float, default=1.0)
    parser.add_argument("--strict-fraction-end", type=float, default=0.0)
    parser.add_argument("--strict-fraction-warmup-iters", type=int, default=5)
    parser.set_defaults(strict_kpi_gate_enabled=True)
    parser.add_argument("--strict-kpi-gate-enabled", dest="strict_kpi_gate_enabled", action="store_true")
    parser.add_argument("--disable-strict-kpi-gate", dest="strict_kpi_gate_enabled", action="store_false")
    parser.add_argument("--strict-kpi-baseline", default="reports/ml/baselines/strict_kpi_baseline.json")
    parser.add_argument("--strict-kpi-round1-games", type=int, default=20)
    parser.add_argument("--strict-kpi-smoke-games", type=int, default=20)
    parser.add_argument("--strict-kpi-max-turns", type=int, default=120)
    parser.add_argument("--strict-kpi-min-round1-pct-15-30", type=float, default=0.2)
    parser.add_argument("--strict-kpi-max-round1-strict-rejected-games", type=int, default=0)
    parser.add_argument("--strict-kpi-max-smoke-strict-rejected-games", type=int, default=0)
    parser.add_argument("--strict-kpi-min-strict-certified-birds", type=int, default=50)
    parser.set_defaults(strict_kpi_require_non_regression=True)
    parser.add_argument("--strict-kpi-require-non-regression", dest="strict_kpi_require_non_regression", action="store_true")
    parser.add_argument("--strict-kpi-disable-non-regression", dest="strict_kpi_require_non_regression", action="store_false")
    parser.add_argument("--strict-kpi-mean-score-tolerance", type=float, default=3.0)
    parser.set_defaults(state_encoder_enable_identity=False)
    parser.add_argument("--state-encoder-enable-identity", dest="state_encoder_enable_identity", action="store_true")
    parser.add_argument("--disable-state-encoder-identity", dest="state_encoder_enable_identity", action="store_false")
    parser.add_argument("--state-encoder-identity-hash-dim", type=int, default=128)
    parser.set_defaults(state_encoder_use_per_slot=False)
    parser.add_argument("--use-per-slot-encoding", dest="state_encoder_use_per_slot", action="store_true")
    parser.add_argument("--disable-per-slot-encoding", dest="state_encoder_use_per_slot", action="store_false")
    parser.add_argument("--max-hand-slots", dest="state_encoder_max_hand_slots", type=int, default=8)
    parser.set_defaults(require_fixed_seed_eval_gate_pass=False)
    parser.add_argument("--require-fixed-seed-eval-gate-pass", dest="require_fixed_seed_eval_gate_pass", action="store_true")
    parser.add_argument("--disable-fixed-seed-eval-gate-pass", dest="require_fixed_seed_eval_gate_pass", action="store_false")
    parser.add_argument("--fixed-seed-eval-gate-games", type=int, default=40)
    parser.add_argument("--fixed-seed-eval-gate-min-win-rate", type=float, default=0.43)
    parser.add_argument("--seed", type=int, default=0)
    # REINFORCE policy-gradient options
    parser.add_argument(
        "--rl-games-per-iter",
        type=int,
        default=0,
        help="Games of RL self-play to collect per iteration (0 = disabled)",
    )
    parser.add_argument("--rl-temperature", type=float, default=1.5)
    parser.add_argument(
        "--rl-warm-start",
        default="",
        dest="rl_warm_start_path",
        help="Seed model for RL episode collection before first promotion",
    )
    parser.set_defaults(rl_use_value_baseline=True)
    parser.add_argument(
        "--rl-use-value-baseline",
        dest="rl_use_value_baseline",
        action="store_true",
    )
    parser.add_argument(
        "--disable-rl-value-baseline",
        dest="rl_use_value_baseline",
        action="store_false",
    )
    args = parser.parse_args()

    manifest = run_auto_improve_factorized(
        out_dir=args.out_dir,
        clean_out_dir=not args.resume,
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
        value_target_score_scale=args.value_target_score_scale,
        value_target_score_bias=args.value_target_score_bias,
        late_round_oversample_factor=args.late_round_oversample_factor,
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
        train_momentum=args.train_momentum,
        train_init_model_path=args.train_init_model_path,
        train_init_first_iter_only=args.train_init_first_iter_only,
        train_early_stop_enabled=args.train_early_stop_enabled,
        train_early_stop_patience=args.train_early_stop_patience,
        train_early_stop_min_delta=args.train_early_stop_min_delta,
        train_early_stop_restore_best=args.train_early_stop_restore_best,
        train_value_weight=args.train_value_weight,
        val_split=args.val_split,
        eval_games=args.eval_games,
        promotion_games=args.promotion_games,
        pool_games_per_opponent=args.pool_games_per_opponent,
        min_pool_win_rate=args.min_pool_win_rate,
        min_pool_mean_score=args.min_pool_mean_score,
        min_pool_rate_ge_100=args.min_pool_rate_ge_100,
        min_pool_rate_ge_120=args.min_pool_rate_ge_120,
        require_pool_non_regression=args.require_pool_non_regression,
        min_gate_win_rate=args.min_gate_win_rate,
        min_gate_mean_score=args.min_gate_mean_score,
        min_gate_rate_ge_100=args.min_gate_rate_ge_100,
        min_gate_rate_ge_120=args.min_gate_rate_ge_120,
        require_non_negative_champion_margin=args.require_non_negative_champion_margin,
        data_accumulation_enabled=args.data_accumulation_enabled,
        data_accumulation_decay=args.data_accumulation_decay,
        max_accumulated_samples=args.max_accumulated_samples,
        dataset_workers=args.dataset_workers,
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
        frozen_opponent_model_path=args.frozen_opponent_model_path,
        best_selection_eval_games=args.best_selection_eval_games,
        proposal_update_policy=args.proposal_update_policy,
        selection_requires_gate_pass=args.selection_requires_gate_pass,
        best_min_win_rate_delta=args.best_min_win_rate_delta,
        best_min_mean_score_delta=args.best_min_mean_score_delta,
        best_max_margin_regression=args.best_max_margin_regression,
        fixed_benchmark_seeds=args.fixed_benchmark_seeds,
        benchmark_seed=args.benchmark_seed,
        engine_teacher_prob=args.engine_teacher_prob,
        engine_time_budget_ms=args.engine_time_budget_ms,
        engine_num_determinizations=args.engine_num_determinizations,
        engine_max_rollout_depth=args.engine_max_rollout_depth,
        adaptive_teacher_depth2_enabled=args.adaptive_teacher_depth2_enabled,
        adaptive_teacher_uncertainty_gap=args.adaptive_teacher_uncertainty_gap,
        adaptive_teacher_top_m=args.adaptive_teacher_top_m,
        hard_replay_games=args.hard_replay_games,
        hard_replay_player=args.hard_replay_player,
        hard_replay_target_player=args.hard_replay_target_player,
        hard_replay_loss_oversample_factor=args.hard_replay_loss_oversample_factor,
        hard_replay_only_losses=args.hard_replay_only_losses,
        rollback_on_fixed_gate_regression=args.rollback_on_fixed_gate_regression,
        fixed_gate_regression_max_drop=args.fixed_gate_regression_max_drop,
        stop_after_consecutive_non_eligible_iters=args.stop_after_consecutive_non_eligible_iters,
        max_consecutive_non_eligible_iters=args.max_consecutive_non_eligible_iters,
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
        strict_curriculum_enabled=args.strict_curriculum_enabled,
        strict_fraction_start=args.strict_fraction_start,
        strict_fraction_end=args.strict_fraction_end,
        strict_fraction_warmup_iters=args.strict_fraction_warmup_iters,
        strict_kpi_gate_enabled=args.strict_kpi_gate_enabled,
        strict_kpi_baseline_path=args.strict_kpi_baseline,
        strict_kpi_round1_games=args.strict_kpi_round1_games,
        strict_kpi_smoke_games=args.strict_kpi_smoke_games,
        strict_kpi_max_turns=args.strict_kpi_max_turns,
        strict_kpi_min_round1_pct_15_30=args.strict_kpi_min_round1_pct_15_30,
        strict_kpi_max_round1_strict_rejected_games=args.strict_kpi_max_round1_strict_rejected_games,
        strict_kpi_max_smoke_strict_rejected_games=args.strict_kpi_max_smoke_strict_rejected_games,
        strict_kpi_min_strict_certified_birds=args.strict_kpi_min_strict_certified_birds,
        strict_kpi_require_non_regression=args.strict_kpi_require_non_regression,
        strict_kpi_mean_score_tolerance=args.strict_kpi_mean_score_tolerance,
        state_encoder_enable_identity=args.state_encoder_enable_identity,
        state_encoder_identity_hash_dim=args.state_encoder_identity_hash_dim,
        state_encoder_use_per_slot=args.state_encoder_use_per_slot,
        state_encoder_max_hand_slots=args.state_encoder_max_hand_slots,
        require_fixed_seed_eval_gate_pass=args.require_fixed_seed_eval_gate_pass,
        fixed_seed_eval_gate_games=args.fixed_seed_eval_gate_games,
        fixed_seed_eval_gate_min_win_rate=args.fixed_seed_eval_gate_min_win_rate,
        seed=args.seed,
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
        rl_games_per_iter=args.rl_games_per_iter,
        rl_temperature=args.rl_temperature,
        rl_use_value_baseline=args.rl_use_value_baseline,
        rl_warm_start_path=args.rl_warm_start_path,
    )

    best = manifest.get("best")
    print(f"auto-improve-factorized complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
