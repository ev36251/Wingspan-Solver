"""Auto-improve loop for factorized BC with strict promotion gating.

Implements:
- Policy improvement data generation (proposal+lookahead)
- BC training with value targets
- 200-game strict promotion gate: promote only if nn_wins > heuristic_wins
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.evaluate_factorized_pool import evaluate_against_pool
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.kpi_gate import run_gate
from backend.ml.strict_kpi_compare import run_compare as run_strict_kpi_compare
from backend.ml.strict_kpi_runner import run_strict_kpi
from backend.ml.train_factorized_bc import train_bc


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
    train_early_stop_enabled: bool = True,
    train_early_stop_patience: int = 5,
    train_early_stop_min_delta: float = 1e-4,
    train_early_stop_restore_best: bool = True,
    train_value_weight: float = 0.5,
    val_split: float = 0.1,
    eval_games: int = 80,
    promotion_games: int = 200,
    pool_games_per_opponent: int = 60,
    min_pool_win_rate: float = 0.0,
    min_pool_mean_score: float = 0.0,
    min_pool_rate_ge_100: float = 0.0,
    min_pool_rate_ge_120: float = 0.0,
    require_pool_non_regression: bool = False,
    min_gate_win_rate: float = 0.20,
    min_gate_mean_score: float = 25.0,
    min_gate_rate_ge_100: float = 0.0,
    min_gate_rate_ge_120: float = 0.0,
    champion_self_play_enabled: bool = True,
    champion_switch_after_first_promotion: bool = True,
    champion_teacher_source: str = "engine_only",
    champion_engine_time_budget_ms: int = 50,
    champion_engine_num_determinizations: int = 8,
    champion_engine_max_rollout_depth: int = 24,
    promotion_primary_opponent: str = "champion",
    engine_teacher_prob: float = 0.15,
    engine_time_budget_ms: int = 25,
    engine_num_determinizations: int = 0,
    engine_max_rollout_depth: int = 24,
    seed: int = 0,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    strict_curriculum_enabled: bool = True,
    strict_fraction_start: float = 1.0,
    strict_fraction_end: float = 0.0,
    strict_fraction_warmup_iters: int = 5,
    strict_kpi_gate_enabled: bool = True,
    strict_kpi_baseline_path: str = "reports/ml/baselines/strict_kpi_baseline.json",
    strict_kpi_round1_games: int = 10,
    strict_kpi_smoke_games: int = 3,
    strict_kpi_max_turns: int = 120,
    strict_kpi_min_round1_pct_15_30: float = 0.2,
    strict_kpi_max_round1_strict_rejected_games: int = 0,
    strict_kpi_max_smoke_strict_rejected_games: int = 0,
    strict_kpi_min_strict_certified_birds: int = 50,
    strict_kpi_require_non_regression: bool = False,
    strict_kpi_mean_score_tolerance: float = 0.5,
    data_accumulation_enabled: bool = True,
    data_accumulation_decay: float = 0.5,
    max_accumulated_samples: int = 200_000,
    clean_out_dir: bool = True,
    train_hidden: int | None = None,
    train_lr: float | None = None,
) -> dict:
    if promotion_primary_opponent not in ("heuristic", "champion"):
        raise ValueError("promotion_primary_opponent must be 'heuristic' or 'champion'")
    if champion_teacher_source not in ("probabilistic_engine", "engine_only"):
        raise ValueError("champion_teacher_source must be 'probabilistic_engine' or 'engine_only'")

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

    data_accumulation_decay = max(0.0, min(1.0, float(data_accumulation_decay)))
    max_accumulated_samples = max(1, int(max_accumulated_samples))

    base = Path(out_dir)
    if clean_out_dir and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / "auto_improve_factorized_manifest.json"
    best_model_path = base / "best_model.npz"
    best_meta_path = base / "best_dataset.meta.json"

    best: dict | None = None
    history: list[dict] = []
    started = time.time()

    proposal_model_path: str | None = None
    accumulation_sources: list[dict] = []

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

        champion_available = best_model_path.exists()
        use_champion_mode = (
            champion_self_play_enabled
            and champion_available
            and champion_switch_after_first_promotion
        )
        generation_mode = "champion_nn_vs_nn" if use_champion_mode else "bootstrap_mixed"
        iter_proposal_model_path = str(best_model_path) if use_champion_mode else proposal_model_path
        iter_opponent_model_path = str(best_model_path) if use_champion_mode else None

        ds_meta = generate_bc_dataset(
            out_jsonl=str(dataset_jsonl),
            out_meta=str(dataset_meta),
            games=games_per_iter,
            players=players,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed,
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
            strict_rules_only=strict_rules_only,
            reject_non_strict_powers=reject_non_strict_powers,
            strict_game_fraction=strict_game_fraction_iter,
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
            early_stop_enabled=train_early_stop_enabled,
            early_stop_patience=train_early_stop_patience,
            early_stop_min_delta=train_early_stop_min_delta,
            early_stop_restore_best=train_early_stop_restore_best,
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
            proposal_top_k=proposal_top_k,
        )
        ev_dict = ev.__dict__
        eval_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

        use_champion_primary_gate = (
            promotion_primary_opponent == "champion"
            and best_model_path.exists()
        )
        if use_champion_primary_gate:
            primary_gate_eval = evaluate_against_pool(
                model_path=str(model_path),
                opponents=[str(best_model_path)],
                games_per_opponent=promotion_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=iter_seed + 777,
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
                "primary_opponent": "champion",
                "opponent_spec": str(best_model_path),
            }
        else:
            gate = evaluate_factorized_vs_heuristic(
                model_path=str(model_path),
                games=promotion_games,
                board_type=board_type,
                max_turns=max_turns,
                seed=iter_seed + 777,
                proposal_top_k=proposal_top_k,
            )
            gate_dict = gate.__dict__
            gate_dict["opponent_wins"] = int(gate_dict.get("heuristic_wins", 0))
            gate_dict["opponent_mean_score"] = float(gate_dict.get("heuristic_mean_score", 0.0))
            gate_dict["primary_opponent"] = "heuristic"
            gate_dict["opponent_spec"] = "heuristic"
        gate_path.write_text(json.dumps(gate_dict, indent=2), encoding="utf-8")

        # Opponent-pool evaluation: heuristic + current best model (if exists).
        opponents = ["heuristic"]
        if best_model_path.exists():
            opponents.append(str(best_model_path))
        pool_eval = evaluate_against_pool(
            model_path=str(model_path),
            opponents=opponents,
            games_per_opponent=pool_games_per_opponent,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed + 999,
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
        promoted = (
            gate_primary_pass
            and gate_secondary_pass
            and bool(kpi_gate["passed"])
            and bool(strict_kpi_compare.get("passed", False))
        )

        row = {
            "iteration": i,
            "seed": iter_seed,
            "generation_mode": generation_mode,
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
            "combined_dataset_summary": accumulation_summary,
            "train_summary": {
                "train_samples": tr["train_samples"],
                "val_samples": tr["val_samples"],
                "last_epoch": tr["history"][-1] if tr["history"] else {},
            },
            "eval_summary": ev_dict,
            "promotion_gate": {
                "games": promotion_games,
                "result": gate_dict,
                "score_primary_pass": gate_primary_pass,
                "win_secondary_pass": gate_secondary_pass,
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
        }
        history.append(row)

        if promoted:
            if best is None or gate_dict["nn_mean_margin"] > best["promotion_gate"]["result"]["nn_mean_margin"]:
                best = row
                shutil.copy2(model_path, best_model_path)
                shutil.copy2(train_dataset_meta, best_meta_path)
                proposal_model_path = str(best_model_path)
        elif not use_champion_mode:
            # Keep bootstrapping with the latest candidate even when strict
            # promotion fails, so proposal quality can still improve.
            proposal_model_path = str(model_path)

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
                "champion_self_play_enabled": champion_self_play_enabled,
                "champion_switch_after_first_promotion": champion_switch_after_first_promotion,
                "champion_teacher_source": champion_teacher_source,
                "champion_engine_time_budget_ms": champion_engine_time_budget_ms,
                "champion_engine_num_determinizations": champion_engine_num_determinizations,
                "champion_engine_max_rollout_depth": champion_engine_max_rollout_depth,
                "promotion_primary_opponent": promotion_primary_opponent,
                "engine_teacher_prob": engine_teacher_prob,
                "engine_time_budget_ms": engine_time_budget_ms,
                "engine_num_determinizations": engine_num_determinizations,
                "engine_max_rollout_depth": engine_max_rollout_depth,
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
                "data_accumulation_enabled": data_accumulation_enabled,
                "data_accumulation_decay": data_accumulation_decay,
                "max_accumulated_samples": max_accumulated_samples,
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
            f"kpi_pass={kpi_gate['passed']} strict_kpi_pass={strict_kpi_compare.get('passed', False)} promoted={promoted}"
        )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-improve factorized BC with strict gating")
    parser.add_argument("--out-dir", default="reports/ml/auto_improve_factorized")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--players", type=int, default=2, choices=[2])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--games-per-iter", type=int, default=800)
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
    parser.add_argument("--eval-games", type=int, default=80)
    parser.add_argument("--promotion-games", type=int, default=200)
    parser.add_argument("--pool-games-per-opponent", type=int, default=60)
    parser.add_argument("--min-pool-win-rate", type=float, default=0.0)
    parser.add_argument("--min-pool-mean-score", type=float, default=0.0)
    parser.add_argument("--min-pool-rate-ge-100", type=float, default=0.0)
    parser.add_argument("--min-pool-rate-ge-120", type=float, default=0.0)
    parser.add_argument("--require-pool-non-regression", action="store_true")
    parser.add_argument("--min-gate-win-rate", type=float, default=0.20)
    parser.add_argument("--min-gate-mean-score", type=float, default=25.0)
    parser.add_argument("--min-gate-rate-ge-100", type=float, default=0.0)
    parser.add_argument("--min-gate-rate-ge-120", type=float, default=0.0)
    parser.set_defaults(data_accumulation_enabled=True)
    parser.add_argument("--data-accumulation-enabled", dest="data_accumulation_enabled", action="store_true")
    parser.add_argument("--disable-data-accumulation", dest="data_accumulation_enabled", action="store_false")
    parser.add_argument("--data-accumulation-decay", type=float, default=0.5)
    parser.add_argument("--max-accumulated-samples", type=int, default=200000)
    parser.set_defaults(champion_self_play_enabled=True, champion_switch_after_first_promotion=True)
    parser.add_argument("--champion-self-play-enabled", dest="champion_self_play_enabled", action="store_true")
    parser.add_argument("--disable-champion-self-play", dest="champion_self_play_enabled", action="store_false")
    parser.add_argument("--champion-switch-after-first-promotion", dest="champion_switch_after_first_promotion", action="store_true")
    parser.add_argument("--disable-champion-switch", dest="champion_switch_after_first_promotion", action="store_false")
    parser.add_argument("--champion-teacher-source", default="engine_only", choices=["probabilistic_engine", "engine_only"])
    parser.add_argument("--champion-engine-time-budget-ms", type=int, default=50)
    parser.add_argument("--champion-engine-num-determinizations", type=int, default=8)
    parser.add_argument("--champion-engine-max-rollout-depth", type=int, default=24)
    parser.add_argument("--promotion-primary-opponent", default="champion", choices=["heuristic", "champion"])
    parser.add_argument("--engine-teacher-prob", type=float, default=0.15)
    parser.add_argument("--engine-time-budget-ms", type=int, default=25)
    parser.add_argument("--engine-num-determinizations", type=int, default=0)
    parser.add_argument("--engine-max-rollout-depth", type=int, default=24)
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
    parser.add_argument("--strict-kpi-round1-games", type=int, default=10)
    parser.add_argument("--strict-kpi-smoke-games", type=int, default=3)
    parser.add_argument("--strict-kpi-max-turns", type=int, default=120)
    parser.add_argument("--strict-kpi-min-round1-pct-15-30", type=float, default=0.2)
    parser.add_argument("--strict-kpi-max-round1-strict-rejected-games", type=int, default=0)
    parser.add_argument("--strict-kpi-max-smoke-strict-rejected-games", type=int, default=0)
    parser.add_argument("--strict-kpi-min-strict-certified-birds", type=int, default=50)
    parser.set_defaults(strict_kpi_require_non_regression=False)
    parser.add_argument("--strict-kpi-require-non-regression", dest="strict_kpi_require_non_regression", action="store_true")
    parser.add_argument("--strict-kpi-disable-non-regression", dest="strict_kpi_require_non_regression", action="store_false")
    parser.add_argument("--strict-kpi-mean-score-tolerance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
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
        data_accumulation_enabled=args.data_accumulation_enabled,
        data_accumulation_decay=args.data_accumulation_decay,
        max_accumulated_samples=args.max_accumulated_samples,
        champion_self_play_enabled=args.champion_self_play_enabled,
        champion_switch_after_first_promotion=args.champion_switch_after_first_promotion,
        champion_teacher_source=args.champion_teacher_source,
        champion_engine_time_budget_ms=args.champion_engine_time_budget_ms,
        champion_engine_num_determinizations=args.champion_engine_num_determinizations,
        champion_engine_max_rollout_depth=args.champion_engine_max_rollout_depth,
        promotion_primary_opponent=args.promotion_primary_opponent,
        engine_teacher_prob=args.engine_teacher_prob,
        engine_time_budget_ms=args.engine_time_budget_ms,
        engine_num_determinizations=args.engine_num_determinizations,
        engine_max_rollout_depth=args.engine_max_rollout_depth,
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
        seed=args.seed,
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
    )

    best = manifest.get("best")
    print(f"auto-improve-factorized complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
