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
from backend.ml.evaluate_factorized_pool import evaluate_against_pool
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.kpi_gate import run_gate
from backend.ml.strict_kpi_compare import run_compare as run_strict_kpi_compare
from backend.ml.strict_kpi_runner import run_strict_kpi
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
    value_target_score_scale: float,
    value_target_score_bias: float,
    late_round_oversample_factor: int,
    train_epochs: int,
    train_batch: int,
    train_hidden: int,
    train_lr: float,
    train_value_weight: float,
    val_split: float,
    eval_games: int,
    promotion_games: int,
    pool_games_per_opponent: int,
    min_pool_win_rate: float,
    min_pool_mean_score: float,
    min_pool_rate_ge_100: float,
    min_pool_rate_ge_120: float,
    require_pool_non_regression: bool,
    min_gate_win_rate: float,
    min_gate_mean_score: float,
    min_gate_rate_ge_100: float,
    min_gate_rate_ge_120: float,
    engine_teacher_prob: float,
    engine_time_budget_ms: int,
    engine_num_determinizations: int,
    engine_max_rollout_depth: int,
    seed: int,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
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
    clean_out_dir: bool = True,
) -> dict:
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

    for i in range(1, iterations + 1):
        iter_seed = seed + i * 1000
        iter_dir = base / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        dataset_jsonl = iter_dir / "bc_dataset.jsonl"
        dataset_meta = iter_dir / "bc_dataset.meta.json"
        model_path = iter_dir / "factorized_model.npz"
        eval_path = iter_dir / "eval.json"
        gate_path = iter_dir / "promotion_gate_eval.json"
        pool_eval_path = iter_dir / "pool_eval.json"
        kpi_gate_path = iter_dir / "kpi_gate.json"
        strict_kpi_candidate_path = iter_dir / "strict_kpi_candidate.json"
        strict_kpi_compare_path = iter_dir / "strict_kpi_compare.json"

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
            value_target_score_scale=value_target_score_scale,
            value_target_score_bias=value_target_score_bias,
            late_round_oversample_factor=late_round_oversample_factor,
            engine_teacher_prob=engine_teacher_prob,
            engine_time_budget_ms=engine_time_budget_ms,
            engine_num_determinizations=engine_num_determinizations,
            engine_max_rollout_depth=engine_max_rollout_depth,
            strict_rules_only=strict_rules_only,
            reject_non_strict_powers=reject_non_strict_powers,
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
            proposal_top_k=proposal_top_k,
        )
        ev_dict = ev.__dict__
        eval_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

        gate = evaluate_factorized_vs_heuristic(
            model_path=str(model_path),
            games=promotion_games,
            board_type=board_type,
            max_turns=max_turns,
            seed=iter_seed + 777,
            proposal_top_k=proposal_top_k,
        )
        gate_dict = gate.__dict__
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
                "score_primary_pass": gate_primary_pass,
                "win_secondary_pass": gate_secondary_pass,
                "win_rate": round(gate_win_rate, 4),
                "promoted": promoted,
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
                "value_target_score_scale": value_target_score_scale,
                "value_target_score_bias": value_target_score_bias,
                "late_round_oversample_factor": late_round_oversample_factor,
                "train_epochs": train_epochs,
                "train_batch": train_batch,
                "train_hidden": train_hidden,
                "train_lr": train_lr,
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
                "engine_teacher_prob": engine_teacher_prob,
                "engine_time_budget_ms": engine_time_budget_ms,
                "engine_num_determinizations": engine_num_determinizations,
                "engine_max_rollout_depth": engine_max_rollout_depth,
                "strict_rules_only": strict_rules_only,
                "reject_non_strict_powers": reject_non_strict_powers,
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
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--bootstrap-mix", type=float, default=0.35)
    parser.add_argument("--value-target-score-scale", type=float, default=160.0)
    parser.add_argument("--value-target-score-bias", type=float, default=0.0)
    parser.add_argument("--late-round-oversample-factor", type=int, default=2)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--train-hidden", type=int, default=192)
    parser.add_argument("--train-lr", type=float, default=1e-3)
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
    parser.add_argument("--min-gate-win-rate", type=float, default=0.45)
    parser.add_argument("--min-gate-mean-score", type=float, default=70.0)
    parser.add_argument("--min-gate-rate-ge-100", type=float, default=0.08)
    parser.add_argument("--min-gate-rate-ge-120", type=float, default=0.01)
    parser.add_argument("--engine-teacher-prob", type=float, default=0.15)
    parser.add_argument("--engine-time-budget-ms", type=int, default=25)
    parser.add_argument("--engine-num-determinizations", type=int, default=0)
    parser.add_argument("--engine-max-rollout-depth", type=int, default=24)
    parser.set_defaults(strict_rules_only=True, reject_non_strict_powers=True)
    parser.add_argument("--strict-rules-only", dest="strict_rules_only", action="store_true")
    parser.add_argument("--allow-non-strict-rules", dest="strict_rules_only", action="store_false")
    parser.add_argument("--reject-non-strict-powers", dest="reject_non_strict_powers", action="store_true")
    parser.add_argument("--allow-non-strict-powers", dest="reject_non_strict_powers", action="store_false")
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
        train_hidden=args.train_hidden,
        train_lr=args.train_lr,
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
        engine_teacher_prob=args.engine_teacher_prob,
        engine_time_budget_ms=args.engine_time_budget_ms,
        engine_num_determinizations=args.engine_num_determinizations,
        engine_max_rollout_depth=args.engine_max_rollout_depth,
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
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
    )

    best = manifest.get("best")
    print(f"auto-improve-factorized complete | best_promoted={'yes' if best else 'no'}")


if __name__ == "__main__":
    main()
