"""v17.1 rules-stable restart orchestrator.

Implements the end-to-end workflow:
1) hard pre-training rules-stability gate
2) strict smoke self-play (10 games)
3) 3-iteration AlphaZero Modal pilot with full feature pack
4) pilot acceptance evaluation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from backend.ml.alphazero_self_play import generate_self_play_dataset
from backend.ml.auto_improve_alphazero import run_auto_improve_alphazero
from backend.ml.strict_kpi_runner import run_strict_kpi
from backend.models.enums import BoardType
from backend.scripts.audit_bird_power_tests import build_audit


DEFAULT_INIT_MODEL_CANDIDATES = [
    "reports/ml/champions/current_champion.npz",
    "reports/ml/alphazero_rules_v17_explicit446_modal_pilot_v3/best_model.npz",
    "reports/ml/alphazero_rules_baseline_modal_pilot_v1/best_model.npz",
]


def _resolve_init_model_path(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--init-model-path not found: {explicit}")
        return str(p)

    for cand in DEFAULT_INIT_MODEL_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No init model found. Set --init-model-path explicitly."
    )


def _run_pytest_gate() -> None:
    cmd = [
        "pytest",
        "backend/tests/test_power_strict_mapping.py",
        "backend/tests/test_power_semantic_batches.py",
        "backend/tests/test_power_semantic_remaining.py",
        "backend/tests/test_score_breakdown_parity.py",
        "backend/tests/test_score_calibration_goldens.py",
    ]
    subprocess.run(cmd, check=True)


def _run_strict_smoke_selfplay_gate(
    *,
    model_path: str,
    games: int,
    mcts_sims: int,
    seed: int,
) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_jsonl = td_path / "strict_smoke_az.jsonl"
        out_meta = td_path / "strict_smoke_az.meta.json"
        meta = generate_self_play_dataset(
            model_path=model_path,
            out_jsonl=str(out_jsonl),
            out_meta=str(out_meta),
            games=games,
            players=2,
            board_type=BoardType.OCEANIA,
            mcts_sims=mcts_sims,
            c_puct=1.5,
            value_blend=0.95,
            rollout_policy="fast",
            root_dirichlet_epsilon=0.25,
            root_dirichlet_alpha=0.3,
            temperature_cutoff=8,
            seed=seed,
            max_turns=240,
            strict_rules_mode=True,
            fail_on_game_exception=True,
        )
    return meta


def run_rules_stability_gate(
    *,
    strict_kpi_out: str,
    init_model_path: str,
    smoke_games: int,
    smoke_mcts_sims: int,
    smoke_seed: int,
    skip_smoke_selfplay: bool,
) -> dict:
    audit = build_audit()
    summary = dict(audit.get("summary", {}))
    if int(summary.get("semantic_untested_count", -1)) != 0:
        raise RuntimeError(
            "rules-stability gate failed: semantic_untested_count != 0"
        )

    strict_kpi = run_strict_kpi(
        out_path=strict_kpi_out,
        players=2,
        board_type=BoardType.OCEANIA,
        round1_games=50,
        smoke_games=10,
        max_turns=120,
        seed=smoke_seed,
        min_round1_pct_15_30=0.20,
        max_round1_strict_rejected_games=0,
        max_smoke_strict_rejected_games=0,
    )
    if not bool(strict_kpi.get("passed", False)):
        raise RuntimeError("rules-stability gate failed: strict_kpi_runner did not pass")

    _run_pytest_gate()

    smoke_meta = {}
    if not skip_smoke_selfplay:
        smoke_meta = _run_strict_smoke_selfplay_gate(
            model_path=init_model_path,
            games=smoke_games,
            mcts_sims=smoke_mcts_sims,
            seed=smoke_seed + 17,
        )
        if int(smoke_meta.get("game_exceptions", 0)) > 0:
            raise RuntimeError(
                "rules-stability gate failed: strict smoke self-play had game exceptions"
            )

    return {
        "audit_summary": summary,
        "strict_kpi": strict_kpi,
        "smoke_selfplay_meta": smoke_meta,
    }


def evaluate_pilot_acceptance(manifest: dict) -> dict:
    history = list(manifest.get("history", []))
    if not history:
        return {
            "accepted": False,
            "reason": "no_history",
        }

    exception_free = True
    eval_margins: list[float] = []
    for h in history:
        ex = h.get("exception_counts", {}) or {}
        if int(ex.get("selfplay", 0)) + int(ex.get("eval", 0)) + int(ex.get("gate", 0)) > 0:
            exception_free = False
        eval_summary = h.get("eval_summary", {}) or {}
        if "nn_mean_margin" in eval_summary:
            try:
                eval_margins.append(float(eval_summary.get("nn_mean_margin", 0.0)))
            except Exception:
                pass

    last = history[-1]
    last_eval = last.get("eval_summary", {}) or {}
    gate_summary = last.get("gate_summary", {}) or {}
    stage1 = gate_summary.get("stage1", {}) or {}
    eval_win_rate = float(last_eval.get("nn_win_rate", 0.0) or 0.0)
    stage1_win_rate = float(stage1.get("nn_win_rate", gate_summary.get("nn_win_rate", 0.0)) or 0.0)
    non_worsening_margin = True
    if len(eval_margins) >= 2:
        non_worsening_margin = bool(eval_margins[-1] >= eval_margins[0])

    accepted = bool(
        exception_free
        and (
            eval_win_rate >= 0.15
            or (stage1_win_rate >= 0.45 and non_worsening_margin)
        )
    )
    return {
        "accepted": accepted,
        "exception_free": exception_free,
        "eval_win_rate": eval_win_rate,
        "stage1_win_rate": stage1_win_rate,
        "non_worsening_eval_margin": non_worsening_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v17.1 rules-stable restart flow")
    parser.add_argument(
        "--out-dir",
        default="reports/ml/alphazero_rules_v17_1_rules_stable_pilot",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--rules-baseline-id", default="rules_2026_03_08_explicit446")
    parser.add_argument("--seed", type=int, default=20260308)
    parser.add_argument("--strict-kpi-out", default="reports/ml/strict_kpi_pretrain.json")
    parser.add_argument("--smoke-games", type=int, default=10)
    parser.add_argument("--smoke-mcts-sims", type=int, default=16)
    parser.add_argument("--skip-smoke-selfplay", action="store_true", default=False)
    parser.add_argument("--only-gates", action="store_true", default=False)
    args = parser.parse_args()

    init_model_path = _resolve_init_model_path(args.init_model_path)
    gate_result = run_rules_stability_gate(
        strict_kpi_out=args.strict_kpi_out,
        init_model_path=init_model_path,
        smoke_games=args.smoke_games,
        smoke_mcts_sims=args.smoke_mcts_sims,
        smoke_seed=args.seed,
        skip_smoke_selfplay=bool(args.skip_smoke_selfplay),
    )
    print(json.dumps({"rules_stability_gate": gate_result}, indent=2))

    if args.only_gates:
        return

    manifest = run_auto_improve_alphazero(
        out_dir=args.out_dir,
        iterations=int(args.iterations),
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=240,
        games_per_iter=96,
        mcts_sims=48,
        c_puct=1.5,
        selfplay_value_blend=0.95,
        gate_value_blend=0.85,
        selfplay_root_dirichlet_epsilon=0.30,
        selfplay_root_dirichlet_alpha=0.40,
        selfplay_fail_on_exception=True,
        rollout_policy="fast",
        temperature_cutoff=8,
        strict_rules_mode=True,
        train_epochs=8,
        train_batch=256,
        train_hidden1=384,
        train_hidden2=192,
        train_dropout=0.2,
        train_lr_init=1e-4,
        train_lr_peak=5e-4,
        train_lr_warmup_epochs=3,
        train_weight_decay=0.0,
        train_score_value_weight=0.25,
        train_win_value_weight=1.0,
        train_init_model_path=init_model_path,
        train_init_first_iter_only=True,
        train_reinit_value_head=False,
        seed_champion_from_init=True,
        eval_games=20,
        eval_use_mcts=True,
        eval_mcts_sims=64,
        eval_mcts_c_puct=1.5,
        eval_mcts_value_blend=0.85,
        eval_mcts_rollout_policy="fast",
        eval_heuristic_policy="greedy",
        promotion_games=40,
        min_promotion_win_rate=0.5,
        gate_mode="champion",
        gate_mcts_sims=64,
        gate_stage_games=20,
        pilot_early_stop_iter=3,
        pilot_min_gate_win_rate=0.20,
        state_encoder_use_per_slot=True,
        state_encoder_use_hand_habitat_features=True,
        state_encoder_use_tray_per_slot=True,
        state_encoder_use_opponent_board=True,
        state_encoder_use_power_features=True,
        data_accumulation_enabled=True,
        data_accumulation_decay=0.5,
        max_accumulated_samples=200_000,
        teacher_games_per_iter=24,
        teacher_mcts_sims=96,
        teacher_value_blend=0.98,
        teacher_root_dirichlet_epsilon=0.10,
        teacher_root_dirichlet_alpha=0.25,
        teacher_rollout_policy="fast",
        value_target_score_scale=120.0,
        tie_value_target=0.5,
        dataset_workers=8,
        use_modal=True,
        modal_cpu_per_worker=2,
        seed=args.seed,
        clean_out_dir=True,
        start_iter=1,
        require_bird_audit_gate=True,
        expected_semantic_untested=0,
        rules_baseline_id=args.rules_baseline_id,
        allow_stale_lineage=False,
    )

    acceptance = evaluate_pilot_acceptance(manifest)
    print(json.dumps({"pilot_acceptance": acceptance}, indent=2))
    if not acceptance.get("accepted", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
