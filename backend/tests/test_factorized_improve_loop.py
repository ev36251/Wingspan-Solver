from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.train_factorized_bc import train_bc
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.auto_improve_factorized import run_auto_improve_factorized


def test_factorized_eval_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "bc.jsonl"
    meta = tmp_path / "bc.meta.json"
    model = tmp_path / "bc_model.npz"

    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=8,
        lookahead_depth=1,
        n_step=1,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden=64,
        lr=1e-3,
        val_split=0.2,
        seed=8,
    )

    ev = evaluate_factorized_vs_heuristic(
        model_path=str(model),
        games=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=8,
    )
    assert ev.games == 2
    assert ev.nn_wins + ev.heuristic_wins + ev.ties == 2


def test_auto_improve_factorized_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        games_per_iter=2,
        proposal_top_k=3,
        lookahead_depth=1,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=1,
        train_epochs=1,
        train_batch=32,
        train_hidden=64,
        train_lr=1e-3,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        promotion_games=4,
        pool_games_per_opponent=2,
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=0.0,
        min_gate_mean_score=0.0,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        engine_teacher_prob=0.0,
        engine_time_budget_ms=25,
        engine_num_determinizations=0,
        engine_max_rollout_depth=24,
        seed=9,
    )

    assert (out_dir / "auto_improve_factorized_manifest.json").exists()
    assert len(manifest["history"]) == 1
    # strict gating field exists
    assert "promotion_gate" in manifest["history"][0]
    assert "pool_eval" in manifest["history"][0]
    assert "kpi_gate" in manifest["history"][0]


def test_generate_bc_dataset_engine_teacher_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "bc_engine.jsonl"
    meta = tmp_path / "bc_engine.meta.json"
    out = generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=10,
        lookahead_depth=0,
        n_step=1,
        engine_teacher_prob=1.0,
        engine_time_budget_ms=5,
        engine_num_determinizations=2,
        engine_max_rollout_depth=16,
    )
    pi = out["policy_improvement"]
    assert pi["engine_teacher_calls"] >= 1
    assert pi["engine_teacher_applied"] >= 1
    assert "value_target_config" in out
