from pathlib import Path

from backend.engine_search.is_mcts import infer_num_determinizations
from backend.models.enums import BoardType
from backend.ml.evaluate_factorized_pool import evaluate_against_pool
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.kpi_gate import run_gate
from backend.ml.train_factorized_bc import train_bc


def test_infer_num_determinizations_monotonic() -> None:
    assert infer_num_determinizations(2000) <= infer_num_determinizations(5000)
    assert infer_num_determinizations(5000) <= infer_num_determinizations(10000)
    assert infer_num_determinizations(10000) <= infer_num_determinizations(20000)


def test_evaluate_against_pool_smoke(tmp_path: Path) -> None:
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
        seed=101,
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
        seed=101,
    )

    res = evaluate_against_pool(
        model_path=str(model),
        opponents=["heuristic"],
        games_per_opponent=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=101,
    )
    assert "summary" in res
    assert res["summary"]["games"] == 2
    assert len(res["by_opponent"]) == 1


def test_kpi_gate_non_regression() -> None:
    candidate = {
        "summary": {
            "nn_win_rate": 0.55,
            "nn_mean_score": 88.0,
            "nn_rate_ge_100": 0.22,
            "nn_rate_ge_120": 0.05,
        }
    }
    baseline = {
        "summary": {
            "nn_win_rate": 0.50,
            "nn_mean_score": 80.0,
            "nn_rate_ge_100": 0.18,
            "nn_rate_ge_120": 0.03,
        }
    }

    p = Path("/tmp/kpi_gate_candidate.json")
    b = Path("/tmp/kpi_gate_baseline.json")
    p.write_text(__import__("json").dumps(candidate), encoding="utf-8")
    b.write_text(__import__("json").dumps(baseline), encoding="utf-8")

    out = run_gate(
        candidate_path=str(p),
        baseline_path=str(b),
        min_win_rate=0.5,
        min_mean_score=70.0,
        min_rate_ge_100=0.1,
        min_rate_ge_120=0.01,
        require_non_regression=True,
    )
    assert out["passed"] is True
