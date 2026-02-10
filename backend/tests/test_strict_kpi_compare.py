import json
from pathlib import Path

from backend.ml.strict_kpi_compare import run_compare


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_kpi_compare_pass(tmp_path: Path):
    base = {
        "power_coverage": {"strict_certified_birds": 50},
        "round1_benchmark": {"totals": {"mean": 20.0, "pct_15_30": 0.4}, "strict_rejected_games": 0},
        "smoke_dataset_meta": {"mean_player_score": 70.0, "strict_rejected_games": 0},
    }
    cand = {
        "power_coverage": {"strict_certified_birds": 51},
        "round1_benchmark": {"totals": {"mean": 20.2, "pct_15_30": 0.41}, "strict_rejected_games": 0},
        "smoke_dataset_meta": {"mean_player_score": 70.1, "strict_rejected_games": 0},
    }
    bp = tmp_path / "base.json"
    cp = tmp_path / "cand.json"
    _write(bp, base)
    _write(cp, cand)
    res = run_compare(str(cp), str(bp), require_non_regression=True)
    assert res["passed"] is True


def test_strict_kpi_compare_fail_on_rejection(tmp_path: Path):
    base = {
        "power_coverage": {"strict_certified_birds": 50},
        "round1_benchmark": {"totals": {"mean": 20.0, "pct_15_30": 0.4}, "strict_rejected_games": 0},
        "smoke_dataset_meta": {"mean_player_score": 70.0, "strict_rejected_games": 0},
    }
    cand = {
        "power_coverage": {"strict_certified_birds": 49},
        "round1_benchmark": {"totals": {"mean": 19.0, "pct_15_30": 0.3}, "strict_rejected_games": 2},
        "smoke_dataset_meta": {"mean_player_score": 60.0, "strict_rejected_games": 1},
    }
    bp = tmp_path / "base.json"
    cp = tmp_path / "cand.json"
    _write(bp, base)
    _write(cp, cand)
    res = run_compare(
        str(cp),
        str(bp),
        max_round1_strict_rejected_games=0,
        max_smoke_strict_rejected_games=0,
        min_strict_certified_birds=50,
        require_non_regression=True,
    )
    assert res["passed"] is False
