from backend.models.enums import BoardType
from backend.solver.round1_benchmark import run_round1_benchmark


def test_round1_benchmark_report_shape():
    rep = run_round1_benchmark(
        games=5,
        players=2,
        board_type=BoardType.OCEANIA,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        seed=42,
    )
    assert rep["games_requested"] == 5
    assert rep["players"] == 2
    assert rep["board_type"] == "oceania"
    assert "totals" in rep
    assert "rows" in rep
    assert rep["samples"] >= 0
    assert 0.0 <= rep["totals"]["pct_15_30"] <= 1.0

