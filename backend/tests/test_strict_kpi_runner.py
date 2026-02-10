from pathlib import Path

from backend.ml.strict_kpi_runner import run_strict_kpi
from backend.models.enums import BoardType


def test_strict_kpi_runner_smoke(tmp_path: Path):
    out = tmp_path / "strict_kpi.json"
    result = run_strict_kpi(
        out_path=str(out),
        players=2,
        board_type=BoardType.OCEANIA,
        round1_games=4,
        smoke_games=2,
        max_turns=60,
        seed=7,
        # permissive thresholds for smoke
        min_round1_pct_15_30=0.0,
        max_round1_strict_rejected_games=4,
        max_smoke_strict_rejected_games=2,
    )
    assert out.exists()
    assert "checks" in result
    assert "power_coverage" in result
    assert "round1_benchmark" in result
    assert "smoke_dataset_meta" in result
    assert isinstance(result["passed"], bool)

