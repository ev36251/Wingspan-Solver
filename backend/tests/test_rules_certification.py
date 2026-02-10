from backend.models.enums import BoardType
from backend.rules.certification import run_conformance_suite, power_mapping_report


def test_power_mapping_report_has_no_fallback() -> None:
    rep = power_mapping_report()
    assert rep["total_birds"] > 0
    assert rep["fallback_count"] == 0
    assert rep["coverage_non_fallback"] == 1.0


def test_conformance_smoke_has_no_critical_issues() -> None:
    result = run_conformance_suite(
        games=3,
        players=2,
        board_type=BoardType.OCEANIA,
        max_steps=220,
        seed=11,
    )
    # Smoke threshold: no critical invariants violated.
    assert result["critical_issues_total"] == 0
    assert result["conformance_rate"] >= 0.99
