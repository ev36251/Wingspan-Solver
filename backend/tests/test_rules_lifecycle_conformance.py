from backend.rules.lifecycle_conformance import run_lifecycle_conformance_checks


def test_lifecycle_conformance_checks_all_pass() -> None:
    result = run_lifecycle_conformance_checks()
    assert result["checks_total"] >= 5
    assert result["checks_failed"] == 0
    assert result["all_passed"] is True
