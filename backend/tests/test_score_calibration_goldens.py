import json
from pathlib import Path

from backend.models.enums import BoardType
from backend.rules.score_calibration import run_constructed_score_case


def _load_fixture() -> dict:
    path = Path(__file__).parent / "fixtures" / "score_calibration_goldens.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_score_calibration_goldens() -> None:
    fixture = _load_fixture()
    board_type = BoardType(fixture["board_type"])
    max_turns = int(fixture["max_turns"])
    players = int(fixture["players"])
    strict_rules_mode = bool(fixture["strict_rules_mode"])
    cases = list(fixture.get("cases", []))
    assert len(cases) >= 20, "score calibration should include at least 20 golden games"

    for case in cases:
        seed = int(case["seed"])
        expected_scores = case["scores"]

        result = run_constructed_score_case(
            seed=seed,
            players=players,
            board_type=board_type,
        )

        assert result["scores"] == expected_scores

        # Keep a structural guard that total always matches component sum.
        for _, bd in result["scores"].items():
            total = int(bd["total"])
            component_sum = sum(int(v) for k, v in bd.items() if k != "total")
            assert total == component_sum
