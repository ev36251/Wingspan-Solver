import json

from backend.models.enums import BoardType


def test_kpi_benchmark_suite_writes_artifacts(tmp_path, monkeypatch):
    from backend.ml import kpi_benchmark_suite as suite

    def fake_round1(**kwargs):
        return {
            "samples": 8,
            "strict_rejected_games": 0,
            "totals": {"mean": 21.5, "pct_15_30": 0.75},
            "rows": [],
        }

    def fake_full(**kwargs):
        return {
            2: {
                "sample_size_player_scores": 6,
                "mean_player_score": 72.0,
                "mean_winner_score": 85.0,
                "rate_ge_100": 0.3333,
                "rate_ge_120": 0.1667,
                "max_winner_score": 128,
                "failed_games": 0,
            },
            "meta": {"board_type": "oceania"},
        }

    monkeypatch.setattr(suite, "run_round1_benchmark", fake_round1)
    monkeypatch.setattr(suite, "run_benchmark", fake_full)

    out = tmp_path / "bench"
    summary = suite.run_kpi_benchmark_suite(
        out_dir=str(out),
        players=2,
        board_type=BoardType.OCEANIA,
        round1_games=4,
        full_games=3,
        max_turns=120,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        seed=42,
    )

    round1_path = out / "round1_benchmark.json"
    full_path = out / "full_game_benchmark.json"
    summary_path = out / "kpi_benchmark_summary.json"

    assert round1_path.exists()
    assert full_path.exists()
    assert summary_path.exists()

    round1_doc = json.loads(round1_path.read_text(encoding="utf-8"))
    full_doc = json.loads(full_path.read_text(encoding="utf-8"))
    summary_doc = json.loads(summary_path.read_text(encoding="utf-8"))

    assert round1_doc["totals"]["mean"] == 21.5
    assert full_doc["2"]["mean_player_score"] == 72.0
    assert summary["round1"]["pct_15_30"] == 0.75
    assert summary["full_game"]["rate_ge_120"] == 0.1667
    assert summary_doc["full_game"]["max_winner_score"] == 128
