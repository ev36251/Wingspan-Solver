import pytest

from backend.solver.replay_video_game import run_scripted_replay


@pytest.mark.parametrize(
    "seed,expected_you,expected_opp,expected_opp_total",
    [
        (
            20260210,
            {
                "bird_vp": 59,
                "eggs": 11,
                "cached_food": 2,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 22,
                "nectar": 5,
                "total": 128,
            },
            {
                "bird_vp": 43,
                "eggs": 8,
                "cached_food": 4,
                "tucked_cards": 0,
                "bonus_cards": 0,
                "round_goals": 19,
                "nectar": 7,
                "total": 81,
            },
            81,
        ),
        (
            20260211,
            {
                "bird_vp": 59,
                "eggs": 12,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 22,
                "nectar": 5,
                "total": 128,
            },
            None,
            None,
        ),
        (
            20260212,
            {
                "bird_vp": 59,
                "eggs": 13,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 24,
                "nectar": 5,
                "total": 131,
            },
            None,
            None,
        ),
    ],
)
def test_video_replay_golden_games(seed, expected_you, expected_opp, expected_opp_total):
    result = run_scripted_replay(seed=seed, max_turns=260)
    assert result["divergence_count"] == 0
    assert result["script_steps_consumed"] == result["script_steps_total"]
    assert result["you_score"] == expected_you
    if expected_opp is not None:
        assert result["opp_score"] == expected_opp
    if expected_opp_total is not None:
        assert result["opp_score"]["total"] == expected_opp_total
