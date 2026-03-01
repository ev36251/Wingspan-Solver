import pytest

from backend.solver.replay_video_game import run_scripted_replay


@pytest.mark.parametrize(
    "seed,expected_divergence,expected_you,expected_opp,expected_opp_total",
    [
        (
            20260210,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 19,
                "nectar": 5,
                "total": 116,
            },
            {
                "bird_vp": 27,
                "eggs": 17,
                "cached_food": 0,
                "tucked_cards": 12,
                "bonus_cards": 3,
                "round_goals": 13,
                "nectar": 10,
                "total": 82,
            },
            82,
        ),
        (
            20260211,
            2,
            {
                "bird_vp": 58,
                "eggs": 4,
                "cached_food": 1,
                "tucked_cards": 24,
                "bonus_cards": 6,
                "round_goals": 19,
                "nectar": 7,
                "total": 119,
            },
            {
                "bird_vp": 42,
                "eggs": 16,
                "cached_food": 25,
                "tucked_cards": 2,
                "bonus_cards": 4,
                "round_goals": 13,
                "nectar": 12,
                "total": 114,
            },
            114,
        ),
        (
            20260212,
            0,
            {
                "bird_vp": 63,
                "eggs": 3,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 16,
                "nectar": 5,
                "total": 117,
            },
            {
                "bird_vp": 30,
                "eggs": 18,
                "cached_food": 23,
                "tucked_cards": 8,
                "bonus_cards": 0,
                "round_goals": 16,
                "nectar": 12,
                "total": 107,
            },
            107,
        ),
    ],
)
def test_video_replay_golden_games(seed, expected_divergence, expected_you, expected_opp, expected_opp_total):
    result = run_scripted_replay(seed=seed, max_turns=260)
    assert result["divergence_count"] == expected_divergence
    assert result["script_steps_consumed"] == result["script_steps_total"]
    assert result["you_score"] == expected_you
    if expected_opp is not None:
        assert result["opp_score"] == expected_opp
    if expected_opp_total is not None:
        assert result["opp_score"]["total"] == expected_opp_total
