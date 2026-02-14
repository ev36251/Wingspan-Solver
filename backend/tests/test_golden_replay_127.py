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
                "bird_vp": 41,
                "eggs": 12,
                "cached_food": 9,
                "tucked_cards": 3,
                "bonus_cards": 4,
                "round_goals": 22,
                "nectar": 12,
                "total": 103,
            },
            103,
        ),
        (
            20260211,
            2,
            {
                "bird_vp": 59,
                "eggs": 4,
                "cached_food": 1,
                "tucked_cards": 24,
                "bonus_cards": 6,
                "round_goals": 20,
                "nectar": 7,
                "total": 121,
            },
            {
                "bird_vp": 37,
                "eggs": 20,
                "cached_food": 24,
                "tucked_cards": 0,
                "bonus_cards": 6,
                "round_goals": 21,
                "nectar": 10,
                "total": 118,
            },
            118,
        ),
        (
            20260212,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 2,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 24,
                "nectar": 5,
                "total": 122,
            },
            {
                "bird_vp": 27,
                "eggs": 0,
                "cached_food": 22,
                "tucked_cards": 10,
                "bonus_cards": 0,
                "round_goals": 17,
                "nectar": 12,
                "total": 88,
            },
            88,
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
