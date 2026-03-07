import pytest

from backend.solver.replay_video_game import run_scripted_replay


@pytest.mark.parametrize(
    "seed,expected_divergence,expected_you,expected_opp,expected_opp_total",
    [
        (
            20260210,
            2,
            {
                "bird_vp": 55,
                "eggs": 3,
                "cached_food": 0,
                "tucked_cards": 24,
                "bonus_cards": 6,
                "round_goals": 17,
                "nectar": 8,
                "total": 113,
            },
            {
                "bird_vp": 52,
                "eggs": 2,
                "cached_food": 7,
                "tucked_cards": 1,
                "bonus_cards": 9,
                "round_goals": 14,
                "nectar": 8,
                "total": 93,
            },
            93,
        ),
        (
            20260211,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 2,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 13,
                "nectar": 5,
                "total": 111,
            },
            {
                "bird_vp": 39,
                "eggs": 24,
                "cached_food": 20,
                "tucked_cards": 9,
                "bonus_cards": 4,
                "round_goals": 19,
                "nectar": 12,
                "total": 127,
            },
            127,
        ),
        (
            20260212,
            4,
            {
                "bird_vp": 59,
                "eggs": 5,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 17,
                "nectar": 5,
                "total": 116,
            },
            {
                "bird_vp": 45,
                "eggs": 7,
                "cached_food": 14,
                "tucked_cards": 7,
                "bonus_cards": 0,
                "round_goals": 14,
                "nectar": 12,
                "total": 99,
            },
            99,
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
