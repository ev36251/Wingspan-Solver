import pytest

from backend.solver.replay_video_game import run_scripted_replay


@pytest.mark.parametrize(
    "seed,expected_divergence,expected_you,expected_opp,expected_opp_total",
    [
        # NOTE: goldens are re-baselined whenever a corrected bird power changes a
        # replay's deck/board, AS LONG AS divergence_count stays 0 (the scripted
        # player's recorded moves still reproduce exactly = the replay is still
        # faithful). Last update: the "exact powers" pass (batches 1-3) made
        # several auto-parsed birds behave per their full rules text.
        (
            20260210,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 22,
                "nectar": 5,
                "total": 119,
            },
            {
                "bird_vp": 48,
                "eggs": 2,
                "cached_food": 19,
                "tucked_cards": 6,
                "bonus_cards": 0,
                "round_goals": 7,
                "nectar": 10,
                "total": 92,
            },
            92,
        ),
        (
            20260211,
            0,
            {
                "bird_vp": 63,
                "eggs": 3,
                "cached_food": 3,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 16,
                "nectar": 5,
                "total": 119,
            },
            {
                "bird_vp": 35,
                "eggs": 19,
                "cached_food": 21,
                "tucked_cards": 4,
                "bonus_cards": 6,
                "round_goals": 16,
                "nectar": 10,
                "total": 111,
            },
            111,
        ),
        (
            20260212,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 1,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 16,
                "nectar": 5,
                "total": 113,
            },
            {
                "bird_vp": 38,
                "eggs": 6,
                "cached_food": 19,
                "tucked_cards": 4,
                "bonus_cards": 0,
                "round_goals": 16,
                "nectar": 12,
                "total": 95,
            },
            95,
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
