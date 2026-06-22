import pytest

from backend.solver.replay_video_game import run_scripted_replay


@pytest.mark.parametrize(
    "seed,expected_divergence,expected_you,expected_opp,expected_opp_total",
    [
        # NOTE: goldens updated after the reset_tray refill fix (actions.py). That
        # fix made the engine reproduce the real recorded games PERFECTLY:
        # scripted-player divergence_count fell from {2,0,4} -> {0,0,0}. The "You"
        # scores below are the now-faithful replay; opponent is stochastic
        # (pick_weighted_random_move, seed-deterministic), so its totals shifted
        # as the corrected reset changed the deck order.
        (
            20260210,
            0,
            {
                "bird_vp": 59,
                "eggs": 3,
                "cached_food": 2,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 19,
                "nectar": 5,
                "total": 117,
            },
            {
                "bird_vp": 43,
                "eggs": 6,
                "cached_food": 18,
                "tucked_cards": 3,
                "bonus_cards": 0,
                "round_goals": 13,
                "nectar": 10,
                "total": 93,
            },
            93,
        ),
        (
            20260211,
            0,
            {
                "bird_vp": 63,
                "eggs": 3,
                "cached_food": 2,
                "tucked_cards": 23,
                "bonus_cards": 6,
                "round_goals": 16,
                "nectar": 5,
                "total": 118,
            },
            {
                "bird_vp": 35,
                "eggs": 10,
                "cached_food": 24,
                "tucked_cards": 4,
                "bonus_cards": 8,
                "round_goals": 16,
                "nectar": 10,
                "total": 107,
            },
            107,
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
