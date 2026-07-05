"""Tests for the batch move scorer and the distilled fast rollout policy."""

from pathlib import Path

import numpy as np
import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType


def setup_module():
    load_all(EXCEL_FILE)


BIG_NET = Path("reports/ml/solo_seed/solo_net_spread.npz")
STUDENT = Path("reports/ml/solo_seed/rollout_student.npz")


def _play_states(n_decisions=25, seed=7):
    """Real mid-game decision states from a quick heuristic game."""
    from backend.ml.two_player import build_2p_game, heuristic_chooser
    from backend.solver.move_generator import generate_all_moves
    from backend.solver.simulation import fast_clone_game, execute_move_on_sim, _refill_tray

    out = []
    game = build_2p_game(seed, BoardType.OCEANIA)
    while len(out) < n_decisions and not game.is_game_over:
        p = game.current_player
        if p.action_cubes_remaining <= 0:
            if all(pl.action_cubes_remaining <= 0 for pl in game.players):
                game.advance_round()
            else:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            continue
        moves = generate_all_moves(game, p)
        if not moves:
            p.action_cubes_remaining = 0
            game.advance_turn()
            continue
        out.append((fast_clone_game(game), game.current_player_idx))
        execute_move_on_sim(game, p, heuristic_chooser(game, p, moves))
        game.advance_turn()
        _refill_tray(game)
    return out


@pytest.mark.skipif(not BIG_NET.exists(), reason="deployed net not present")
def test_score_moves_matches_score_move():
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.solver.move_generator import generate_all_moves

    model = FactorizedPolicyModel(BIG_NET)
    enc = StateEncoder.resolve_for_model(model.meta)

    checked = 0
    for game, idx in _play_states():
        p = game.players[idx]
        moves = generate_all_moves(game, p)
        if not moves:
            continue
        st = np.asarray(enc.encode(game, idx), dtype=np.float32)
        logits, _ = model.forward(st)
        old = [model.score_move(st, m, p, logits=logits) for m in moves]
        new = model.score_moves(st, moves, p, logits=logits)
        assert len(old) == len(new)
        np.testing.assert_allclose(new, old, rtol=1e-4, atol=1e-5)
        assert int(np.argmax(old)) == int(np.argmax(new))
        checked += 1
    assert checked >= 10


def test_lean_score_key_matches_full_targets():
    """The lean per-head key must equal the key built from the full
    encode_factorized_targets() dict for every generated move — otherwise
    score_moves() would silently diverge from the training-time encoding."""
    from backend.ml.factorized_policy import (
        encode_factorized_targets,
        encode_factorized_score_key,
        RELEVANT_HEADS_BY_ACTION,
        ACTION_TYPE_TO_ID,
    )
    from backend.solver.move_generator import generate_all_moves

    checked = 0
    for game, idx in _play_states(n_decisions=25):
        p = game.players[idx]
        for m in generate_all_moves(game, p):
            t = encode_factorized_targets(m, p)
            aid = ACTION_TYPE_TO_ID[m.action_type]
            heads = RELEVANT_HEADS_BY_ACTION.get(aid, [])
            full_key = (aid,) + tuple(int(t[h]) for h in heads)
            assert encode_factorized_score_key(m, p) == full_key
            checked += 1
    assert checked >= 50


def test_fast_rollout_encoder_shape_and_bounds():
    from backend.ml.rollout_student import FastRolloutEncoder

    enc = FastRolloutEncoder()
    names = enc.feature_names()
    for game, idx in _play_states(n_decisions=8):
        v = enc.encode(game, idx)
        assert len(v) == len(names)
        assert all(-0.001 <= x <= 1.001 for x in v)


@pytest.mark.skipif(not BIG_NET.exists(), reason="deployed net not present")
def test_strong_move_respects_time_budget_on_turn_one():
    """Turn-1 decisions have the longest playouts; a heavy preset used to run
    a single determinization for many minutes past its budget (frozen app)."""
    import time
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.ml.rollout_student import load_rollout_student
    from backend.ml.two_player import build_2p_game
    from backend.solver.strong_move import strong_engine_best_move

    model = FactorizedPolicyModel(BIG_NET)
    enc = StateEncoder.resolve_for_model(model.meta)
    pair = load_rollout_student(STUDENT) if STUDENT.exists() else None
    r_model, r_enc = pair if pair else (None, None)
    game = build_2p_game(11, BoardType.OCEANIA)  # fresh game = worst case

    budget = 6.0
    t0 = time.time()
    best, _ = strong_engine_best_move(
        game, 0, model, enc,
        top_k=20, rollouts=36,  # the old runaway config
        temperature=0.3, time_budget_s=budget, max_determinizations=3, seed=0,
        rollout_model=r_model, rollout_encoder=r_enc,
    )
    elapsed = time.time() - t0
    assert best is not None
    # Allow generous overshoot for one in-flight playout + slow CI machines,
    # but nothing like the unbounded minutes-long runaway.
    assert elapsed < budget * 5, f"deadline not enforced: {elapsed:.1f}s"


@pytest.mark.skipif(not STUDENT.exists(), reason="student not trained yet")
def test_student_loads_and_scores_moves():
    from backend.ml.rollout_student import load_rollout_student
    from backend.solver.move_generator import generate_all_moves

    pair = load_rollout_student(STUDENT)
    assert pair is not None
    model, enc = pair
    assert model.use_move_value_head is False  # students skip the per-move dot
    game, idx = _play_states(n_decisions=3)[-1]
    p = game.players[idx]
    moves = generate_all_moves(game, p)
    st = np.asarray(enc.encode(game, idx), dtype=np.float32)
    logits, _ = model.forward(st)
    scores = model.score_moves(st, moves, p, logits=logits)
    assert len(scores) == len(moves)
    assert all(np.isfinite(s) for s in scores)
