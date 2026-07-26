"""Tests for process-parallel candidate rollout evaluation.

The parallel branch must: return a legal move, be deterministic given seeds
(scheduling order must not matter), and leave the sequential path untouched
when no pool is passed.
"""

from pathlib import Path

import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType

STUDENT = Path("reports/ml/solo_seed/rollout_student.npz")
BIG_NET = Path("reports/ml/solo_seed/solo_net_spread.npz")


def setup_module():
    load_all(EXCEL_FILE)


@pytest.mark.skipif(
    not (STUDENT.exists() and BIG_NET.exists()), reason="model files not present"
)
def test_parallel_chooser_is_deterministic_and_legal():
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.ml.rollout_student import load_rollout_student
    from backend.ml.two_player import build_2p_game, make_search_chooser, heuristic_chooser
    from backend.ml.parallel_rollouts import _init_worker, shutdown_pool
    from backend.solver.move_generator import generate_all_moves
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    model = FactorizedPolicyModel(BIG_NET)
    enc = StateEncoder.resolve_for_model(model.meta)
    sm, se = load_rollout_student(STUDENT)

    pool = ProcessPoolExecutor(
        max_workers=2,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(STUDENT),),
    )
    try:
        game = build_2p_game(21, BoardType.OCEANIA)
        player = game.players[0]
        moves = generate_all_moves(game, player)
        assert len(moves) > 1

        def pick():
            chooser = make_search_chooser(
                model, enc, 4, 0, heuristic_chooser,
                objective="selfish", rollouts=3, temperature=0.3,
                determinize=True, rollout_model=sm, rollout_encoder=se,
                pool=pool, pool_seed=5,
            )
            return chooser(game, player, moves)

        first = pick()
        second = pick()
        assert first in moves
        # Deterministic given seeds: scheduling order must not change the pick.
        assert first.description == second.description
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        shutdown_pool()


def test_sequential_path_unchanged_without_pool():
    """pool=None takes the exact pre-existing sequential code path."""
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.ml.two_player import build_2p_game, make_search_chooser, heuristic_chooser
    from backend.solver.move_generator import generate_all_moves

    if not BIG_NET.exists():
        pytest.skip("net not present")
    model = FactorizedPolicyModel(BIG_NET)
    enc = StateEncoder.resolve_for_model(model.meta)
    game = build_2p_game(22, BoardType.OCEANIA)
    player = game.players[0]
    moves = generate_all_moves(game, player)
    chooser = make_search_chooser(
        model, enc, 3, 0, heuristic_chooser,
        objective="selfish", rollouts=2, temperature=0.3, determinize=True,
    )
    assert chooser(game, player, moves) in moves


def test_default_workers_is_opt_in(monkeypatch):
    from backend.ml import parallel_rollouts as pr

    # Unset -> sequential (the safe default; the pool can hang on some setups).
    monkeypatch.delenv("WINGSPAN_PARALLEL", raising=False)
    monkeypatch.delenv("WINGSPAN_PARALLEL_WORKERS", raising=False)
    assert pr.default_workers() == 0

    # Explicit off stays off.
    monkeypatch.setenv("WINGSPAN_PARALLEL", "off")
    assert pr.default_workers() == 0

    # Opt in with a flag (auto-sizes) or an explicit count.
    monkeypatch.setenv("WINGSPAN_PARALLEL", "on")
    assert pr.default_workers() >= 0  # 0 on <=2-core machines, else cores-1
    monkeypatch.delenv("WINGSPAN_PARALLEL")
    monkeypatch.setenv("WINGSPAN_PARALLEL_WORKERS", "3")
    assert pr.default_workers() == 3
