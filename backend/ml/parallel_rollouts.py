"""Process-parallel candidate evaluation for the rollout search.

The search scores each candidate move by playing whole games to the end; the
candidates are independent, so they parallelize perfectly across CPU cores.
This module provides:

  - a lazily created, process-wide worker pool (spawn context; each worker
    loads the registries and the distilled rollout student once), and
  - `eval_candidate(payload)`, the worker-side function that runs one
    candidate's rollouts with its own deterministic RNG seeds.

The sequential path in `make_search_chooser` is untouched when no pool is
passed, so gates/tests keep their exact historical behavior. Results in
parallel mode are deterministic given the seeds: each candidate's value
depends only on its own payload, never on scheduling order.

Kill switch: WINGSPAN_PARALLEL=off (or workers <= 1 / <= 2 CPUs) disables
pooling and the advisor falls back to the sequential path.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor

# ─── Worker side ─────────────────────────────────────────────────────────────

_W_STUDENT = None  # (model, encoder) inside a worker process


def _init_worker(student_path: str) -> None:
    """Runs once per worker process (spawn): load data + the rollout student."""
    # Keep BLAS single-threaded inside workers: the matrices are tiny and
    # oversubscription (workers x BLAS threads) would thrash 4 cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from backend.config import EXCEL_FILE
    from backend.data.registries import load_all
    load_all(EXCEL_FILE)

    global _W_STUDENT
    from backend.ml.rollout_student import load_rollout_student
    _W_STUDENT = load_rollout_student(student_path)


def eval_candidate(payload: dict) -> tuple[int, float, int]:
    """Run one candidate's rollouts. Returns (cand_idx, value_total, n_ok).

    Deterministic given payload["seed"]; independent of the other candidates.
    """
    from backend.ml.two_player import (
        make_net_sampling_chooser,
        play_multi,
        _refill_tray,
    )
    from backend.solver.simulation import execute_move_on_sim, fast_clone_game

    game = pickle.loads(payload["game"])
    move = pickle.loads(payload["move"])
    my_idx = payload["my_idx"]
    lam = payload["lam"]
    n_roll = payload["n_roll"]
    deadline = payload["deadline"]
    determinize = payload["determinize"]
    seed = payload["seed"]

    sm, se = _W_STUDENT
    roll_me = make_net_sampling_chooser(sm, se, payload["temperature"], seed=seed * 3 + 1)
    roll_opp = make_net_sampling_chooser(sm, se, payload["temperature"], seed=seed * 3 + 2)
    det_rng = random.Random(seed * 3)

    def _obj(scores):
        if lam == 0.0:
            return scores[my_idx]
        return scores[my_idx] - lam * max(
            scores[j] for j in range(len(scores)) if j != my_idx
        )

    total = 0.0
    n_ok = 0
    for _ in range(n_roll):
        if deadline is not None and n_ok > 0 and time.time() >= deadline:
            break
        sim = fast_clone_game(game)
        if determinize:
            deck = getattr(sim, "_deck_cards", None)
            if isinstance(deck, list) and len(deck) > 1:
                det_rng.shuffle(deck)
        sp = sim.current_player
        if not execute_move_on_sim(sim, sp, move):
            continue
        sim.advance_turn()
        _refill_tray(sim)
        choosers = [roll_me if i == my_idx else roll_opp for i in range(sim.num_players)]
        total += _obj(play_multi(sim, choosers))
        n_ok += 1
    return payload["cand_idx"], total, n_ok


# ─── Parent side ─────────────────────────────────────────────────────────────

_POOL: ProcessPoolExecutor | None = None
_POOL_STUDENT_PATH: str | None = None


def default_workers() -> int:
    if os.getenv("WINGSPAN_PARALLEL", "").lower() in {"off", "0", "none"}:
        return 0
    env = os.getenv("WINGSPAN_PARALLEL_WORKERS", "")
    if env.isdigit():
        return int(env)
    cpus = os.cpu_count() or 1
    if cpus <= 2:
        return 0  # not worth the worker startup on tiny machines
    return min(cpus - 1, 6)


def get_rollout_pool(student_path: str) -> ProcessPoolExecutor | None:
    """Process-wide pool whose workers hold the given student. None = run
    sequentially. Recreated if a different student path is requested."""
    global _POOL, _POOL_STUDENT_PATH
    workers = default_workers()
    if workers <= 1 or not student_path or not os.path.exists(student_path):
        return None
    if _POOL is not None and _POOL_STUDENT_PATH == student_path:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
    import multiprocessing as mp

    _POOL = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(student_path,),
    )
    _POOL_STUDENT_PATH = student_path
    return _POOL


def shutdown_pool() -> None:
    global _POOL, _POOL_STUDENT_PATH
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
        _POOL_STUDENT_PATH = None


def eval_candidates_parallel(
    pool: ProcessPoolExecutor,
    game,
    candidates: list,
    *,
    my_idx: int,
    lam: float,
    n_roll: int,
    temperature: float,
    determinize: bool,
    deadline: float | None,
    pool_seed: int,
) -> list[tuple[float, int]]:
    """Evaluate all candidates on the pool. Returns [(value_total, n_ok)] in
    candidate order. A crashed/failed candidate gets (0.0, 0)."""
    game_bytes = pickle.dumps(game, protocol=5)
    futures = []
    for i, move in enumerate(candidates):
        futures.append(
            pool.submit(
                eval_candidate,
                {
                    "game": game_bytes,
                    "move": pickle.dumps(move, protocol=5),
                    "cand_idx": i,
                    "my_idx": my_idx,
                    "lam": lam,
                    "n_roll": n_roll,
                    "temperature": temperature,
                    "determinize": determinize,
                    "deadline": deadline,
                    "seed": pool_seed * 1_000_003 + i * 7919 + 17,
                },
            )
        )
    results: list[tuple[float, int]] = [(0.0, 0)] * len(candidates)
    for fut in futures:
        try:
            # Workers enforce the deadline internally (they stop starting new
            # playouts past it), so a generous outer timeout is just a hang guard.
            timeout = None
            if deadline is not None:
                timeout = max(10.0, deadline - time.time() + 60.0)
            idx, total, n_ok = fut.result(timeout=timeout)
            results[idx] = (total, n_ok)
        except Exception:
            continue
    return results
