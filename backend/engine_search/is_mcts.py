"""IS-MCTS-style root search for Wingspan best-move analysis.

This is a practical phase-1 implementation:
- Information-set via determinization sampling.
- Root-level PUCT allocation across legal actions.
- Heuristic priors + rollout final-score evaluation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from backend.engine_search.belief import sample_hidden_state
from backend.engine_search.time_manager import TimeManager
from backend.engine_search.transposition import RootTransposition
from backend.models.game_state import GameState
from backend.solver.heuristics import rank_moves
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import execute_move_on_sim, simulate_playout, _refill_tray
from backend.ml.action_codec import action_signature


@dataclass
class EngineConfig:
    time_budget_ms: int = 15000
    num_determinizations: int = 0  # <=0 means auto-tune from budget
    max_rollout_depth: int = 24
    c_puct: float = 1.4
    top_k: int = 5
    seed: int = 0


@dataclass
class EngineMoveStat:
    move: Move
    visit_count: int
    mean_vp: float
    prior: float
    ucb: float


@dataclass
class EngineResult:
    best_move: Move | None
    top_k_moves: list[EngineMoveStat]
    nodes: int
    simulations: int
    determinizations: int
    elapsed_ms: float


def _rollout_score_after_move(
    game: GameState,
    player_name: str,
    move: Move,
    max_rollout_depth: int,
) -> float | None:
    sim_player = game.current_player
    if sim_player.name != player_name:
        return None
    if not execute_move_on_sim(game, sim_player, move):
        return None
    game.advance_turn()
    _refill_tray(game)
    finals = simulate_playout(game, max_turns=max_rollout_depth)
    if player_name not in finals:
        return None
    return float(finals[player_name])


def _heuristic_priors(game: GameState, player_name: str, moves: list[Move]) -> dict[str, float]:
    priors: dict[str, float] = {}
    player = game.get_player(player_name)
    if player is None:
        uni = 1.0 / max(1, len(moves))
        return {action_signature(m): uni for m in moves}

    ranked = rank_moves(game, player)
    by_sig: dict[str, float] = {action_signature(rm.move): max(0.01, float(rm.score)) for rm in ranked}

    vals = []
    for m in moves:
        vals.append(by_sig.get(action_signature(m), 0.01))
    total = sum(vals)
    if total <= 0:
        uni = 1.0 / max(1, len(moves))
        return {action_signature(m): uni for m in moves}
    for m, v in zip(moves, vals):
        priors[action_signature(m)] = v / total
    return priors


def infer_num_determinizations(time_budget_ms: int) -> int:
    """Heuristic budget-to-determinizations mapping for root IS-MCTS."""
    if time_budget_ms <= 2_000:
        return 12
    if time_budget_ms <= 5_000:
        return 24
    if time_budget_ms <= 10_000:
        return 40
    if time_budget_ms <= 20_000:
        return 64
    if time_budget_ms <= 40_000:
        return 96
    return 128


def search_best_move(game: GameState, player_idx: int, cfg: EngineConfig) -> EngineResult:
    rng = random.Random(cfg.seed)
    tm = TimeManager(cfg.time_budget_ms)

    if game.is_game_over:
        return EngineResult(None, [], 0, 0, 0, tm.elapsed_ms())
    if player_idx < 0 or player_idx >= game.num_players:
        return EngineResult(None, [], 0, 0, 0, tm.elapsed_ms())

    player = game.players[player_idx]
    root_moves = generate_all_moves(game, player)
    if not root_moves:
        return EngineResult(None, [], 0, 0, 0, tm.elapsed_ms())

    priors = _heuristic_priors(game, player.name, root_moves)
    stats = RootTransposition()
    root_visits = 0
    simulations = 0
    nodes = 1  # root
    det_used = 0

    det_limit = cfg.num_determinizations if cfg.num_determinizations > 0 else infer_num_determinizations(cfg.time_budget_ms)

    while not tm.expired() and det_used < det_limit:
        det_used += 1
        hidden = sample_hidden_state(game, rng)

        # Run several root playouts per determinization while we have time.
        local_budget = min(8, max(1, len(root_moves)))
        for _ in range(local_budget):
            if tm.expired():
                break

            # Root PUCT selection.
            best_move = None
            best_score = -1e30
            for m in root_moves:
                sig = action_signature(m)
                st = stats.get(sig)
                prior = priors.get(sig, 1.0 / max(1, len(root_moves)))
                q = st.mean_value
                u = cfg.c_puct * prior * math.sqrt(max(1, root_visits)) / (1 + st.visits)
                s = q + u
                if s > best_score:
                    best_score = s
                    best_move = m

            if best_move is None:
                break

            rollout_game = sample_hidden_state(hidden, rng)
            vp = _rollout_score_after_move(
                rollout_game,
                player_name=player.name,
                move=best_move,
                max_rollout_depth=cfg.max_rollout_depth,
            )
            if vp is None:
                continue

            sig = action_signature(best_move)
            st = stats.get(sig)
            st.visits += 1
            st.value_sum += vp
            root_visits += 1
            simulations += 1
            nodes += 1

    out: list[EngineMoveStat] = []
    for m in root_moves:
        sig = action_signature(m)
        st = stats.get(sig)
        prior = priors.get(sig, 1.0 / max(1, len(root_moves)))
        q = st.mean_value
        u = cfg.c_puct * prior * math.sqrt(max(1, root_visits)) / (1 + st.visits)
        out.append(
            EngineMoveStat(
                move=m,
                visit_count=st.visits,
                mean_vp=round(q, 3),
                prior=round(prior, 6),
                ucb=round(q + u, 6),
            )
        )

    out.sort(key=lambda x: (x.visit_count, x.mean_vp, x.prior), reverse=True)
    best = out[0].move if out else None
    return EngineResult(
        best_move=best,
        top_k_moves=out[: max(1, cfg.top_k)],
        nodes=nodes,
        simulations=simulations,
        determinizations=det_used,
        elapsed_ms=round(tm.elapsed_ms(), 1),
    )
