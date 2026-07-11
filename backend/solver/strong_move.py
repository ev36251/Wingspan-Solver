"""Pick a move with the literal ~91-benchmarked rollout-search agent.

The advisor's default move-picker is the API's IS-MCTS engine. This module wires
in the *exact* agent measured at ~91 in memory/SOLO_SEED_FINDINGS.md -- the
two_player rollout search (net-sampling playouts, selfish objective, k10/r12/t0.3,
determinized). It plays full games per candidate, so one decision is ~10-20s.

Companion-mode hidden information (unknown deck order + opponent hands) is handled
by determinizing the state a few times (engine_search.belief.sample_hidden_state)
and majority-voting the agent's chosen move across determinizations, all inside a
wall-clock budget so it stays under a real turn.
"""

from __future__ import annotations

import random
import time
from collections import Counter

from backend.engine_search.belief import sample_hidden_state
from backend.ml.action_codec import action_signature
from backend.solver.move_generator import Move, generate_all_moves


def strong_engine_best_move(
    game,
    player_idx: int,
    model,
    encoder,
    *,
    top_k: int = 10,
    rollouts: int = 12,
    temperature: float = 0.3,
    time_budget_s: float = 90.0,
    max_determinizations: int = 5,
    seed: int = 0,
    rollout_model=None,
    rollout_encoder=None,
    pool=None,
) -> tuple[Move | None, int]:
    """Return (best_move_on_`game`, determinizations_used).

    Runs the ~91 agent on up to `max_determinizations` plausible concretizations
    of the hidden state and votes; stops early once `time_budget_s` elapses (after
    at least one determinization). Votes are matched back to the *original* game's
    legal moves by action signature -- the asking player's own hand/board is known,
    so their legal moves are identical across determinizations.
    """
    from backend.ml.two_player import make_search_chooser, heuristic_chooser

    if player_idx < 0 or player_idx >= game.num_players:
        return None, 0
    base_moves = generate_all_moves(game, game.players[player_idx])
    if not base_moves:
        return None, 0
    if len(base_moves) == 1:
        return base_moves[0], 0
    sig_to_move = {action_signature(m): m for m in base_moves}

    votes: Counter[str] = Counter()
    start = time.time()
    # Hard wall-clock cap enforced INSIDE the search (per candidate/rollout),
    # not just between determinizations — a single turn-1 determinization at a
    # heavy preset could otherwise run for many minutes with no way to stop.
    deadline = start + time_budget_s
    used = 0
    for d in range(max(1, max_determinizations)):
        if d > 0 and (time.time() - start) >= time_budget_s:
            break
        det = sample_hidden_state(game, random.Random(seed * 1009 + d))
        dp = det.players[player_idx]
        dmoves = generate_all_moves(det, dp)
        if not dmoves:
            used += 1
            continue
        chooser = make_search_chooser(
            model, encoder, top_k, player_idx, heuristic_chooser,
            objective="selfish", rollouts=rollouts, temperature=temperature,
            determinize=True,
            rollout_model=rollout_model, rollout_encoder=rollout_encoder,
            deadline=deadline,
            pool=pool, pool_seed=seed * 1009 + d,
        )
        best = chooser(det, dp, dmoves)
        votes[action_signature(best)] += 1
        used += 1

    if not votes:
        return base_moves[0], used
    best_sig = votes.most_common(1)[0][0]
    return sig_to_move.get(best_sig, base_moves[0]), used
