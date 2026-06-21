"""Net-guided rollout search for strong single-game solo play.

Judges moves by actually finishing the game (rollouts), not the unreliable value
head. Three knobs, combinable:
  1. top-k        : how many policy-ranked moves to expand at the first ply.
  2. depth        : plies of explicit lookahead before rolling out (depth>1
                    expands `inner-top-k` moves at each deeper ply).
  3. n-drafts     : also search the top-N opening drafts, keep the best game.

The search value of a state = the best final score reachable by expanding the
branching schedule and rolling out the policy at the leaves. fast_clone_game
makes the per-node clones cheap.

Usage:
    python -m backend.ml.solo_search --model reports/ml/solo_seed/solo_net_spread.npz \
        --seeds 4000-4019 --top-k 8 --depth 2 --inner-top-k 3 --n-drafts 3
"""

from __future__ import annotations

import argparse
import statistics as st
import time

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import fast_clone_game, execute_move_on_sim, _refill_tray
from backend.engine.scoring import calculate_score
from backend.ml.solo_eval import (
    heuristic_chooser, make_net_chooser, play_with, _parse_seeds,
)
from backend.ml.solo_seed_optimizer import (
    deal_seed, draft_candidates, build_game_from_draft,
    _food_dict_from_rec, _bonus_by_name,
)


def _advance_to_decision(game):
    """Advance past exhausted players / round ends to the next real decision.
    Returns (player, moves) or (None, None) if the game is over."""
    while not game.is_game_over:
        player = game.current_player
        if player.action_cubes_remaining <= 0:
            game.advance_round()
            continue
        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.advance_turn()
            continue
        return player, moves
    return None, None


def _apply(game, move):
    """Apply a move on an owned clone, advancing the turn. True if it stuck."""
    sp = game.current_player
    if not execute_move_on_sim(game, sp, move):
        return False
    game.advance_turn()
    _refill_tray(game)
    return True


def make_search_chooser(model, encoder, k_schedule=(8,), rollout="net"):
    roll_choose = make_net_chooser(model, encoder) if rollout == "net" else heuristic_chooser

    def _ranked(game, player, moves, k):
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        return sorted(moves, key=lambda m: -model.score_move(state, m, player, logits=logits))[:max(1, k)]

    def value_of_state(game, level):
        """Best final score reachable from `game`, expanding k_schedule[level:]."""
        if level >= len(k_schedule):
            return play_with(game, roll_choose)        # leaf: roll out to the end
        player, moves = _advance_to_decision(game)
        if moves is None:
            return calculate_score(game, game.players[0]).total
        best = -1e18
        for m in _ranked(game, player, moves, k_schedule[level]):
            sim = fast_clone_game(game)
            if not _apply(sim, m):
                continue
            best = max(best, value_of_state(sim, level + 1))
        return best

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        best_m, best_v = None, -1e18
        for m in _ranked(game, player, moves, k_schedule[0]):
            sim = fast_clone_game(game)
            if not _apply(sim, m):
                continue
            v = value_of_state(sim, 1)
            if v > best_v:
                best_m, best_v = m, v
        return best_m if best_m is not None else moves[0]

    return choose


def play_seed_with_draft_search(deal, chooser, n_drafts):
    """Play the seed once per top-N draft with the search chooser; keep the best."""
    recs = draft_candidates(deal, top_k=max(1, n_drafts))
    best = -1
    for rec in recs[:max(1, n_drafts)]:
        game = build_game_from_draft(deal, rec.birds_to_keep, _food_dict_from_rec(rec),
                                     _bonus_by_name(deal, rec.bonus_card))
        best = max(best, play_with(game, chooser))
    return best


def evaluate(model_path, seeds, board, k_schedule, rollout, n_drafts):
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)
    search_choose = make_search_chooser(model, encoder, k_schedule, rollout)

    net_s, search_s, secs = [], [], []
    for seed in seeds:
        deal = deal_seed(seed, board)
        rec0 = draft_candidates(deal, top_k=1)[0]
        food = _food_dict_from_rec(rec0)
        bonus = _bonus_by_name(deal, rec0.bonus_card)
        n = play_with(build_game_from_draft(deal, rec0.birds_to_keep, food, bonus), net_choose)

        t0 = time.perf_counter()
        s = play_seed_with_draft_search(deal, search_choose, n_drafts)
        dt = time.perf_counter() - t0
        net_s.append(n)
        search_s.append(s)
        secs.append(dt)
        print(f"seed {seed:>4}:  net={n:>3}  search={s:>3}  ({dt:.1f}s)")

    nN = len(net_s)
    print("\n==================== SUMMARY ====================")
    print(f"held-out {seeds[0]}-{seeds[-1]} (n={nN})  k_schedule={tuple(k_schedule)} "
          f"rollout={rollout} n_drafts={n_drafts}")
    print(f"net greedy     : mean={st.mean(net_s):.1f}")
    print(f"rollout search : mean={st.mean(search_s):.1f}  (+{st.mean(search_s)-st.mean(net_s):.1f} vs net)")
    wins = sum(1 for a, b in zip(search_s, net_s) if a > b)
    print(f"search > net in {wins}/{nN}   |   avg {st.mean(secs):.1f}s/game   max {max(secs):.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="4000-4019")
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--top-k", type=int, default=8, help="moves expanded at ply 1")
    ap.add_argument("--depth", type=int, default=1, help="plies of lookahead before rollout")
    ap.add_argument("--inner-top-k", type=int, default=3, help="moves expanded at deeper plies")
    ap.add_argument("--n-drafts", type=int, default=3, help="top opening drafts to search")
    ap.add_argument("--rollout", choices=["net", "heuristic"], default="net")
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    k_schedule = [args.top_k] + [args.inner_top_k] * max(0, args.depth - 1)
    evaluate(args.model, _parse_seeds(args.seeds), board, k_schedule, args.rollout, args.n_drafts)


if __name__ == "__main__":
    main()
