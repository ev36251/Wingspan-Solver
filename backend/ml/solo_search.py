"""Net-guided rollout search for strong single-game solo play.

At each decision: rank moves by the policy net, take the top-K, play each out
to the end with the policy (a "rollout"), and keep the move whose rollout
reaches the best *real* final score. This sidesteps the (unreliable) value head
entirely -- it judges moves by actually finishing the game. fast_clone_game
makes the per-move rollouts affordable.

Usage:
    python -m backend.ml.solo_search --model reports/ml/solo_seed/solo_net_spread.npz \
        --seeds 4000-4019 --top-k 5 --rollout net
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
from backend.solver.simulation import fast_clone_game, execute_move_on_sim, _refill_tray
from backend.ml.solo_eval import (
    heuristic_chooser, make_net_chooser, play_with, _parse_seeds,
)
from backend.ml.solo_seed_optimizer import (
    deal_seed, draft_candidates, build_game_from_draft,
    _food_dict_from_rec, _bonus_by_name,
)


def make_rollout_search_chooser(model, encoder, top_k: int = 5, rollout: str = "net"):
    net_choose = make_net_chooser(model, encoder)
    roll_choose = net_choose if rollout == "net" else heuristic_chooser

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        ranked = sorted(moves, key=lambda m: -model.score_move(state, m, player, logits=logits))
        cands = ranked[:max(1, top_k)]
        best, best_sc = cands[0], -1e18
        for m in cands:
            sim = fast_clone_game(game)
            sp = sim.current_player
            if not execute_move_on_sim(sim, sp, m):
                continue
            sim.advance_turn()
            _refill_tray(sim)
            sc = play_with(sim, roll_choose)   # finish the game -> real score
            if sc > best_sc:
                best, best_sc = m, sc
        return best

    return choose


def evaluate(model_path, seeds, board, top_k, rollout):
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)
    search_choose = make_rollout_search_chooser(model, encoder, top_k, rollout)

    net_s, search_s, secs = [], [], []
    for seed in seeds:
        deal = deal_seed(seed, board)
        rec = draft_candidates(deal, top_k=1)[0]
        food = _food_dict_from_rec(rec)
        bonus = _bonus_by_name(deal, rec.bonus_card)

        n = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), net_choose)
        t0 = time.perf_counter()
        s = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), search_choose)
        dt = time.perf_counter() - t0
        net_s.append(n)
        search_s.append(s)
        secs.append(dt)
        print(f"seed {seed:>4}:  net={n:>3}  search={s:>3}  ({dt:.1f}s)")

    nN = len(net_s)
    print("\n==================== SUMMARY ====================")
    print(f"held-out seeds {seeds[0]}-{seeds[-1]} (n={nN})  top_k={top_k} rollout={rollout}")
    print(f"net greedy       : mean={st.mean(net_s):.1f}")
    print(f"rollout search   : mean={st.mean(search_s):.1f}  (+{st.mean(search_s)-st.mean(net_s):.1f} vs net)")
    wins = sum(1 for a, b in zip(search_s, net_s) if a > b)
    print(f"search > net in {wins}/{nN} games   |   avg {st.mean(secs):.1f}s/game")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="4000-4019")
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rollout", choices=["net", "heuristic"], default="net")
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    evaluate(args.model, _parse_seeds(args.seeds), board, args.top_k, args.rollout)


if __name__ == "__main__":
    main()
