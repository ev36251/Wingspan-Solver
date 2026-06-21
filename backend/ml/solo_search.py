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
import json
import statistics as st
import time
from pathlib import Path

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


def play_with_record(game, choose, max_turns: int = 400):
    """Like play_with but records the move trajectory (for training data)."""
    traj = []
    turns = 0
    while not game.is_game_over and turns < max_turns:
        player = game.current_player
        if player.action_cubes_remaining <= 0:
            game.advance_round()
            continue
        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.advance_turn()
            continue
        mv = choose(game, player, moves)
        if execute_move_on_sim(game, player, mv):
            traj.append(f"R{game.current_round} {mv.action_type.value}: {mv.description}")
            game.advance_turn()
            _refill_tray(game)
        else:
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1
    p = game.players[0]
    bd = calculate_score(game, p)
    return bd.total, traj, bd.as_dict(), [b.name for b in p.board.all_birds()]


def search_seed_row(seed, model, encoder, k_schedule, rollout, n_drafts, board):
    """Run the net-guided search over the top-N drafts; return the best line as a
    training row (same schema as solo_seed_optimizer.result_to_row, role=policy)."""
    chooser = make_search_chooser(model, encoder, k_schedule, rollout)
    deal = deal_seed(seed, board)
    recs = draft_candidates(deal, top_k=max(1, n_drafts))
    best = None
    for rec in recs[:max(1, n_drafts)]:
        game = build_game_from_draft(deal, rec.birds_to_keep, _food_dict_from_rec(rec),
                                     _bonus_by_name(deal, rec.bonus_card))
        score, traj, bd, birds = play_with_record(game, chooser)
        if best is None or score > best["score"]:
            best = {
                "seed": int(seed), "role": "policy", "score": score,
                "breakdown": bd, "actions": {},
                "draft": {
                    "birds_kept": rec.birds_to_keep, "food_kept": rec.food_to_keep,
                    "bonus": rec.bonus_card, "num_birds_kept": len(rec.birds_to_keep),
                },
                "board_birds": birds, "trajectory": traj,
            }
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


def generate(model_path, seeds, board, k_schedule, rollout, n_drafts,
             out_path, use_modal, seeds_per_shard):
    """Generate best search lines (training rows) for the flywheel's next iter."""
    load_all(EXCEL_FILE)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if use_modal:
        from backend.ml.modal_solo import dispatch_search_modal
        rows = dispatch_search_modal(seeds, model_path, board, k_schedule,
                                     rollout=rollout, n_drafts=n_drafts,
                                     seeds_per_shard=seeds_per_shard)
    else:
        model = FactorizedPolicyModel(model_path)
        encoder = StateEncoder.resolve_for_model(model.meta)
        rows = []
        for s in seeds:
            r = search_seed_row(s, model, encoder, k_schedule, rollout, n_drafts, board)
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["seed"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    scores = [r["score"] for r in rows]
    print(f"generated {len(rows)} lines -> {out_path}")
    print(f"mean score {st.mean(scores):.1f}  min {min(scores)}  max {max(scores)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
        p.add_argument("--seeds", default="4000-4019")
        p.add_argument("--board", choices=["oceania", "base"], default="oceania")
        p.add_argument("--top-k", type=int, default=8)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--inner-top-k", type=int, default=3)
        p.add_argument("--n-drafts", type=int, default=3)
        p.add_argument("--rollout", choices=["net", "heuristic"], default="net")

    e = sub.add_parser("eval"); _common(e)
    g = sub.add_parser("gen"); _common(g)
    g.add_argument("--out", default="reports/ml/solo_seed/best_lines_iter2.jsonl")
    g.add_argument("--use-modal", action="store_true")
    g.add_argument("--seeds-per-shard", type=int, default=10)

    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    k_schedule = [args.top_k] + [args.inner_top_k] * max(0, args.depth - 1)

    if args.cmd == "eval":
        evaluate(args.model, _parse_seeds(args.seeds), board, k_schedule,
                 args.rollout, args.n_drafts)
    else:
        generate(args.model, _parse_seeds(args.seeds), board, k_schedule,
                 args.rollout, args.n_drafts, args.out, args.use_modal,
                 args.seeds_per_shard)


if __name__ == "__main__":
    main()
