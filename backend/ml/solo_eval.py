"""Evaluate a trained solo policy net by actually playing games.

Plays full deterministic solo games with the greedy net policy and compares to
the greedy rule-based heuristic, head-to-head from the *same* opening draft, on
**fresh seeds the net never trained on** (default 4000+). This tests whether
behavioral cloning of the best-of-N search produced a policy that (a) beats the
heuristic and (b) generalizes beyond the training seeds.

Usage:
    python -m backend.ml.solo_eval --model reports/ml/solo_seed/solo_net_spread.npz \
        --seeds 4000-4099
"""

from __future__ import annotations

import argparse
import statistics as st

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, _refill_tray, deep_copy_game
from backend.solver.heuristics import dynamic_weights, _estimate_move_value
from backend.engine.scoring import calculate_score
from backend.ml.solo_seed_optimizer import (
    deal_seed, draft_candidates, build_game_from_draft,
    _food_dict_from_rec, _bonus_by_name,
)


def heuristic_chooser(game, player, moves):
    w = dynamic_weights(game)
    return max(moves, key=lambda m: _estimate_move_value(game, player, m, w))


def make_net_chooser(model: FactorizedPolicyModel, encoder: StateEncoder):
    def choose(game, player, moves):
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        scores = model.score_moves(state, moves, player, logits=logits)
        return moves[max(range(len(moves)), key=scores.__getitem__)]
    return choose


def make_net_sampling_chooser(model: FactorizedPolicyModel, encoder: StateEncoder,
                              temp: float = 0.7, seed: int = 0):
    """Stochastic policy: sample a move ~ softmax(score/temp). For exploring
    rollouts in score-maximizing search."""
    import math
    import random as _random
    rng = _random.Random(seed)

    def choose(game, player, moves):
        if len(moves) == 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        scores = model.score_moves(state, moves, player, logits=logits)
        mx = max(scores)
        weights = [math.exp((s - mx) / max(1e-6, temp)) for s in scores]
        tot = sum(weights)
        r = rng.random() * tot
        u = 0.0
        for m, w in zip(moves, weights):
            u += w
            if u >= r:
                return m
        return moves[-1]
    return choose


def make_value_lookahead_chooser(model: FactorizedPolicyModel, encoder: StateEncoder,
                                 policy_blend: float = 0.0):
    """1-ply value search: score each move by the net's predicted final-score
    value of the state it leads to (optionally blended with the policy score)."""
    def choose(game, player, moves):
        if len(moves) == 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        best, best_score = moves[0], -1e18
        for m in moves:
            sim = deep_copy_game(game)
            sp = sim.current_player
            if execute_move_on_sim(sim, sp, m):
                sim.advance_turn()
                _refill_tray(sim)
            else:
                continue
            if sim.is_game_over:
                val = calculate_score(sim, sim.players[0]).total / 100.0
            else:
                s2 = np.asarray(encoder.encode(sim, 0), dtype=np.float32)
                _, v = model.forward(s2)
                val = float(v) if v is not None else 0.0
            score = val
            if policy_blend > 0.0:
                score += policy_blend * model.score_move(state, m, player, logits=logits)
            if score > best_score:
                best, best_score = m, score
        return best
    return choose


def play_with(game, choose, max_turns: int = 400) -> int:
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
            game.advance_turn()
            _refill_tray(game)
        else:
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1
    return calculate_score(game, game.players[0]).total


def evaluate(model_path: str, seeds, board=BoardType.OCEANIA, lookahead=False,
             policy_blend=0.0):
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)
    look_choose = make_value_lookahead_chooser(model, encoder, policy_blend) if lookahead else None

    heur_scores, net_scores, look_scores, wins = [], [], [], 0
    for seed in seeds:
        deal = deal_seed(seed, board)
        rec = draft_candidates(deal, top_k=1)[0]          # same opening for all
        food = _food_dict_from_rec(rec)
        bonus = _bonus_by_name(deal, rec.bonus_card)

        h = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), heuristic_chooser)
        n = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), net_choose)
        heur_scores.append(h)
        net_scores.append(n)
        if n > h:
            wins += 1
        line = f"seed {seed:>4}:  heur={h:>3}  net={n:>3}"
        if look_choose is not None:
            lk = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), look_choose)
            look_scores.append(lk)
            line += f"  lookahead={lk:>3}"
        print(line)

    nN = len(net_scores)
    print("\n==================== SUMMARY ====================")
    print(f"fresh seeds (held out): {seeds[0]}–{seeds[-1]}  (n={nN})")
    def row(name, xs):
        print(f"{name:<16}: mean={st.mean(xs):5.1f}  median={st.median(xs):5.1f}  min={min(xs):>3}  max={max(xs):>3}")
    row("heuristic", heur_scores)
    row("net greedy", net_scores)
    if look_scores:
        row("net+1ply value", look_scores)
    print(f"\nnet - heuristic   : {st.mean(net_scores)-st.mean(heur_scores):+.1f} mean pts  "
          f"(net wins {wins}/{nN})")
    if look_scores:
        print(f"1ply - net greedy : {st.mean(look_scores)-st.mean(net_scores):+.1f} mean pts  "
              f"(value-head signal)")
        print(f"1ply - heuristic  : {st.mean(look_scores)-st.mean(heur_scores):+.1f} mean pts")


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net.npz")
    ap.add_argument("--seeds", default="4000-4099", help="held-out seeds (>=4000)")
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--lookahead", action="store_true",
                    help="also run 1-ply value-head search")
    ap.add_argument("--policy-blend", type=float, default=0.0,
                    help="blend policy score into 1-ply value search")
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    evaluate(args.model, _parse_seeds(args.seeds), board,
             lookahead=args.lookahead, policy_blend=args.policy_blend)


if __name__ == "__main__":
    main()
