"""Two-player net-guided rollout search (differential objective).

Extends the solo search to 2 players. At the agent's decision it ranks moves by
the policy, and scores each candidate by rolling the *whole game out for both
players* with the policy, valuing the result as (my_score - opponent_score).
Competitive behavior (goal racing, not feeding the leader) emerges from that
differential objective -- 2-player games already use real placement goal +
nectar-majority scoring in the engine.

    python -m backend.ml.two_player --seeds 0-19
"""

from __future__ import annotations

import argparse
import random
import statistics as st

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.solver.self_play import create_training_game
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, fast_clone_game, _refill_tray
from backend.engine.scoring import calculate_score
from backend.solver.heuristics import dynamic_weights, _estimate_move_value
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.solo_seed_optimizer import SeededBirdfeeder
from backend.ml.solo_eval import make_net_chooser, make_net_sampling_chooser


def build_2p_game(seed: int, board: BoardType = BoardType.OCEANIA):
    """Deterministic 2-player setup (real draft for both seats + seeded feeder)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    game = create_training_game(num_players=2, board_type=board, setup_mode="real5_softmax")
    game.birdfeeder = SeededBirdfeeder(board, seed ^ 0x5EED)
    game.birdfeeder.reroll()
    return game


def heuristic_chooser(game, player, moves):
    w = dynamic_weights(game)
    return max(moves, key=lambda m: _estimate_move_value(game, player, m, w))


def play_multi(game, choosers, max_turns: int = 600):
    """Play a multiplayer game to the end; each seat acts via choosers[idx]."""
    turns = 0
    while not game.is_game_over and turns < max_turns:
        idx = game.current_player_idx
        p = game.current_player
        if p.action_cubes_remaining <= 0:
            if all(pl.action_cubes_remaining <= 0 for pl in game.players):
                game.advance_round()
            else:
                game.current_player_idx = (idx + 1) % game.num_players
            continue
        moves = generate_all_moves(game, p)
        if not moves:
            p.action_cubes_remaining = 0
            game.advance_turn()
            turns += 1
            continue
        mv = choosers[idx](game, p, moves)
        if execute_move_on_sim(game, p, mv):
            game.advance_turn()
            _refill_tray(game)
        else:
            p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1
    return [calculate_score(game, pl).total for pl in game.players]


def make_search_chooser(model, encoder, top_k, my_idx, opp_chooser, objective="diff",
                        rollouts=1, temperature=0.0, determinize=False):
    """1-ply rollout search for seat `my_idx`.

    objective="diff"    -> maximize (my_score - best_opponent_score)  [competitive]
    objective="selfish" -> maximize my_score only                    [score-max]

    With temperature<=0 each candidate gets one greedy rollout (opponent modeled
    by `opp_chooser`). With temperature>0 both seats sample ~softmax(score/temp)
    in rollouts and each candidate is averaged over `rollouts` trajectories,
    which cuts the variance of the noisy point estimate.

    determinize=True reshuffles the face-down deck order before each rollout, so
    the search plans over many *plausible* futures instead of peeking at the one
    true deck order -- honest play under hidden information. Best with rollouts>1
    so the candidate value averages over different imagined decks.
    """
    stochastic = temperature > 0
    roll_me = (make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 1)
               if stochastic else make_net_chooser(model, encoder))
    roll_opp = (make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 2)
                if stochastic else opp_chooser)
    n_roll = rollouts if (stochastic or determinize) else 1
    det_rng = random.Random(my_idx * 104729 + 13)

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        ranked = sorted(moves, key=lambda m: -model.score_move(state, m, player, logits=logits))[:top_k]
        best_m, best_v = ranked[0], -1e18
        for m in ranked:
            total = 0.0
            n_ok = 0
            for _ in range(n_roll):
                sim = fast_clone_game(game)
                if determinize:
                    deck = getattr(sim, "_deck_cards", None)
                    if isinstance(deck, list) and len(deck) > 1:
                        det_rng.shuffle(deck)  # unseen deck order is hidden info
                sp = sim.current_player
                if not execute_move_on_sim(sim, sp, m):
                    continue
                sim.advance_turn()
                _refill_tray(sim)
                choosers = [roll_me if i == my_idx else roll_opp for i in range(sim.num_players)]
                scores = play_multi(sim, choosers)
                if objective == "selfish":
                    total += scores[my_idx]
                else:
                    total += scores[my_idx] - max(scores[j] for j in range(len(scores)) if j != my_idx)
                n_ok += 1
            if n_ok == 0:
                continue
            val = total / n_ok
            if val > best_v:
                best_m, best_v = m, val
        return best_m

    return choose


def make_diff_search_chooser(model, encoder, top_k, my_idx, opp_chooser):
    return make_search_chooser(model, encoder, top_k, my_idx, opp_chooser, objective="diff")


# mode -> (label for the agent-of-interest "a", label for the comparator "b")
MODE_LABELS = {
    "heuristic": ("agent", "heuristic"),
    "ablation": ("differential", "score-max"),
    "selfplay": ("improved", "baseline"),
}


def play_one(mode, seed, model, encoder, board, cfg):
    """Play one 2-player game for `mode` and return {seed, a_idx, a, b}.

    "a" is always the agent of interest, "b" the comparator. Seats alternate by
    seed parity so seat advantage cancels across an even seed range. Shared by
    the local loop and the Modal shards.
    """
    net_choose = make_net_chooser(model, encoder)
    a_idx, b_idx = seed % 2, 1 - (seed % 2)
    tk, ro, tp, det = cfg["top_k"], cfg["rollouts"], cfg["temperature"], cfg["determinize"]
    # Default deployed objective is "selfish" (pure score-max): the Modal n=100
    # ablation showed it beats the differential objective 60/40, because the
    # differential term subtracts a noisy/biased opponent-score estimate. Revisit
    # once the net can see the opponent's board (then denial value is reliable).
    obj = cfg.get("objective", "selfish")
    if mode == "heuristic":
        a_ch = make_search_chooser(model, encoder, tk, a_idx, heuristic_chooser, obj, ro, tp, det)
        b_ch = heuristic_chooser
    elif mode == "ablation":
        a_ch = make_search_chooser(model, encoder, tk, a_idx, net_choose, "diff")
        b_ch = make_search_chooser(model, encoder, tk, b_idx, net_choose, "selfish")
    elif mode == "selfplay":
        a_ch = make_search_chooser(model, encoder, tk, a_idx, net_choose, obj, ro, tp, det)
        b_ch = make_search_chooser(model, encoder, tk, b_idx, net_choose, obj)
    else:
        raise ValueError(f"unknown mode {mode}")
    game = build_2p_game(seed, board)
    choosers = [None, None]
    choosers[a_idx], choosers[b_idx] = a_ch, b_ch
    scores = play_multi(game, choosers)
    return {"seed": seed, "a_idx": a_idx, "a": scores[a_idx], "b": scores[b_idx]}


def summarize(mode, results, cfg):
    from math import comb
    results = sorted(results, key=lambda r: r["seed"])
    la, lb = MODE_LABELS[mode]
    a_scores = [r["a"] for r in results]
    b_scores = [r["b"] for r in results]
    n = len(results)
    wins = sum(1 for r in results if r["a"] > r["b"])
    dec = sum(1 for r in results if r["a"] != r["b"])
    for r in results:
        tag = la.upper() if r["a"] > r["b"] else (lb if r["a"] < r["b"] else "tie")
        print(f"seed {r['seed']:>3} ({la}@seat {r['a_idx']}):  {la}={r['a']:>3}  {lb}={r['b']:>3}  {tag}")
    p = sum(comb(dec, i) for i in range(wins, dec + 1)) / 2 ** dec if dec > 0 else 1.0
    print("\n==================== SUMMARY ====================")
    print(f"mode={mode} | {n} games | top_k={cfg['top_k']} rollouts={cfg['rollouts']} "
          f"temp={cfg['temperature']} determinize={cfg['determinize']}")
    print(f"{la:<13}: mean={st.mean(a_scores):.1f}")
    print(f"{lb:<13}: mean={st.mean(b_scores):.1f}")
    print(f"{la} win rate: {wins}/{n} ({100*wins/n:.0f}%)  one-sided binomial p={p:.3f}")


def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--mode", choices=["heuristic", "ablation", "selfplay"], default="heuristic")
    ap.add_argument("--rollouts", type=int, default=4, help="stochastic rollouts/candidate")
    ap.add_argument("--temperature", type=float, default=0.6, help="rollout sampling temp")
    ap.add_argument("--determinize", action="store_true",
                    help="reshuffle the unseen deck each rollout (honest hidden-info play)")
    ap.add_argument("--use-modal", action="store_true", help="fan games across Modal containers")
    ap.add_argument("--seeds-per-shard", type=int, default=2)
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)
    cfg = {"top_k": args.top_k, "rollouts": args.rollouts,
           "temperature": args.temperature, "determinize": args.determinize}

    if args.use_modal:
        from backend.ml.modal_two_player import dispatch_2p_eval_modal
        results = dispatch_2p_eval_modal(args.mode, seeds, args.model, board, cfg,
                                         args.seeds_per_shard)
    else:
        load_all(EXCEL_FILE)
        model = FactorizedPolicyModel(args.model)
        encoder = StateEncoder.resolve_for_model(model.meta)
        results = [play_one(args.mode, s, model, encoder, board, cfg) for s in seeds]
    summarize(args.mode, results, cfg)


if __name__ == "__main__":
    main()
