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


def evaluate(model_path, seeds, board, top_k, rollouts=1, temperature=0.0, determinize=False):
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)

    agent_wins = 0
    agent_scores, opp_scores = [], []
    for seed in seeds:
        # Alternate seats so seat advantage cancels.
        my_idx = seed % 2
        game = build_2p_game(seed, board)
        agent = make_search_chooser(model, encoder, top_k, my_idx, heuristic_chooser,
                                    "diff", rollouts=rollouts, temperature=temperature,
                                    determinize=determinize)
        choosers = [agent if i == my_idx else heuristic_chooser for i in range(2)]
        scores = play_multi(game, choosers)
        a, o = scores[my_idx], scores[1 - my_idx]
        agent_scores.append(a)
        opp_scores.append(o)
        if a > o:
            agent_wins += 1
        print(f"seed {seed:>3} (seat {my_idx}):  agent={a:>3}  heuristic={o:>3}  "
              f"{'WIN' if a > o else 'loss' if a < o else 'tie'}")

    n = len(seeds)
    print("\n==================== SUMMARY ====================")
    print(f"2-player: differential search agent vs heuristic | {n} games")
    print(f"agent     : mean={st.mean(agent_scores):.1f}")
    print(f"heuristic : mean={st.mean(opp_scores):.1f}")
    print(f"agent win rate: {agent_wins}/{n} ({100*agent_wins/n:.0f}%)")


def ablation(model_path, seeds, board, top_k):
    """Head-to-head: differential objective vs pure score-max (both full search).

    Both agents model the opponent with the net policy inside their rollouts, so
    the only difference is the objective. Seats alternate by seed parity.
    """
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)

    diff_wins = 0
    diff_scores, selfish_scores = [], []
    for seed in seeds:
        diff_idx = seed % 2
        selfish_idx = 1 - diff_idx
        diff_agent = make_search_chooser(model, encoder, top_k, diff_idx, net_choose, "diff")
        selfish_agent = make_search_chooser(model, encoder, top_k, selfish_idx, net_choose, "selfish")
        game = build_2p_game(seed, board)
        choosers = [None, None]
        choosers[diff_idx] = diff_agent
        choosers[selfish_idx] = selfish_agent
        scores = play_multi(game, choosers)
        d, s = scores[diff_idx], scores[selfish_idx]
        diff_scores.append(d)
        selfish_scores.append(s)
        if d > s:
            diff_wins += 1
        print(f"seed {seed:>3} (diff@seat {diff_idx}):  diff={d:>3}  selfish={s:>3}  "
              f"{'DIFF' if d > s else 'selfish' if d < s else 'tie'}")

    n = len(seeds)
    print("\n==================== SUMMARY ====================")
    print(f"ABLATION: differential (my-opp) vs pure score-max (my only) | {n} games")
    print(f"differential : mean={st.mean(diff_scores):.1f}")
    print(f"score-max    : mean={st.mean(selfish_scores):.1f}")
    print(f"differential win rate: {diff_wins}/{n} ({100*diff_wins/n:.0f}%)")


def selfplay(model_path, seeds, board, top_k, rollouts, temperature, determinize=False):
    """Head-to-head: improved search (averaged stochastic rollouts) vs the
    baseline single greedy-rollout search. Both use the differential objective;
    the only difference is rollout quality. Seats alternate by seed parity."""
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)

    new_wins = 0
    new_scores, base_scores = [], []
    for seed in seeds:
        new_idx = seed % 2
        base_idx = 1 - new_idx
        new_agent = make_search_chooser(model, encoder, top_k, new_idx, net_choose,
                                        "diff", rollouts=rollouts, temperature=temperature,
                                        determinize=determinize)
        base_agent = make_search_chooser(model, encoder, top_k, base_idx, net_choose, "diff")
        game = build_2p_game(seed, board)
        choosers = [None, None]
        choosers[new_idx] = new_agent
        choosers[base_idx] = base_agent
        scores = play_multi(game, choosers)
        nw, bs = scores[new_idx], scores[base_idx]
        new_scores.append(nw)
        base_scores.append(bs)
        if nw > bs:
            new_wins += 1
        print(f"seed {seed:>3} (new@seat {new_idx}):  new={nw:>3}  base={bs:>3}  "
              f"{'NEW' if nw > bs else 'base' if nw < bs else 'tie'}")

    n = len(seeds)
    print("\n==================== SUMMARY ====================")
    print(f"SELF-PLAY: improved search (rollouts={rollouts}, temp={temperature}) "
          f"vs baseline (1 greedy rollout) | {n} games")
    print(f"improved : mean={st.mean(new_scores):.1f}")
    print(f"baseline : mean={st.mean(base_scores):.1f}")
    print(f"improved win rate: {new_wins}/{n} ({100*new_wins/n:.0f}%)")


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
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)
    if args.mode == "ablation":
        ablation(args.model, seeds, board, args.top_k)
    elif args.mode == "selfplay":
        selfplay(args.model, seeds, board, args.top_k, args.rollouts, args.temperature,
                 args.determinize)
    else:
        evaluate(args.model, seeds, board, args.top_k, args.rollouts, args.temperature,
                 args.determinize)


if __name__ == "__main__":
    main()
