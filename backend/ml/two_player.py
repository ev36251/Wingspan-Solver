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


# Named search-budget presets for the 2-player rollout-search agent. Budget is
# the one lever that scales agent strength (memory/SOLO_SEED_FINDINGS.md). The
# ladder vs heuristic (n=100, paired) plateaus at ~96: r6=87.5 < r12=93.2 <
# r24=96.2 == r36=96.3. So "strong" (r24) is the diminishing-returns knee and the
# value pick; "max" (r36) is the highest measured point but adds nothing over
# "strong" for ~1.5x compute. Each preset pins temperature=0.3 + determinize, the
# config those numbers were measured at.
BUDGET_PRESETS = {
    "default": dict(top_k=10, rollouts=12, temperature=0.3, determinize=True),
    "strong":  dict(top_k=16, rollouts=24, temperature=0.3, determinize=True),
    "max":     dict(top_k=20, rollouts=36, temperature=0.3, determinize=True),
}


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
                        rollouts=1, temperature=0.0, determinize=False, opp_weight=1.0,
                        depth=1, branch=3, rollout_model=None, rollout_encoder=None,
                        deadline=None, pool=None, pool_seed=0):
    """1-ply rollout search for seat `my_idx`.

    Objective value of a rollout = my_score - lambda * best_opponent_score, where
    lambda is the *denial weight*:
        objective="selfish" -> lambda = 0     (pure score-max)
        objective="diff"    -> lambda = opp_weight (default 1.0 = full denial)
    Sweeping opp_weight in (0,1) tests whether a *little* denial awareness helps
    while full denial hurts -- the n=100 ablations put full denial (lambda=1) at
    break-even, so the sweet spot is expected to be small.

    With temperature<=0 each candidate gets one greedy rollout (opponent modeled
    by `opp_chooser`). With temperature>0 both seats sample ~softmax(score/temp)
    in rollouts and each candidate is averaged over `rollouts` trajectories,
    which cuts the variance of the noisy point estimate.

    determinize=True reshuffles the face-down deck order before each rollout, so
    the search plans over many *plausible* futures instead of peeking at the one
    true deck order -- honest play under hidden information. Best with rollouts>1
    so the candidate value averages over different imagined decks.
    """
    lam = 0.0 if objective == "selfish" else float(opp_weight)
    stochastic = temperature > 0
    # Rollout policy: optionally a cheaper distilled student (rollout_model);
    # the full `model` always keeps the root candidate ranking.
    r_model = rollout_model if rollout_model is not None else model
    r_enc = rollout_encoder if rollout_encoder is not None else encoder
    roll_me = (make_net_sampling_chooser(r_model, r_enc, temperature, seed=my_idx * 7919 + 1)
               if stochastic else make_net_chooser(r_model, r_enc))
    roll_opp = (make_net_sampling_chooser(r_model, r_enc, temperature, seed=my_idx * 7919 + 2)
                if stochastic else opp_chooser)
    n_roll = rollouts if (stochastic or determinize) else 1
    det_rng = random.Random(my_idx * 104729 + 13)

    def _obj(scores):
        if lam == 0.0:
            return scores[my_idx]
        return scores[my_idx] - lam * max(scores[j] for j in range(len(scores)) if j != my_idx)

    def _step_until_my_turn(sim):
        """Advance opponents one turn each (via roll_opp) until it is my turn
        with a legal move, or the game ends."""
        while not sim.is_game_over:
            idx = sim.current_player_idx
            p = sim.current_player
            if p.action_cubes_remaining <= 0:
                if all(pl.action_cubes_remaining <= 0 for pl in sim.players):
                    sim.advance_round()
                else:
                    sim.current_player_idx = (idx + 1) % sim.num_players
                continue
            if idx == my_idx:
                return
            mv_list = generate_all_moves(sim, p)
            if not mv_list:
                p.action_cubes_remaining = 0
                sim.advance_turn()
                continue
            mv = roll_opp(sim, p, mv_list)
            if execute_move_on_sim(sim, p, mv):
                sim.advance_turn()
                _refill_tray(sim)
            else:
                p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1)
                sim.advance_turn()

    def _rollout_value(sim):
        choosers = [roll_me if i == my_idx else roll_opp for i in range(sim.num_players)]
        return _obj(play_multi(sim, choosers))

    def _value_at_my_turn(sim, remaining):
        """Best objective value from one of my decisions, searching `remaining`
        further of my moves (branch-wide) before rolling out."""
        if remaining <= 0 or sim.is_game_over:
            return _rollout_value(sim)
        p = sim.current_player
        moves = generate_all_moves(sim, p)
        if not moves:
            return _rollout_value(sim)
        st = np.asarray(encoder.encode(sim, sim.current_player_idx), dtype=np.float32)
        lg, _ = model.forward(st)
        sc = model.score_moves(st, moves, p, logits=lg)
        order = sorted(range(len(moves)), key=lambda i: -sc[i])
        cand = [moves[i] for i in order[:branch]]
        best = -1e18
        for m in cand:
            s2 = fast_clone_game(sim)
            sp = s2.current_player
            if not execute_move_on_sim(s2, sp, m):
                continue
            s2.advance_turn()
            _refill_tray(s2)
            _step_until_my_turn(s2)
            best = max(best, _value_at_my_turn(s2, remaining - 1))
        return best if best > -1e17 else _rollout_value(sim)

    import time as _time

    # Parallel candidate evaluation is only wired for the plain depth-1
    # stochastic search with a distilled student (the advisor's config); the
    # sequential path below stays byte-identical when pool is None.
    use_pool = (pool is not None and depth == 1 and stochastic
                and rollout_model is not None)

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        sc = model.score_moves(state, moves, player, logits=logits)
        order = sorted(range(len(moves)), key=lambda i: -sc[i])
        ranked = [moves[i] for i in order[:top_k]]
        best_m, best_v = ranked[0], -1e18

        if use_pool:
            from backend.ml.parallel_rollouts import eval_candidates_parallel
            results = eval_candidates_parallel(
                pool, game, ranked,
                my_idx=my_idx, lam=lam, n_roll=n_roll,
                temperature=temperature, determinize=determinize,
                deadline=deadline, pool_seed=pool_seed,
            )
            for m, (total, n_ok) in zip(ranked, results):
                if n_ok == 0:
                    continue
                val = total / n_ok
                if val > best_v:
                    best_m, best_v = m, val
            return best_m
        # Wall-clock budgeting: identical behavior when deadline is None; with
        # a deadline, stop starting new candidates once past it (returning the
        # best so far) and cut a candidate's remaining rollouts. Without this,
        # one decision at a high preset could run for many minutes (turn-1
        # playouts are the longest) with no way to stop it.
        candidates_done = 0
        for m in ranked:
            if (deadline is not None and candidates_done > 0
                    and _time.time() >= deadline):
                break
            total = 0.0
            n_ok = 0
            for _ in range(n_roll):
                if (deadline is not None and n_ok > 0
                        and _time.time() >= deadline):
                    break
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
                if depth >= 2:
                    # Look ahead to my next decision(s) before rolling out.
                    _step_until_my_turn(sim)
                    total += _value_at_my_turn(sim, depth - 1)
                else:
                    total += _rollout_value(sim)
                n_ok += 1
            candidates_done += 1
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
    dep, brn = cfg.get("depth", 1), cfg.get("branch", 3)
    if mode == "heuristic":
        a_ch = make_search_chooser(model, encoder, tk, a_idx, heuristic_chooser, obj, ro, tp, det,
                                   depth=dep, branch=brn)
        b_ch = heuristic_chooser
    elif mode == "ablation":
        # a = denial-weighted (lambda = cfg opp_weight); b = pure selfish baseline.
        a_ch = make_search_chooser(model, encoder, tk, a_idx, net_choose, "diff",
                                   opp_weight=cfg.get("opp_weight", 1.0))
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
    from backend.determinism import ensure_deterministic_hashing
    ensure_deterministic_hashing()
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
    ap.add_argument("--opp-weight", type=float, default=1.0,
                    help="denial weight lambda for the ablation 'a' agent (0=selfish, 1=full)")
    ap.add_argument("--depth", type=int, default=1, help="plies of my-move lookahead (>=2 enables it)")
    ap.add_argument("--branch", type=int, default=3, help="candidates expanded at deeper plies")
    ap.add_argument("--strength", choices=sorted(BUDGET_PRESETS),
                    help="named search-budget preset; overrides --top-k/--rollouts/"
                         "--temperature/--determinize (default|strong|max)")
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)
    if args.strength:
        preset = BUDGET_PRESETS[args.strength]
        args.top_k, args.rollouts = preset["top_k"], preset["rollouts"]
        args.temperature, args.determinize = preset["temperature"], preset["determinize"]
        print(f"  [strength={args.strength}] top_k={args.top_k} rollouts={args.rollouts} "
              f"temp={args.temperature} determinize={args.determinize}")
    cfg = {"top_k": args.top_k, "rollouts": args.rollouts,
           "temperature": args.temperature, "determinize": args.determinize,
           "opp_weight": args.opp_weight, "depth": args.depth, "branch": args.branch}

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
