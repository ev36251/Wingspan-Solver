"""Evaluate factorized BC model against heuristic policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType, BoardType
from backend.solver.move_generator import generate_all_moves, Move
from backend.solver.self_play import create_training_game
from backend.solver.simulation import (
    _refill_tray,
    deep_copy_game,
    execute_move_on_sim,
    pick_weighted_random_move,
    pick_best_heuristic_move,
)
from backend.ml.factorized_inference import FactorizedPolicyModel, load_policy_model
from backend.ml.mcts import MCTS
from backend.ml.state_encoder import StateEncoder


@dataclass
class EvalResult:
    games: int
    nn_wins: int
    heuristic_wins: int
    ties: int
    nn_mean_score: float
    heuristic_mean_score: float
    nn_mean_margin: float
    nn_rate_ge_100: float
    nn_rate_ge_120: float
    heuristic_rate_ge_100: float
    heuristic_rate_ge_120: float
    nn_max_score: int
    nn_min_score: int
    heuristic_max_score: int
    heuristic_min_score: int
    nn_rate_lt_80: float
    nn_rate_80_99: float
    nn_rate_100_119: float
    nn_rate_ge_120_bucket: float
    heuristic_rate_lt_80: float
    heuristic_rate_80_99: float
    heuristic_rate_100_119: float
    heuristic_rate_ge_120_bucket: float


def evaluate_factorized_vs_heuristic(
    model_path: str,
    games: int = 200,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    seed: int = 0,
    proposal_top_k: int = 5,
    nn_use_mcts: bool = False,
    nn_mcts_sims: int = 40,
    nn_c_puct: float = 1.5,
    nn_value_blend: float = 0.5,
    nn_rollout_policy: str = "fast",
    heuristic_policy: str = "greedy",
) -> EvalResult:
    load_all(EXCEL_FILE)

    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    model = FactorizedPolicyModel(model_path)
    enc = StateEncoder.resolve_for_model(model.meta)
    nn_mcts: MCTS | None = None
    if nn_use_mcts:
        nn_mcts = MCTS(
            model=model,
            encoder=enc,
            num_sims=max(1, int(nn_mcts_sims)),
            c_puct=float(nn_c_puct),
            value_blend=float(nn_value_blend),
            rollout_policy=nn_rollout_policy,
        )

    nn_wins = 0
    h_wins = 0
    ties = 0
    nn_scores: list[int] = []
    h_scores: list[int] = []
    margins: list[int] = []

    def _rate(scores: list[int], low: int | None = None, high: int | None = None) -> float:
        n = max(1, len(scores))
        if low is None and high is None:
            return 0.0
        if low is None:
            return round(sum(1 for s in scores if s < high) / n, 4)
        if high is None:
            return round(sum(1 for s in scores if s >= low) / n, 4)
        return round(sum(1 for s in scores if low <= s < high) / n, 4)

    heuristic_policy_norm = str(heuristic_policy).strip().lower()
    if heuristic_policy_norm not in {"greedy", "weighted_random"}:
        raise ValueError(
            f"Unsupported heuristic_policy={heuristic_policy!r}; expected 'greedy' or 'weighted_random'"
        )

    for g in range(1, games + 1):
        game = create_training_game(num_players=2, board_type=board_type)
        nn_idx = (g + seed) % 2
        h_idx = 1 - nn_idx

        turns = 0
        while not game.is_game_over and turns < max_turns:
            p = game.current_player
            pi = game.current_player_idx

            if p.action_cubes_remaining <= 0:
                if all(x.action_cubes_remaining <= 0 for x in game.players):
                    game.advance_round()
                else:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            moves = generate_all_moves(game, p)
            if not moves:
                p.action_cubes_remaining = 0
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            if pi == nn_idx:
                if nn_use_mcts and nn_mcts is not None:
                    move = nn_mcts.get_best_move(game, pi, temperature=0.0)
                    if move is None:
                        move = moves[0]
                else:
                    state = np.asarray(enc.encode(game, pi), dtype=np.float32)
                    logits, _ = model.forward(state)
                    scored = sorted(
                        ((m, model.score_move(state, m, p, logits=logits)) for m in moves),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    candidates = [m for m, _ in scored[: max(1, min(proposal_top_k, len(scored)))]]
                    if len(candidates) == 1:
                        move = candidates[0]
                    else:
                        reranked: list[tuple[Move, float]] = []
                        for cand in candidates:
                            sim = deep_copy_game(game)
                            sp = sim.players[pi]
                            if not execute_move_on_sim(sim, sp, cand):
                                reranked.append((cand, -1e9))
                                continue
                            sim.advance_turn()
                            _refill_tray(sim)
                            s2 = np.asarray(enc.encode(sim, pi), dtype=np.float32)
                            _, v2 = model.forward(s2)
                            score_est = model.value_to_expected_score(v2)
                            immediate = float(calculate_score(sim, sp).total)
                            reranked.append((cand, 0.85 * score_est + 0.15 * immediate))
                        move = max(reranked, key=lambda x: x[1])[0]
            else:
                if heuristic_policy_norm == "greedy":
                    move = pick_best_heuristic_move(moves, game, p)
                else:
                    move = pick_weighted_random_move(moves, game, p)

            success = execute_move_on_sim(game, p, move)
            if success:
                game.advance_turn()
                _refill_tray(game)
            else:
                fallback = False
                for m in moves:
                    if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                        if execute_move_on_sim(game, p, m):
                            game.advance_turn()
                            _refill_tray(game)
                            fallback = True
                            break
                if not fallback:
                    game.advance_turn()
                    _refill_tray(game)

            turns += 1

        final_scores = [int(calculate_score(game, pl).total) for pl in game.players]
        nn_score = final_scores[nn_idx]
        h_score = final_scores[h_idx]

        nn_scores.append(nn_score)
        h_scores.append(h_score)
        margins.append(nn_score - h_score)

        if nn_score > h_score:
            nn_wins += 1
        elif h_score > nn_score:
            h_wins += 1
        else:
            ties += 1

        if g % 20 == 0 or g == games:
            print(
                f"game {g}/{games} | nn_wins={nn_wins} h_wins={h_wins} ties={ties} | "
                f"nn_avg={sum(nn_scores)/len(nn_scores):.1f} h_avg={sum(h_scores)/len(h_scores):.1f}"
            )

    return EvalResult(
        games=games,
        nn_wins=nn_wins,
        heuristic_wins=h_wins,
        ties=ties,
        nn_mean_score=round(sum(nn_scores) / max(1, len(nn_scores)), 3),
        heuristic_mean_score=round(sum(h_scores) / max(1, len(h_scores)), 3),
        nn_mean_margin=round(sum(margins) / max(1, len(margins)), 3),
        nn_rate_ge_100=round(sum(1 for s in nn_scores if s >= 100) / max(1, len(nn_scores)), 4),
        nn_rate_ge_120=round(sum(1 for s in nn_scores if s >= 120) / max(1, len(nn_scores)), 4),
        heuristic_rate_ge_100=round(sum(1 for s in h_scores if s >= 100) / max(1, len(h_scores)), 4),
        heuristic_rate_ge_120=round(sum(1 for s in h_scores if s >= 120) / max(1, len(h_scores)), 4),
        nn_max_score=max(nn_scores) if nn_scores else 0,
        nn_min_score=min(nn_scores) if nn_scores else 0,
        heuristic_max_score=max(h_scores) if h_scores else 0,
        heuristic_min_score=min(h_scores) if h_scores else 0,
        nn_rate_lt_80=_rate(nn_scores, high=80),
        nn_rate_80_99=_rate(nn_scores, low=80, high=100),
        nn_rate_100_119=_rate(nn_scores, low=100, high=120),
        nn_rate_ge_120_bucket=_rate(nn_scores, low=120),
        heuristic_rate_lt_80=_rate(h_scores, high=80),
        heuristic_rate_80_99=_rate(h_scores, low=80, high=100),
        heuristic_rate_100_119=_rate(h_scores, low=100, high=120),
        heuristic_rate_ge_120_bucket=_rate(h_scores, low=120),
    )


def evaluate_nn_vs_nn(
    model_a_path: str,
    model_b_path: str,
    games: int = 50,
    board_type: BoardType = BoardType.OCEANIA,
    mcts_sims: int = 20,
    c_puct: float = 1.5,
    value_blend: float = 0.5,
    rollout_policy: str = "fast",
    max_turns: int = 240,
    seed: int = 0,
) -> dict:
    """Evaluate two NN models against each other using MCTS.

    model_a = candidate (reported as 'nn' in the returned dict)
    model_b = champion  (reported as 'heuristic' for naming compatibility)

    Returns a dict with the same keys as _eval_to_summary() so it can be used
    directly as gate_summary in auto_improve_alphazero.py.
    """
    load_all(EXCEL_FILE)

    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    model_a = load_policy_model(model_a_path)
    model_b = load_policy_model(model_b_path)
    enc_a = StateEncoder.resolve_for_model(model_a.meta)
    enc_b = StateEncoder.resolve_for_model(model_b.meta)

    mcts_a = MCTS(
        model_a, enc_a,
        num_sims=mcts_sims, c_puct=c_puct,
        value_blend=value_blend, rollout_policy=rollout_policy,
    )
    mcts_b = MCTS(
        model_b, enc_b,
        num_sims=mcts_sims, c_puct=c_puct,
        value_blend=value_blend, rollout_policy=rollout_policy,
    )

    a_wins = 0
    b_wins = 0
    ties = 0
    a_scores: list[int] = []
    b_scores: list[int] = []
    margins: list[int] = []

    for g in range(1, games + 1):
        game = create_training_game(num_players=2, board_type=board_type)
        # Alternate seat assignments so neither model has a consistent first-mover advantage
        a_idx = (g + seed) % 2
        b_idx = 1 - a_idx

        turns = 0
        while not game.is_game_over and turns < max_turns:
            p = game.current_player
            pi = game.current_player_idx

            if p.action_cubes_remaining <= 0:
                if all(x.action_cubes_remaining <= 0 for x in game.players):
                    game.advance_round()
                else:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            moves = generate_all_moves(game, p)
            if not moves:
                p.action_cubes_remaining = 0
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

            if pi == a_idx:
                move = mcts_a.get_best_move(game, pi, temperature=0.0)
            else:
                move = mcts_b.get_best_move(game, pi, temperature=0.0)

            if move is None:
                move = moves[0]

            success = execute_move_on_sim(game, p, move)
            if success:
                game.advance_turn()
                _refill_tray(game)
            else:
                fallback = False
                for m in moves:
                    if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                        if execute_move_on_sim(game, p, m):
                            game.advance_turn()
                            _refill_tray(game)
                            fallback = True
                            break
                if not fallback:
                    game.advance_turn()
                    _refill_tray(game)

            turns += 1

        final_scores = [int(calculate_score(game, pl).total) for pl in game.players]
        a_score = final_scores[a_idx]
        b_score = final_scores[b_idx]

        a_scores.append(a_score)
        b_scores.append(b_score)
        margins.append(a_score - b_score)

        if a_score > b_score:
            a_wins += 1
        elif b_score > a_score:
            b_wins += 1
        else:
            ties += 1

        if g % 10 == 0 or g == games:
            print(
                f"game {g}/{games} | a_wins={a_wins} b_wins={b_wins} ties={ties} | "
                f"a_avg={sum(a_scores)/len(a_scores):.1f} b_avg={sum(b_scores)/len(b_scores):.1f}"
            )

    n = max(1, len(a_scores))
    return {
        "games": games,
        "nn_wins": a_wins,
        "heuristic_wins": b_wins,
        "ties": ties,
        "nn_mean_score": round(sum(a_scores) / n, 3),
        "heuristic_mean_score": round(sum(b_scores) / n, 3),
        "nn_mean_margin": round(sum(margins) / n, 3),
        "nn_win_rate": round(a_wins / max(1, games), 4),
        "nn_rate_ge_100": round(sum(1 for s in a_scores if s >= 100) / n, 4),
        "nn_rate_ge_120": round(sum(1 for s in a_scores if s >= 120) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factorized BC vs heuristic")
    parser.add_argument("--model", default="reports/ml/factorized_bc_2p_v1.npz")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nn-use-mcts", action="store_true", default=False)
    parser.add_argument("--nn-mcts-sims", type=int, default=40)
    parser.add_argument("--nn-c-puct", type=float, default=1.5)
    parser.add_argument("--nn-value-blend", type=float, default=0.5)
    parser.add_argument("--nn-rollout-policy", default="fast")
    parser.add_argument(
        "--heuristic-policy",
        default="greedy",
        choices=["greedy", "weighted_random"],
        help="Opponent policy in eval: deterministic greedy heuristic (default) or legacy weighted_random.",
    )
    parser.add_argument("--out", default="reports/ml/factorized_bc_eval.json")
    args = parser.parse_args()

    result = evaluate_factorized_vs_heuristic(
        model_path=args.model,
        games=args.games,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
        nn_use_mcts=args.nn_use_mcts,
        nn_mcts_sims=args.nn_mcts_sims,
        nn_c_puct=args.nn_c_puct,
        nn_value_blend=args.nn_value_blend,
        nn_rollout_policy=args.nn_rollout_policy,
        heuristic_policy=args.heuristic_policy,
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
