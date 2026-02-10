"""Evaluate a factorized model against an opponent policy pool."""

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
from backend.solver.simulation import _refill_tray, deep_copy_game, execute_move_on_sim, pick_weighted_random_move
from backend.ml.factorized_inference import FactorizedPolicyModel, score_move_with_factorized_model
from backend.ml.state_encoder import StateEncoder


@dataclass
class OpponentEval:
    opponent: str
    games: int
    nn_wins: int
    opponent_wins: int
    ties: int
    nn_mean_score: float
    opponent_mean_score: float
    nn_mean_margin: float
    nn_rate_ge_100: float
    nn_rate_ge_120: float


def _pick_model_move(model: FactorizedPolicyModel, enc: StateEncoder, game, pi: int, proposal_top_k: int = 5):
    p = game.players[pi]
    moves = generate_all_moves(game, p)
    if not moves:
        return None, moves
    state = np.asarray(enc.encode(game, pi), dtype=np.float32)
    logits, _ = model.forward(state)
    scored = sorted(
        ((m, score_move_with_factorized_model(logits, m, p)) for m in moves),
        key=lambda x: x[1],
        reverse=True,
    )
    candidates = [m for m, _ in scored[: max(1, min(proposal_top_k, len(scored)))]]
    if len(candidates) == 1:
        return candidates[0], moves

    reranked: list[tuple[Move, float]] = []
    for cand in candidates:
        sim = deep_copy_game(game)
        sp = sim.players[pi]
        if not execute_move_on_sim(sim, sp, cand):
            reranked.append((cand, -1e9))
            continue
        sim.advance_turn()
        _refill_tray(sim)
        state2 = np.asarray(enc.encode(sim, pi), dtype=np.float32)
        _, v2 = model.forward(state2)
        score_est = model.value_to_expected_score(v2)
        immediate = float(calculate_score(sim, sp).total)
        reranked.append((cand, 0.85 * score_est + 0.15 * immediate))
    return max(reranked, key=lambda x: x[1])[0], moves


def _evaluate_vs_one_opponent(
    candidate_model: FactorizedPolicyModel,
    opponent_spec: str,
    games: int,
    board_type: BoardType,
    max_turns: int,
    seed: int,
    proposal_top_k: int,
) -> OpponentEval:
    enc = StateEncoder()
    opp_model = None if opponent_spec == "heuristic" else FactorizedPolicyModel(opponent_spec)

    nn_wins = 0
    opp_wins = 0
    ties = 0
    nn_scores: list[int] = []
    opp_scores: list[int] = []
    margins: list[int] = []

    for g in range(1, games + 1):
        game = create_training_game(num_players=2, board_type=board_type)
        nn_idx = (g + seed) % 2
        opp_idx = 1 - nn_idx

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

            if pi == nn_idx:
                move, moves = _pick_model_move(candidate_model, enc, game, pi, proposal_top_k=proposal_top_k)
            else:
                if opp_model is not None:
                    move, moves = _pick_model_move(opp_model, enc, game, pi, proposal_top_k=proposal_top_k)
                else:
                    moves = generate_all_moves(game, p)
                    move = pick_weighted_random_move(moves, game, p) if moves else None

            if not moves:
                p.action_cubes_remaining = 0
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                turns += 1
                continue

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
        ns = final_scores[nn_idx]
        os = final_scores[opp_idx]
        nn_scores.append(ns)
        opp_scores.append(os)
        margins.append(ns - os)
        if ns > os:
            nn_wins += 1
        elif os > ns:
            opp_wins += 1
        else:
            ties += 1

    n = max(1, len(nn_scores))
    return OpponentEval(
        opponent=opponent_spec,
        games=games,
        nn_wins=nn_wins,
        opponent_wins=opp_wins,
        ties=ties,
        nn_mean_score=round(sum(nn_scores) / n, 3),
        opponent_mean_score=round(sum(opp_scores) / n, 3),
        nn_mean_margin=round(sum(margins) / n, 3),
        nn_rate_ge_100=round(sum(1 for s in nn_scores if s >= 100) / n, 4),
        nn_rate_ge_120=round(sum(1 for s in nn_scores if s >= 120) / n, 4),
    )


def evaluate_against_pool(
    model_path: str,
    opponents: list[str],
    games_per_opponent: int = 100,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    seed: int = 0,
    proposal_top_k: int = 5,
) -> dict:
    load_all(EXCEL_FILE)
    candidate = FactorizedPolicyModel(model_path)

    rows: list[OpponentEval] = []
    for i, opp in enumerate(opponents):
        row = _evaluate_vs_one_opponent(
            candidate_model=candidate,
            opponent_spec=opp,
            games=games_per_opponent,
            board_type=board_type,
            max_turns=max_turns,
            seed=seed + i * 1000,
            proposal_top_k=proposal_top_k,
        )
        rows.append(row)
        print(
            f"opp={opp} | nn_wins={row.nn_wins}/{row.games} | "
            f"margin={row.nn_mean_margin:.2f} | nn_ge100={row.nn_rate_ge_100:.3f}"
        )

    total_games = sum(r.games for r in rows)
    total_wins = sum(r.nn_wins for r in rows)
    total_opp_wins = sum(r.opponent_wins for r in rows)
    total_ties = sum(r.ties for r in rows)
    weighted_margin = (
        sum(r.nn_mean_margin * r.games for r in rows) / max(1, total_games)
    )
    weighted_score = (
        sum(r.nn_mean_score * r.games for r in rows) / max(1, total_games)
    )
    weighted_ge100 = (
        sum(r.nn_rate_ge_100 * r.games for r in rows) / max(1, total_games)
    )
    weighted_ge120 = (
        sum(r.nn_rate_ge_120 * r.games for r in rows) / max(1, total_games)
    )

    return {
        "model": model_path,
        "board_type": board_type.value,
        "games_per_opponent": games_per_opponent,
        "opponents": opponents,
        "summary": {
            "games": total_games,
            "nn_wins": total_wins,
            "opponent_wins": total_opp_wins,
            "ties": total_ties,
            "nn_win_rate": round(total_wins / max(1, total_games), 4),
            "nn_mean_score": round(weighted_score, 3),
            "nn_mean_margin": round(weighted_margin, 3),
            "nn_rate_ge_100": round(weighted_ge100, 4),
            "nn_rate_ge_120": round(weighted_ge120, 4),
        },
        "by_opponent": [r.__dict__ for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factorized model vs opponent pool")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--opponents",
        default="heuristic",
        help="Comma-separated: 'heuristic' and/or factorized model .npz paths",
    )
    parser.add_argument("--games-per-opponent", type=int, default=100)
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proposal-top-k", type=int, default=5)
    parser.add_argument("--out", default="reports/ml/factorized_pool_eval.json")
    args = parser.parse_args()

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if not opponents:
        raise ValueError("At least one opponent must be provided")

    result = evaluate_against_pool(
        model_path=args.model,
        opponents=opponents,
        games_per_opponent=args.games_per_opponent,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
        proposal_top_k=args.proposal_top_k,
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
