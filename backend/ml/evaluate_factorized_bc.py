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
from backend.solver.simulation import _refill_tray, deep_copy_game, execute_move_on_sim, pick_weighted_random_move
from backend.ml.factorized_inference import FactorizedPolicyModel
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


def evaluate_factorized_vs_heuristic(
    model_path: str,
    games: int = 200,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    seed: int = 0,
    proposal_top_k: int = 5,
) -> EvalResult:
    load_all(EXCEL_FILE)

    model = FactorizedPolicyModel(model_path)
    enc = StateEncoder()

    nn_wins = 0
    h_wins = 0
    ties = 0
    nn_scores: list[int] = []
    h_scores: list[int] = []
    margins: list[int] = []

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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factorized BC vs heuristic")
    parser.add_argument("--model", default="reports/ml/factorized_bc_2p_v1.npz")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/ml/factorized_bc_eval.json")
    args = parser.parse_args()

    result = evaluate_factorized_vs_heuristic(
        model_path=args.model,
        games=args.games,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
