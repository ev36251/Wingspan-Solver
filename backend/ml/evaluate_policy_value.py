"""Evaluate a trained policy-value model against the heuristic policy."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType, BoardType
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import _refill_tray, execute_move_on_sim, pick_weighted_random_move
from backend.ml.action_codec import action_signature
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


def _load_model(model_path: str | Path):
    z = np.load(model_path, allow_pickle=True)
    W1 = z["W1"]
    b1 = z["b1"]
    Wp = z["Wp"]
    bp = z["bp"]
    Wv = z["Wv"]
    bv = float(z["bv"][0])
    meta_json = z["metadata_json"][0]
    if isinstance(meta_json, bytes):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(str(meta_json))
    return (W1, b1, Wp, bp, Wv, bv, meta)


def _nn_forward(state: np.ndarray, params) -> tuple[np.ndarray, float]:
    W1, b1, Wp, bp, Wv, bv, _ = params
    h_pre = state @ W1 + b1
    h = np.maximum(h_pre, 0.0)
    logits = h @ Wp + bp
    value_raw = float(h @ Wv + bv)
    value = 1.0 / (1.0 + np.exp(-value_raw))
    return logits, float(value)


def _pick_nn_move(
    state_encoder: StateEncoder,
    params,
    action_id_lookup: dict[str, int],
    game,
    player_idx: int,
    moves: list[Move],
) -> Move:
    state = np.asarray(state_encoder.encode(game, player_idx), dtype=np.float32)
    logits, _ = _nn_forward(state, params)

    best_move = None
    best_score = -1e30

    for m in moves:
        sig = action_signature(m)
        aid = action_id_lookup.get(sig, -1)
        if 0 <= aid < len(logits):
            s = float(logits[aid])
        else:
            # unseen action signatures fallback below
            s = -1e20
        if s > best_score:
            best_score = s
            best_move = m

    if best_move is not None and best_score > -1e19:
        return best_move

    # Fallback to heuristic if all candidate actions are unseen in the NN action space.
    return pick_weighted_random_move(moves, game, game.players[player_idx])


def evaluate_nn_vs_heuristic(
    model_path: str,
    dataset_meta_path: str,
    games: int = 200,
    board_type: BoardType = BoardType.OCEANIA,
    max_turns: int = 240,
    seed: int = 0,
) -> EvalResult:
    load_all(EXCEL_FILE)

    params = _load_model(model_path)
    meta = json.loads(Path(dataset_meta_path).read_text(encoding="utf-8"))
    id_to_sig = meta["action_space"]["id_to_signature"]
    action_id_lookup = {sig: i for i, sig in enumerate(id_to_sig)}

    rnd = random.Random(seed)
    enc = StateEncoder()

    nn_wins = 0
    h_wins = 0
    ties = 0
    nn_scores: list[int] = []
    h_scores: list[int] = []
    margins: list[int] = []

    for g in range(1, games + 1):
        game = create_training_game(num_players=2, board_type=board_type)

        # Alternate seat to avoid first-player/seat bias.
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
                move = _pick_nn_move(enc, params, action_id_lookup, game, pi, moves)
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NN policy vs heuristic policy")
    parser.add_argument("--model", default="reports/ml/policy_value_2p_bootstrap_e8.npz")
    parser.add_argument("--meta", default="reports/ml/self_play_dataset_2p_bootstrap.meta.json")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/ml/nn_vs_heuristic_eval.json")
    args = parser.parse_args()

    result = evaluate_nn_vs_heuristic(
        model_path=args.model,
        dataset_meta_path=args.meta,
        games=args.games,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
