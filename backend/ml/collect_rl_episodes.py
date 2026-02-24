"""Collect REINFORCE episodes for policy gradient training.

Runs self-play with temperature sampling, computes per-turn advantages
using the value-head baseline, and writes JSONL with is_rl=true + rl_advantage.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import BoardType
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.factorized_policy import encode_factorized_targets, ACTION_TYPE_TO_ID
from backend.ml.state_encoder import StateEncoder
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import _refill_tray, execute_move_on_sim


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _action_type_mask(moves: list) -> list[int]:
    mask = [0, 0, 0, 0]
    for m in moves:
        mask[ACTION_TYPE_TO_ID[m.action_type]] = 1
    return mask


@dataclass
class _RLDecision:
    state: list[float]
    targets: dict[str, int]
    legal_action_type_mask: list[int]
    player_index: int
    round_num: int
    turn_in_round: int
    value_pred_raw: float | None  # raw sigmoid/linear value from model.forward()


def collect_rl_episodes(
    model_path: str,
    n_games: int,
    out_path: str,
    board_type: str = "base",
    players: int = 2,
    temperature: float = 1.5,
    use_value_baseline: bool = True,
    seed: int = 0,
    max_turns: int = 120,
) -> dict:
    """Collect REINFORCE self-play episodes from a trained policy model.

    Each turn's move is sampled via softmax(scores / temperature) rather than
    argmax, enabling exploration of move combos the heuristic teacher wouldn't
    choose. Advantages = final_score - value_baseline, normalized globally.

    Returns dict with keys: games, samples, mean_score, mean_advantage.
    """
    load_all(EXCEL_FILE)
    rng = random.Random(seed)  # noqa: F841 — kept for reproducible future use
    np_rng = np.random.default_rng(seed)

    model = FactorizedPolicyModel(model_path)
    enc = StateEncoder.resolve_for_model(model.meta)
    bt = BoardType(board_type)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    all_scores: list[int] = []
    # Per-game: (decisions_list, finals_list)
    game_records: list[tuple[list[_RLDecision], list[int]]] = []

    # ------------------------------------------------------------------ #
    # Phase 1: play games and collect per-turn decisions + final scores   #
    # ------------------------------------------------------------------ #
    for g in range(1, n_games + 1):
        game = create_training_game(players, bt, strict_rules_mode=True)
        decisions: list[_RLDecision] = []
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

            # Encode state before move
            state_vec = enc.encode(game, pi)
            state_np = np.asarray(state_vec, dtype=np.float32)
            logits, value_pred = model.forward(state_np)

            # Score all legal moves with the model
            scores = np.array(
                [model.score_move(state_np, m, p, logits=logits) for m in moves],
                dtype=np.float64,
            )

            # Temperature sampling (temperature > 1 = more exploratory)
            probs = _softmax(scores / max(1e-6, float(temperature)))
            choice_idx = int(np_rng.choice(len(moves), p=probs))
            chosen = moves[choice_idx]

            # Record targets (must be computed before executing the move)
            targets = encode_factorized_targets(chosen, p)
            mask = _action_type_mask(moves)
            decision = _RLDecision(
                state=state_vec,
                targets=targets,
                legal_action_type_mask=mask,
                player_index=pi,
                round_num=game.current_round,
                turn_in_round=game.turn_in_round,
                value_pred_raw=value_pred,
            )

            # Execute chosen move
            ok = execute_move_on_sim(game, p, chosen)
            if not ok:
                # Fallback: try any legal move and update recorded targets
                fell_back = False
                for m in moves:
                    if execute_move_on_sim(game, p, m):
                        decision = _RLDecision(
                            state=state_vec,
                            targets=encode_factorized_targets(m, p),
                            legal_action_type_mask=mask,
                            player_index=pi,
                            round_num=game.current_round,
                            turn_in_round=game.turn_in_round,
                            value_pred_raw=value_pred,
                        )
                        fell_back = True
                        break
                if not fell_back:
                    # No move executable; skip this turn entirely (no sample)
                    game.advance_turn()
                    _refill_tray(game)
                    turns += 1
                    continue

            decisions.append(decision)
            game.advance_turn()
            _refill_tray(game)
            turns += 1

        finals = [int(calculate_score(game, pl).total) for pl in game.players]
        all_scores.extend(finals)
        game_records.append((decisions, finals))

        if g % 10 == 0 or g == n_games:
            total_decisions = sum(len(d) for d, _ in game_records)
            print(
                f"rl_collect game {g}/{n_games} | "
                f"decisions_so_far={total_decisions} | "
                f"mean_score={sum(all_scores)/max(1,len(all_scores)):.1f}"
            )

    # ------------------------------------------------------------------ #
    # Phase 2: compute raw advantages                                     #
    # ------------------------------------------------------------------ #
    raw_advantages: list[float] = []
    for decisions, finals in game_records:
        for d in decisions:
            final_score = float(finals[d.player_index])
            if use_value_baseline and d.value_pred_raw is not None and model.has_value_head:
                baseline = model.value_to_expected_score(d.value_pred_raw)
            else:
                baseline = 0.0
            raw_advantages.append(final_score - baseline)

    # ------------------------------------------------------------------ #
    # Phase 3: normalize advantages across ALL turns in ALL games         #
    # ------------------------------------------------------------------ #
    adv_arr = np.array(raw_advantages, dtype=np.float64)
    mean_adv = float(adv_arr.mean()) if len(adv_arr) > 0 else 0.0
    std_adv = float(adv_arr.std()) if len(adv_arr) > 0 else 1.0
    norm_adv = (adv_arr - mean_adv) / (std_adv + 1e-8)

    # ------------------------------------------------------------------ #
    # Phase 4: write JSONL                                                #
    # ------------------------------------------------------------------ #
    value_scale = float(model.meta.get("value_score_scale", 150.0))
    value_bias = float(model.meta.get("value_score_bias", 0.0))

    samples = 0
    adv_idx = 0
    with outp.open("w", encoding="utf-8") as f:
        for decisions, finals in game_records:
            for d in decisions:
                final_score = float(finals[d.player_index])
                # Normalized value target (same convention as BC dataset)
                value_target_norm = max(
                    0.0, min(1.0, (final_score - value_bias) / max(1.0, value_scale))
                )
                row = {
                    "state": d.state,
                    "targets": d.targets,
                    "legal_action_type_mask": d.legal_action_type_mask,
                    "value_target": round(value_target_norm, 6),
                    "value_target_score": round(final_score, 4),
                    "player_index": d.player_index,
                    "round_num": d.round_num,
                    "turn_in_round": d.turn_in_round,
                    "is_rl": True,
                    "rl_advantage": round(float(norm_adv[adv_idx]), 6),
                }
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                samples += 1
                adv_idx += 1

    elapsed = time.time() - started
    mean_score = float(np.mean(all_scores)) if all_scores else 0.0
    mean_norm_adv = float(norm_adv.mean()) if len(norm_adv) > 0 else 0.0

    result = {
        "games": n_games,
        "samples": samples,
        "players": players,
        "board_type": board_type,
        "temperature": temperature,
        "use_value_baseline": use_value_baseline,
        "mean_score": round(mean_score, 3),
        "mean_advantage": round(mean_norm_adv, 6),
        "adv_mean_raw": round(mean_adv, 3),
        "adv_std_raw": round(std_adv, 3),
        "elapsed_sec": round(elapsed, 2),
    }
    print(
        f"rl_collect done | games={n_games} samples={samples} "
        f"mean_score={mean_score:.1f} adv_std={std_adv:.2f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect REINFORCE RL episodes")
    parser.add_argument("--model", required=True, help="Path to factorized model .npz")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(use_value_baseline=True)
    parser.add_argument("--use-value-baseline", dest="use_value_baseline", action="store_true")
    parser.add_argument("--disable-value-baseline", dest="use_value_baseline", action="store_false")
    args = parser.parse_args()

    meta = collect_rl_episodes(
        model_path=args.model,
        n_games=args.games,
        out_path=args.out,
        board_type=args.board_type,
        players=args.players,
        temperature=args.temperature,
        use_value_baseline=args.use_value_baseline,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    print(
        f"collect_rl_episodes complete | samples={meta['samples']} "
        f"mean_score={meta['mean_score']:.1f} adv_std={meta['adv_std_raw']:.2f}"
    )


if __name__ == "__main__":
    main()
