"""Generate policy/value training data from Wingspan self-play.

Phase 1 pipeline:
- Uses current rules engine as environment.
- Uses current heuristic policy for action selection.
- Records (state, legal action IDs, chosen action, final reward).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType, BoardType
from backend.solver.heuristics import _estimate_move_value, dynamic_weights
from backend.solver.move_generator import generate_all_moves, Move
from backend.solver.self_play import create_training_game
from backend.solver.simulation import (
    _refill_tray,
    deep_copy_game,
    execute_move_on_sim,
    pick_weighted_random_move,
)
from backend.powers.registry import get_power_source, is_strict_power_source_allowed
from backend.ml.action_codec import ActionCodec
from backend.ml.replay_buffer import JsonlReplayWriter, TrainingSample
from backend.ml.state_encoder import StateEncoder


@dataclass
class _PendingDecision:
    state: list[float]
    legal_action_ids: list[int]
    action_id: int
    player_index: int
    num_players: int
    round_num: int
    turn_in_round: int


REWARD_WIN_WEIGHT = 0.1
REWARD_SCORE_WEIGHT = 0.7
REWARD_MARGIN_WEIGHT = 0.2


def _compute_outcomes(game) -> dict[int, tuple[float, int, int]]:
    """Return player_idx -> (reward, final_score, won)."""
    finals = [int(calculate_score(game, p).total) for p in game.players]
    best = max(finals) if finals else 0

    out: dict[int, tuple[float, int, int]] = {}
    for i, score in enumerate(finals):
        others = [s for j, s in enumerate(finals) if j != i]
        mean_others = (sum(others) / len(others)) if others else score
        margin = score - mean_others

        win = 1 if score == best else 0
        score_component = max(0.0, min(1.0, score / 150.0))
        margin_component = max(0.0, min(1.0, (margin + 60.0) / 120.0))

        reward = (
            REWARD_WIN_WEIGHT * float(win)
            + REWARD_SCORE_WEIGHT * score_component
            + REWARD_MARGIN_WEIGHT * margin_component
        )
        out[i] = (round(reward, 6), score, win)

    return out


def _heuristic_score_move(game, player, move: Move) -> float:
    return float(_estimate_move_value(game, player, move, dynamic_weights(game)))


def _rerank_with_lookahead(game, player_idx: int, moves: list[Move], depth: int) -> dict[int, float]:
    """Evaluate candidate moves with short rollouts and choose by actor final score."""
    out: dict[int, float] = {}
    actor_name = game.players[player_idx].name

    for move in moves:
        sim = deep_copy_game(game)
        actor = sim.players[player_idx]
        if actor.name != sim.current_player.name:
            out[id(move)] = -1e9
            continue

        if not execute_move_on_sim(sim, actor, move):
            out[id(move)] = -1e9
            continue

        sim.advance_turn()
        _refill_tray(sim)

        if depth >= 2 and not sim.is_game_over:
            opp = sim.current_player
            opp_moves = generate_all_moves(sim, opp)
            if opp_moves:
                best_opp = max(opp_moves, key=lambda m: _heuristic_score_move(sim, opp, m))
                if execute_move_on_sim(sim, opp, best_opp):
                    sim.advance_turn()
                    _refill_tray(sim)

        actor_after = sim.get_player(actor_name)
        out[id(move)] = float(calculate_score(sim, actor_after).total) if actor_after else -1e9

    return out


def _pick_teacher_move(
    game,
    player,
    moves: list[Move],
    teacher_policy: str,
    proposal_top_k: int,
    lookahead_depth: int,
) -> Move:
    if teacher_policy == "epsilon_heuristic":
        return pick_weighted_random_move(moves, game, player)

    scored = [(m, _heuristic_score_move(game, player, m)) for m in moves]
    scored.sort(key=lambda x: -x[1])
    k = max(1, min(proposal_top_k, len(scored)))
    candidates = [m for m, _ in scored[:k]]

    if teacher_policy == "heuristic_topk":
        return candidates[0]

    if lookahead_depth > 0 and len(candidates) > 1:
        look = _rerank_with_lookahead(game, game.current_player_idx, candidates, lookahead_depth)
        candidates.sort(key=lambda m: (look.get(id(m), -1e9), _heuristic_score_move(game, player, m)), reverse=True)
    return candidates[0]


def _play_one_game(
    num_players: int,
    board_type: BoardType,
    action_codec: ActionCodec,
    encoder: StateEncoder,
    max_turns: int,
    teacher_policy: str,
    proposal_top_k: int,
    lookahead_depth: int,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    max_round: int = 4,
) -> tuple[list[TrainingSample], dict]:
    game = create_training_game(
        num_players,
        board_type,
        strict_rules_mode=strict_rules_only,
    )

    pending: list[_PendingDecision] = []
    turns = 0
    strict_violations: list[str] = []

    while not game.is_game_over and turns < max_turns:
        if game.current_round > max_round:
            break

        if reject_non_strict_powers:
            for p in game.players:
                for b in p.board.all_birds():
                    src = get_power_source(b)
                    if not is_strict_power_source_allowed(src):
                        strict_violations.append(f"{b.name}:{src}")
            if strict_violations:
                break

        player = game.current_player
        player_idx = game.current_player_idx

        if player.action_cubes_remaining <= 0:
            if all(p.action_cubes_remaining <= 0 for p in game.players):
                game.advance_round()
                continue
            game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            continue

        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            continue

        state_vec = encoder.encode(game, player_idx)
        legal_action_ids = action_codec.encode_moves(moves)

        try:
            move = _pick_teacher_move(
                game=game,
                player=player,
                moves=moves,
                teacher_policy=teacher_policy,
                proposal_top_k=proposal_top_k,
                lookahead_depth=lookahead_depth,
            )
        except RuntimeError as e:
            if "Strict rules mode rejected" in str(e):
                strict_violations.append(str(e))
                break
            raise
        action_id = action_codec.encode_move(move)

        pending.append(
            _PendingDecision(
                state=state_vec,
                legal_action_ids=legal_action_ids,
                action_id=action_id,
                player_index=player_idx,
                num_players=game.num_players,
                round_num=game.current_round,
                turn_in_round=game.turn_in_round,
            )
        )

        try:
            success = execute_move_on_sim(game, player, move)
        except RuntimeError as e:
            if "Strict rules mode rejected" in str(e):
                strict_violations.append(str(e))
                break
            raise

        if success:
            game.advance_turn()
            _refill_tray(game)
        else:
            fallback_executed = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    try:
                        ok_fb = execute_move_on_sim(game, player, m)
                    except RuntimeError as e:
                        if "Strict rules mode rejected" in str(e):
                            strict_violations.append(str(e))
                            ok_fb = False
                        else:
                            raise
                    if ok_fb:
                        game.advance_turn()
                        _refill_tray(game)
                        fallback_executed = True
                        break
            if not fallback_executed:
                player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
                game.advance_turn()
                _refill_tray(game)

        turns += 1

    outcomes = _compute_outcomes(game)
    samples: list[TrainingSample] = []
    for d in pending:
        reward, final_score, won = outcomes[d.player_index]
        samples.append(
            TrainingSample(
                state=d.state,
                legal_action_ids=d.legal_action_ids,
                action_id=d.action_id,
                reward=reward,
                final_score=final_score,
                won=won,
                player_index=d.player_index,
                num_players=d.num_players,
                round_num=d.round_num,
                turn_in_round=d.turn_in_round,
            )
        )

    game_summary = {
        "turns": turns,
        "scores": [int(calculate_score(game, p).total) for p in game.players],
        "winner_score": max(int(calculate_score(game, p).total) for p in game.players),
        "score_breakdown": [
            calculate_score(game, p).as_dict()
            for p in game.players
        ],
        "strict_certified": len(strict_violations) == 0,
        "strict_violations": strict_violations,
    }
    return samples, game_summary


def generate_dataset(
    output_jsonl: str,
    metadata_path: str,
    games: int,
    players: int,
    board_type: BoardType,
    max_turns: int,
    seed: int | None,
    teacher_policy: str = "lookahead",
    proposal_top_k: int = 6,
    lookahead_depth: int = 2,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    max_round: int = 4,
    emit_score_breakdown: bool = True,
) -> dict:
    if seed is not None:
        random.seed(seed)

    load_all(EXCEL_FILE)

    writer = JsonlReplayWriter(output_jsonl, overwrite=True)
    encoder = StateEncoder()
    codec = ActionCodec()

    started = time.time()
    all_scores: list[int] = []
    winner_scores: list[int] = []
    sample_count = 0
    strict_rejected_games = 0

    for g in range(1, games + 1):
        samples, summary = _play_one_game(
            num_players=players,
            board_type=board_type,
            action_codec=codec,
            encoder=encoder,
            max_turns=max_turns,
            teacher_policy=teacher_policy,
            proposal_top_k=proposal_top_k,
            lookahead_depth=lookahead_depth,
            strict_rules_only=strict_rules_only,
            reject_non_strict_powers=reject_non_strict_powers,
            max_round=max_round,
        )
        if reject_non_strict_powers and not summary.get("strict_certified", False):
            strict_rejected_games += 1
            continue
        writer.write_many(samples)
        sample_count += len(samples)

        all_scores.extend(summary["scores"])
        winner_scores.append(summary["winner_score"])

        if g % 10 == 0 or g == games:
            print(
                f"game {g}/{games} | samples={sample_count} | "
                f"mean_player_score={sum(all_scores)/max(1, len(all_scores)):.1f} | "
                f"mean_winner_score={sum(winner_scores)/max(1, len(winner_scores)):.1f}"
            )

    meta = {
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "elapsed_sec": round(time.time() - started, 2),
        "games": games,
        "players": players,
        "board_type": board_type.value,
        "max_turns": max_turns,
        "samples": sample_count,
        "feature_dim": len(encoder.feature_names()),
        "feature_names": encoder.feature_names(),
        "action_space": codec.to_dict(),
        "teacher_policy": teacher_policy,
        "proposal_top_k": proposal_top_k,
        "lookahead_depth": lookahead_depth,
        "strict_rules_only": strict_rules_only,
        "reject_non_strict_powers": reject_non_strict_powers,
        "max_round": max_round,
        "emit_score_breakdown": emit_score_breakdown,
        "strict_rejected_games": strict_rejected_games,
        "reward_weights": {
            "win": REWARD_WIN_WEIGHT,
            "score": REWARD_SCORE_WEIGHT,
            "margin": REWARD_MARGIN_WEIGHT,
        },
        "mean_player_score": round(sum(all_scores) / max(1, len(all_scores)), 3),
        "mean_winner_score": round(sum(winner_scores) / max(1, len(winner_scores)), 3),
    }

    p = Path(metadata_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wingspan self-play training dataset")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--out", default="reports/ml/self_play_dataset.jsonl")
    parser.add_argument("--meta", default="reports/ml/self_play_dataset.meta.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--teacher-policy",
        default="lookahead",
        choices=["epsilon_heuristic", "heuristic_topk", "lookahead"],
    )
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.set_defaults(strict_rules_only=True, reject_non_strict_powers=True)
    parser.add_argument("--strict-rules-only", dest="strict_rules_only", action="store_true")
    parser.add_argument("--allow-non-strict-rules", dest="strict_rules_only", action="store_false")
    parser.add_argument("--reject-non-strict-powers", dest="reject_non_strict_powers", action="store_true")
    parser.add_argument("--allow-non-strict-powers", dest="reject_non_strict_powers", action="store_false")
    parser.add_argument("--max-round", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--emit-score-breakdown", action="store_true")
    args = parser.parse_args()

    meta = generate_dataset(
        output_jsonl=args.out,
        metadata_path=args.meta,
        games=args.games,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
        teacher_policy=args.teacher_policy,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
        max_round=args.max_round,
        emit_score_breakdown=args.emit_score_breakdown,
    )

    print(
        "dataset complete | "
        f"samples={meta['samples']} | feature_dim={meta['feature_dim']} | "
        f"actions={meta['action_space']['num_actions']}"
    )


if __name__ == "__main__":
    main()
