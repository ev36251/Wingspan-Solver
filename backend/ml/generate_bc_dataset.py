"""Generate behavioral-cloning dataset from improved policy targets.

Supports:
- Heuristic teacher behavioral cloning.
- Optional model-proposed candidate moves.
- 1-2 ply lookahead re-ranking for policy improvement.
- N-step bootstrapped value targets.
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
from backend.solver.heuristics import _estimate_move_value, dynamic_weights
from backend.solver.move_generator import generate_all_moves, Move
from backend.solver.self_play import create_training_game
from backend.solver.simulation import _refill_tray, deep_copy_game, execute_move_on_sim
from backend.engine_search import EngineConfig, search_best_move
from backend.ml.factorized_policy import encode_factorized_targets, ACTION_TYPE_TO_ID
from backend.ml.factorized_inference import FactorizedPolicyModel, score_move_with_factorized_model
from backend.ml.state_encoder import StateEncoder
from backend.powers.registry import get_power_source, is_strict_power_source_allowed


@dataclass
class _PendingDecision:
    state: list[float]
    targets: dict[str, int]
    legal_action_type_mask: list[int]
    player_index: int
    round_num: int
    turn_in_round: int
    ordinal_for_player: int
    score_now: int


def _heuristic_score_move(game, player, move: Move) -> float:
    w = dynamic_weights(game)
    return float(_estimate_move_value(game, player, move, w))


def _action_type_mask(moves: list[Move]) -> list[int]:
    mask = [0, 0, 0, 0]
    for m in moves:
        mask[ACTION_TYPE_TO_ID[m.action_type]] = 1
    return mask


def _rerank_with_lookahead(game, player_idx: int, moves: list[Move], depth: int) -> dict[int, float]:
    """Evaluate candidate moves by short lookahead and resulting score."""
    out: dict[int, float] = {}
    actor_name = game.players[player_idx].name

    for move in moves:
        sim = deep_copy_game(game)
        actor = sim.players[player_idx]

        if actor.name != sim.current_player.name:
            out[id(move)] = -1e9
            continue

        ok = execute_move_on_sim(sim, actor, move)
        if not ok:
            out[id(move)] = -1e9
            continue

        sim.advance_turn()
        _refill_tray(sim)

        if depth >= 2 and not sim.is_game_over:
            p2 = sim.current_player
            moves2 = generate_all_moves(sim, p2)
            if moves2:
                # Opponent best heuristic response.
                best2 = max(moves2, key=lambda m: _heuristic_score_move(sim, p2, m))
                if execute_move_on_sim(sim, p2, best2):
                    sim.advance_turn()
                    _refill_tray(sim)

        actor_after = sim.get_player(actor_name)
        if actor_after is None:
            out[id(move)] = -1e9
            continue

        out[id(move)] = float(calculate_score(sim, actor_after).total)

    return out


def _select_improved_move(
    game,
    player,
    moves: list[Move],
    proposal_model: FactorizedPolicyModel | None,
    encoder: StateEncoder,
    proposal_top_k: int,
    lookahead_depth: int,
) -> Move:
    if not moves:
        raise ValueError("No moves")

    # Proposal phase: model-guided if available, else heuristic-guided.
    scored: list[tuple[Move, float]] = []
    if proposal_model is not None:
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = proposal_model.forward(state)
        for m in moves:
            scored.append((m, score_move_with_factorized_model(logits, m, player)))
    else:
        for m in moves:
            scored.append((m, _heuristic_score_move(game, player, m)))

    scored.sort(key=lambda x: -x[1])
    k = max(1, min(proposal_top_k, len(scored)))
    candidates = [m for m, _ in scored[:k]]

    # Improvement phase: short lookahead reranking.
    if lookahead_depth > 0 and len(candidates) > 1:
        look = _rerank_with_lookahead(game, game.current_player_idx, candidates, lookahead_depth)
        candidates.sort(key=lambda m: (look.get(id(m), -1e9), _heuristic_score_move(game, player, m)), reverse=True)

    # Additional value-head reranking for proposal-model candidates.
    if proposal_model is not None and len(candidates) > 1:
        value_scores: dict[int, float] = {}
        for m in candidates:
            sim = deep_copy_game(game)
            actor = sim.players[game.current_player_idx]
            if not execute_move_on_sim(sim, actor, m):
                value_scores[id(m)] = -1e9
                continue
            sim.advance_turn()
            _refill_tray(sim)
            state2 = np.asarray(encoder.encode(sim, game.current_player_idx), dtype=np.float32)
            _, v2 = proposal_model.forward(state2)
            v_score = proposal_model.value_to_expected_score(v2)
            immediate = float(calculate_score(sim, actor).total)
            value_scores[id(m)] = 0.85 * v_score + 0.15 * immediate
        candidates.sort(
            key=lambda m: (
                value_scores.get(id(m), -1e9),
                _heuristic_score_move(game, player, m),
            ),
            reverse=True,
        )

    return candidates[0]


def _select_engine_teacher_move(
    game,
    player_idx: int,
    *,
    time_budget_ms: int,
    num_determinizations: int,
    max_rollout_depth: int,
    seed: int,
) -> Move | None:
    cfg = EngineConfig(
        time_budget_ms=max(1, int(time_budget_ms)),
        num_determinizations=int(num_determinizations),
        max_rollout_depth=max(1, int(max_rollout_depth)),
        seed=int(seed),
    )
    result = search_best_move(game, player_idx=player_idx, cfg=cfg)
    return result.best_move


def generate_bc_dataset(
    out_jsonl: str,
    out_meta: str,
    games: int,
    players: int,
    board_type: BoardType,
    max_turns: int,
    seed: int | None,
    proposal_model_path: str | None = None,
    proposal_top_k: int = 6,
    lookahead_depth: int = 2,
    n_step: int = 2,
    gamma: float = 0.97,
    bootstrap_mix: float = 0.35,
    value_target_score_scale: float = 160.0,
    value_target_score_bias: float = 0.0,
    late_round_oversample_factor: int = 2,
    engine_teacher_prob: float = 0.0,
    engine_time_budget_ms: int = 25,
    engine_num_determinizations: int = 0,
    engine_max_rollout_depth: int = 24,
    strict_rules_only: bool = True,
    reject_non_strict_powers: bool = True,
    max_round: int = 4,
    emit_score_breakdown: bool = True,
) -> dict:
    if seed is not None:
        random.seed(seed)
    engine_teacher_prob = max(0.0, min(1.0, float(engine_teacher_prob)))
    value_target_score_scale = max(1.0, float(value_target_score_scale))
    late_round_oversample_factor = max(1, int(late_round_oversample_factor))

    load_all(EXCEL_FILE)

    proposal_model = FactorizedPolicyModel(proposal_model_path) if proposal_model_path else None
    enc = StateEncoder()
    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    samples = 0
    all_scores: list[int] = []
    engine_teacher_calls = 0
    engine_teacher_applied = 0
    strict_rejected_games = 0

    with outp.open("w", encoding="utf-8") as f:
        for g in range(1, games + 1):
            game = create_training_game(
                players,
                board_type,
                strict_rules_mode=strict_rules_only,
            )
            turns = 0
            pending: list[_PendingDecision] = []
            ordinals = [0 for _ in range(players)]
            strict_violations: list[str] = []

            while not game.is_game_over and turns < max_turns:
                if game.current_round > max_round:
                    break

                if reject_non_strict_powers:
                    for pl in game.players:
                        for b in pl.board.all_birds():
                            src = get_power_source(b)
                            if not is_strict_power_source_allowed(src):
                                strict_violations.append(f"{b.name}:{src}")
                    if strict_violations:
                        break

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

                best = _select_improved_move(
                    game=game,
                    player=p,
                    moves=moves,
                    proposal_model=proposal_model,
                    encoder=enc,
                    proposal_top_k=proposal_top_k,
                    lookahead_depth=lookahead_depth,
                )

                if engine_teacher_prob > 0.0 and random.random() < engine_teacher_prob:
                    engine_teacher_calls += 1
                    teacher_move = _select_engine_teacher_move(
                        game=game,
                        player_idx=pi,
                        time_budget_ms=engine_time_budget_ms,
                        num_determinizations=engine_num_determinizations,
                        max_rollout_depth=engine_max_rollout_depth,
                        seed=(seed or 0) + g * 10000 + turns,
                    )
                    if teacher_move is not None:
                        best = teacher_move
                        engine_teacher_applied += 1

                score_now = int(calculate_score(game, p).total)
                pending.append(
                    _PendingDecision(
                        state=enc.encode(game, pi),
                        targets=encode_factorized_targets(best, p),
                        legal_action_type_mask=_action_type_mask(moves),
                        player_index=pi,
                        round_num=game.current_round,
                        turn_in_round=game.turn_in_round,
                        ordinal_for_player=ordinals[pi],
                        score_now=score_now,
                    )
                )
                ordinals[pi] += 1

                success = execute_move_on_sim(game, p, best)
                if success:
                    game.advance_turn()
                    _refill_tray(game)
                else:
                    fallback = False
                    for m in moves:
                        if execute_move_on_sim(game, p, m):
                            game.advance_turn()
                            _refill_tray(game)
                            fallback = True
                            break
                    if not fallback:
                        game.advance_turn()
                        _refill_tray(game)

                turns += 1

            finals = [int(calculate_score(game, pl).total) for pl in game.players]
            if reject_non_strict_powers and strict_violations:
                strict_rejected_games += 1
                continue
            all_scores.extend(finals)

            # Build fast lookup for n-step bootstrap from future own decisions.
            score_by_player_ord: dict[tuple[int, int], int] = {}
            for d in pending:
                score_by_player_ord[(d.player_index, d.ordinal_for_player)] = d.score_now

            for d in pending:
                final_score = float(finals[d.player_index])

                future_ord = d.ordinal_for_player + max(0, n_step)
                future_score = score_by_player_ord.get((d.player_index, future_ord), finals[d.player_index])
                bootstrap_score = float(future_score)

                boot = (gamma ** max(0, n_step)) * bootstrap_score
                value_target_score = (1.0 - bootstrap_mix) * final_score + bootstrap_mix * boot
                value_target = (
                    value_target_score - value_target_score_bias
                ) / value_target_score_scale
                value_target = max(0.0, min(1.0, value_target))

                row = {
                    "state": d.state,
                    "targets": d.targets,
                    "legal_action_type_mask": d.legal_action_type_mask,
                    "value_target": round(value_target, 6),
                    "value_target_score": round(value_target_score, 4),
                    "player_index": d.player_index,
                    "round_num": d.round_num,
                    "turn_in_round": d.turn_in_round,
                }
                copies = 1
                if d.round_num >= 3:
                    copies = late_round_oversample_factor
                for _ in range(copies):
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    samples += 1

            if g % 10 == 0 or g == games:
                print(
                    f"game {g}/{games} | samples={samples} | "
                    f"mean_player_score={sum(all_scores)/max(1, len(all_scores)):.1f}"
                )

    meta = {
        "version": 2,
        "mode": "behavioral_cloning_improved_policy",
        "generated_at_epoch": int(time.time()),
        "elapsed_sec": round(time.time() - started, 2),
        "games": games,
        "players": players,
        "board_type": board_type.value,
        "max_turns": max_turns,
        "samples": samples,
        "feature_dim": len(enc.feature_names()),
        "feature_names": enc.feature_names(),
        "target_heads": {
            "action_type": 4,
            "play_habitat": 4,
            "gain_food_primary": 7,
            "draw_mode": 4,
            "lay_eggs_bin": 11,
            "play_cost_bin": 7,
            "play_power_color": 7,
        },
        "has_value_target": True,
        "value_target_config": {
            "n_step": n_step,
            "gamma": gamma,
            "bootstrap_mix": bootstrap_mix,
            "score_scale": value_target_score_scale,
            "score_bias": value_target_score_bias,
        },
        "policy_improvement": {
            "proposal_model_path": proposal_model_path,
            "proposal_top_k": proposal_top_k,
            "lookahead_depth": lookahead_depth,
            "engine_teacher_prob": engine_teacher_prob,
            "engine_time_budget_ms": engine_time_budget_ms,
            "engine_num_determinizations": engine_num_determinizations,
            "engine_max_rollout_depth": engine_max_rollout_depth,
            "late_round_oversample_factor": late_round_oversample_factor,
            "engine_teacher_calls": engine_teacher_calls,
            "engine_teacher_applied": engine_teacher_applied,
        },
        "strict_rules_only": strict_rules_only,
        "reject_non_strict_powers": reject_non_strict_powers,
        "max_round": max_round,
        "emit_score_breakdown": emit_score_breakdown,
        "strict_rejected_games": strict_rejected_games,
        "mean_player_score": round(sum(all_scores) / max(1, len(all_scores)), 3),
    }
    Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate factorized BC dataset")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--board-type", default="oceania", choices=["base", "oceania"])
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--out", default="reports/ml/bc_dataset.jsonl")
    parser.add_argument("--meta", default="reports/ml/bc_dataset.meta.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--proposal-model", default=None)
    parser.add_argument("--proposal-top-k", type=int, default=6)
    parser.add_argument("--lookahead-depth", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--bootstrap-mix", type=float, default=0.35)
    parser.add_argument("--value-target-score-scale", type=float, default=160.0)
    parser.add_argument("--value-target-score-bias", type=float, default=0.0)
    parser.add_argument("--late-round-oversample-factor", type=int, default=2)
    parser.add_argument("--engine-teacher-prob", type=float, default=0.0)
    parser.add_argument("--engine-time-budget-ms", type=int, default=25)
    parser.add_argument("--engine-num-determinizations", type=int, default=0)
    parser.add_argument("--engine-max-rollout-depth", type=int, default=24)
    parser.set_defaults(strict_rules_only=True, reject_non_strict_powers=True)
    parser.add_argument("--strict-rules-only", dest="strict_rules_only", action="store_true")
    parser.add_argument("--allow-non-strict-rules", dest="strict_rules_only", action="store_false")
    parser.add_argument("--reject-non-strict-powers", dest="reject_non_strict_powers", action="store_true")
    parser.add_argument("--allow-non-strict-powers", dest="reject_non_strict_powers", action="store_false")
    parser.add_argument("--max-round", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--emit-score-breakdown", action="store_true")
    args = parser.parse_args()

    meta = generate_bc_dataset(
        out_jsonl=args.out,
        out_meta=args.meta,
        games=args.games,
        players=args.players,
        board_type=BoardType(args.board_type),
        max_turns=args.max_turns,
        seed=args.seed,
        proposal_model_path=args.proposal_model,
        proposal_top_k=args.proposal_top_k,
        lookahead_depth=args.lookahead_depth,
        n_step=args.n_step,
        gamma=args.gamma,
        bootstrap_mix=args.bootstrap_mix,
        value_target_score_scale=args.value_target_score_scale,
        value_target_score_bias=args.value_target_score_bias,
        late_round_oversample_factor=args.late_round_oversample_factor,
        engine_teacher_prob=args.engine_teacher_prob,
        engine_time_budget_ms=args.engine_time_budget_ms,
        engine_num_determinizations=args.engine_num_determinizations,
        engine_max_rollout_depth=args.engine_max_rollout_depth,
        strict_rules_only=args.strict_rules_only,
        reject_non_strict_powers=args.reject_non_strict_powers,
        max_round=args.max_round,
        emit_score_breakdown=args.emit_score_breakdown,
    )
    print(
        f"bc dataset complete | samples={meta['samples']} | feature_dim={meta['feature_dim']}"
    )


if __name__ == "__main__":
    main()
