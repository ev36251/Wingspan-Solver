"""AlphaZero-style self-play game generator.

Generates JSONL training data via MCTS self-play.  Output schema is compatible
with train_factorized_bc.py — no changes to the training code are required.

Key design decisions:
  - Both players use MCTS for their real moves (fair self-play).
  - Within each MCTS simulation, opponents play greedy NN (no tree expansion).
  - Value target = my_final_score - best_opponent_final_score (delta, not raw).
    Delta is normalized by /80 during training (typical range ≈ ±80).
  - Temperature schedule: τ=1.0 first `temperature_cutoff` turns per player
    (diverse openings), then τ=0.0 (exploit learned strategies).
  - Records are tagged with player_name so value targets are filled per-player.

CLI (single-process):
    python -m backend.ml.alphazero_self_play \\
        --model reports/ml/phase4_slot_v3/best_model.npz \\
        --out-jsonl reports/ml/az_test/dataset.jsonl \\
        --out-meta reports/ml/az_test/dataset.meta.json \\
        --games 10 --mcts-sims 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import calculate_score
from backend.ml.factorized_inference import FactorizedPolicyModel, load_policy_model
from backend.ml.factorized_policy import encode_factorized_targets
from backend.ml.mcts import MCTS
from backend.ml.move_features import MOVE_FEATURE_DIM, encode_move_features
from backend.ml.state_encoder import StateEncoder
from backend.models.enums import BoardType
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import (
    _refill_tray,
    _score_round_goal,
    execute_move_on_sim,
)

# Delta value target scale: typical delta range is ≈ ±80 in 2-player Oceania.
# train_factorized_bc.py reads this from meta["value_target_config"]["score_scale"].
DELTA_SCALE = 80.0
DELTA_BIAS = 0.0


def generate_alphazero_game(
    model: FactorizedPolicyModel,
    encoder: StateEncoder,
    mcts: MCTS,
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    temperature_cutoff: int = 8,
    max_turns: int = 240,
    strict_rules_mode: bool = False,
    training_encoder: StateEncoder | None = None,
) -> list[dict]:
    """Play one self-play game using MCTS; return per-decision training records.

    All players use MCTS for their own moves.  Within each MCTS simulation,
    opponents are modeled as greedy NN (handled inside MCTS._advance_through_opponents).

    value_target_score is filled after the game ends with the delta for that player.
    """
    game = create_training_game(
        num_players=players,
        board_type=board_type,
        strict_rules_mode=strict_rules_mode,
    )

    records: list[dict] = []
    # Track per-player turn count for temperature schedule
    player_turns: dict[str, int] = {p.name: 0 for p in game.players}
    turns_total = 0

    while not game.is_game_over and turns_total < max_turns:
        player = game.current_player
        player_idx = game.current_player_idx

        # Handle exhausted player (round transition)
        if player.action_cubes_remaining <= 0:
            if all(p.action_cubes_remaining <= 0 for p in game.players):
                _score_round_goal(game, game.current_round)
                game.advance_round()
            else:
                game.current_player_idx = (
                    (game.current_player_idx + 1) % game.num_players
                )
            turns_total += 1
            continue

        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.current_player_idx = (
                (game.current_player_idx + 1) % game.num_players
            )
            turns_total += 1
            continue

        # Encode state BEFORE the move is made.
        # Use training_encoder if provided (may include identity features not in the
        # inference encoder whose dim must match the current model weights).
        store_enc = training_encoder if training_encoder is not None else encoder
        state_vec = np.asarray(store_enc.encode(game, player_idx), dtype=np.float32)

        # Temperature: high early (exploration), zero late (exploitation)
        tau = 1.0 if player_turns[player.name] < temperature_cutoff else 0.0

        # Run MCTS to get policy and sample a move
        mcts_moves, mcts_probs = mcts.get_policy(game, player_idx, temperature=tau)

        if mcts_moves:
            chosen = random.choices(mcts_moves, weights=mcts_probs, k=1)[0]
        else:
            # Fallback: heuristic argmax (should rarely happen)
            from backend.solver.heuristics import _estimate_move_value, dynamic_weights
            w = dynamic_weights(game)
            chosen = max(moves, key=lambda m: _estimate_move_value(game, player, m, w))

        # Encode targets for the chosen move
        targets = encode_factorized_targets(chosen, player)

        # Move-ranking features: positive sample + up to 4 negatives
        move_pos = list(encode_move_features(chosen, player))
        other_moves = [m for m in moves if m.description != chosen.description]
        neg_samples = random.sample(other_moves, min(4, len(other_moves)))
        move_negs = [list(encode_move_features(m, player)) for m in neg_samples]

        records.append(
            {
                "state": state_vec.tolist(),
                "targets": targets,
                "player_name": player.name,
                "value_target_score": None,   # filled after game ends
                "has_value_target": True,
                "move_pos": move_pos,
                "move_negs": move_negs,
            }
        )

        # Execute the chosen move
        success = execute_move_on_sim(game, player, chosen)
        if not success:
            # Move failed — consume the cube without game advancement
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
        else:
            _refill_tray(game)
            game.advance_turn()

        player_turns[player.name] = player_turns.get(player.name, 0) + 1
        turns_total += 1

    # Fill delta value targets: my_score - best_opponent_score
    final_scores = {p.name: calculate_score(game, p).total for p in game.players}
    for rec in records:
        my_name = rec["player_name"]
        my_score = float(final_scores.get(my_name, 0))
        opp_scores = [s for n, s in final_scores.items() if n != my_name]
        best_opp = float(max(opp_scores)) if opp_scores else 0.0
        rec["value_target_score"] = my_score - best_opp

    return records


def generate_self_play_dataset(
    model_path: str,
    out_jsonl: str,
    out_meta: str,
    games: int,
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    mcts_sims: int = 150,
    c_puct: float = 1.5,
    value_blend: float = 0.5,
    rollout_policy: str = "fast",
    temperature_cutoff: int = 8,
    seed: int = 0,
    max_turns: int = 240,
    strict_rules_mode: bool = False,
    value_target_score_scale: float = DELTA_SCALE,
    value_target_score_bias: float = DELTA_BIAS,
    # Optional training-encoder overrides. When set, states stored in the JSONL
    # use these settings (allowing identity features to be added even when the
    # current inference model was trained without them). MCTS inference always
    # uses the model's own encoder (matching its weight dimensions).
    enable_identity_features: bool | None = None,
    identity_hash_dim: int | None = None,
    use_per_slot_encoding: bool | None = None,
    use_hand_habitat_features: bool | None = None,
    use_tray_per_slot_encoding: bool | None = None,
    use_opponent_board_encoding: bool | None = None,
) -> dict:
    """Generate a self-play dataset using MCTS.

    Writes `out_jsonl` + `out_meta` compatible with train_factorized_bc.py.
    Returns the meta dict.
    """
    random.seed(seed)
    np.random.seed(seed)
    load_all(EXCEL_FILE)

    # Limit BLAS to 1 thread — OpenBLAS thread management overhead inflates
    # numpy GEMV time from 0.1ms to ~150ms on machines with multi-threaded BLAS.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    model = load_policy_model(model_path)
    # Inference encoder: must match model's weight dimensions for MCTS forward pass.
    enc = StateEncoder.resolve_for_model(model.meta)

    mcts = MCTS(
        model=model,
        encoder=enc,
        num_sims=mcts_sims,
        c_puct=c_puct,
        value_blend=value_blend,
        rollout_policy=rollout_policy,
    )

    # Training encoder: may differ from inference encoder when identity features
    # are requested but the current model was trained without them.  MCTS runs
    # with `enc` (model-compatible); stored states use `training_enc` so the
    # next model trained on this data will see the full feature set.
    meta_enc_cfg: dict = (model.meta or {}).get("state_encoder", {}) if isinstance(model.meta, dict) else {}
    training_enc_overrides = dict(meta_enc_cfg)
    needs_training_enc = False
    if enable_identity_features is not None and enable_identity_features != bool(meta_enc_cfg.get("enable_identity_features", False)):
        training_enc_overrides["enable_identity_features"] = enable_identity_features
        needs_training_enc = True
    if identity_hash_dim is not None:
        training_enc_overrides["identity_hash_dim"] = identity_hash_dim
    if use_per_slot_encoding is not None and use_per_slot_encoding != bool(meta_enc_cfg.get("use_per_slot_encoding", False)):
        training_enc_overrides["use_per_slot_encoding"] = use_per_slot_encoding
        needs_training_enc = True
    if use_hand_habitat_features is not None and use_hand_habitat_features != bool(meta_enc_cfg.get("use_hand_habitat_features", False)):
        training_enc_overrides["use_hand_habitat_features"] = use_hand_habitat_features
        needs_training_enc = True
    if use_tray_per_slot_encoding is not None and use_tray_per_slot_encoding != bool(meta_enc_cfg.get("use_tray_per_slot_encoding", False)):
        training_enc_overrides["use_tray_per_slot_encoding"] = use_tray_per_slot_encoding
        needs_training_enc = True
    if use_opponent_board_encoding is not None and use_opponent_board_encoding != bool(meta_enc_cfg.get("use_opponent_board_encoding", False)):
        training_enc_overrides["use_opponent_board_encoding"] = use_opponent_board_encoding
        needs_training_enc = True

    training_enc: StateEncoder | None = None
    if needs_training_enc:
        training_enc = StateEncoder.from_metadata({"state_encoder": training_enc_overrides})
        # Compute feature_dim for the training encoder via a dummy encode
        _dummy = create_training_game(num_players=players, board_type=board_type)
        feature_dim = len(training_enc.encode(_dummy, 0))
        del _dummy
        state_encoder_cfg = training_enc_overrides
    else:
        feature_dim = int(model.feature_dim)
        state_encoder_cfg = meta_enc_cfg or {"enable_identity_features": False, "identity_hash_dim": 128}

    head_dims = dict(model.meta.get("head_dims") or model.meta.get("target_heads") or {})

    out_jsonl_path = Path(out_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    total_samples = 0
    delta_values: list[float] = []
    final_scores_all: list[float] = []

    with out_jsonl_path.open("w", encoding="utf-8") as fout:
        for g in range(1, games + 1):
            try:
                game_records = generate_alphazero_game(
                    model=model,
                    encoder=enc,
                    mcts=mcts,
                    players=players,
                    board_type=board_type,
                    temperature_cutoff=temperature_cutoff,
                    max_turns=max_turns,
                    strict_rules_mode=strict_rules_mode,
                    training_encoder=training_enc,
                )
            except Exception as exc:
                print(f"  [warn] game {g} failed: {exc}")
                continue

            for rec in game_records:
                delta = float(rec.get("value_target_score") or 0.0)
                delta_values.append(delta)
                fout.write(json.dumps(rec) + "\n")
                total_samples += 1

            if g % 5 == 0 or g == games:
                elapsed = time.time() - started
                mean_d = float(np.mean(delta_values)) if delta_values else 0.0
                print(
                    f"  [az-selfplay] game {g}/{games} | "
                    f"samples={total_samples} | "
                    f"mean_delta={mean_d:.1f} | "
                    f"elapsed={elapsed:.1f}s"
                )

    elapsed = time.time() - started
    mean_delta = float(np.mean(delta_values)) if delta_values else 0.0
    std_delta = float(np.std(delta_values)) if delta_values else 0.0

    meta: dict = {
        "feature_dim": feature_dim,
        "target_heads": head_dims,
        "games": games,
        "samples": total_samples,
        "elapsed_sec": round(elapsed, 2),
        "players": players,
        "board_type": board_type.value,
        "mcts_sims": mcts_sims,
        "c_puct": c_puct,
        "value_blend": value_blend,
        "rollout_policy": rollout_policy,
        "temperature_cutoff": temperature_cutoff,
        "seed": seed,
        "mean_delta": round(mean_delta, 3),
        "std_delta": round(std_delta, 3),
        # This is read by _load_as_arrays to normalise value_target_score
        "value_target_config": {
            "mode": "delta",
            "score_scale": value_target_score_scale,
            "score_bias": value_target_score_bias,
        },
        "state_encoder": state_encoder_cfg,
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# CLI for single-process testing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AlphaZero self-play dataset (single process)"
    )
    parser.add_argument(
        "--model", required=True, help="Path to .npz model file"
    )
    parser.add_argument(
        "--out-jsonl",
        default="reports/ml/az_selfplay/dataset.jsonl",
    )
    parser.add_argument(
        "--out-meta",
        default="reports/ml/az_selfplay/dataset.meta.json",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--board-type", default="oceania")
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--value-blend", type=float, default=0.5)
    parser.add_argument("--rollout-policy", default="fast")
    parser.add_argument("--temperature-cutoff", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bt = BoardType.OCEANIA if args.board_type.lower() == "oceania" else BoardType.BASE
    meta = generate_self_play_dataset(
        model_path=args.model,
        out_jsonl=args.out_jsonl,
        out_meta=args.out_meta,
        games=args.games,
        players=args.players,
        board_type=bt,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        value_blend=args.value_blend,
        rollout_policy=args.rollout_policy,
        temperature_cutoff=args.temperature_cutoff,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "state_encoder"}, indent=2))


if __name__ == "__main__":
    main()
