"""Generate opponent-aware 2-player BC training data.

Bootstrap step 1: play 2-player games with our strong *selfish* rollout search
(driven by the existing solo net) and record every decision as a training row
whose STATE is encoded with opponent-board features ON. The search picks strong
moves with the 1693-dim solo net; we store the state with a separate 2443-dim
training encoder (own board + the leading opponent's full board). Training on
these rows yields a net of similar strength that can *see* across the table --
usable as a realistic opponent model inside the rollouts (which is what the
n=100 ablation showed the differential objective was missing).

    # local smoke
    python -m backend.ml.two_player_dataset --seeds 0-1 --out /tmp/probe.jsonl
    # many seeds across Modal
    python -m backend.ml.two_player_dataset --seeds 0-499 --use-modal \
        --seeds-per-shard 4 --out reports/ml/two_player/opp_aware.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.engine.scoring import calculate_score
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, _refill_tray
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.factorized_policy import encode_factorized_targets, HeadDims
from backend.ml.move_features import encode_move_features
from backend.ml.solo_eval import make_net_chooser
from backend.ml.two_player import build_2p_game, make_search_chooser

# Opponent-aware training encoder: same flags as solo_net_spread PLUS the
# opponent-board block (+750 dims -> feature_dim 2443).
OPP_ENCODER_CFG = {
    "enable_identity_features": True,
    "identity_hash_dim": 128,
    "use_per_slot_encoding": True,
    "max_hand_slots": 8,
    "use_hand_habitat_features": True,
    "use_tray_per_slot_encoding": True,
    "use_opponent_board_encoding": True,
    "use_power_features": True,
}


def generate_game_rows(seed, model, model_encoder, train_encoder, board, top_k=6):
    """Play one 2-player selfish-search game; return opponent-aware BC rows.

    Both seats are played by the strong selfish search (using `model_encoder` for
    move selection). Each decision is stored with `train_encoder` (opponent board
    on). value_target_score is filled with that seat's final score.
    """
    net_choose = make_net_chooser(model, model_encoder)
    choosers = [
        make_search_chooser(model, model_encoder, top_k, idx, net_choose, "selfish")
        for idx in range(2)
    ]
    game = build_2p_game(seed, board)

    pending = []  # (player_idx, state, targets, move_pos, move_negs)
    turns = 0
    while not game.is_game_over and turns < 600:
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

        state = np.asarray(train_encoder.encode(game, idx), dtype=np.float32)
        mv = choosers[idx](game, p, moves)
        move_pos = list(encode_move_features(mv, p))
        move_negs = [list(encode_move_features(m, p))
                     for m in moves if m.description != mv.description][:4]
        pending.append((idx, state, encode_factorized_targets(mv, p), move_pos, move_negs))

        if execute_move_on_sim(game, p, mv):
            game.advance_turn()
            _refill_tray(game)
        else:
            p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1

    finals = [calculate_score(game, pl).total for pl in game.players]
    rows = []
    for idx, state, targets, move_pos, move_negs in pending:
        rows.append({
            "state": [float(x) for x in state],
            "targets": targets,
            "value_target_score": float(finals[idx]),
            "record_policy": True,
            "move_pos": move_pos,
            "move_negs": move_negs,
        })
    return rows


def seed_rows(seed, model, model_encoder, train_encoder, board, top_k=6):
    return generate_game_rows(seed, model, model_encoder, train_encoder, board, top_k)


def build_encoders(model_path):
    model = FactorizedPolicyModel(model_path)
    model_encoder = StateEncoder.resolve_for_model(model.meta)
    train_encoder = StateEncoder.from_metadata({"state_encoder": OPP_ENCODER_CFG})
    return model, model_encoder, train_encoder


def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="0-1")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--out", default="reports/ml/two_player/opp_aware.jsonl")
    ap.add_argument("--out-meta", default=None)
    ap.add_argument("--use-modal", action="store_true")
    ap.add_argument("--seeds-per-shard", type=int, default=4)
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)
    out_meta = args.out_meta or (args.out.rsplit(".", 1)[0] + ".meta.json")

    if args.use_modal:
        from backend.ml.modal_two_player import dispatch_2p_dataset_modal
        rows = dispatch_2p_dataset_modal(seeds, args.model, board, args.top_k,
                                         args.seeds_per_shard)
    else:
        load_all(EXCEL_FILE)
        model, model_encoder, train_encoder = build_encoders(args.model)
        rows = []
        for s in seeds:
            rows.extend(seed_rows(s, model, model_encoder, train_encoder, board, args.top_k))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    feature_dim = len(rows[0]["state"]) if rows else 0
    heads = HeadDims()
    meta = {
        "feature_dim": feature_dim,
        "target_heads": {
            "action_type": heads.action_type,
            "play_habitat": heads.play_habitat,
            "gain_food_primary": heads.gain_food_primary,
            "draw_mode": heads.draw_mode,
            "lay_eggs_bin": heads.lay_eggs_bin,
            "play_cost_bin": heads.play_cost_bin,
            "play_power_color": heads.play_power_color,
        },
        "samples": len(rows),
        "n_games": len(seeds),
        "board_type": board.value,
        "players": 2,
        "source": "two_player_selfish_search",
        "value_target_config": {
            "mode": "absolute",
            "score_scale": 100.0,
            "score_bias": 0.0,
            "win_target_mode": "none",
            "breakdown_components": [],
            "breakdown_scale": 40.0,
        },
        "state_encoder": OPP_ENCODER_CFG,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(rows)} samples (feature_dim={feature_dim}) from {len(seeds)} games "
          f"-> {args.out}\nmeta -> {out_meta}")


if __name__ == "__main__":
    main()
