"""Build a factorized-BC training dataset from solo best-line trajectories (Option A).

Replays each best line deterministically: rebuild the seed's fixed deal + the
recorded draft, then match each stored move by its description against the
freshly generated legal moves. At every decision it emits one training sample in
the schema `train_factorized_bc` expects:
  - state vector (StateEncoder, solo config)
  - factorized head targets for the chosen move (encode_factorized_targets)
  - move-ranking features (positive + up to 4 negatives)
  - value target = the line's final score (absolute)

Usage:
    python -m backend.ml.solo_bc_dataset \
        --in reports/ml/solo_seed/best_lines_4000_q250.jsonl \
        --out-jsonl reports/ml/solo_seed/solo_bc.jsonl \
        --out-meta  reports/ml/solo_seed/solo_bc.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType, FoodType
from backend.ml.state_encoder import StateEncoder
from backend.ml.factorized_policy import encode_factorized_targets, HeadDims
from backend.ml.move_features import encode_move_features
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, _refill_tray
from backend.ml.solo_seed_optimizer import deal_seed, build_game_from_draft

# Solo encoder: own-board per-slot + power-effect + identity features; no
# opponent-board block (there is no opponent in solo).
ENCODER_CFG = {
    "enable_identity_features": True,
    "identity_hash_dim": 128,
    "use_per_slot_encoding": True,
    "max_hand_slots": 8,
    "use_hand_habitat_features": True,
    "use_tray_per_slot_encoding": True,
    "use_opponent_board_encoding": False,
    "use_power_features": True,
}

SCORE_SCALE = 100.0
SCORE_BIAS = 0.0


def _desc_of(traj_entry: str) -> str:
    """'R1 play_bird: Play X in grassland (...)' -> 'Play X in grassland (...)'."""
    return traj_entry.split(": ", 1)[1] if ": " in traj_entry else traj_entry


def replay_row(row: dict, encoder: StateEncoder) -> list[dict] | None:
    """Replay one line; return its training samples, or None on a mismatch.

    role="policy"     -> samples train policy + value (+ move-value) heads
    role="value_only" -> samples train only the value head (record_policy=False,
                         no move features) so weak lines give the value head
                         low/medium outcomes without polluting the policy.
    """
    seed = int(row["seed"])
    policy_role = row.get("role", "policy") == "policy"
    draft = row.get("draft") or {}
    if not draft.get("birds_kept") and draft.get("num_birds_kept", 0) != 0:
        return None

    deal = deal_seed(seed, BoardType.OCEANIA)
    food = {}
    for k, v in (draft.get("food_kept") or {}).items():
        try:
            food[FoodType(k)] = int(v)
        except ValueError:
            continue
    bonus = next((b for b in deal.dealt_bonus if b.name == draft.get("bonus")),
                 deal.dealt_bonus[0])
    game = build_game_from_draft(deal, draft.get("birds_kept", []), food, bonus)

    final_score = float(row["score"])
    descs = [_desc_of(t) for t in row.get("trajectory", [])]
    samples: list[dict] = []
    di = 0
    turns = 0
    while not game.is_game_over and turns < 400 and di < len(descs):
        player = game.current_player
        if player.action_cubes_remaining <= 0:
            game.advance_round()
            continue
        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.advance_turn()
            continue

        want = descs[di]
        chosen = next((m for m in moves if m.description == want), None)
        if chosen is None:
            return None  # description didn't reproduce — skip the whole row

        # Encode the state BEFORE applying the move.
        state = encoder.encode(game, game.current_player_idx)
        targets = encode_factorized_targets(chosen, player)
        sample = {
            "state": state,
            "targets": targets,                 # required by loader even if unused
            "value_target_score": final_score,
            "record_policy": policy_role,
        }
        if policy_role:
            sample["move_pos"] = list(encode_move_features(chosen, player))
            sample["move_negs"] = [
                list(encode_move_features(m, player))
                for m in moves if m.description != want
            ][:4]
        samples.append(sample)

        di += 1
        if execute_move_on_sim(game, player, chosen):
            game.advance_turn()
            _refill_tray(game)
        else:
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1

    return samples


def build(in_path: str, out_jsonl: str, out_meta: str) -> dict:
    load_all(EXCEL_FILE)
    encoder = StateEncoder.from_metadata({"state_encoder": ENCODER_CFG})

    # feature_dim from a dummy encode.
    probe_deal = deal_seed(0, BoardType.OCEANIA)
    probe_game = build_game_from_draft(
        probe_deal, [b.name for b in probe_deal.dealt_birds[:1]], {}, probe_deal.dealt_bonus[0]
    )
    feature_dim = len(encoder.encode(probe_game, 0))

    rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    n_ok = 0
    n_skip = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for row in rows:
            samples = replay_row(row, encoder)
            if samples is None:
                n_skip += 1
                continue
            n_ok += 1
            for s in samples:
                fout.write(json.dumps(s) + "\n")
                n_samples += 1

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
        "samples": n_samples,
        "source_lines": len(rows),
        "lines_used": n_ok,
        "lines_skipped": n_skip,
        "board_type": BoardType.OCEANIA.value,
        "players": 1,
        "value_target_config": {
            "mode": "absolute",
            "score_scale": SCORE_SCALE,
            "score_bias": SCORE_BIAS,
            "win_target_mode": "none",
            "breakdown_components": [],
            "breakdown_scale": 40.0,
        },
        "state_encoder": ENCODER_CFG,
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"feature_dim={feature_dim}")
    print(f"lines: {n_ok} used, {n_skip} skipped (description mismatch) of {len(rows)}")
    print(f"samples written: {n_samples}  ({n_samples / max(1, n_ok):.1f}/line)")
    print(f"-> {out_jsonl}\n-> {out_meta}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="reports/ml/solo_seed/best_lines_4000_q250.jsonl")
    ap.add_argument("--out-jsonl", default="reports/ml/solo_seed/solo_bc.jsonl")
    ap.add_argument("--out-meta", default="reports/ml/solo_seed/solo_bc.meta.json")
    args = ap.parse_args()
    build(args.in_path, args.out_jsonl, args.out_meta)


if __name__ == "__main__":
    main()
