"""Self-play loop step 1: soft-target data generator (expert iteration).

Differs from two_player_dataset.py (which produced the null opp_aware nets) in two
ways that matter:
  1. STRONG teacher. The old data used make_search_chooser's default (1 greedy
     rollout). Here both seats use the deployed search (rollouts>1, determinize,
     temperature) so the teacher is genuinely stronger than its own prior.
  2. SOFT targets. Instead of the argmax move, we log the search's value-weighted
     distribution over the top_k candidates (softmax of their averaged rollout
     returns) as factorized soft policy targets -- the AlphaZero "improved policy"
     signal none of the prior BC attempts used.

State is encoded with the opponent-board training encoder (dim 2443); the search
itself uses the model's own encoder. Output rows feed train_factorized_bc
unchanged (it already reads `policy_visit_targets`).

    python -m backend.ml.selfplay_soft --seeds 0-499 --use-modal \
        --seeds-per-shard 4 --rollouts 6 --top-k 8 \
        --out reports/ml/two_player/soft_iter1.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.engine.scoring import calculate_score
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, fast_clone_game, _refill_tray
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.factorized_policy import encode_factorized_targets, HeadDims
from backend.ml.move_features import encode_move_features
from backend.ml.solo_eval import make_net_sampling_chooser
from backend.ml.two_player import build_2p_game, play_multi
from backend.ml.two_player_dataset import OPP_ENCODER_CFG, build_encoders
from backend.ml.alphazero_self_play import _encode_policy_visit_targets

_HERE = Path(__file__).parent
_XLSX = _HERE.parent.parent / "wingspan-20260128.xlsx"


def _softmax(v, temp):
    a = np.asarray(v, dtype=np.float64)
    a = (a - a.max()) / max(1e-6, temp)
    e = np.exp(a)
    return e / max(1e-12, e.sum())


def generate_soft_rows(seed, model, model_encoder, train_encoder, board,
                       top_k=8, rollouts=6, temp_roll=0.3, temp_soft=3.0,
                       determinize=True):
    """Both seats play the strong search; log soft policy targets per decision."""
    roll_a = make_net_sampling_chooser(model, model_encoder, temp_roll, seed=1)
    roll_b = make_net_sampling_chooser(model, model_encoder, temp_roll, seed=2)
    det_rng = random.Random(seed * 104729 + 13)

    def _rollout_value(sim, my_idx):
        ch = [roll_a if i == my_idx else roll_b for i in range(sim.num_players)]
        return play_multi(sim, ch)[my_idx]

    def _eval_candidates(game, my_idx, p, moves):
        sm = np.asarray(model_encoder.encode(game, my_idx), dtype=np.float32)
        logits, _ = model.forward(sm)
        ranked = sorted(moves, key=lambda m: -model.score_move(sm, m, p, logits=logits))[:top_k]
        vals = []
        for m in ranked:
            tot, n = 0.0, 0
            for _ in range(max(1, rollouts)):
                sim = fast_clone_game(game)
                if determinize:
                    deck = getattr(sim, "_deck_cards", None)
                    if isinstance(deck, list) and len(deck) > 1:
                        det_rng.shuffle(deck)
                sp = sim.current_player
                if not execute_move_on_sim(sim, sp, m):
                    continue
                sim.advance_turn(); _refill_tray(sim)
                tot += _rollout_value(sim, my_idx); n += 1
            vals.append(tot / n if n else -1e9)
        return ranked, vals

    pending = []  # (idx, state_train, hard_targets, visit_targets, move_pos, move_negs)
    game = build_2p_game(seed, board)
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
            p.action_cubes_remaining = 0; game.advance_turn(); turns += 1; continue

        if len(moves) == 1:
            best = moves[0]
        else:
            ranked, vals = _eval_candidates(game, idx, p, moves)
            best = ranked[int(np.argmax(vals))]
            probs = _softmax(vals, temp_soft)
            state_train = np.asarray(train_encoder.encode(game, idx), dtype=np.float32)
            hard = encode_factorized_targets(best, p)
            visit, _, _ = _encode_policy_visit_targets(p, ranked, list(probs), hard)
            move_pos = list(encode_move_features(best, p))
            move_negs = [list(encode_move_features(m, p))
                         for m in ranked if m.description != best.description][:4]
            pending.append((idx, state_train, hard, visit, move_pos, move_negs))

        if execute_move_on_sim(game, p, best):
            game.advance_turn(); _refill_tray(game)
        else:
            p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1); game.advance_turn()
        turns += 1

    finals = [calculate_score(game, pl).total for pl in game.players]
    rows = []
    for idx, state, hard, visit, mp, mn in pending:
        rows.append({
            "state": [float(x) for x in state],
            "targets": hard,
            "policy_visit_targets": visit,
            "value_target_score": float(finals[idx]),
            "record_policy": True,
            "move_pos": mp,
            "move_negs": mn,
        })
    return rows


# --------------------------------------------------------------------------- #
# Modal dispatch
# --------------------------------------------------------------------------- #
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore

if _MODAL_AVAILABLE:
    _image = (
        modal.Image.debian_slim(python_version="3.12")
        .env({"PYTHONHASHSEED": "0"})
        .pip_install("numpy", "openpyxl", "threadpoolctl")
        .add_local_file(str(_XLSX), remote_path="/root/wingspan-20260128.xlsx")
    )
    app = modal.App("wingspan-selfplay-soft", image=_image)

    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_soft_shard_remote(task: dict) -> dict:
        import os, tempfile
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(k, "1")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass
        load_all(EXCEL_FILE)
        mb = task.pop("model_bytes")
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.write(mb); tmp.close()
        try:
            model, model_encoder, train_encoder = build_encoders(tmp.name)
            board = BoardType(task["board_type"])
            c = task["cfg"]
            rows = []
            for s in task["seeds"]:
                rows.extend(generate_soft_rows(int(s), model, model_encoder, train_encoder,
                                               board, c["top_k"], c["rollouts"],
                                               c["temp_roll"], c["temp_soft"], c["determinize"]))
        finally:
            os.unlink(tmp.name)
        return {"rows_gz": gzip.compress(json.dumps(rows).encode("utf-8"), 6)}


def dispatch_soft_modal(seeds, model_path, board, cfg, seeds_per_shard):
    model_bytes = Path(model_path).read_bytes()
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [{"seeds": s, "board_type": board.value, "model_bytes": model_bytes, "cfg": cfg}
             for s in shards]
    print(f"  [modal] dispatching {len(tasks)} soft-data shards ({len(seeds)} games) …")
    rows = []
    with modal.enable_output(), app.run():
        for result in run_soft_shard_remote.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                rows.extend(json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return rows


def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    from backend.determinism import ensure_deterministic_hashing
    ensure_deterministic_hashing()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--temp-roll", type=float, default=0.3)
    ap.add_argument("--temp-soft", type=float, default=3.0)
    ap.add_argument("--no-determinize", action="store_true")
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--out", default="reports/ml/two_player/soft_iter1.jsonl")
    ap.add_argument("--use-modal", action="store_true")
    ap.add_argument("--seeds-per-shard", type=int, default=4)
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)
    cfg = {"top_k": args.top_k, "rollouts": args.rollouts, "temp_roll": args.temp_roll,
           "temp_soft": args.temp_soft, "determinize": not args.no_determinize}
    if args.use_modal:
        rows = dispatch_soft_modal(seeds, args.model, board, cfg, args.seeds_per_shard)
    else:
        load_all(EXCEL_FILE)
        model, model_encoder, train_encoder = build_encoders(args.model)
        rows = []
        for s in seeds:
            rows.extend(generate_soft_rows(s, model, model_encoder, train_encoder, board,
                                           cfg["top_k"], cfg["rollouts"], cfg["temp_roll"],
                                           cfg["temp_soft"], cfg["determinize"]))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    feature_dim = len(rows[0]["state"]) if rows else 0
    heads = HeadDims()
    meta = {"feature_dim": feature_dim, "state_encoder": OPP_ENCODER_CFG,
            "target_heads": {"action_type": heads.action_type, "play_habitat": heads.play_habitat,
                             "gain_food_primary": heads.gain_food_primary, "draw_mode": heads.draw_mode,
                             "lay_eggs_bin": heads.lay_eggs_bin, "play_cost_bin": heads.play_cost_bin,
                             "play_power_color": heads.play_power_color}}
    json.dump(meta, open(out.with_suffix(".meta.json"), "w"), indent=2)
    print(f"  saved {len(rows)} rows (dim={feature_dim}) -> {out}")


if __name__ == "__main__":
    main()
