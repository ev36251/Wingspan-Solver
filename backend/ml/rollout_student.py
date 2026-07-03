"""Distilled fast rollout policy (the "student").

Profiling the deployed rollout policy (solo_net_spread + full StateEncoder)
showed ~76% of a full playout is the per-ply policy cost, and most of THAT is
the Python feature encoding (1.38 ms of a ~2 ms ply), not the numpy forward
(0.15 ms). So the speed lever is a policy with cheap features, not a smaller
copy of the same net.

This module provides:
  - FastRolloutEncoder: ~40 cheap counter features (~0.05 ms vs 1.38 ms).
  - Distillation dataset generation: states are sampled from big-net-policy
    playouts (the exact distribution rollouts visit) and labeled with the
    teacher's per-head softmax distributions.
  - A tiny student MLP trained with per-head soft-target cross-entropy
    (torch, CPU) and exported in the FactorizedPolicyModel .npz format
    (policy heads only — no value / move-value heads, which also removes the
    per-move move-feature cost from score_moves).
  - load_rollout_student(): loads the student (model, encoder) pair for use as
    the rollout policy inside the search (the big net keeps the root ranking).

CLI:
  python -m backend.ml.rollout_student gen   --games 800 --out reports/ml/solo_seed/distill_states.npz
  python -m backend.ml.rollout_student train --data reports/ml/solo_seed/distill_states.npz \
      --out reports/ml/solo_seed/rollout_student.npz
  python -m backend.ml.rollout_student bench --model reports/ml/solo_seed/rollout_student.npz
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from backend.models.enums import BoardType, FoodType, Habitat

STUDENT_ENCODER_TAG = "fast_rollout_v1"

_FOOD_TYPES = [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
               FoodType.FRUIT, FoodType.RODENT, FoodType.NECTAR]
_HABITATS = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND]


class FastRolloutEncoder:
    """Cheap counter features for the rollout policy. Same .encode(game, idx)
    interface as StateEncoder, ~25x faster (no bird identities, no per-slot
    blocks — the student's job is plausible rollout moves, not deep reads)."""

    def feature_names(self) -> list[str]:
        names = ["round", "turn", "my_cubes", "opp_cubes", "board_type_oceania"]
        names += [f"my_food_{ft.value}" for ft in _FOOD_TYPES]
        for h in _HABITATS:
            names += [f"my_{h.value}_birds", f"my_{h.value}_eggs",
                      f"my_{h.value}_egg_space", f"my_{h.value}_nectar_spent"]
        names += ["my_cached_food", "my_tucked", "my_hand", "my_unknown_hand",
                  "my_bonus_cards", "my_food_total"]
        names += ["opp_birds", "opp_eggs", "opp_hand_total", "opp_food_total",
                  "opp_cached_plus_tucked"]
        names += [f"feeder_{ft.value}" for ft in _FOOD_TYPES]
        names += ["feeder_dice", "tray_cards", "deck_remaining"]
        return names

    def encode(self, game, player_idx: int) -> list[float]:
        me = game.players[player_idx]
        f: list[float] = [
            game.current_round / 4.0,
            min(game.turn_in_round, 10) / 10.0,
            me.action_cubes_remaining / 8.0,
            0.0,  # opp cubes, filled below
            1.0 if game.board_type == BoardType.OCEANIA else 0.0,
        ]
        fs = me.food_supply
        f += [min(fs.get(ft), 6) / 6.0 for ft in _FOOD_TYPES]

        cached = tucked = 0
        for h in _HABITATS:
            row = me.board.get_row(h)
            birds = 0
            eggs = 0
            space = 0
            for s in row.slots:
                if s.bird is not None:
                    birds += 1
                    eggs += s.eggs
                    space += max(0, s.bird.egg_limit - s.eggs)
                    cached += s.total_cached_food
                    tucked += s.tucked_cards
            f += [birds / 5.0, eggs / 12.0, space / 12.0,
                  min(row.nectar_spent, 6) / 6.0]

        f += [min(cached, 20) / 20.0, min(tucked, 25) / 25.0,
              min(len(me.hand), 8) / 8.0, min(me.unknown_hand_count, 8) / 8.0,
              min(len(me.bonus_cards), 4) / 4.0, min(fs.total(), 15) / 15.0]

        opp_birds = opp_eggs = opp_hand = opp_food = opp_engine = 0
        opp_cubes = 0.0
        for j, p in enumerate(game.players):
            if j == player_idx:
                continue
            opp_cubes = max(opp_cubes, p.action_cubes_remaining / 8.0)
            opp_birds += p.board.total_birds()
            opp_eggs += p.board.total_eggs()
            opp_hand += len(p.hand) + p.unknown_hand_count
            opp_food += p.food_supply.total()
            for _, _, s in p.board.all_slots():
                opp_engine += s.total_cached_food + s.tucked_cards
        f[3] = opp_cubes
        f += [min(opp_birds, 15) / 15.0, min(opp_eggs, 30) / 30.0,
              min(opp_hand, 10) / 10.0, min(opp_food, 15) / 15.0,
              min(opp_engine, 30) / 30.0]

        feeder_counts = {ft: 0 for ft in _FOOD_TYPES}
        for die in game.birdfeeder.dice:
            if isinstance(die, tuple):
                for ft in die:
                    feeder_counts[ft] = feeder_counts.get(ft, 0) + 1
            else:
                feeder_counts[die] = feeder_counts.get(die, 0) + 1
        f += [min(feeder_counts[ft], 5) / 5.0 for ft in _FOOD_TYPES]
        f += [len(game.birdfeeder.dice) / 5.0,
              len(game.card_tray.face_up) / 3.0,
              min(game.deck_remaining, 250) / 250.0]
        return f


# ── Distillation dataset ─────────────────────────────────────────────────────

def _teacher_head_probs(logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for hn, lg in logits.items():
        e = np.exp(lg - float(np.max(lg)))
        out[hn] = (e / e.sum()).astype(np.float32)
    return out


def generate_distill_dataset(games: int, out_path: str, temp: float = 0.3,
                             seed0: int = 5000, board: BoardType = BoardType.OCEANIA):
    """Play big-net-policy games (the rollout distribution) and log
    (fast_features, teacher per-head softmax) at every decision of both seats."""
    import math
    import random as _random
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.ml.two_player import build_2p_game
    from backend.solver.move_generator import generate_all_moves
    from backend.solver.simulation import execute_move_on_sim, _refill_tray

    teacher = FactorizedPolicyModel("reports/ml/solo_seed/solo_net_spread.npz")
    big_enc = StateEncoder.resolve_for_model(teacher.meta)
    fast_enc = FastRolloutEncoder()
    head_names = sorted(teacher.head_dims)
    rng = _random.Random(1234)

    feats: list[list[float]] = []
    probs: dict[str, list[np.ndarray]] = {hn: [] for hn in head_names}

    t0 = time.time()
    for gi in range(games):
        game = build_2p_game(seed0 + gi, board)
        turns = 0
        while not game.is_game_over and turns < 300:
            p = game.current_player
            if p.action_cubes_remaining <= 0:
                if all(pl.action_cubes_remaining <= 0 for pl in game.players):
                    game.advance_round()
                else:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue
            moves = generate_all_moves(game, p)
            if not moves:
                p.action_cubes_remaining = 0
                game.advance_turn()
                turns += 1
                continue
            st = np.asarray(big_enc.encode(game, game.current_player_idx), dtype=np.float32)
            logits, _ = teacher.forward(st)

            feats.append(fast_enc.encode(game, game.current_player_idx))
            hp = _teacher_head_probs(logits)
            for hn in head_names:
                probs[hn].append(hp[hn])

            # Sample the next move exactly like the deployed rollout policy.
            scores = teacher.score_moves(st, moves, p, logits=logits)
            mx = max(scores)
            weights = [math.exp((s - mx) / max(1e-6, temp)) for s in scores]
            tot = sum(weights)
            r = rng.random() * tot
            u, mv = 0.0, moves[-1]
            for m, w in zip(moves, weights):
                u += w
                if u >= r:
                    mv = m
                    break
            if execute_move_on_sim(game, p, mv):
                game.advance_turn()
                _refill_tray(game)
            else:
                p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1)
                game.advance_turn()
            turns += 1
        if (gi + 1) % 100 == 0:
            print(f"  {gi+1}/{games} games, {len(feats)} states, {time.time()-t0:.0f}s", flush=True)

    arrs = {"features": np.asarray(feats, dtype=np.float32)}
    for hn in head_names:
        arrs[f"probs_{hn}"] = np.stack(probs[hn])
    arrs["metadata_json"] = np.array([json.dumps({
        "encoder": STUDENT_ENCODER_TAG,
        "feature_names": fast_enc.feature_names(),
        "head_dims": {hn: int(teacher.head_dims[hn]) for hn in head_names},
        "teacher": "solo_net_spread.npz",
        "games": games, "temp": temp, "seed0": seed0,
    })])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    print(f"saved {len(feats)} states -> {out_path}")


# ── Training (torch, CPU) ────────────────────────────────────────────────────

def train_student(data_path: str, out_path: str, hidden1: int = 128, hidden2: int = 64,
                  epochs: int = 40, lr: float = 1e-3, batch: int = 512, seed: int = 0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    z = np.load(data_path, allow_pickle=True)
    meta = json.loads(str(z["metadata_json"][0]))
    head_dims: dict[str, int] = meta["head_dims"]
    head_names = sorted(head_dims)
    X = torch.from_numpy(z["features"])
    Y = {hn: torch.from_numpy(z[f"probs_{hn}"]) for hn in head_names}
    n = X.shape[0]
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"{n} states, {X.shape[1]} features, train {len(tr_idx)} / val {len(val_idx)}")

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(X.shape[1], hidden1)
            self.l2 = nn.Linear(hidden1, hidden2)
            self.heads = nn.ModuleDict({hn: nn.Linear(hidden2, head_dims[hn]) for hn in head_names})

        def forward(self, x):
            h = torch.relu(self.l2(torch.relu(self.l1(x))))
            return {hn: self.heads[hn](h) for hn in head_names}

    net = Student()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def soft_ce(logits, target_probs):
        return -(target_probs * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def eval_val():
        net.eval()
        with torch.no_grad():
            out = net(X[val_idx])
            loss = sum(soft_ce(out[hn], Y[hn][val_idx]) for hn in head_names)
            # top-1 agreement with the teacher's argmax, per head
            agree = {hn: float((out[hn].argmax(1) == Y[hn][val_idx].argmax(1)).float().mean())
                     for hn in head_names}
        net.train()
        return float(loss), agree

    best_val, best_state = 1e18, None
    for ep in range(1, epochs + 1):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            out = net(X[b])
            loss = sum(soft_ce(out[hn], Y[hn][b]) for hn in head_names)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl, agree = eval_val()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            ag = " ".join(f"{hn}={agree[hn]:.2f}" for hn in head_names)
            print(f"epoch {ep:3d}  val_loss {vl:.4f}  agree: {ag}", flush=True)

    net.load_state_dict(best_state)
    vl, agree = eval_val()
    print(f"best val_loss {vl:.4f}")

    # Export in FactorizedPolicyModel npz format (policy heads only).
    sd = net.state_dict()
    arrs = {
        "W1": sd["l1.weight"].numpy().T.astype(np.float32),
        "b1": sd["l1.bias"].numpy().astype(np.float32),
        "W2": sd["l2.weight"].numpy().T.astype(np.float32),
        "b2": sd["l2.bias"].numpy().astype(np.float32),
    }
    for hn in head_names:
        arrs[f"W_{hn}"] = sd[f"heads.{hn}.weight"].numpy().T.astype(np.float32)
        arrs[f"b_{hn}"] = sd[f"heads.{hn}.bias"].numpy().astype(np.float32)
    arrs["metadata_json"] = np.array([json.dumps({
        "head_dims": head_dims,
        "state_encoder": {"custom": STUDENT_ENCODER_TAG},
        "feature_names": meta["feature_names"],
        "distilled_from": meta.get("teacher", "solo_net_spread.npz"),
        "val_loss": vl,
        "val_head_agreement": agree,
        "hidden": [hidden1, hidden2],
        "train_states": int(n),
    })])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    print(f"saved student -> {out_path}")


# ── Loading / wiring ─────────────────────────────────────────────────────────

def is_student_model(model) -> bool:
    enc_meta = (model.meta or {}).get("state_encoder")
    return isinstance(enc_meta, dict) and enc_meta.get("custom") == STUDENT_ENCODER_TAG


def load_rollout_student(path: str | Path):
    """Load the student as a (model, encoder) pair, or None if missing/invalid."""
    from backend.ml.factorized_inference import FactorizedPolicyModel
    p = Path(path)
    if not p.exists():
        return None
    model = FactorizedPolicyModel(p)
    if not is_student_model(model):
        return None
    return model, FastRolloutEncoder()


DEFAULT_STUDENT_PATH = "reports/ml/solo_seed/rollout_student.npz"


# ── Quality gate (paired A/B on the real search agent) ──────────────────────

_GATE = {}


def _gate_worker_init(student_path: str):
    from backend.config import EXCEL_FILE
    from backend.data.registries import load_all
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    load_all(EXCEL_FILE)
    big = FactorizedPolicyModel("reports/ml/solo_seed/solo_net_spread.npz")
    _GATE["big"] = big
    _GATE["big_enc"] = StateEncoder.resolve_for_model(big.meta)
    _GATE["student"] = load_rollout_student(student_path)


def _gate_play_one(job):
    """One agent-vs-heuristic game. job = (seed, arm, rollouts, top_k, temp)."""
    from backend.ml.two_player import (build_2p_game, play_multi,
                                       make_search_chooser, heuristic_chooser)
    seed, arm, rollouts, top_k, temp = job
    big, big_enc = _GATE["big"], _GATE["big_enc"]
    stu = _GATE["student"]
    r_model, r_enc = (stu if arm.startswith("student") and stu else (None, None))
    a_idx = seed % 2
    a_ch = make_search_chooser(big, big_enc, top_k, a_idx, heuristic_chooser,
                               objective="selfish", rollouts=rollouts,
                               temperature=temp, determinize=True,
                               rollout_model=r_model, rollout_encoder=r_enc)
    choosers = [None, None]
    choosers[a_idx] = a_ch
    choosers[1 - a_idx] = heuristic_chooser
    game = build_2p_game(seed, BoardType.OCEANIA)
    t0 = time.time()
    scores = play_multi(game, choosers)
    return {"seed": seed, "arm": arm, "rollouts": rollouts,
            "agent": scores[a_idx], "opp": scores[1 - a_idx],
            "secs": round(time.time() - t0, 1)}


def run_gate(seeds: list[int], student_path: str, rollouts: int = 8, top_k: int = 10,
             temp: float = 0.3, procs: int = 4, matched_rollouts: int | None = None,
             out_path: str = "reports/ml/solo_seed/rollout_student_gate.json"):
    """Paired 3-arm gate on the same seeds:
      baseline        — big-net rollouts at r
      student_same_r  — student rollouts at the same r  (quality-cost check)
      student_matched — student rollouts at wall-clock-matched higher r (strength)
    """
    from multiprocessing import get_context

    jobs = []
    for s in seeds:
        jobs.append((s, "baseline", rollouts, top_k, temp))
        jobs.append((s, "student_same_r", rollouts, top_k, temp))
        if matched_rollouts and matched_rollouts > rollouts:
            jobs.append((s, "student_matched", matched_rollouts, top_k, temp))

    ctx = get_context("spawn")
    results = []
    t0 = time.time()
    with ctx.Pool(procs, initializer=_gate_worker_init, initargs=(student_path,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_gate_play_one, jobs), 1):
            results.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} games, {time.time()-t0:.0f}s", flush=True)

    by_arm: dict[str, dict[int, dict]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r

    summary = {}
    base = by_arm.get("baseline", {})
    for arm, rows in sorted(by_arm.items()):
        agent = [rows[s]["agent"] for s in sorted(rows)]
        secs = [rows[s]["secs"] for s in sorted(rows)]
        entry = {
            "n": len(agent),
            "mean": round(float(np.mean(agent)), 2),
            "floor": int(min(agent)),
            "median": float(np.median(agent)),
            "wins": sum(1 for s in rows if rows[s]["agent"] > rows[s]["opp"]),
            "mean_secs_per_game": round(float(np.mean(secs)), 1),
        }
        if arm != "baseline" and base:
            common = sorted(set(rows) & set(base))
            diffs = [rows[s]["agent"] - base[s]["agent"] for s in common]
            up = sum(1 for d in diffs if d > 0)
            dn = sum(1 for d in diffs if d < 0)
            entry["paired_mean_diff_vs_baseline"] = round(float(np.mean(diffs)), 2)
            entry["paired_up_down"] = f"{up}up/{dn}dn/{len(diffs)-up-dn}tie"
        summary[arm] = entry
        print(f"{arm:16s} mean {entry['mean']:6.2f}  floor {entry['floor']:3d}  "
              f"wins {entry['wins']}/{entry['n']}  {entry['mean_secs_per_game']}s/game"
              + (f"  paired diff {entry.get('paired_mean_diff_vs_baseline')}"
                 f" ({entry.get('paired_up_down')})" if arm != "baseline" else ""))

    payload = {"config": {"seeds": seeds, "rollouts": rollouts, "top_k": top_k,
                          "temp": temp, "matched_rollouts": matched_rollouts,
                          "student": student_path},
               "summary": summary, "games": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"gate results -> {out_path}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from backend.config import EXCEL_FILE
    from backend.data.registries import load_all
    load_all(EXCEL_FILE)

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=800)
    g.add_argument("--out", default="reports/ml/solo_seed/distill_states.npz")
    g.add_argument("--temp", type=float, default=0.3)
    g.add_argument("--seed0", type=int, default=5000)
    t = sub.add_parser("train")
    t.add_argument("--data", default="reports/ml/solo_seed/distill_states.npz")
    t.add_argument("--out", default=DEFAULT_STUDENT_PATH)
    t.add_argument("--hidden1", type=int, default=128)
    t.add_argument("--hidden2", type=int, default=64)
    t.add_argument("--epochs", type=int, default=40)
    b = sub.add_parser("bench")
    b.add_argument("--model", default=DEFAULT_STUDENT_PATH)
    b.add_argument("--playouts", type=int, default=8)
    q = sub.add_parser("gate")
    q.add_argument("--seeds", default="0-39")
    q.add_argument("--student", default=DEFAULT_STUDENT_PATH)
    q.add_argument("--rollouts", type=int, default=8)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--matched-rollouts", type=int, default=0,
                   help="student arm at this higher r (wall-clock matched); 0 = skip")
    q.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "gen":
        generate_distill_dataset(args.games, args.out, temp=args.temp, seed0=args.seed0)
    elif args.cmd == "train":
        train_student(args.data, args.out, hidden1=args.hidden1, hidden2=args.hidden2,
                      epochs=args.epochs)
    elif args.cmd == "bench":
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder
        from backend.ml.two_player import build_2p_game, play_multi
        from backend.ml.solo_eval import make_net_sampling_chooser
        from backend.solver.simulation import fast_clone_game

        big = FactorizedPolicyModel("reports/ml/solo_seed/solo_net_spread.npz")
        big_enc = StateEncoder.resolve_for_model(big.meta)
        pair = load_rollout_student(args.model)
        if pair is None:
            raise SystemExit(f"no student at {args.model}")
        stu, stu_enc = pair

        def bench(model, enc, label):
            tot = 0.0
            for s in range(args.playouts):
                g = build_2p_game(100 + s, BoardType.OCEANIA)
                sim = fast_clone_game(g)
                ch = make_net_sampling_chooser(model, enc, 0.3, seed=s)
                t0 = time.perf_counter()
                play_multi(sim, [ch, ch])
                tot += time.perf_counter() - t0
            print(f"{label}: {tot/args.playouts*1000:.0f} ms/playout")
            return tot / args.playouts

        tb = bench(big, big_enc, "teacher playout")
        ts = bench(stu, stu_enc, "student playout")
        print(f"speedup: {tb/ts:.2f}x")
    elif args.cmd == "gate":
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
        run_gate(seeds, args.student, rollouts=args.rollouts, top_k=args.top_k,
                 matched_rollouts=args.matched_rollouts or None, procs=args.procs)


if __name__ == "__main__":
    main()
