"""Value-net go/no-go gate for the "AlphaZero of search value" direction.

The 2-player honest agent evaluates each candidate move by rolling the game to
the end with the net-sampling policy and averaging `rollouts` returns (see
`two_player.make_search_chooser`). That leaf evaluator is the bottleneck: at
r12/k10 we spend 120 full playouts/move and still sit at ~92 mean because each
playout is a high-variance sample of the same quantity.

A learned value V(state) -> expected final score could replace/bootstrap that
rollout. The GATE this module answers, cheaply and before any training loop:

    Is V a *better single-shot estimator* of a state's rollout return than one
    actual playout?

We log, as a free byproduct of real agent play, every candidate leaf state and
its M rollout-return samples. Then:
    sigma_bar   = sqrt(mean_state Var(samples))      # one-playout RMSE vs truth
    V_rmse      = RMSE(V(state), mean(samples))       # learned-eval RMSE vs truth
    R*          = (sigma_bar / V_rmse)^2              # "playouts V is worth"
GREEN if R* > 1 (V beats a single playout); R* >~ rollouts means V alone matches
the current full-search leaf. (V_rmse is bias-corrected for label-mean noise
sigma^2/M.)

Usage
-----
    # generate dataset (real agent leaves + M rollout returns), fanned on Modal
    python -m backend.ml.value_gate gen --seeds 0-99 --use-modal \
        --top-k 10 --rollouts 16 --temperature 0.3 --determinize \
        --out reports/ml/value_gate/data.npz

    # train V and print the gate
    python -m backend.ml.value_gate gate --data reports/ml/value_gate/data.npz
"""

from __future__ import annotations

import argparse
import gzip
import io
import math
import random
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, fast_clone_game, _refill_tray
from backend.engine.scoring import calculate_score
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.solo_eval import make_net_chooser, make_net_sampling_chooser
from backend.ml.two_player import build_2p_game, play_multi, heuristic_chooser

_HERE = Path(__file__).parent
_XLSX = _HERE.parent.parent / "wingspan-20260128.xlsx"


# --------------------------------------------------------------------------- #
# Value model: numpy-only serve-time inference (no torch on Modal shards)
# --------------------------------------------------------------------------- #
class ValueModel:
    """Tiny MLP V(state) -> expected final score. Standardization (mu/sd) is baked
    in, so predict() takes the RAW encoder vector. Loaded from a .npz exported by
    train_value()."""

    def __init__(self, path_or_bytes):
        if isinstance(path_or_bytes, (bytes, bytearray)):
            d = np.load(io.BytesIO(bytes(path_or_bytes)))
        else:
            d = np.load(path_or_bytes)
        self.W = [d["W0"].astype(np.float32), d["W1"].astype(np.float32), d["W2"].astype(np.float32)]
        self.b = [d["b0"].astype(np.float32), d["b1"].astype(np.float32), d["b2"].astype(np.float32)]
        self.mu = d["mu"].astype(np.float32)
        self.sd = d["sd"].astype(np.float32)
        self.scale = float(d["score_scale"])

    def predict(self, x) -> float:
        h = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        h = np.maximum(0.0, h @ self.W[0] + self.b[0])
        h = np.maximum(0.0, h @ self.W[1] + self.b[1])
        return float((h @ self.W[2] + self.b[2])[0]) * self.scale


def train_value(data_path, out_path, hidden=(256, 64), epochs=60, lr=1e-3,
                weight_decay=1e-5, seed=0):
    """Train an MLP on (state -> mean rollout return), export numpy weights."""
    import torch
    d = np.load(data_path)
    states = d["states"].astype(np.float32)
    y = d["samples"].mean(axis=1).astype(np.float32)
    n, D = states.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.9 * n)
    tr, te = idx[:cut], idx[cut:]
    mu = states[tr].mean(axis=0)
    sd = states[tr].std(axis=0) + 1e-6
    Xs = (states - mu) / sd
    scale = 120.0

    torch.manual_seed(seed)
    h1, h2 = hidden
    net = torch.nn.Sequential(
        torch.nn.Linear(D, h1), torch.nn.ReLU(),
        torch.nn.Linear(h1, h2), torch.nn.ReLU(),
        torch.nn.Linear(h2, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.MSELoss()
    Xt = torch.tensor(Xs[tr]); yt = torch.tensor((y[tr] / scale)).unsqueeze(1)
    Xv = torch.tensor(Xs[te])
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward(); opt.step()
    with torch.no_grad():
        pv = net(Xv).squeeze(1).numpy() * scale
    val_rmse = float(np.sqrt(np.mean((pv - y[te]) ** 2)))

    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    arrs = {"score_scale": np.float32(scale),
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    for k, m in enumerate(lin):
        arrs[f"W{k}"] = m.weight.detach().numpy().T.astype(np.float32)  # (in,out)
        arrs[f"b{k}"] = m.bias.detach().numpy().astype(np.float32)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrs)
    print(f"  trained V on {len(tr)} states, val RMSE={val_rmse:.2f} -> {out}")
    return val_rmse


# --------------------------------------------------------------------------- #
# Dataset generation: real search agent, logging every candidate leaf + returns
# --------------------------------------------------------------------------- #
def make_logging_search_chooser(model, encoder, top_k, my_idx, M, temperature,
                                determinize, sink):
    """Depth-1 rollout-search chooser (selfish objective) that LOGS, for every
    top_k candidate, (encoded leaf state, list of M rollout returns) into `sink`,
    then advances by the candidate with the best mean return (the real agent)."""
    roll_me = make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 1)
    roll_opp = make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 2)
    det_rng = random.Random(my_idx * 104729 + 13)

    def _rollout_return(sim):
        choosers = [roll_me if i == my_idx else roll_opp for i in range(sim.num_players)]
        return play_multi(sim, choosers)[my_idx]

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        ranked = sorted(moves, key=lambda m: -model.score_move(state, m, player, logits=logits))[:top_k]
        best_m, best_mean = ranked[0], -1e18
        for m in ranked:
            # Encode the leaf once (deck order is hidden info; encoder ignores it).
            leaf = fast_clone_game(game)
            lp = leaf.current_player
            if not execute_move_on_sim(leaf, lp, m):
                continue
            leaf.advance_turn()
            _refill_tray(leaf)
            leaf_vec = np.asarray(encoder.encode(leaf, my_idx), dtype=np.float16)
            samples = []
            for _ in range(M):
                sim = fast_clone_game(game)
                if determinize:
                    deck = getattr(sim, "_deck_cards", None)
                    if isinstance(deck, list) and len(deck) > 1:
                        det_rng.shuffle(deck)
                sp = sim.current_player
                if not execute_move_on_sim(sim, sp, m):
                    continue
                sim.advance_turn()
                _refill_tray(sim)
                samples.append(_rollout_return(sim))
            if not samples:
                continue
            sink.append((leaf_vec, np.asarray(samples, dtype=np.int16)))
            mean = float(np.mean(samples))
            if mean > best_mean:
                best_m, best_mean = m, mean
        return best_m

    return choose


def make_eval_chooser(model, encoder, value_model, top_k, my_idx, leaf_mode,
                      bootstrap_plies, M, temperature, determinize):
    """Depth-1 search chooser (selfish) whose LEAF evaluator is selectable:
        leaf_mode="full" -> play to game end (the rollout-search baseline)
        leaf_mode="v"    -> roll `bootstrap_plies` of my turns, then V(leaf)
                            (bootstrap_plies=0 = pure value-net instinct).
    """
    roll_me = make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 1)
    roll_opp = make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 2)
    det_rng = random.Random(my_idx * 104729 + 13)

    def _step_until_my_turn(sim):
        while not sim.is_game_over:
            idx = sim.current_player_idx
            p = sim.current_player
            if p.action_cubes_remaining <= 0:
                if all(pl.action_cubes_remaining <= 0 for pl in sim.players):
                    sim.advance_round()
                else:
                    sim.current_player_idx = (idx + 1) % sim.num_players
                continue
            if idx == my_idx:
                return
            mvs = generate_all_moves(sim, p)
            if not mvs:
                p.action_cubes_remaining = 0; sim.advance_turn(); continue
            mv = roll_opp(sim, p, mvs)
            if execute_move_on_sim(sim, p, mv):
                sim.advance_turn(); _refill_tray(sim)
            else:
                p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1); sim.advance_turn()

    def _full_value(sim):
        choosers = [roll_me if i == my_idx else roll_opp for i in range(sim.num_players)]
        return play_multi(sim, choosers)[my_idx]

    def _truncated_value(sim):
        plies = 0
        while not sim.is_game_over and plies < bootstrap_plies:
            _step_until_my_turn(sim)
            if sim.is_game_over:
                break
            p = sim.current_player
            mvs = generate_all_moves(sim, p)
            if not mvs:
                p.action_cubes_remaining = 0; sim.advance_turn(); continue
            mv = roll_me(sim, p, mvs)
            if execute_move_on_sim(sim, p, mv):
                sim.advance_turn(); _refill_tray(sim)
            else:
                p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1); sim.advance_turn()
            plies += 1
        if sim.is_game_over:
            return calculate_score(sim, sim.players[my_idx]).total
        return value_model.predict(encoder.encode(sim, my_idx))

    leaf_fn = _full_value if leaf_mode == "full" else _truncated_value

    def choose(game, player, moves):
        if len(moves) <= 1:
            return moves[0]
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        ranked = sorted(moves, key=lambda m: -model.score_move(state, m, player, logits=logits))[:top_k]
        best_m, best_v = ranked[0], -1e18
        for m in ranked:
            total, n_ok = 0.0, 0
            for _ in range(max(1, M)):
                sim = fast_clone_game(game)
                if determinize:
                    deck = getattr(sim, "_deck_cards", None)
                    if isinstance(deck, list) and len(deck) > 1:
                        det_rng.shuffle(deck)
                sp = sim.current_player
                if not execute_move_on_sim(sim, sp, m):
                    continue
                sim.advance_turn(); _refill_tray(sim)
                total += leaf_fn(sim); n_ok += 1
            if n_ok == 0:
                continue
            val = total / n_ok
            if val > best_v:
                best_m, best_v = m, val
        return best_m

    return choose


class _PUCTNode:
    __slots__ = ("action", "prior", "N", "W", "children", "expanded")

    def __init__(self, action=None, prior=0.0):
        self.action = action
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children = []
        self.expanded = False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


def make_mcts_chooser(model, encoder, value_model, my_idx, n_sims=400, c_puct=1.5,
                      temperature=0.3, determinize=True, dirichlet_eps=0.0,
                      dirichlet_alpha=0.3, seed=0):
    """Determinized, score-maximizing PUCT-MCTS for seat my_idx.

    Priors from the policy net (softmax of score_move), leaf value from the
    trained V (value_model), opponent modeled by net-sampling (matching the
    rollout baseline). Each simulation reshuffles the unseen deck (honest play).
    Robust to the optimizer's curse: the policy PRIOR + visit-count averaging +
    backups keep search from blindly walking into V's over-estimates.
    """
    rng = random.Random(seed * 2654435761 + my_idx * 13 + 1)
    np_rng = np.random.default_rng((seed * 40503 + my_idx) % (2 ** 32))
    roll_opp = make_net_sampling_chooser(model, encoder, temperature, seed=my_idx * 7919 + 2)

    def _opp_until_me(sim):
        guard = 0
        while not sim.is_game_over and guard < 500:
            guard += 1
            idx = sim.current_player_idx
            p = sim.current_player
            if p.action_cubes_remaining <= 0:
                if all(pl.action_cubes_remaining <= 0 for pl in sim.players):
                    sim.advance_round()
                else:
                    sim.current_player_idx = (idx + 1) % sim.num_players
                continue
            if idx == my_idx:
                return
            mvs = generate_all_moves(sim, p)
            if not mvs:
                p.action_cubes_remaining = 0; sim.advance_turn(); continue
            mv = roll_opp(sim, p, mvs)
            if execute_move_on_sim(sim, p, mv):
                sim.advance_turn(); _refill_tray(sim)
            else:
                p.action_cubes_remaining = max(0, p.action_cubes_remaining - 1); sim.advance_turn()

    def _priors_and_moves(sim):
        p = sim.players[my_idx]
        moves = generate_all_moves(sim, p)
        if not moves:
            return [], []
        state = np.asarray(encoder.encode(sim, my_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        sc = np.array([model.score_move(state, m, p, logits=logits) for m in moves], dtype=np.float64)
        e = np.exp(sc - sc.max())
        return moves, e / max(1e-12, e.sum())

    def _leaf_value(sim):
        if sim.is_game_over:
            return float(calculate_score(sim, sim.players[my_idx]).total)
        return float(value_model.predict(encoder.encode(sim, my_idx)))

    def choose(game, player, moves0):
        if len(moves0) <= 1:
            return moves0[0]
        rmoves, rpr = _priors_and_moves(game)
        if not rmoves:
            return moves0[0]
        if dirichlet_eps > 0 and len(rpr) > 1:
            noise = np_rng.dirichlet([dirichlet_alpha] * len(rpr))
            rpr = (1 - dirichlet_eps) * rpr + dirichlet_eps * noise
        root = _PUCTNode()
        root.children = [_PUCTNode(m, float(p)) for m, p in zip(rmoves, rpr)]
        root.expanded = True
        qlo, qhi = [1e18], [-1e18]

        def _norm(q):
            if qhi[0] <= qlo[0]:
                return 0.5
            return (q - qlo[0]) / (qhi[0] - qlo[0])

        for _ in range(n_sims):
            sim = fast_clone_game(game)
            if determinize:
                deck = getattr(sim, "_deck_cards", None)
                if isinstance(deck, list) and len(deck) > 1:
                    rng.shuffle(deck)
            node = root
            path = [root]
            while node.expanded and node.children and not sim.is_game_over:
                pn = node.N
                best, bestu = None, -1e18
                for c in node.children:
                    u = _norm(c.Q) + c_puct * c.prior * math.sqrt(pn + 1) / (1 + c.N)
                    if u > bestu:
                        bestu, best = u, c
                p = sim.players[my_idx]
                if p.action_cubes_remaining <= 0:
                    break
                if not execute_move_on_sim(sim, p, best.action):
                    break
                sim.advance_turn(); _refill_tray(sim)
                _opp_until_me(sim)
                node = best
                path.append(node)
            if (not node.expanded and not sim.is_game_over
                    and sim.current_player_idx == my_idx
                    and sim.players[my_idx].action_cubes_remaining > 0):
                mv, pr = _priors_and_moves(sim)
                if mv:
                    node.children = [_PUCTNode(m, float(p)) for m, p in zip(mv, pr)]
                node.expanded = True
            val = _leaf_value(sim)
            for n in path:
                n.N += 1
                n.W += val
                q = n.W / n.N
                if q < qlo[0]:
                    qlo[0] = q
                if q > qhi[0]:
                    qhi[0] = q
        return max(root.children, key=lambda c: c.N).action

    return choose


def compare_one_game(seed, model, encoder, value_model, board, cfg):
    """Play one game: search agent (leaf=cfg) vs heuristic. Returns scores + secs."""
    import time
    my_idx = seed % 2
    if cfg["leaf_mode"] == "mcts":
        ch = make_mcts_chooser(model, encoder, value_model, my_idx,
                               n_sims=cfg["n_sims"], c_puct=cfg.get("c_puct", 1.5),
                               temperature=cfg["temperature"],
                               determinize=cfg["determinize"], seed=seed)
    else:
        ch = make_eval_chooser(model, encoder, value_model, cfg["top_k"], my_idx,
                               cfg["leaf_mode"], cfg.get("bootstrap_plies", 0),
                               cfg["rollouts"], cfg["temperature"], cfg["determinize"])
    choosers = [None, None]
    choosers[my_idx] = ch
    choosers[1 - my_idx] = heuristic_chooser
    game = build_2p_game(seed, board)
    t = time.time()
    scores = play_multi(game, choosers)
    return {"seed": seed, "a_idx": my_idx, "a": scores[my_idx],
            "b": scores[1 - my_idx], "secs": round(time.time() - t, 2)}


def tiering_detail_one_game(seed, model, encoder, board, top_k=6):
    """Play one 2p selfish-search game; return per-bird rows with habitat + the
    points sitting on that bird (eggs / cached food / tucked cards) + game score."""
    from backend.ml.two_player import make_search_chooser
    net = make_net_chooser(model, encoder)
    agents = [make_search_chooser(model, encoder, top_k, i, net, "selfish")
              for i in range(2)]
    game = build_2p_game(seed, board)
    scores = play_multi(game, agents)
    rows = []
    for i, p in enumerate(game.players):
        for hab, _idx, slot in p.board.all_slots():
            if slot.bird is None:
                continue
            b = slot.bird
            rows.append({
                "seed": seed, "player": i, "score": int(scores[i]),
                "bird": b.name, "habitat": hab.value, "eggs": int(slot.eggs),
                "cached": int(slot.total_cached_food),
                "tucked": int(slot.tucked_cards), "vp": int(b.victory_points),
            })
    return rows


def gen_one_game(seed, model, encoder, board, top_k, M, temperature, determinize):
    """Play one heuristic-mode game (search agent vs heuristic) logging leaves."""
    sink: list = []
    my_idx = seed % 2
    log_ch = make_logging_search_chooser(model, encoder, top_k, my_idx, M,
                                         temperature, determinize, sink)
    choosers = [None, None]
    choosers[my_idx] = log_ch
    choosers[1 - my_idx] = heuristic_chooser
    game = build_2p_game(seed, board)
    play_multi(game, choosers)
    return sink


def _pack_rows(rows):
    """rows: list[(state_f16[D], samples_i16[M])] -> gzipped npz bytes."""
    if not rows:
        return None
    states = np.stack([r[0] for r in rows]).astype(np.float16)
    samples = np.stack([r[1] for r in rows]).astype(np.int16)
    buf = io.BytesIO()
    np.savez(buf, states=states, samples=samples)
    return gzip.compress(buf.getvalue(), 5)


def _unpack_rows(blob):
    d = np.load(io.BytesIO(gzip.decompress(blob)))
    return d["states"], d["samples"]


# --------------------------------------------------------------------------- #
# Modal dispatch (mirrors modal_two_player)
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
        .pip_install("numpy", "openpyxl", "threadpoolctl")
        .add_local_file(str(_XLSX), remote_path="/root/wingspan-20260128.xlsx")
    )
    app = modal.App("wingspan-value-gate", image=_image)

    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_value_shard_remote(task: dict) -> dict:
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
            model = FactorizedPolicyModel(tmp.name)
            encoder = StateEncoder.resolve_for_model(model.meta)
            board = BoardType(task["board_type"])
            c = task["cfg"]
            rows = []
            for s in task["seeds"]:
                rows.extend(gen_one_game(int(s), model, encoder, board,
                                         c["top_k"], c["rollouts"],
                                         c["temperature"], c["determinize"]))
        finally:
            os.unlink(tmp.name)
        return {"blob": _pack_rows(rows), "n": len(rows)}


if _MODAL_AVAILABLE:
    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_compare_shard_remote(task: dict) -> dict:
        import os, tempfile, json as _json, gzip as _gz
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(k, "1")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass
        load_all(EXCEL_FILE)
        mb = task.pop("model_bytes")
        vb = task.pop("value_bytes")
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.write(mb); tmp.close()
        try:
            model = FactorizedPolicyModel(tmp.name)
            encoder = StateEncoder.resolve_for_model(model.meta)
            vm = ValueModel(bytes(vb)) if vb else None
            board = BoardType(task["board_type"])
            cfg = task["cfg"]
            rows = [compare_one_game(int(s), model, encoder, vm, board, cfg)
                    for s in task["seeds"]]
        finally:
            os.unlink(tmp.name)
        return {"rows_gz": _gz.compress(_json.dumps(rows).encode("utf-8"), 6)}


def dispatch_compare_modal(seeds, model_path, value_path, board, cfg, seeds_per_shard):
    if not _MODAL_AVAILABLE:
        raise RuntimeError("Modal not installed")
    import json as _json
    model_bytes = Path(model_path).read_bytes()
    value_bytes = Path(value_path).read_bytes() if value_path else b""
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [{"seeds": s, "board_type": board.value, "model_bytes": model_bytes,
              "value_bytes": value_bytes, "cfg": cfg} for s in shards]
    print(f"  [modal] dispatching {len(tasks)} compare shards ({len(seeds)} games, "
          f"leaf={cfg['leaf_mode']} H={cfg.get('bootstrap_plies')}) …")
    rows = []
    with modal.enable_output(), app.run():
        for result in run_compare_shard_remote.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                rows.extend(_json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return rows


def _summarize_compare(rows, cfg):
    import statistics as st
    from math import comb
    rows = sorted(rows, key=lambda r: r["seed"])
    A = [r["a"] for r in rows]; B = [r["b"] for r in rows]
    secs = [r.get("secs", 0) for r in rows]
    n = len(A)
    wins = sum(1 for r in rows if r["a"] > r["b"])
    dec = sum(1 for r in rows if r["a"] != r["b"])
    p = sum(comb(dec, i) for i in range(wins, dec + 1)) / 2 ** dec if dec else 1.0
    ge100 = sum(1 for a in A if a >= 100)
    print("\n==================== COMPARE ====================")
    if cfg["leaf_mode"] == "mcts":
        print(f"leaf=mcts n_sims={cfg.get('n_sims')} c_puct={cfg.get('c_puct')} "
              f"temp={cfg['temperature']} det={cfg['determinize']} | n={n}")
    else:
        print(f"leaf={cfg['leaf_mode']} H={cfg.get('bootstrap_plies')} top_k={cfg['top_k']} "
              f"M={cfg['rollouts']} temp={cfg['temperature']} det={cfg['determinize']} | n={n}")
    print(f"  agent mean={st.mean(A):.1f}  floor={min(A)}  median={st.median(A):.1f}  "
          f"max={max(A)}  %>=100={100*ge100/n:.0f}%")
    print(f"  heuristic mean={st.mean(B):.1f}  | agent wins {wins}/{n} ({100*wins/n:.0f}%) p={p:.4f}")
    print(f"  compute: mean {st.mean(secs):.1f} s/game  (total {sum(secs):.0f}s)")


if _MODAL_AVAILABLE:
    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_detail_shard_remote(task: dict) -> dict:
        import os, tempfile, json as _json, gzip as _gz
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
            model = FactorizedPolicyModel(tmp.name)
            encoder = StateEncoder.resolve_for_model(model.meta)
            board = BoardType(task["board_type"])
            rows = []
            for s in task["seeds"]:
                rows.extend(tiering_detail_one_game(int(s), model, encoder, board, task.get("top_k", 6)))
        finally:
            os.unlink(tmp.name)
        return {"rows_gz": _gz.compress(_json.dumps(rows).encode("utf-8"), 6)}


def dispatch_detail_modal(seeds, model_path, board, top_k, seeds_per_shard):
    import json as _json
    model_bytes = Path(model_path).read_bytes()
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [{"seeds": s, "board_type": board.value, "model_bytes": model_bytes,
              "top_k": top_k} for s in shards]
    print(f"  [modal] dispatching {len(tasks)} detail shards ({len(seeds)} games) …")
    rows = []
    with modal.enable_output(), app.run():
        for result in run_detail_shard_remote.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                rows.extend(_json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return rows


def dispatch_gen_modal(seeds, model_path, board, cfg, seeds_per_shard):
    if not _MODAL_AVAILABLE:
        raise RuntimeError("Modal not installed")
    model_bytes = Path(model_path).read_bytes()
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [{"seeds": s, "board_type": board.value,
              "model_bytes": model_bytes, "cfg": cfg} for s in shards]
    print(f"  [modal] dispatching {len(tasks)} value-gen shards ({len(seeds)} games) …")
    states_all, samples_all = [], []
    with modal.enable_output(), app.run():
        for result in run_value_shard_remote.map(tasks):
            blob = result.get("blob")
            if blob:
                st, sm = _unpack_rows(bytes(blob))
                states_all.append(st); samples_all.append(sm)
    if not states_all:
        return np.empty((0, 0), np.float16), np.empty((0, 0), np.int16)
    return np.concatenate(states_all), np.concatenate(samples_all)


# --------------------------------------------------------------------------- #
# Gate: train V, compare V_rmse to one-playout sigma
# --------------------------------------------------------------------------- #
def _ridge_fit(X, y, lam):
    # closed-form ridge on standardized features
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(A, X.T @ y)
    return w


def run_gate(states, samples, score_scale=120.0, seed=0):
    rng = np.random.default_rng(seed)
    n, D = states.shape
    M = samples.shape[1]
    X = states.astype(np.float64)
    mean = samples.mean(axis=1)                      # per-state label (truth est.)
    var = samples.var(axis=1, ddof=1)                # per-state playout variance
    sigma_bar = float(np.sqrt(var.mean()))           # one-playout RMSE vs truth
    label_noise_var = float((var / M).mean())        # noise in mean() as truth est.

    # train/test split
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    # standardize on train
    mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    Xtr = np.hstack([Xs[tr], np.ones((len(tr), 1))])
    Xte = np.hstack([Xs[te], np.ones((len(te), 1))])
    ytr, yte = mean[tr], mean[te]

    results = {}
    # 1) trivial baseline: predict global mean
    base_rmse = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    results["predict_mean_rmse"] = base_rmse

    # 2) ridge linear V
    best = None
    for lam in (1.0, 10.0, 100.0, 1000.0):
        w = _ridge_fit(Xtr, ytr, lam)
        pred = Xte @ w
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        if best is None or rmse < best[1]:
            best = (lam, rmse)
    results["ridge_lam"], results["ridge_rmse_raw"] = best

    # 3) torch MLP V (if available)
    mlp_rmse = None
    try:
        import torch
        torch.manual_seed(seed)
        dev = "cpu"
        net = torch.nn.Sequential(
            torch.nn.Linear(D, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = torch.nn.MSELoss()
        Xt = torch.tensor((Xs[tr]).astype(np.float32))
        yt = torch.tensor((ytr / score_scale).astype(np.float32)).unsqueeze(1)
        Xv = torch.tensor((Xs[te]).astype(np.float32))
        bs = 512
        for ep in range(40):
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(Xt[b]), yt[b])
                loss.backward(); opt.step()
        with torch.no_grad():
            pv = net(Xv).squeeze(1).numpy() * score_scale
        mlp_rmse = float(np.sqrt(np.mean((pv - yte) ** 2)))
    except ImportError:
        pass
    results["mlp_rmse_raw"] = mlp_rmse

    # bias-correct V_rmse for label-mean noise (sigma^2/M), pick best available V
    v_raw = mlp_rmse if mlp_rmse is not None else results["ridge_rmse_raw"]
    v_corr = float(np.sqrt(max(1e-9, v_raw ** 2 - label_noise_var)))
    results["sigma_bar_one_playout"] = sigma_bar
    results["label_noise_sd"] = float(np.sqrt(label_noise_var))
    results["V_rmse_corrected"] = v_corr
    results["R_star_playouts_V_is_worth"] = float((sigma_bar / v_corr) ** 2) if v_corr > 0 else float("inf")
    results["n_states"], results["M"], results["state_dim"] = int(n), int(M), int(D)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--seeds", default="0-19")
    g.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    g.add_argument("--board", choices=["oceania", "base"], default="oceania")
    g.add_argument("--top-k", type=int, default=10)
    g.add_argument("--rollouts", type=int, default=16, help="M playouts/candidate (label samples)")
    g.add_argument("--temperature", type=float, default=0.3)
    g.add_argument("--determinize", action="store_true")
    g.add_argument("--use-modal", action="store_true")
    g.add_argument("--seeds-per-shard", type=int, default=2)
    g.add_argument("--out", default="reports/ml/value_gate/data.npz")

    ga = sub.add_parser("gate")
    ga.add_argument("--data", default="reports/ml/value_gate/data.npz")

    tr = sub.add_parser("train")
    tr.add_argument("--data", default="reports/ml/value_gate/data.npz")
    tr.add_argument("--out", default="reports/ml/value_gate/value_v1.npz")
    tr.add_argument("--epochs", type=int, default=60)

    cp = sub.add_parser("compare")
    cp.add_argument("--seeds", default="0-99")
    cp.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    cp.add_argument("--value", default="reports/ml/value_gate/value_v1.npz")
    cp.add_argument("--board", choices=["oceania", "base"], default="oceania")
    cp.add_argument("--leaf-mode", choices=["v", "full", "mcts"], default="v")
    cp.add_argument("--bootstrap-plies", type=int, default=8)
    cp.add_argument("--top-k", type=int, default=10)
    cp.add_argument("--rollouts", type=int, default=3, help="M leaf evals/candidate")
    cp.add_argument("--n-sims", type=int, default=400, help="MCTS simulations/decision")
    cp.add_argument("--c-puct", type=float, default=1.5)
    cp.add_argument("--temperature", type=float, default=0.3)
    cp.add_argument("--determinize", action="store_true")
    cp.add_argument("--use-modal", action="store_true")
    cp.add_argument("--seeds-per-shard", type=int, default=2)

    args = ap.parse_args()
    if args.cmd == "train":
        train_value(args.data, args.out, epochs=args.epochs)
        return
    if args.cmd == "compare":
        board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
        seeds = _parse_seeds(args.seeds)
        cfg = {"leaf_mode": args.leaf_mode, "bootstrap_plies": args.bootstrap_plies,
               "top_k": args.top_k, "rollouts": args.rollouts,
               "n_sims": args.n_sims, "c_puct": args.c_puct,
               "temperature": args.temperature, "determinize": args.determinize}
        value_path = args.value if args.leaf_mode in ("v", "mcts") else None
        if args.use_modal:
            rows = dispatch_compare_modal(seeds, args.model, value_path, board, cfg, args.seeds_per_shard)
        else:
            load_all(EXCEL_FILE)
            model = FactorizedPolicyModel(args.model)
            encoder = StateEncoder.resolve_for_model(model.meta)
            vm = ValueModel(value_path) if value_path else None
            rows = [compare_one_game(s, model, encoder, vm, board, cfg) for s in seeds]
        for r in sorted(rows, key=lambda r: r["seed"]):
            tag = "AGENT" if r["a"] > r["b"] else ("heur" if r["a"] < r["b"] else "tie")
            print(f"seed {r['seed']:>3} (agent@{r['a_idx']}): agent={r['a']:>3} heur={r['b']:>3} {tag} ({r.get('secs','?')}s)")
        _summarize_compare(rows, cfg)
        return
    if args.cmd == "gen":
        board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
        seeds = _parse_seeds(args.seeds)
        cfg = {"top_k": args.top_k, "rollouts": args.rollouts,
               "temperature": args.temperature, "determinize": args.determinize}
        if args.use_modal:
            states, samples = dispatch_gen_modal(seeds, args.model, board, cfg, args.seeds_per_shard)
        else:
            load_all(EXCEL_FILE)
            model = FactorizedPolicyModel(args.model)
            encoder = StateEncoder.resolve_for_model(model.meta)
            rows = []
            for s in seeds:
                rows.extend(gen_one_game(s, model, encoder, board, args.top_k,
                                         args.rollouts, args.temperature, args.determinize))
            states, samples = _unpack_rows(_pack_rows(rows)) if rows else (
                np.empty((0, 0), np.float16), np.empty((0, 0), np.int16))
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, states=states, samples=samples)
        print(f"  saved {states.shape[0]} states (dim={states.shape[1] if states.size else 0}, "
              f"M={samples.shape[1] if samples.size else 0}) -> {out}")
    elif args.cmd == "gate":
        d = np.load(args.data)
        res = run_gate(d["states"], d["samples"])
        print("\n==================== VALUE GATE ====================")
        for k, v in res.items():
            print(f"  {k:32}: {v}")
        rstar = res["R_star_playouts_V_is_worth"]
        verdict = "GREEN -- V beats one playout" if rstar > 1 else "RED -- V worse than one playout"
        print(f"\n  GATE: R*={rstar:.2f}  ->  {verdict}")
        if rstar > 1:
            print(f"  V is worth ~{rstar:.1f} playouts; current leaf uses ~rollouts playouts.")


if __name__ == "__main__":
    main()
