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
import random
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, fast_clone_game, _refill_tray
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.solo_eval import make_net_chooser, make_net_sampling_chooser
from backend.ml.two_player import build_2p_game, play_multi, heuristic_chooser

_HERE = Path(__file__).parent
_XLSX = _HERE.parent.parent / "wingspan-20260128.xlsx"


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

    args = ap.parse_args()
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
