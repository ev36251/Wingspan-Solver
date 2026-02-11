"""Train a factorized behavioral-cloning policy network (NumPy).

Supports an auxiliary value head when `value_target` is present in data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from backend.ml.factorized_policy import RELEVANT_HEADS_BY_ACTION
from backend.ml.move_features import MOVE_FEATURE_DIM


HEAD_NAMES = [
    "action_type",
    "play_habitat",
    "gain_food_primary",
    "draw_mode",
    "lay_eggs_bin",
    "play_cost_bin",
    "play_power_color",
]


class BCModel:
    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        head_dims: dict[str, int],
        has_value_head: bool,
        has_move_value_head: bool,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / feature_dim), size=(feature_dim, hidden1)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.head_W: dict[str, np.ndarray] = {}
        self.head_b: dict[str, np.ndarray] = {}
        for hn, d in head_dims.items():
            self.head_W[hn] = rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2, d)).astype(np.float32)
            self.head_b[hn] = np.zeros(d, dtype=np.float32)

        self.has_value_head = has_value_head
        self.W_value = (
            rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2,)).astype(np.float32)
            if has_value_head
            else None
        )
        self.b_value = np.float32(0.0)
        self.has_move_value_head = has_move_value_head
        self.W_move_value = (
            rng.normal(0.0, np.sqrt(2.0 / (feature_dim + MOVE_FEATURE_DIM)), size=(feature_dim + MOVE_FEATURE_DIM,)).astype(np.float32)
            if has_move_value_head
            else None
        )
        self.b_move_value = np.float32(0.0)
        self.hidden1 = hidden1
        self.hidden2 = hidden2

    def forward(
        self,
        x: np.ndarray,
        *,
        train: bool = False,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        h1_pre = x @ self.W1 + self.b1
        h1 = np.maximum(h1_pre, 0.0)

        keep = 1.0 - float(dropout)
        if train and dropout > 0.0 and rng is not None and keep > 0.0:
            mask1 = (rng.random(h1.shape, dtype=np.float32) < keep).astype(np.float32) / np.float32(keep)
            h1_drop = h1 * mask1
        else:
            mask1 = np.ones_like(h1, dtype=np.float32)
            h1_drop = h1

        h2_pre = h1_drop @ self.W2 + self.b2
        h2 = np.maximum(h2_pre, 0.0)
        if train and dropout > 0.0 and rng is not None and keep > 0.0:
            mask2 = (rng.random(h2.shape, dtype=np.float32) < keep).astype(np.float32) / np.float32(keep)
            h2_drop = h2 * mask2
        else:
            mask2 = np.ones_like(h2, dtype=np.float32)
            h2_drop = h2

        logits = {hn: h2_drop @ self.head_W[hn] + self.head_b[hn] for hn in self.head_W}
        value = None
        if self.has_value_head:
            value = float(h2_drop @ self.W_value + self.b_value)
        return h1_pre, h1, mask1, h1_drop, h2_pre, h2, mask2, h2_drop, logits, value

    def save(self, path: str | Path, metadata: dict) -> None:
        out = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "metadata_json": np.asarray([json.dumps(metadata)], dtype=object),
        }
        for hn in self.head_W:
            out[f"W_{hn}"] = self.head_W[hn]
            out[f"b_{hn}"] = self.head_b[hn]
        if self.has_value_head and self.W_value is not None:
            out["W_value"] = self.W_value
            out["b_value"] = np.asarray([self.b_value], dtype=np.float32)
        if self.has_move_value_head and self.W_move_value is not None:
            out["W_move_value"] = self.W_move_value
            out["b_move_value"] = np.asarray([self.b_move_value], dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)


def _softmax(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    if s <= 0:
        return np.full_like(ex, 1.0 / len(ex))
    return ex / s


def _load_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lr_for_epoch(
    ep: int,
    *,
    lr_init: float,
    lr_peak: float,
    lr_warmup_epochs: int,
    lr_decay_every: int,
    lr_decay_factor: float,
) -> float:
    warm = max(0, int(lr_warmup_epochs))
    if warm > 0 and ep <= warm:
        if warm == 1:
            return float(lr_peak)
        t = float(ep - 1) / float(warm - 1)
        return float(lr_init + t * (lr_peak - lr_init))
    if lr_decay_every <= 0:
        return float(lr_peak)
    k = max(0, (ep - warm - 1) // lr_decay_every)
    return float(lr_peak * (lr_decay_factor ** k))


def train_bc(
    dataset_jsonl: str,
    meta_json: str,
    out_model: str,
    epochs: int,
    batch_size: int,
    hidden1: int = 256,
    hidden2: int = 128,
    dropout: float = 0.15,
    lr_init: float = 1e-4,
    lr_peak: float = 1e-3,
    lr_warmup_epochs: int = 2,
    lr_decay_every: int = 3,
    lr_decay_factor: float = 0.5,
    val_split: float = 0.1,
    seed: int = 0,
    value_loss_weight: float = 0.5,
    move_value_enabled: bool = True,
    move_value_loss_weight: float = 0.2,
    move_value_num_negatives: int = 4,
    early_stop_enabled: bool = True,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 1e-4,
    early_stop_restore_best: bool = True,
    hidden: int | None = None,
    lr: float | None = None,
) -> dict:
    started = time.time()
    rnd = random.Random(seed)
    np_rng = np.random.default_rng(seed + 17)
    deprecated_args_used: list[str] = []

    if hidden is not None:
        hidden1 = int(hidden)
        hidden2 = int(hidden)
        deprecated_args_used.append("hidden")
    if lr is not None:
        lr_peak = float(lr)
        deprecated_args_used.append("lr")
    if deprecated_args_used:
        print(f"warning: deprecated args used: {', '.join(deprecated_args_used)}")

    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    feature_dim = int(meta["feature_dim"])
    head_dims = meta["target_heads"]

    rows = _load_rows(dataset_jsonl)
    if not rows:
        raise ValueError("Empty BC dataset")

    has_value_target = any(("value_target_score" in r) or ("value_target" in r) for r in rows)
    has_move_value_data = any(("move_pos" in r) and ("move_negs" in r) for r in rows)
    move_value_num_negatives = max(0, int(move_value_num_negatives))
    move_value_loss_weight = max(0.0, float(move_value_loss_weight))
    effective_move_value = bool(move_value_enabled and has_move_value_data)
    value_target_scale = float(meta.get("value_target_config", {}).get("score_scale", 150.0))
    value_target_bias = float(meta.get("value_target_config", {}).get("score_bias", 0.0))

    def target_score(row: dict) -> float:
        if "value_target_score" in row:
            return float(row["value_target_score"])
        if "value_target" in row:
            return float(value_target_bias + value_target_scale * float(row["value_target"]))
        return 0.0

    rnd.shuffle(rows)
    val_n = max(1, int(len(rows) * val_split)) if len(rows) >= 10 else 1
    val = rows[:val_n]
    train_rows = rows[val_n:] if len(rows) > val_n else rows

    model = BCModel(
        feature_dim,
        hidden1,
        hidden2,
        head_dims,
        has_value_target,
        effective_move_value,
        seed,
    )
    early_stop_patience = max(1, int(early_stop_patience))
    early_stop_min_delta = max(0.0, float(early_stop_min_delta))

    def _snapshot_model() -> dict:
        snap = {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy(),
            "head_W": {k: v.copy() for k, v in model.head_W.items()},
            "head_b": {k: v.copy() for k, v in model.head_b.items()},
            "b_value": np.float32(model.b_value),
            "b_move_value": np.float32(model.b_move_value),
        }
        if model.has_value_head and model.W_value is not None:
            snap["W_value"] = model.W_value.copy()
        if model.has_move_value_head and model.W_move_value is not None:
            snap["W_move_value"] = model.W_move_value.copy()
        return snap

    def _restore_model(snap: dict) -> None:
        model.W1[...] = snap["W1"]
        model.b1[...] = snap["b1"]
        model.W2[...] = snap["W2"]
        model.b2[...] = snap["b2"]
        for k in model.head_W:
            model.head_W[k][...] = snap["head_W"][k]
            model.head_b[k][...] = snap["head_b"][k]
        model.b_value = np.float32(snap["b_value"])
        if model.has_value_head and model.W_value is not None and "W_value" in snap:
            model.W_value[...] = snap["W_value"]
        model.b_move_value = np.float32(snap["b_move_value"])
        if model.has_move_value_head and model.W_move_value is not None and "W_move_value" in snap:
            model.W_move_value[...] = snap["W_move_value"]

    def eval_split(split_rows: list[dict]) -> dict:
        total_loss = 0.0
        total = 0
        action_correct = 0
        value_mse = 0.0
        value_n = 0
        move_rank_loss = 0.0
        move_rank_n = 0
        move_margin_sum = 0.0
        move_pair_correct = 0
        move_pair_total = 0

        for r in split_rows:
            x = np.asarray(r["state"], dtype=np.float32)
            t = r["targets"]
            _, _, _, _, _, _, _, _, logits, value_pred = model.forward(x, train=False)

            # Always action_type
            p_action = _softmax(logits["action_type"])
            ta = int(t["action_type"])
            total_loss += -math.log(max(1e-9, float(p_action[ta])))
            if int(np.argmax(p_action)) == ta:
                action_correct += 1
            total += 1

            for hn in RELEVANT_HEADS_BY_ACTION.get(ta, []):
                p = _softmax(logits[hn])
                tv = int(t[hn])
                total_loss += -math.log(max(1e-9, float(p[tv])))

            if model.has_value_head and value_pred is not None and (("value_target_score" in r) or ("value_target" in r)):
                dv = float(value_pred) - target_score(r)
                value_mse += dv * dv
                value_n += 1

            if (
                model.has_move_value_head
                and model.W_move_value is not None
                and "move_pos" in r
                and "move_negs" in r
                and r["move_negs"]
            ):
                pos = np.asarray(r["move_pos"], dtype=np.float32)
                negs = [np.asarray(nf, dtype=np.float32) for nf in r["move_negs"]]
                x_pos = np.concatenate([x, pos], axis=0)
                s_pos = float(x_pos @ model.W_move_value + model.b_move_value)
                s_negs = []
                for nf in negs:
                    x_neg = np.concatenate([x, nf], axis=0)
                    s_neg = float(x_neg @ model.W_move_value + model.b_move_value)
                    s_negs.append(s_neg)
                    delta = s_pos - s_neg
                    move_rank_loss += float(np.logaddexp(0.0, -delta))
                    move_rank_n += 1
                    move_pair_total += 1
                    if s_pos > s_neg:
                        move_pair_correct += 1
                move_margin_sum += s_pos - (sum(s_negs) / max(1, len(s_negs)))

        cls_loss = total_loss / max(1, total)
        v_mse = value_mse / max(1, value_n) if value_n > 0 else 0.0
        mv_loss = move_rank_loss / max(1, move_rank_n) if move_rank_n > 0 else 0.0
        return {
            "loss": cls_loss + (value_loss_weight * v_mse) + (move_value_loss_weight * mv_loss),
            "action_acc": action_correct / max(1, total),
            "value_mse": v_mse,
            "move_rank_loss": mv_loss,
            "move_rank_margin_mean": move_margin_sum / max(1, total),
            "move_pair_acc": move_pair_correct / max(1, move_pair_total) if move_pair_total > 0 else 0.0,
        }

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    stopped_early = False
    best_snapshot: dict | None = None
    epochs_completed = 0

    for ep in range(1, epochs + 1):
        lr_epoch = _lr_for_epoch(
            ep,
            lr_init=lr_init,
            lr_peak=lr_peak,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_decay_every=lr_decay_every,
            lr_decay_factor=lr_decay_factor,
        )
        rnd.shuffle(train_rows)
        seen = 0
        loss_sum = 0.0
        action_correct = 0
        value_mse_sum = 0.0
        value_seen = 0
        move_rank_loss_sum = 0.0
        move_rank_seen = 0
        move_margin_sum = 0.0
        move_pair_correct = 0
        move_pair_total = 0

        for i in range(0, len(train_rows), batch_size):
            batch = train_rows[i:i + batch_size]
            if not batch:
                continue

            dW1 = np.zeros_like(model.W1)
            db1 = np.zeros_like(model.b1)
            dW2 = np.zeros_like(model.W2)
            db2 = np.zeros_like(model.b2)
            dWh = {hn: np.zeros_like(model.head_W[hn]) for hn in HEAD_NAMES}
            dbh = {hn: np.zeros_like(model.head_b[hn]) for hn in HEAD_NAMES}
            dWv = np.zeros_like(model.W_value) if model.has_value_head and model.W_value is not None else None
            dbv = 0.0
            dWmv = np.zeros_like(model.W_move_value) if model.has_move_value_head and model.W_move_value is not None else None
            dbmv = 0.0

            for r in batch:
                x = np.asarray(r["state"], dtype=np.float32)
                t = r["targets"]

                h1_pre, h1, mask1, h1_drop, h2_pre, h2, mask2, h2_drop, logits, value_pred = model.forward(
                    x,
                    train=True,
                    dropout=dropout,
                    rng=np_rng,
                )
                ta = int(t["action_type"])

                dh2_drop = np.zeros_like(h2_drop)

                # Action head (always)
                pa = _softmax(logits["action_type"])
                loss_sum += -math.log(max(1e-9, float(pa[ta])))
                if int(np.argmax(pa)) == ta:
                    action_correct += 1

                da = pa.copy()
                da[ta] -= 1.0
                dWh["action_type"] += np.outer(h2_drop, da)
                dbh["action_type"] += da
                dh2_drop += model.head_W["action_type"] @ da

                # Relevant sub-heads for this action type
                for hn in RELEVANT_HEADS_BY_ACTION.get(ta, []):
                    tv = int(t[hn])
                    p = _softmax(logits[hn])
                    loss_sum += -math.log(max(1e-9, float(p[tv])))
                    d = p.copy()
                    d[tv] -= 1.0
                    dWh[hn] += np.outer(h2_drop, d)
                    dbh[hn] += d
                    dh2_drop += model.head_W[hn] @ d

                # Optional value head regression
                if (
                    model.has_value_head
                    and dWv is not None
                    and value_pred is not None
                    and (("value_target_score" in r) or ("value_target" in r))
                ):
                    vt = target_score(r)
                    dv = float(value_pred) - vt
                    value_mse_sum += dv * dv
                    value_seen += 1

                    dvr = value_loss_weight * 2.0 * dv
                    dWv += h2_drop * dvr
                    dbv += dvr
                    dh2_drop += model.W_value * dvr

                # Optional move-value ranking head on concat(state, move_features).
                if (
                    model.has_move_value_head
                    and dWmv is not None
                    and "move_pos" in r
                    and "move_negs" in r
                    and r["move_negs"]
                ):
                    pos = np.asarray(r["move_pos"], dtype=np.float32)
                    negs = [np.asarray(nf, dtype=np.float32) for nf in r["move_negs"]]
                    x_pos = np.concatenate([x, pos], axis=0)
                    s_pos = float(x_pos @ model.W_move_value + model.b_move_value)
                    s_negs: list[float] = []
                    grad_s_pos = 0.0
                    grad_s_neg: list[float] = []
                    for nf in negs:
                        x_neg = np.concatenate([x, nf], axis=0)
                        s_neg = float(x_neg @ model.W_move_value + model.b_move_value)
                        s_negs.append(s_neg)
                        delta = s_pos - s_neg
                        move_rank_loss_sum += float(np.logaddexp(0.0, -delta))
                        move_rank_seen += 1
                        move_pair_total += 1
                        if s_pos > s_neg:
                            move_pair_correct += 1
                        g = -1.0 / (1.0 + math.exp(delta))  # d/d(delta) softplus(-delta)
                        grad_s_pos += g
                        grad_s_neg.append(-g)

                    move_margin_sum += s_pos - (sum(s_negs) / max(1, len(s_negs)))
                    nneg = max(1, len(negs))
                    grad_s_pos = move_value_loss_weight * (grad_s_pos / nneg)
                    dWmv += grad_s_pos * x_pos
                    dbmv += grad_s_pos
                    for nf, gs in zip(negs, grad_s_neg):
                        x_neg = np.concatenate([x, nf], axis=0)
                        gs_scaled = move_value_loss_weight * (gs / nneg)
                        dWmv += gs_scaled * x_neg
                        dbmv += gs_scaled

                dh2 = dh2_drop * mask2
                dh2[h2_pre <= 0.0] = 0.0

                dW2 += np.outer(h1_drop, dh2)
                db2 += dh2

                dh1_drop = model.W2 @ dh2
                dh1 = dh1_drop * mask1
                dh1[h1_pre <= 0.0] = 0.0

                dW1 += np.outer(x, dh1)
                db1 += dh1
                seen += 1

            scale = 1.0 / max(1, len(batch))
            model.W1 -= lr_epoch * dW1 * scale
            model.b1 -= lr_epoch * db1 * scale
            model.W2 -= lr_epoch * dW2 * scale
            model.b2 -= lr_epoch * db2 * scale
            for hn in HEAD_NAMES:
                model.head_W[hn] -= lr_epoch * dWh[hn] * scale
                model.head_b[hn] -= lr_epoch * dbh[hn] * scale
            if model.has_value_head and dWv is not None and model.W_value is not None:
                model.W_value -= lr_epoch * dWv * scale
                model.b_value = np.float32(float(model.b_value - lr_epoch * dbv * scale))
            if model.has_move_value_head and dWmv is not None and model.W_move_value is not None:
                model.W_move_value -= lr_epoch * dWmv * scale
                model.b_move_value = np.float32(float(model.b_move_value - lr_epoch * dbmv * scale))

        train_cls_loss = loss_sum / max(1, seen)
        train_value_mse = value_mse_sum / max(1, value_seen) if value_seen > 0 else 0.0
        train_move_rank_loss = move_rank_loss_sum / max(1, move_rank_seen) if move_rank_seen > 0 else 0.0
        train_metrics = {
            "loss": train_cls_loss + (value_loss_weight * train_value_mse) + (move_value_loss_weight * train_move_rank_loss),
            "action_acc": action_correct / max(1, seen),
            "value_mse": train_value_mse,
            "move_rank_loss": train_move_rank_loss,
            "move_rank_margin_mean": move_margin_sum / max(1, seen),
            "move_pair_acc": move_pair_correct / max(1, move_pair_total) if move_pair_total > 0 else 0.0,
        }
        val_metrics = eval_split(val)

        row = {
            "epoch": ep,
            "lr_epoch": round(lr_epoch, 8),
            "train_loss": round(train_metrics["loss"], 6),
            "train_action_acc": round(train_metrics["action_acc"], 6),
            "train_value_mse": round(train_metrics["value_mse"], 6),
            "train_move_rank_loss": round(train_metrics["move_rank_loss"], 6),
            "train_move_rank_margin_mean": round(train_metrics["move_rank_margin_mean"], 6),
            "train_move_pair_acc": round(train_metrics["move_pair_acc"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_action_acc": round(val_metrics["action_acc"], 6),
            "val_value_mse": round(val_metrics["value_mse"], 6),
            "val_move_rank_loss": round(val_metrics["move_rank_loss"], 6),
            "val_move_rank_margin_mean": round(val_metrics["move_rank_margin_mean"], 6),
            "val_move_pair_acc": round(val_metrics["move_pair_acc"], 6),
            "is_best_epoch": False,
        }
        current_val_loss = float(val_metrics["loss"])
        if best_val_loss - current_val_loss >= early_stop_min_delta:
            best_val_loss = current_val_loss
            best_epoch = ep
            no_improve_epochs = 0
            row["is_best_epoch"] = True
            if early_stop_restore_best:
                best_snapshot = _snapshot_model()
        else:
            no_improve_epochs += 1

        history.append(row)
        epochs_completed = ep
        print(
            f"epoch {ep}/{epochs} | lr={row['lr_epoch']:.6f} | train_loss={row['train_loss']:.4f} train_acc={row['train_action_acc']:.3f} "
            f"train_v={row['train_value_mse']:.4f} train_mv={row['train_move_rank_loss']:.4f} | val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_action_acc']:.3f} val_v={row['val_value_mse']:.4f} val_mv={row['val_move_rank_loss']:.4f}"
        )
        if early_stop_enabled and no_improve_epochs >= early_stop_patience:
            stopped_early = True
            break

    if early_stop_enabled and early_stop_restore_best and best_snapshot is not None:
        _restore_model(best_snapshot)

    result = {
        "version": 3,
        "format_version": 3,
        "mode": "factorized_behavioral_cloning",
        "model_arch": "mlp_2layer",
        "elapsed_sec": round(time.time() - started, 3),
        "dataset": dataset_jsonl,
        "meta": meta_json,
        "feature_dim": feature_dim,
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "dropout": float(dropout),
        "lr_init": float(lr_init),
        "lr_peak": float(lr_peak),
        "lr_warmup_epochs": int(lr_warmup_epochs),
        "lr_decay_every": int(lr_decay_every),
        "lr_decay_factor": float(lr_decay_factor),
        "early_stop_enabled": bool(early_stop_enabled),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_restore_best": bool(early_stop_restore_best),
        "stopped_early": bool(stopped_early),
        "best_val_loss_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss if best_epoch > 0 else history[-1]["val_loss"] if history else 0.0),
        "epochs_completed": int(epochs_completed),
        "deprecated_args_used": deprecated_args_used,
        "head_dims": head_dims,
        "has_value_head": has_value_target,
        "value_prediction_mode": "score_linear" if has_value_target else "none",
        "value_score_scale": value_target_scale,
        "value_score_bias": value_target_bias,
        "has_move_value_head": bool(model.has_move_value_head),
        "move_feature_dim": MOVE_FEATURE_DIM,
        "move_value_loss_weight": float(move_value_loss_weight),
        "move_value_training": {
            "num_negatives": int(move_value_num_negatives),
            "sampling": "topk_then_random",
        },
        "train_samples": len(train_rows),
        "val_samples": len(val),
        "history": history,
    }
    model.save(out_model, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train factorized BC policy")
    parser.add_argument("--dataset", default="reports/ml/bc_dataset.jsonl")
    parser.add_argument("--meta", default="reports/ml/bc_dataset.meta.json")
    parser.add_argument("--out", default="reports/ml/factorized_bc_model.npz")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr-init", type=float, default=1e-4)
    parser.add_argument("--lr-peak", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-epochs", type=int, default=2)
    parser.add_argument("--lr-decay-every", type=int, default=3)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.set_defaults(early_stop_enabled=True, early_stop_restore_best=True)
    parser.add_argument("--early-stop-enabled", dest="early_stop_enabled", action="store_true")
    parser.add_argument("--disable-early-stop", dest="early_stop_enabled", action="store_false")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-restore-best", dest="early_stop_restore_best", action="store_true")
    parser.add_argument("--no-early-stop-restore-best", dest="early_stop_restore_best", action="store_false")
    parser.add_argument("--hidden", type=int, default=None, help="Deprecated: maps hidden1=hidden2=hidden")
    parser.add_argument("--lr", type=float, default=None, help="Deprecated: maps lr_peak=lr")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.set_defaults(move_value_enabled=True)
    parser.add_argument("--move-value-enabled", dest="move_value_enabled", action="store_true")
    parser.add_argument("--disable-move-value", dest="move_value_enabled", action="store_false")
    parser.add_argument("--move-value-loss-weight", type=float, default=0.2)
    parser.add_argument("--move-value-num-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train_bc(
        dataset_jsonl=args.dataset,
        meta_json=args.meta,
        out_model=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        lr_init=args.lr_init,
        lr_peak=args.lr_peak,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        early_stop_enabled=args.early_stop_enabled,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_restore_best=args.early_stop_restore_best,
        val_split=args.val_split,
        seed=args.seed,
        value_loss_weight=args.value_weight,
        move_value_enabled=args.move_value_enabled,
        move_value_loss_weight=args.move_value_loss_weight,
        move_value_num_negatives=args.move_value_num_negatives,
        hidden=args.hidden,
        lr=args.lr,
    )
    print(
        f"factorized BC complete | model={args.out} | "
        f"train={out['train_samples']} val={out['val_samples']}"
    )


if __name__ == "__main__":
    main()
