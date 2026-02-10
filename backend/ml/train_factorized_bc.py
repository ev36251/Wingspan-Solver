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
    def __init__(self, feature_dim: int, hidden: int, head_dims: dict[str, int], has_value_head: bool, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / feature_dim), size=(feature_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.head_W: dict[str, np.ndarray] = {}
        self.head_b: dict[str, np.ndarray] = {}
        for hn, d in head_dims.items():
            self.head_W[hn] = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, d)).astype(np.float32)
            self.head_b[hn] = np.zeros(d, dtype=np.float32)

        self.has_value_head = has_value_head
        self.W_value = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden,)).astype(np.float32) if has_value_head else None
        self.b_value = np.float32(0.0)

    def forward(self, x: np.ndarray):
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = {hn: h @ self.head_W[hn] + self.head_b[hn] for hn in self.head_W}
        value = None
        if self.has_value_head:
            vr = float(h @ self.W_value + self.b_value)
            value = 1.0 / (1.0 + math.exp(-vr))
        return h_pre, h, logits, value

    def save(self, path: str | Path, metadata: dict) -> None:
        out = {"W1": self.W1, "b1": self.b1, "metadata_json": np.asarray([json.dumps(metadata)], dtype=object)}
        for hn in self.head_W:
            out[f"W_{hn}"] = self.head_W[hn]
            out[f"b_{hn}"] = self.head_b[hn]
        if self.has_value_head and self.W_value is not None:
            out["W_value"] = self.W_value
            out["b_value"] = np.asarray([self.b_value], dtype=np.float32)
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


def train_bc(
    dataset_jsonl: str,
    meta_json: str,
    out_model: str,
    epochs: int,
    batch_size: int,
    hidden: int,
    lr: float,
    val_split: float,
    seed: int,
    value_loss_weight: float = 0.5,
) -> dict:
    started = time.time()
    rnd = random.Random(seed)

    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    feature_dim = int(meta["feature_dim"])
    head_dims = meta["target_heads"]

    rows = _load_rows(dataset_jsonl)
    if not rows:
        raise ValueError("Empty BC dataset")

    has_value_target = any("value_target" in r for r in rows)

    rnd.shuffle(rows)
    val_n = max(1, int(len(rows) * val_split)) if len(rows) >= 10 else 1
    val = rows[:val_n]
    train_rows = rows[val_n:] if len(rows) > val_n else rows

    model = BCModel(feature_dim, hidden, head_dims, has_value_target, seed)

    def eval_split(split_rows: list[dict]) -> dict:
        total_loss = 0.0
        total = 0
        action_correct = 0
        value_mse = 0.0
        value_n = 0

        for r in split_rows:
            x = np.asarray(r["state"], dtype=np.float32)
            t = r["targets"]
            _, _, logits, value_pred = model.forward(x)

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

            if model.has_value_head and "value_target" in r and value_pred is not None:
                dv = float(value_pred) - float(r["value_target"])
                value_mse += dv * dv
                value_n += 1

        return {
            "loss": total_loss / max(1, total),
            "action_acc": action_correct / max(1, total),
            "value_mse": value_mse / max(1, value_n) if value_n > 0 else 0.0,
        }

    history: list[dict] = []

    for ep in range(1, epochs + 1):
        rnd.shuffle(train_rows)
        seen = 0
        loss_sum = 0.0
        action_correct = 0
        value_mse_sum = 0.0
        value_seen = 0

        for i in range(0, len(train_rows), batch_size):
            batch = train_rows[i:i + batch_size]
            if not batch:
                continue

            dW1 = np.zeros_like(model.W1)
            db1 = np.zeros_like(model.b1)
            dWh = {hn: np.zeros_like(model.head_W[hn]) for hn in HEAD_NAMES}
            dbh = {hn: np.zeros_like(model.head_b[hn]) for hn in HEAD_NAMES}
            dWv = np.zeros_like(model.W_value) if model.has_value_head and model.W_value is not None else None
            dbv = 0.0

            for r in batch:
                x = np.asarray(r["state"], dtype=np.float32)
                t = r["targets"]

                h_pre, h, logits, value_pred = model.forward(x)
                ta = int(t["action_type"])

                dh = np.zeros_like(h)

                # Action head (always)
                pa = _softmax(logits["action_type"])
                loss_sum += -math.log(max(1e-9, float(pa[ta])))
                if int(np.argmax(pa)) == ta:
                    action_correct += 1

                da = pa.copy()
                da[ta] -= 1.0
                dWh["action_type"] += np.outer(h, da)
                dbh["action_type"] += da
                dh += model.head_W["action_type"] @ da

                # Relevant sub-heads for this action type
                for hn in RELEVANT_HEADS_BY_ACTION.get(ta, []):
                    tv = int(t[hn])
                    p = _softmax(logits[hn])
                    loss_sum += -math.log(max(1e-9, float(p[tv])))
                    d = p.copy()
                    d[tv] -= 1.0
                    dWh[hn] += np.outer(h, d)
                    dbh[hn] += d
                    dh += model.head_W[hn] @ d

                # Optional value head regression
                if model.has_value_head and dWv is not None and "value_target" in r and value_pred is not None:
                    vt = float(r["value_target"])
                    dv = float(value_pred) - vt
                    value_mse_sum += dv * dv
                    value_seen += 1

                    dvr = value_loss_weight * 2.0 * dv * float(value_pred) * (1.0 - float(value_pred))
                    dWv += h * dvr
                    dbv += dvr
                    dh += model.W_value * dvr

                dh[h_pre <= 0.0] = 0.0
                dW1 += np.outer(x, dh)
                db1 += dh
                seen += 1

            scale = 1.0 / max(1, len(batch))
            model.W1 -= lr * dW1 * scale
            model.b1 -= lr * db1 * scale
            for hn in HEAD_NAMES:
                model.head_W[hn] -= lr * dWh[hn] * scale
                model.head_b[hn] -= lr * dbh[hn] * scale
            if model.has_value_head and dWv is not None and model.W_value is not None:
                model.W_value -= lr * dWv * scale
                model.b_value = np.float32(float(model.b_value - lr * dbv * scale))

        train_metrics = {
            "loss": loss_sum / max(1, seen),
            "action_acc": action_correct / max(1, seen),
            "value_mse": value_mse_sum / max(1, value_seen) if value_seen > 0 else 0.0,
        }
        val_metrics = eval_split(val)

        row = {
            "epoch": ep,
            "train_loss": round(train_metrics["loss"], 6),
            "train_action_acc": round(train_metrics["action_acc"], 6),
            "train_value_mse": round(train_metrics["value_mse"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_action_acc": round(val_metrics["action_acc"], 6),
            "val_value_mse": round(val_metrics["value_mse"], 6),
        }
        history.append(row)
        print(
            f"epoch {ep}/{epochs} | train_loss={row['train_loss']:.4f} train_acc={row['train_action_acc']:.3f} "
            f"train_v={row['train_value_mse']:.4f} | val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_action_acc']:.3f} val_v={row['val_value_mse']:.4f}"
        )

    result = {
        "version": 2,
        "mode": "factorized_behavioral_cloning",
        "elapsed_sec": round(time.time() - started, 3),
        "dataset": dataset_jsonl,
        "meta": meta_json,
        "feature_dim": feature_dim,
        "head_dims": head_dims,
        "has_value_head": has_value_target,
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
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train_bc(
        dataset_jsonl=args.dataset,
        meta_json=args.meta,
        out_model=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        value_loss_weight=args.value_weight,
    )
    print(
        f"factorized BC complete | model={args.out} | "
        f"train={out['train_samples']} val={out['val_samples']}"
    )


if __name__ == "__main__":
    main()
