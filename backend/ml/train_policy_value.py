"""Train a simple policy+value neural network from self-play JSONL data.

Model: MLP(state -> hidden -> [policy logits, value scalar]).
Backend: NumPy (no external ML framework required).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    state: np.ndarray
    legal_action_ids: list[int]
    action_id: int
    reward: float


def _load_samples(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                Sample(
                    state=np.asarray(row["state"], dtype=np.float32),
                    legal_action_ids=[int(x) for x in row["legal_action_ids"]],
                    action_id=int(row["action_id"]),
                    reward=float(row["reward"]),
                )
            )
    return samples


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    if s <= 0:
        return np.full_like(ex, 1.0 / len(ex))
    return ex / s


class PolicyValueMLP:
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)

        # Xavier-ish init
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / feature_dim), size=(feature_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.Wp = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, action_dim)).astype(np.float32)
        self.bp = np.zeros(action_dim, dtype=np.float32)

        self.Wv = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim,)).astype(np.float32)
        self.bv = np.float32(0.0)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        h_pre = x @ self.W1 + self.b1
        h = _relu(h_pre)
        logits = h @ self.Wp + self.bp
        value_raw = float(h @ self.Wv + self.bv)
        value = _sigmoid(value_raw)
        return h_pre, h, logits, value

    def save(self, path: str | Path, metadata: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            W1=self.W1,
            b1=self.b1,
            Wp=self.Wp,
            bp=self.bp,
            Wv=self.Wv,
            bv=np.asarray([self.bv], dtype=np.float32),
            metadata_json=np.asarray([json.dumps(metadata)], dtype=object),
        )


def _evaluate(model: PolicyValueMLP, dataset: list[Sample]) -> dict[str, float]:
    if not dataset:
        return {"policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}

    policy_loss = 0.0
    value_loss = 0.0
    correct = 0

    for s in dataset:
        _, _, logits, value = model.forward(s.state)
        legal = s.legal_action_ids
        legal_logits = logits[legal]
        probs = _softmax_1d(legal_logits)

        # Policy loss
        try:
            idx = legal.index(s.action_id)
            p = max(1e-9, float(probs[idx]))
            policy_loss += -math.log(p)
        except ValueError:
            policy_loss += 10.0

        # Accuracy on legal mask argmax
        pred_local = int(np.argmax(probs))
        pred_action = legal[pred_local]
        if pred_action == s.action_id:
            correct += 1

        # Value loss
        dv = value - s.reward
        value_loss += dv * dv

    n = float(len(dataset))
    return {
        "policy_loss": policy_loss / n,
        "value_loss": value_loss / n,
        "accuracy": correct / n,
    }


def train(
    dataset_jsonl: str,
    metadata_json: str,
    out_model: str,
    epochs: int = 5,
    batch_size: int = 64,
    hidden_dim: int = 128,
    learning_rate: float = 1e-3,
    value_loss_weight: float = 0.5,
    val_split: float = 0.1,
    seed: int = 0,
) -> dict:
    start = time.time()
    rnd = random.Random(seed)

    meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    feature_dim = int(meta["feature_dim"])
    action_dim = int(meta["action_space"]["num_actions"])

    samples = _load_samples(dataset_jsonl)
    if not samples:
        raise ValueError("No samples found in dataset")

    # Basic shape checks
    for i, s in enumerate(samples[:50]):
        if len(s.state) != feature_dim:
            raise ValueError(f"Sample {i} feature dim mismatch: {len(s.state)} != {feature_dim}")
        if s.action_id >= action_dim:
            raise ValueError(f"Sample {i} action id out of range: {s.action_id} >= {action_dim}")

    rnd.shuffle(samples)
    val_n = max(1, int(len(samples) * val_split)) if len(samples) >= 10 else 1
    val = samples[:val_n]
    train_set = samples[val_n:] if len(samples) > val_n else samples

    model = PolicyValueMLP(feature_dim=feature_dim, action_dim=action_dim, hidden_dim=hidden_dim, seed=seed)

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        rnd.shuffle(train_set)

        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        seen = 0

        for bi in range(0, len(train_set), batch_size):
            batch = train_set[bi: bi + batch_size]
            if not batch:
                continue

            dW1 = np.zeros_like(model.W1)
            db1 = np.zeros_like(model.b1)
            dWp = np.zeros_like(model.Wp)
            dbp = np.zeros_like(model.bp)
            dWv = np.zeros_like(model.Wv)
            dbv = 0.0

            for s in batch:
                h_pre, h, logits, value = model.forward(s.state)

                legal = s.legal_action_ids
                legal_logits = logits[legal]
                probs = _softmax_1d(legal_logits)

                # dlogits over full action dimension
                dlogits = np.zeros_like(logits)
                try:
                    chosen_local = legal.index(s.action_id)
                except ValueError:
                    chosen_local = -1

                for local_i, aid in enumerate(legal):
                    dlogits[aid] += probs[local_i]
                if chosen_local >= 0:
                    dlogits[legal[chosen_local]] -= 1.0

                # Metrics
                if chosen_local >= 0:
                    p = max(1e-9, float(probs[chosen_local]))
                    train_policy_loss += -math.log(p)
                else:
                    train_policy_loss += 10.0

                pred_local = int(np.argmax(probs))
                pred_action = legal[pred_local]
                if pred_action == s.action_id:
                    train_correct += 1

                value_err = value - s.reward
                train_value_loss += value_err * value_err

                dvalue_raw = value_loss_weight * 2.0 * value_err * value * (1.0 - value)

                # Backprop heads
                dWp += np.outer(h, dlogits)
                dbp += dlogits

                dWv += h * dvalue_raw
                dbv += dvalue_raw

                dh = (model.Wp @ dlogits) + (model.Wv * dvalue_raw)
                dh[h_pre <= 0.0] = 0.0

                dW1 += np.outer(s.state, dh)
                db1 += dh

                seen += 1

            scale = 1.0 / max(1, len(batch))
            model.W1 -= learning_rate * dW1 * scale
            model.b1 -= learning_rate * db1 * scale
            model.Wp -= learning_rate * dWp * scale
            model.bp -= learning_rate * dbp * scale
            model.Wv -= learning_rate * dWv * scale
            model.bv = np.float32(float(model.bv - learning_rate * dbv * scale))

        train_metrics = {
            "policy_loss": train_policy_loss / max(1, seen),
            "value_loss": train_value_loss / max(1, seen),
            "accuracy": train_correct / max(1, seen),
        }
        val_metrics = _evaluate(model, val)

        row = {
            "epoch": epoch,
            "train_policy_loss": round(train_metrics["policy_loss"], 6),
            "train_value_loss": round(train_metrics["value_loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_policy_loss": round(val_metrics["policy_loss"], 6),
            "val_value_loss": round(val_metrics["value_loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} | "
            f"train_pi={row['train_policy_loss']:.4f} train_v={row['train_value_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.3f} | "
            f"val_pi={row['val_policy_loss']:.4f} val_v={row['val_value_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.3f}"
        )

    out_meta = {
        "version": 1,
        "model_type": "numpy_mlp_policy_value",
        "created_at_epoch": int(time.time()),
        "elapsed_sec": round(time.time() - start, 3),
        "dataset_jsonl": str(dataset_jsonl),
        "metadata_json": str(metadata_json),
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_loss_weight": value_loss_weight,
        "train_samples": len(train_set),
        "val_samples": len(val),
        "history": history,
    }

    model.save(out_model, out_meta)
    return out_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Wingspan policy+value MLP")
    parser.add_argument("--dataset", default="reports/ml/self_play_dataset.jsonl")
    parser.add_argument("--meta", default="reports/ml/self_play_dataset.meta.json")
    parser.add_argument("--out", default="reports/ml/policy_value_model.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train(
        dataset_jsonl=args.dataset,
        metadata_json=args.meta,
        out_model=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden,
        learning_rate=args.lr,
        value_loss_weight=args.value_weight,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(
        f"training complete | model={args.out} | "
        f"train={out['train_samples']} val={out['val_samples']}"
    )


if __name__ == "__main__":
    main()
