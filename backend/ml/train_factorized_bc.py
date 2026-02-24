"""Train a factorized behavioral-cloning policy network (PyTorch).

Supports an auxiliary value head when `value_target` is present in data.
Uses MPS (Apple GPU) or CUDA when available for accelerated training.
Saves model in .npz format compatible with FactorizedPolicyModel inference.

Key improvement over the numpy version: _load_as_arrays() converts each JSON
row's state vector directly into a pre-allocated numpy matrix, avoiding the
~1GB of Python float objects that caused hour-long load times at 1072 features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

# Inverse of RELEVANT_HEADS_BY_ACTION: head_name → list of action IDs that activate it
_HEAD_RELEVANT_ACTIONS: dict[str, list[int]] = {}
for _aid, _heads in RELEVANT_HEADS_BY_ACTION.items():
    for _hn in _heads:
        _HEAD_RELEVANT_ACTIONS.setdefault(_hn, []).append(_aid)

MAX_MOVE_NEGS = 4


def _grad_softplus_neg_delta(delta: float) -> float:
    """Return d/d(delta) softplus(-delta) with overflow-safe branches.

    Kept for backward compatibility with tests; the PyTorch training loop
    uses F.softplus(-delta) directly for efficient vectorized computation.
    """
    if delta >= 0.0:
        exp_neg = math.exp(-delta)
        return -exp_neg / (1.0 + exp_neg)
    return -1.0 / (1.0 + math.exp(delta))


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _load_as_arrays(
    path: str | Path,
    meta: dict,
    value_target_scale: float,
    value_target_bias: float,
) -> dict:
    """Load JSONL dataset into compact numpy arrays.

    Unlike the previous list[dict] approach which stored every float as a Python
    object (~1 GB RAM for 30k × 1072 features), this pre-allocates float32 numpy
    arrays and copies each row in-place, keeping peak memory ~125 MB.
    """
    feature_dim = int(meta["feature_dim"])
    head_names = list(meta["target_heads"].keys())

    # First pass: count rows (no JSON parsing, just newline counting)
    n_rows = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_rows += 1

    if n_rows == 0:
        raise ValueError(f"Empty dataset: {path}")

    # Pre-allocate compact arrays
    states = np.empty((n_rows, feature_dim), dtype=np.float32)
    targets = {hn: np.zeros(n_rows, dtype=np.int64) for hn in head_names}
    value_targets = np.zeros(n_rows, dtype=np.float32)
    has_value = np.zeros(n_rows, dtype=np.bool_)
    is_rl = np.zeros(n_rows, dtype=np.bool_)
    rl_advantage = np.ones(n_rows, dtype=np.float32)
    has_move = np.zeros(n_rows, dtype=np.bool_)
    move_pos = np.zeros((n_rows, MOVE_FEATURE_DIM), dtype=np.float32)
    move_negs = np.zeros((n_rows, MAX_MOVE_NEGS, MOVE_FEATURE_DIM), dtype=np.float32)
    num_move_negs = np.zeros(n_rows, dtype=np.int64)

    # Second pass: fill arrays (each Python list freed immediately after copy)
    scale = max(1e-9, float(value_target_scale))
    i = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            states[i] = r["state"]  # bulk list→numpy copy; no per-float Python objects retained
            t = r["targets"]
            for hn in head_names:
                targets[hn][i] = int(t[hn])
            if "value_target_score" in r:
                value_targets[i] = (float(r["value_target_score"]) - value_target_bias) / scale
                has_value[i] = True
            elif "value_target" in r:
                value_targets[i] = float(r["value_target"])
                has_value[i] = True
            if r.get("is_rl"):
                is_rl[i] = True
                rl_advantage[i] = float(r.get("rl_advantage", 1.0))
            if "move_pos" in r and r.get("move_negs"):
                negs = r["move_negs"][:MAX_MOVE_NEGS]
                n = len(negs)
                if n > 0:
                    has_move[i] = True
                    move_pos[i] = r["move_pos"]
                    for k, neg in enumerate(negs):
                        move_negs[i, k] = neg
                    num_move_negs[i] = n
            i += 1

    return {
        "states": states[:i],
        "targets": {hn: arr[:i] for hn, arr in targets.items()},
        "value_targets": value_targets[:i],
        "has_value": has_value[:i],
        "is_rl": is_rl[:i],
        "rl_advantage": rl_advantage[:i],
        "has_move": has_move[:i],
        "move_pos": move_pos[:i],
        "move_negs": move_negs[:i],
        "num_move_negs": num_move_negs[:i],
        "n_rows": i,
        "has_value_target_any": bool(has_value[:i].any()),
        "has_move_data_any": bool(has_move[:i].any()),
    }


class BCModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        head_dims: dict[str, int],
        has_value_head: bool,
        has_move_value_head: bool,
        dropout: float,
        batch_norm_enabled: bool,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.batch_norm_enabled = bool(batch_norm_enabled)

        self.layer1 = nn.Linear(feature_dim, hidden1)
        self.bn1 = (
            nn.BatchNorm1d(hidden1, eps=batch_norm_eps, momentum=batch_norm_momentum)
            if batch_norm_enabled
            else None
        )
        self.drop1 = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.bn2 = (
            nn.BatchNorm1d(hidden2, eps=batch_norm_eps, momentum=batch_norm_momentum)
            if batch_norm_enabled
            else None
        )
        self.drop2 = nn.Dropout(p=dropout)
        self.heads = nn.ModuleDict({hn: nn.Linear(hidden2, dim) for hn, dim in head_dims.items()})
        self.value_head = nn.Linear(hidden2, 1) if has_value_head else None
        self.move_value_head = (
            nn.Linear(feature_dim + MOVE_FEATURE_DIM, 1) if has_move_value_head else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        h = self.layer1(x)
        if self.bn1 is not None:
            h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.layer2(h)
        if self.bn2 is not None:
            h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)
        logits = {hn: head(h) for hn, head in self.heads.items()}
        value = self.value_head(h).squeeze(-1) if self.value_head is not None else None
        return logits, value


def _warm_start_from_npz(
    model: BCModel, init_path: Path, device: torch.device
) -> dict:
    """Copy overlapping weights from a .npz checkpoint into the PyTorch model."""
    z = np.load(init_path, allow_pickle=True)
    applied: list[dict] = []
    skipped: list[dict] = []

    def try_copy_weight_2d(layer: nn.Linear, key: str) -> None:
        """npz stores (in_dim, out_dim); nn.Linear.weight is (out_dim, in_dim)."""
        if key not in z:
            skipped.append({"param": key, "reason": "src_missing"})
            return
        src_t = torch.from_numpy(z[key].T.astype(np.float32)).to(device)
        rows = min(src_t.shape[0], layer.weight.data.shape[0])
        cols = min(src_t.shape[1], layer.weight.data.shape[1])
        with torch.no_grad():
            layer.weight.data[:rows, :cols] = src_t[:rows, :cols]
        applied.append({"param": key, "copied": rows * cols})

    def try_copy_1d(param: nn.Parameter, key: str) -> None:
        if key not in z:
            skipped.append({"param": key, "reason": "src_missing"})
            return
        src = torch.from_numpy(z[key].astype(np.float32)).to(device)
        n = min(len(src), len(param.data))
        with torch.no_grad():
            param.data[:n] = src[:n]
        applied.append({"param": key, "copied": n})

    try_copy_weight_2d(model.layer1, "W1")
    try_copy_1d(model.layer1.bias, "b1")
    try_copy_weight_2d(model.layer2, "W2")
    try_copy_1d(model.layer2.bias, "b2")

    for hn in HEAD_NAMES:
        if hn in model.heads:
            try_copy_weight_2d(model.heads[hn], f"W_{hn}")
            try_copy_1d(model.heads[hn].bias, f"b_{hn}")

    if model.value_head is not None:
        if "W_value" in z:
            src = torch.from_numpy(z["W_value"].astype(np.float32)).to(device)
            n = min(len(src), model.value_head.weight.data.shape[1])
            with torch.no_grad():
                model.value_head.weight.data[0, :n] = src[:n]
            applied.append({"param": "W_value", "copied": n})
        else:
            skipped.append({"param": "W_value", "reason": "src_missing"})
        if "b_value" in z:
            with torch.no_grad():
                model.value_head.bias.data[0] = float(z["b_value"][0])
            applied.append({"param": "b_value", "copied": 1})
        else:
            skipped.append({"param": "b_value", "reason": "src_missing"})

    if model.move_value_head is not None:
        if "W_move_value" in z:
            src = torch.from_numpy(z["W_move_value"].astype(np.float32)).to(device)
            n = min(len(src), model.move_value_head.weight.data.shape[1])
            with torch.no_grad():
                model.move_value_head.weight.data[0, :n] = src[:n]
            applied.append({"param": "W_move_value", "copied": n})
        else:
            skipped.append({"param": "W_move_value", "reason": "src_missing"})
        if "b_move_value" in z:
            with torch.no_grad():
                model.move_value_head.bias.data[0] = float(z["b_move_value"][0])
            applied.append({"param": "b_move_value", "copied": 1})
        else:
            skipped.append({"param": "b_move_value", "reason": "src_missing"})

    if model.batch_norm_enabled:
        for prefix, bn_layer in [("bn1", model.bn1), ("bn2", model.bn2)]:
            if bn_layer is None:
                continue
            try_copy_1d(bn_layer.weight, f"{prefix}_gamma")
            try_copy_1d(bn_layer.bias, f"{prefix}_beta")
            for stat_key, buf in [
                ("running_mean", bn_layer.running_mean),
                ("running_var", bn_layer.running_var),
            ]:
                npz_key = f"{prefix}_{stat_key}"
                if npz_key in z:
                    src_arr = z[npz_key].astype(np.float32)
                    n = min(len(src_arr), len(buf))
                    with torch.no_grad():
                        buf[:n] = torch.from_numpy(src_arr[:n]).to(device)
                    applied.append({"param": npz_key, "copied": n})
                else:
                    skipped.append({"param": npz_key, "reason": "src_missing"})

    return {
        "requested": True,
        "init_model_path": str(init_path),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
    }


def _save_to_npz(model: BCModel, out_path: str | Path, meta_dict: dict) -> None:
    """Save PyTorch model weights to .npz format (compatible with FactorizedPolicyModel)."""
    d: dict[str, np.ndarray] = {}

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    # Inference format: (in_dim, out_dim) so eval does state @ W + b
    d["W1"] = _np(model.layer1.weight.T)
    d["b1"] = _np(model.layer1.bias)
    d["W2"] = _np(model.layer2.weight.T)
    d["b2"] = _np(model.layer2.bias)

    if model.batch_norm_enabled:
        for prefix, bn_layer in [("bn1", model.bn1), ("bn2", model.bn2)]:
            if bn_layer is None:
                continue
            d[f"{prefix}_gamma"] = _np(bn_layer.weight)
            d[f"{prefix}_beta"] = _np(bn_layer.bias)
            d[f"{prefix}_running_mean"] = _np(bn_layer.running_mean)
            d[f"{prefix}_running_var"] = _np(bn_layer.running_var)

    for hn, head_layer in model.heads.items():
        d[f"W_{hn}"] = _np(head_layer.weight.T)
        d[f"b_{hn}"] = _np(head_layer.bias)

    if model.value_head is not None:
        d["W_value"] = _np(model.value_head.weight[0])
        d["b_value"] = np.array([float(_np(model.value_head.bias)[0])], dtype=np.float32)

    if model.move_value_head is not None:
        d["W_move_value"] = _np(model.move_value_head.weight[0])
        d["b_move_value"] = np.array(
            [float(_np(model.move_value_head.bias)[0])], dtype=np.float32
        )

    d["metadata_json"] = np.array([json.dumps(meta_dict).encode("utf-8")])
    np.savez_compressed(out_path, **d)


def _eval_pass(
    model: BCModel,
    device: torch.device,
    states: torch.Tensor,
    targets: dict[str, torch.Tensor],
    value_targets: torch.Tensor,
    has_value: torch.Tensor,
    has_move: torch.Tensor,
    move_pos: torch.Tensor,
    move_negs: torch.Tensor,
    num_move_negs: torch.Tensor,
    value_loss_weight: float,
    move_value_loss_weight: float,
    batch_size: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    action_correct = 0
    total = 0
    value_mse = 0.0
    value_n = 0
    move_rank_loss = 0.0
    move_rank_n = 0
    move_margin_sum = 0.0
    move_margin_rows = 0
    move_pair_correct = 0
    move_pair_total = 0
    n = states.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xs = states[start:end].to(device)
            action_ids = targets["action_type"][start:end].to(device)
            logits, value = model(xs)
            B = xs.shape[0]

            ce = F.cross_entropy(logits["action_type"], action_ids, reduction="none")
            total_loss += ce.sum().item()
            action_correct += (logits["action_type"].argmax(dim=1) == action_ids).sum().item()
            total += B

            for hn, rel_aids in _HEAD_RELEVANT_ACTIONS.items():
                mask = torch.zeros(B, device=device)
                for aid in rel_aids:
                    mask = mask + (action_ids == aid).float()
                if mask.sum() < 0.5:
                    continue
                tgt = targets[hn][start:end].to(device)
                head_ce = F.cross_entropy(logits[hn], tgt, reduction="none")
                total_loss += (head_ce * mask).sum().item()

            if model.value_head is not None and value is not None:
                hv = has_value[start:end].to(device)
                if hv.any():
                    vt = value_targets[start:end].to(device)
                    value_mse += ((value - vt) ** 2 * hv.float()).sum().item()
                    value_n += int(hv.sum().item())

            if model.move_value_head is not None:
                hm = has_move[start:end]
                move_idx = hm.nonzero(as_tuple=True)[0]
                M = len(move_idx)
                if M > 0:
                    s_m = states[start:end][move_idx].to(device)
                    mp = move_pos[start:end][move_idx].to(device)
                    mn = move_negs[start:end][move_idx].to(device)
                    nn_arr = num_move_negs[start:end][move_idx].to(device)
                    max_neg = mn.shape[1]
                    neg_mask = (
                        torch.arange(max_neg, device=device).unsqueeze(0) < nn_arr.unsqueeze(1)
                    )
                    pos_x = torch.cat([s_m, mp], dim=1)
                    s_pos = model.move_value_head(pos_x).squeeze(-1)
                    s_neg_all = torch.stack(
                        [
                            model.move_value_head(
                                torch.cat([s_m, mn[:, k, :]], dim=1)
                            ).squeeze(-1)
                            for k in range(max_neg)
                        ],
                        dim=1,
                    )
                    delta = s_pos.unsqueeze(1) - s_neg_all
                    rank_l = F.softplus(-delta) * neg_mask.float()
                    cnt = neg_mask.float().sum(dim=1).clamp(min=1)
                    move_rank_loss += (rank_l.sum(dim=1) / cnt).sum().item()
                    move_rank_n += M
                    move_margin_sum += (
                        s_pos - (s_neg_all * neg_mask.float()).sum(dim=1) / cnt
                    ).sum().item()
                    move_margin_rows += M
                    move_pair_correct += ((delta > 0) & neg_mask).sum().item()
                    move_pair_total += neg_mask.sum().item()

    cls_loss = total_loss / max(1, total)
    v_mse = value_mse / max(1, value_n) if value_n > 0 else 0.0
    mv_loss = move_rank_loss / max(1, move_rank_n) if move_rank_n > 0 else 0.0
    return {
        "loss": cls_loss + value_loss_weight * v_mse + move_value_loss_weight * mv_loss,
        "action_acc": action_correct / max(1, total),
        "value_mse": v_mse,
        "move_rank_loss": mv_loss,
        "move_rank_margin_mean": move_margin_sum / max(1, move_margin_rows),
        "move_pair_acc": (
            move_pair_correct / max(1, move_pair_total) if move_pair_total > 0 else 0.0
        ),
    }


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
    momentum: float = 0.9,
    val_split: float = 0.1,
    seed: int = 0,
    value_loss_weight: float = 0.5,
    move_value_enabled: bool = True,
    move_value_loss_weight: float = 0.2,
    move_value_num_negatives: int = 4,
    batch_norm_enabled: bool = True,
    batch_norm_momentum: float = 0.1,
    batch_norm_eps: float = 1e-5,
    early_stop_enabled: bool = True,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
    early_stop_restore_best: bool = True,
    init_model_path: str | None = None,
    hidden: int | None = None,
    lr: float | None = None,
) -> dict:
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    value_target_scale = float(meta.get("value_target_config", {}).get("score_scale", 150.0))
    value_target_bias = float(meta.get("value_target_config", {}).get("score_bias", 0.0))

    print(f"Loading dataset: {dataset_jsonl}")
    data = _load_as_arrays(Path(dataset_jsonl), meta, value_target_scale, value_target_bias)
    n_rows = data["n_rows"]
    print(f"Loaded {n_rows} samples | feature_dim={feature_dim}")

    has_value_target = bool(data["has_value_target_any"])
    has_move_value_data = bool(data["has_move_data_any"])
    effective_move_value = bool(move_value_enabled and has_move_value_data)

    # Shuffle and split
    perm = np.random.default_rng(seed).permutation(n_rows)
    val_n = max(1, int(n_rows * val_split)) if n_rows >= 10 else 1
    val_idx = perm[:val_n]
    train_idx = perm[val_n:] if n_rows > val_n else perm

    # Build val tensors (CPU; moved to device in eval pass)
    val_states = torch.from_numpy(data["states"][val_idx])
    val_targets = {hn: torch.from_numpy(data["targets"][hn][val_idx]) for hn in head_dims}
    val_value_targets = torch.from_numpy(data["value_targets"][val_idx])
    val_has_value = torch.from_numpy(data["has_value"][val_idx])
    val_has_move = torch.from_numpy(data["has_move"][val_idx])
    val_move_pos = torch.from_numpy(data["move_pos"][val_idx])
    val_move_negs = torch.from_numpy(data["move_negs"][val_idx])
    val_num_negs = torch.from_numpy(data["num_move_negs"][val_idx])

    # Build train tensors
    train_states = torch.from_numpy(data["states"][train_idx])
    train_targets = {hn: torch.from_numpy(data["targets"][hn][train_idx]) for hn in head_dims}
    train_value_targets = torch.from_numpy(data["value_targets"][train_idx])
    train_has_value = torch.from_numpy(data["has_value"][train_idx])
    train_rl_adv = torch.from_numpy(data["rl_advantage"][train_idx])
    train_has_move = torch.from_numpy(data["has_move"][train_idx])
    train_move_pos = torch.from_numpy(data["move_pos"][train_idx])
    train_move_negs = torch.from_numpy(data["move_negs"][train_idx])
    train_num_negs = torch.from_numpy(data["num_move_negs"][train_idx])
    n_train = train_states.shape[0]

    # DataLoader over indices; use drop_last to avoid single-sample batches with BatchNorm
    loader = DataLoader(
        TensorDataset(torch.arange(n_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(batch_norm_enabled and batch_size > 1 and n_train > batch_size),
    )

    device = _select_device()
    print(f"Training device: {device}")

    model = BCModel(
        feature_dim,
        hidden1,
        hidden2,
        head_dims,
        has_value_target,
        effective_move_value,
        dropout,
        batch_norm_enabled,
        batch_norm_eps,
        batch_norm_momentum,
    ).to(device)

    warm_start_summary: dict = {"requested": False}
    if init_model_path:
        init_path = Path(init_model_path)
        if not init_path.exists():
            raise FileNotFoundError(f"init_model_path does not exist: {init_model_path}")
        warm_start_summary = _warm_start_from_npz(model, init_path, device)
        print(
            f"warm-start from {init_path} | "
            f"applied={warm_start_summary['applied_count']} "
            f"skipped={warm_start_summary['skipped_count']}"
        )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr_peak, momentum=max(0.0, min(0.999, float(momentum)))
    )
    early_stop_patience = max(1, int(early_stop_patience))
    early_stop_min_delta = max(0.0, float(early_stop_min_delta))

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    stopped_early = False
    best_state_dict: dict | None = None
    epochs_completed = 0
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        # Set LR for this epoch
        lr_epoch = _lr_for_epoch(
            ep,
            lr_init=lr_init,
            lr_peak=lr_peak,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_decay_every=lr_decay_every,
            lr_decay_factor=lr_decay_factor,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_epoch

        model.train()
        loss_sum = 0.0
        action_correct = 0
        value_mse_sum = 0.0
        value_seen = 0
        move_rank_loss_sum = 0.0
        move_rank_n = 0
        move_margin_sum = 0.0
        move_margin_rows = 0
        move_pair_correct = 0
        move_pair_total = 0
        seen = 0

        for (batch_idx_t,) in loader:
            bidx = batch_idx_t.numpy()
            xs = train_states[bidx].to(device)
            action_ids = train_targets["action_type"][bidx].to(device)
            rl_adv = train_rl_adv[bidx].to(device)
            B = xs.shape[0]

            logits, value = model(xs)

            # Action type CE (always active, scaled by RL advantage)
            ce_action = F.cross_entropy(logits["action_type"], action_ids, reduction="none")
            loss = (ce_action * rl_adv).mean()
            loss_sum += ce_action.sum().item()
            action_correct += (logits["action_type"].argmax(dim=1) == action_ids).sum().item()

            # Sub-head CE (only active for their relevant action types)
            for hn, rel_aids in _HEAD_RELEVANT_ACTIONS.items():
                mask = torch.zeros(B, device=device)
                for aid in rel_aids:
                    mask = mask + (action_ids == aid).float()
                mask_sum = mask.sum()
                if mask_sum < 0.5:
                    continue
                tgt = train_targets[hn][bidx].to(device)
                head_ce = F.cross_entropy(logits[hn], tgt, reduction="none")
                loss = loss + (head_ce * mask * rl_adv).sum() / mask_sum.clamp(min=1)
                loss_sum += (head_ce * mask).sum().item()

            # Value head (MSE regression)
            if model.value_head is not None and value is not None:
                hv = train_has_value[bidx].to(device)
                if hv.any():
                    vt = train_value_targets[bidx].to(device)
                    dv2 = (value - vt) ** 2
                    hv_f = hv.float()
                    v_loss = (dv2 * hv_f).sum() / hv_f.sum().clamp(min=1)
                    loss = loss + value_loss_weight * v_loss
                    value_mse_sum += (dv2 * hv_f).sum().item()
                    value_seen += int(hv.sum().item())

            # Move-value ranking head (pairwise softplus loss)
            if model.move_value_head is not None:
                hm = train_has_move[bidx]
                move_idx = hm.nonzero(as_tuple=True)[0]
                M = len(move_idx)
                if M > 0:
                    s_m = train_states[bidx][move_idx].to(device)
                    mp = train_move_pos[bidx][move_idx].to(device)
                    mn = train_move_negs[bidx][move_idx].to(device)
                    nn_arr = train_num_negs[bidx][move_idx].to(device)
                    max_neg = mn.shape[1]
                    neg_mask = (
                        torch.arange(max_neg, device=device).unsqueeze(0) < nn_arr.unsqueeze(1)
                    )
                    pos_x = torch.cat([s_m, mp], dim=1)
                    s_pos = model.move_value_head(pos_x).squeeze(-1)
                    s_neg_all = torch.stack(
                        [
                            model.move_value_head(
                                torch.cat([s_m, mn[:, k, :]], dim=1)
                            ).squeeze(-1)
                            for k in range(max_neg)
                        ],
                        dim=1,
                    )
                    delta = s_pos.unsqueeze(1) - s_neg_all
                    rank_l = F.softplus(-delta) * neg_mask.float()
                    cnt = neg_mask.float().sum(dim=1).clamp(min=1)
                    mv_loss = (rank_l.sum(dim=1) / cnt).mean()
                    loss = loss + move_value_loss_weight * mv_loss
                    move_rank_loss_sum += (rank_l.sum(dim=1) / cnt).sum().item()
                    move_rank_n += M
                    move_margin_sum += (
                        s_pos - (s_neg_all * neg_mask.float()).sum(dim=1) / cnt
                    ).sum().item()
                    move_margin_rows += M
                    move_pair_correct += ((delta > 0) & neg_mask).sum().item()
                    move_pair_total += int(neg_mask.sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seen += B

        train_cls_loss = loss_sum / max(1, seen)
        train_v_mse = value_mse_sum / max(1, value_seen) if value_seen > 0 else 0.0
        train_mv_loss = move_rank_loss_sum / max(1, move_rank_n) if move_rank_n > 0 else 0.0
        train_metrics = {
            "loss": train_cls_loss + value_loss_weight * train_v_mse + move_value_loss_weight * train_mv_loss,
            "action_acc": action_correct / max(1, seen),
            "value_mse": train_v_mse,
            "move_rank_loss": train_mv_loss,
            "move_rank_margin_mean": move_margin_sum / max(1, move_margin_rows),
            "move_pair_acc": (
                move_pair_correct / max(1, move_pair_total) if move_pair_total > 0 else 0.0
            ),
        }

        val_metrics = _eval_pass(
            model, device,
            val_states, val_targets, val_value_targets, val_has_value,
            val_has_move, val_move_pos, val_move_negs, val_num_negs,
            value_loss_weight, move_value_loss_weight, batch_size,
        )
        model.train()

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
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve_epochs += 1

        history.append(row)
        epochs_completed = ep
        print(
            f"epoch {ep}/{epochs} | lr={row['lr_epoch']:.6f} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_action_acc']:.3f} "
            f"train_v={row['train_value_mse']:.4f} train_mv={row['train_move_rank_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_action_acc']:.3f} "
            f"val_v={row['val_value_mse']:.4f} val_mv={row['val_move_rank_loss']:.4f}"
        )
        if early_stop_enabled and no_improve_epochs >= early_stop_patience:
            stopped_early = True
            break

    if early_stop_enabled and early_stop_restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    result = {
        "version": 5,
        "format_version": 5,
        "mode": "factorized_behavioral_cloning",
        "model_arch": "mlp_2layer",
        "elapsed_sec": round(time.time() - started, 3),
        "dataset": dataset_jsonl,
        "meta": meta_json,
        "feature_dim": feature_dim,
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "dropout": float(dropout),
        "batch_norm_enabled": bool(model.batch_norm_enabled),
        "batch_norm_momentum": float(batch_norm_momentum),
        "batch_norm_eps": float(batch_norm_eps),
        "lr_init": float(lr_init),
        "lr_peak": float(lr_peak),
        "lr_warmup_epochs": int(lr_warmup_epochs),
        "lr_decay_every": int(lr_decay_every),
        "lr_decay_factor": float(lr_decay_factor),
        "momentum": float(momentum),
        "early_stop_enabled": bool(early_stop_enabled),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_restore_best": bool(early_stop_restore_best),
        "stopped_early": bool(stopped_early),
        "best_val_loss_epoch": int(best_epoch),
        "best_val_loss": float(
            best_val_loss if best_epoch > 0 else (history[-1]["val_loss"] if history else 0.0)
        ),
        "epochs_completed": int(epochs_completed),
        "deprecated_args_used": deprecated_args_used,
        "head_dims": head_dims,
        "state_encoder": meta.get(
            "state_encoder",
            {"enable_identity_features": False, "identity_hash_dim": 128},
        ),
        "has_value_head": has_value_target,
        "value_prediction_mode": "score_norm_linear" if has_value_target else "none",
        "value_score_scale": value_target_scale,
        "value_score_bias": value_target_bias,
        "has_move_value_head": bool(model.move_value_head is not None),
        "move_feature_dim": MOVE_FEATURE_DIM,
        "move_value_loss_weight": float(move_value_loss_weight),
        "move_value_training": {
            "num_negatives": int(move_value_num_negatives),
            "sampling": "topk_then_random",
        },
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "warm_start": warm_start_summary,
        "history": history,
        "training_backend": "pytorch",
        "training_device": str(device),
    }
    _save_to_npz(model, out_model, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train factorized BC policy (PyTorch)")
    parser.add_argument("--dataset", default="reports/ml/bc_dataset.jsonl")
    parser.add_argument("--meta", default="reports/ml/bc_dataset.meta.json")
    parser.add_argument("--out", default="reports/ml/factorized_bc_model.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.set_defaults(batch_norm_enabled=True)
    parser.add_argument("--batch-norm-enabled", dest="batch_norm_enabled", action="store_true")
    parser.add_argument("--disable-batch-norm", dest="batch_norm_enabled", action="store_false")
    parser.add_argument("--batch-norm-momentum", type=float, default=0.1)
    parser.add_argument("--batch-norm-eps", type=float, default=1e-5)
    parser.add_argument("--lr-init", type=float, default=1e-4)
    parser.add_argument("--lr-peak", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-epochs", type=int, default=2)
    parser.add_argument("--lr-decay-every", type=int, default=3)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.set_defaults(early_stop_enabled=True, early_stop_restore_best=True)
    parser.add_argument("--early-stop-enabled", dest="early_stop_enabled", action="store_true")
    parser.add_argument("--disable-early-stop", dest="early_stop_enabled", action="store_false")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-restore-best", dest="early_stop_restore_best", action="store_true"
    )
    parser.add_argument(
        "--no-early-stop-restore-best", dest="early_stop_restore_best", action="store_false"
    )
    parser.add_argument("--init-model", default=None, help="Optional warm-start model (.npz)")
    parser.add_argument("--hidden", type=int, default=None, help="Deprecated: sets hidden1=hidden2")
    parser.add_argument("--lr", type=float, default=None, help="Deprecated: sets lr_peak")
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
        momentum=args.momentum,
        val_split=args.val_split,
        value_loss_weight=args.value_weight,
        move_value_enabled=args.move_value_enabled,
        move_value_loss_weight=args.move_value_loss_weight,
        move_value_num_negatives=args.move_value_num_negatives,
        batch_norm_enabled=args.batch_norm_enabled,
        batch_norm_momentum=args.batch_norm_momentum,
        batch_norm_eps=args.batch_norm_eps,
        early_stop_enabled=args.early_stop_enabled,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_restore_best=args.early_stop_restore_best,
        init_model_path=args.init_model,
        hidden=args.hidden,
        lr=args.lr,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
