"""Inference helpers for factorized behavioral-cloning models."""

from __future__ import annotations

import json
import math as _math
from pathlib import Path

import numpy as np

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _Func
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from backend.models.player import Player
from backend.solver.move_generator import Move
from backend.ml.factorized_policy import encode_factorized_targets, RELEVANT_HEADS_BY_ACTION
from backend.ml.move_features import encode_move_features


class FactorizedPolicyModel:
    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=True)
        self.W1 = z["W1"]
        self.b1 = z["b1"]
        self.feature_dim = int(self.W1.shape[0])
        self.has_second_layer = "W2" in z and "b2" in z
        self.W2 = z["W2"] if self.has_second_layer else None
        self.b2 = z["b2"] if self.has_second_layer else None

        meta_json = z["metadata_json"][0]
        if isinstance(meta_json, bytes):
            meta_json = meta_json.decode("utf-8")
        self.meta = json.loads(str(meta_json))
        self.batch_norm_eps = float(self.meta.get("batch_norm_eps", 1e-5))

        self.has_bn1 = all(k in z for k in ("bn1_gamma", "bn1_beta", "bn1_running_mean", "bn1_running_var"))
        self.bn1_gamma = z["bn1_gamma"] if self.has_bn1 else None
        self.bn1_beta = z["bn1_beta"] if self.has_bn1 else None
        self.bn1_running_mean = z["bn1_running_mean"] if self.has_bn1 else None
        self.bn1_running_var = z["bn1_running_var"] if self.has_bn1 else None
        self.has_bn2 = self.has_second_layer and all(
            k in z for k in ("bn2_gamma", "bn2_beta", "bn2_running_mean", "bn2_running_var")
        )
        self.bn2_gamma = z["bn2_gamma"] if self.has_bn2 else None
        self.bn2_beta = z["bn2_beta"] if self.has_bn2 else None
        self.bn2_running_mean = z["bn2_running_mean"] if self.has_bn2 else None
        self.bn2_running_var = z["bn2_running_var"] if self.has_bn2 else None

        self.head_dims = self.meta.get("head_dims") or self.meta.get("target_heads") or {}
        self.head_W: dict[str, np.ndarray] = {}
        self.head_b: dict[str, np.ndarray] = {}
        for hn in self.head_dims:
            self.head_W[hn] = z[f"W_{hn}"]
            self.head_b[hn] = z[f"b_{hn}"]

        self.has_value_head = "W_value" in z and "b_value" in z
        self.W_value = z["W_value"] if self.has_value_head else None
        self.b_value = float(z["b_value"][0]) if self.has_value_head else 0.0
        self.value_prediction_mode = str(self.meta.get("value_prediction_mode", "sigmoid_norm"))
        self.value_score_scale = float(self.meta.get("value_score_scale", 150.0))
        self.value_score_bias = float(self.meta.get("value_score_bias", 0.0))
        self.has_move_value_head = "W_move_value" in z and "b_move_value" in z
        self.W_move_value = z["W_move_value"] if self.has_move_value_head else None
        self.b_move_value = float(z["b_move_value"][0]) if self.has_move_value_head else 0.0
        self.move_value_blend_alpha = float(self.meta.get("move_value_inference_blend_alpha", 0.35))
        self.move_value_min_pair_acc = float(self.meta.get("move_value_inference_min_pair_acc", 0.52))
        self.move_value_min_margin = float(self.meta.get("move_value_inference_min_margin", 0.01))
        self.use_move_value_head = self._infer_move_value_reliability()

    def _infer_move_value_reliability(self) -> bool:
        """Enable move-value at inference only when validation ranking is strong."""
        if not self.has_move_value_head or self.W_move_value is None:
            return False

        hist = self.meta.get("history")
        if not isinstance(hist, list) or not hist:
            return False

        best_epoch = int(self.meta.get("best_val_loss_epoch", 0))
        best_row = None
        if best_epoch > 0:
            for row in hist:
                if int(row.get("epoch", 0)) == best_epoch:
                    best_row = row
                    break
        if best_row is None:
            best_row = hist[-1]

        pair_acc = float(best_row.get("val_move_pair_acc", 0.0))
        margin = float(best_row.get("val_move_rank_margin_mean", 0.0))
        return pair_acc >= self.move_value_min_pair_acc and margin >= self.move_value_min_margin

    def _apply_batch_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
    ) -> np.ndarray:
        inv_std = 1.0 / np.sqrt(np.maximum(running_var, 0.0) + self.batch_norm_eps)
        x_hat = (x - running_mean) * inv_std
        return x_hat * gamma + beta

    def forward(self, state: np.ndarray) -> tuple[dict[str, np.ndarray], float | None]:
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim != 1:
            raise ValueError(f"FactorizedPolicyModel.forward expects 1D state; got shape {state_arr.shape}")
        if int(state_arr.shape[0]) != self.feature_dim:
            raise ValueError(
                f"State feature dimension mismatch: model expects {self.feature_dim}, got {int(state_arr.shape[0])}"
            )

        h1_pre = state_arr @ self.W1 + self.b1
        if (
            self.has_bn1
            and self.bn1_gamma is not None
            and self.bn1_beta is not None
            and self.bn1_running_mean is not None
            and self.bn1_running_var is not None
        ):
            h1_pre = self._apply_batch_norm(
                h1_pre,
                self.bn1_gamma,
                self.bn1_beta,
                self.bn1_running_mean,
                self.bn1_running_var,
            )
        h1 = np.maximum(h1_pre, 0.0)
        if self.has_second_layer and self.W2 is not None and self.b2 is not None:
            h2_pre = h1 @ self.W2 + self.b2
            if (
                self.has_bn2
                and self.bn2_gamma is not None
                and self.bn2_beta is not None
                and self.bn2_running_mean is not None
                and self.bn2_running_var is not None
            ):
                h2_pre = self._apply_batch_norm(
                    h2_pre,
                    self.bn2_gamma,
                    self.bn2_beta,
                    self.bn2_running_mean,
                    self.bn2_running_var,
                )
            h = np.maximum(h2_pre, 0.0)
        else:
            h = h1
        logits = {hn: h @ self.head_W[hn] + self.head_b[hn] for hn in self.head_W}
        value = None
        if self.has_value_head:
            vr = float(h @ self.W_value + self.b_value)
            if self.value_prediction_mode in {"score_linear", "score_norm_linear"}:
                value = vr
            else:
                value = 1.0 / (1.0 + np.exp(-vr))
        return logits, value

    def value_to_expected_score(self, value: float | None) -> float:
        if value is None:
            return 0.0
        if self.value_prediction_mode == "score_linear":
            return float(value)
        if self.value_prediction_mode == "score_norm_linear":
            return float(self.value_score_bias + self.value_score_scale * float(value))
        return float(self.value_score_bias + self.value_score_scale * float(value))

    def score_move(
        self,
        state: np.ndarray,
        move: Move,
        player: Player,
        logits: dict[str, np.ndarray] | None = None,
    ) -> float:
        """Score a move; blend move-value with logits when move head is reliable."""
        if logits is None:
            logits, _ = self.forward(state.astype(np.float32, copy=False))
        base_score = score_move_with_factorized_model(logits, move, player)

        if self.use_move_value_head and self.W_move_value is not None:
            move_f = np.asarray(encode_move_features(move, player), dtype=np.float32)
            x = np.concatenate([state.astype(np.float32, copy=False), move_f], axis=0)
            move_value_score = float(x @ self.W_move_value + self.b_move_value)
            return float(base_score + (self.move_value_blend_alpha * move_value_score))
        return base_score


def score_move_with_factorized_model(
    logits: dict[str, np.ndarray],
    move: Move,
    player: Player,
) -> float:
    """Score a move from factorized logits by summing relevant head logits."""
    t = encode_factorized_targets(move, player)
    action_id = int(t["action_type"])

    score = float(logits["action_type"][action_id])
    for hn in RELEVANT_HEADS_BY_ACTION.get(action_id, []):
        tv = int(t[hn])
        score += float(logits[hn][tv])
    return score


# ─── Optional PyTorch MPS acceleration ───────────────────────────────────────

if _TORCH_AVAILABLE:
    class _TorchInferenceNet(_nn.Module):
        """Minimal PyTorch MLP for fast inference (mirrors BCModel architecture)."""

        def __init__(
            self,
            feature_dim: int,
            hidden1: int,
            hidden2: int,
            head_dims: dict[str, int],
            has_value: bool,
            has_bn1: bool,
            has_bn2: bool,
            batch_norm_eps: float = 1e-5,
        ):
            super().__init__()
            self.fc1 = _nn.Linear(feature_dim, hidden1)
            self.bn1 = _nn.BatchNorm1d(hidden1, eps=batch_norm_eps) if has_bn1 else None
            self.fc2 = _nn.Linear(hidden1, hidden2)
            self.bn2 = _nn.BatchNorm1d(hidden2, eps=batch_norm_eps) if has_bn2 else None
            self.heads = _nn.ModuleDict(
                {k: _nn.Linear(hidden2, d) for k, d in head_dims.items()}
            )
            self.value_head = _nn.Linear(hidden2, 1) if has_value else None

        def forward(self, x: "_torch.Tensor"):
            h = self.fc1(x)
            if self.bn1 is not None:
                h = self.bn1(h)
            h = _Func.relu(h)
            h = self.fc2(h)
            if self.bn2 is not None:
                h = self.bn2(h)
            h = _Func.relu(h)
            logits = {k: head(h) for k, head in self.heads.items()}
            value = self.value_head(h).squeeze(-1) if self.value_head is not None else None
            return logits, value

    class FactorizedPolicyModelTorch(FactorizedPolicyModel):
        """Drop-in replacement using PyTorch MPS (Apple Silicon GPU) for fast inference.

        On M1/M2/M3 Macs, MPS reduces forward-pass time dramatically compared to
        numpy BLAS in multiprocessing workers (no thread contention).
        Supports batched inference via ``forward_batch()`` for MCTS throughput.
        """

        def __init__(self, path: str | Path, device: str = "auto"):
            super().__init__(path)
            self._init_torch(device)

        def _init_torch(self, device: str) -> None:
            """Build the PyTorch model using already-loaded numpy weights."""
            if not self.has_second_layer or self.W2 is None:
                raise ValueError(
                    "FactorizedPolicyModelTorch requires a 2-layer model (W2/b2 missing)"
                )

            hidden1 = int(self.W1.shape[1])
            hidden2 = int(self.W2.shape[1])

            net = _TorchInferenceNet(
                feature_dim=self.feature_dim,
                hidden1=hidden1,
                hidden2=hidden2,
                head_dims=dict(self.head_dims),
                has_value=self.has_value_head,
                has_bn1=self.has_bn1,
                has_bn2=self.has_bn2,
                batch_norm_eps=self.batch_norm_eps,
            )

            with _torch.no_grad():
                # npz stores W as (in_dim, out_dim); PyTorch Linear.weight is (out_dim, in_dim)
                net.fc1.weight.data = _torch.from_numpy(self.W1.T.astype(np.float32))
                net.fc1.bias.data = _torch.from_numpy(self.b1.astype(np.float32))
                net.fc2.weight.data = _torch.from_numpy(self.W2.T.astype(np.float32))
                net.fc2.bias.data = _torch.from_numpy(self.b2.astype(np.float32))

                if self.has_bn1 and self.bn1_gamma is not None:
                    net.bn1.weight.data = _torch.from_numpy(self.bn1_gamma.astype(np.float32))
                    net.bn1.bias.data = _torch.from_numpy(self.bn1_beta.astype(np.float32))
                    net.bn1.running_mean.copy_(_torch.from_numpy(self.bn1_running_mean.astype(np.float32)))
                    net.bn1.running_var.copy_(_torch.from_numpy(self.bn1_running_var.astype(np.float32)))

                if self.has_bn2 and self.bn2_gamma is not None:
                    net.bn2.weight.data = _torch.from_numpy(self.bn2_gamma.astype(np.float32))
                    net.bn2.bias.data = _torch.from_numpy(self.bn2_beta.astype(np.float32))
                    net.bn2.running_mean.copy_(_torch.from_numpy(self.bn2_running_mean.astype(np.float32)))
                    net.bn2.running_var.copy_(_torch.from_numpy(self.bn2_running_var.astype(np.float32)))

                for hn, W_h in self.head_W.items():
                    net.heads[hn].weight.data = _torch.from_numpy(W_h.T.astype(np.float32))
                    net.heads[hn].bias.data = _torch.from_numpy(self.head_b[hn].astype(np.float32))

                if self.has_value_head and self.W_value is not None:
                    # W_value in npz is (hidden2,); Linear(hidden2, 1).weight is (1, hidden2)
                    net.value_head.weight.data = _torch.from_numpy(
                        np.asarray(self.W_value, dtype=np.float32)
                    ).unsqueeze(0)
                    net.value_head.bias.data = _torch.tensor(
                        [self.b_value], dtype=_torch.float32
                    )

            if device == "auto":
                if _torch.backends.mps.is_available():
                    self._device = _torch.device("mps")
                elif _torch.cuda.is_available():
                    self._device = _torch.device("cuda")
                else:
                    self._device = _torch.device("cpu")
            else:
                self._device = _torch.device(device)

            self._net = net.to(self._device)
            self._net.eval()

        def forward(
            self, state: np.ndarray
        ) -> tuple[dict[str, np.ndarray], float | None]:
            """Single-sample forward pass on MPS/GPU (or CPU fallback)."""
            state_arr = np.asarray(state, dtype=np.float32)
            with _torch.no_grad():
                x = _torch.from_numpy(state_arr).to(self._device).unsqueeze(0)
                logits_t, value_t = self._net(x)
                logits = {k: v[0].cpu().numpy() for k, v in logits_t.items()}
                value: float | None = None
                if value_t is not None:
                    vr = float(value_t[0].cpu().item())
                    if self.value_prediction_mode in {"score_linear", "score_norm_linear"}:
                        value = vr
                    else:
                        value = float(1.0 / (1.0 + _math.exp(-vr)))
            return logits, value

        def forward_batch(
            self, states: np.ndarray
        ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
            """Batched forward pass on a (N, feature_dim) array.

            Returns:
                logits: dict of (N, head_dim) arrays
                values: (N,) array of value estimates (post-sigmoid if applicable), or None
            """
            states_arr = np.asarray(states, dtype=np.float32)
            with _torch.no_grad():
                x = _torch.from_numpy(states_arr).to(self._device)
                logits_t, value_t = self._net(x)
                logits = {k: v.cpu().numpy() for k, v in logits_t.items()}
                values: np.ndarray | None = None
                if value_t is not None:
                    raw = value_t.cpu().numpy()
                    if self.value_prediction_mode in {"score_linear", "score_norm_linear"}:
                        values = raw
                    else:
                        values = 1.0 / (1.0 + np.exp(-raw))
            return logits, values


def load_policy_model(
    path: str | Path,
    prefer_torch: bool = True,
    device: str = "auto",
) -> "FactorizedPolicyModel":
    """Load a factorized policy model, preferring PyTorch MPS inference when available.

    PyTorch on MPS (Apple Silicon) gives a large speedup over numpy in multiprocessing
    workers (avoids BLAS thread contention between parallel processes).
    Falls back to numpy if PyTorch is unavailable or fails to initialize.
    """
    if prefer_torch and _TORCH_AVAILABLE:
        try:
            return FactorizedPolicyModelTorch(path, device=device)  # type: ignore[return-value]
        except Exception:
            pass
    return FactorizedPolicyModel(path)
