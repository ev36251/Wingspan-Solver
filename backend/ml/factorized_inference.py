"""Inference helpers for factorized behavioral-cloning models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.models.player import Player
from backend.solver.move_generator import Move
from backend.ml.factorized_policy import encode_factorized_targets, RELEVANT_HEADS_BY_ACTION
from backend.ml.move_features import encode_move_features


class FactorizedPolicyModel:
    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=True)
        self.W1 = z["W1"]
        self.b1 = z["b1"]
        self.has_second_layer = "W2" in z and "b2" in z
        self.W2 = z["W2"] if self.has_second_layer else None
        self.b2 = z["b2"] if self.has_second_layer else None

        meta_json = z["metadata_json"][0]
        if isinstance(meta_json, bytes):
            meta_json = meta_json.decode("utf-8")
        self.meta = json.loads(str(meta_json))

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

    def forward(self, state: np.ndarray) -> tuple[dict[str, np.ndarray], float | None]:
        h1_pre = state @ self.W1 + self.b1
        h1 = np.maximum(h1_pre, 0.0)
        if self.has_second_layer and self.W2 is not None and self.b2 is not None:
            h2_pre = h1 @ self.W2 + self.b2
            h = np.maximum(h2_pre, 0.0)
        else:
            h = h1
        logits = {hn: h @ self.head_W[hn] + self.head_b[hn] for hn in self.head_W}
        value = None
        if self.has_value_head:
            vr = float(h @ self.W_value + self.b_value)
            if self.value_prediction_mode == "score_linear":
                value = vr
            else:
                value = 1.0 / (1.0 + np.exp(-vr))
        return logits, value

    def value_to_expected_score(self, value: float | None) -> float:
        if value is None:
            return 0.0
        if self.value_prediction_mode == "score_linear":
            return float(value)
        return float(self.value_score_bias + self.value_score_scale * float(value))

    def score_move(
        self,
        state: np.ndarray,
        move: Move,
        player: Player,
        logits: dict[str, np.ndarray] | None = None,
    ) -> float:
        """Score a move with move-value head when available, else legacy logits."""
        if self.has_move_value_head and self.W_move_value is not None:
            move_f = np.asarray(encode_move_features(move, player), dtype=np.float32)
            x = np.concatenate([state.astype(np.float32, copy=False), move_f], axis=0)
            return float(x @ self.W_move_value + self.b_move_value)
        if logits is None:
            logits, _ = self.forward(state.astype(np.float32, copy=False))
        return score_move_with_factorized_model(logits, move, player)


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
