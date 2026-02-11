from pathlib import Path

import json
import numpy as np

from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.solver.move_generator import Move
from backend.ml.factorized_policy import encode_factorized_targets
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.train_factorized_bc import train_bc


def test_encode_factorized_targets_smoke() -> None:
    move = Move(
        action_type=ActionType.PLAY_BIRD,
        description="x",
        bird_name=None,
        habitat=Habitat.FOREST,
        food_payment={FoodType.SEED: 1, FoodType.NECTAR: 1},
    )

    class DummyPlayer:
        hand = []

    t = encode_factorized_targets(move, DummyPlayer())
    assert t["action_type"] == 0
    assert t["play_habitat"] == 0
    assert t["play_cost_bin"] == 2


def test_generate_and_train_factorized_bc(tmp_path: Path) -> None:
    ds = tmp_path / "bc.jsonl"
    meta = tmp_path / "bc.meta.json"
    model = tmp_path / "bc_model.npz"

    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=5,
    )

    out = train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden1=64,
        hidden2=32,
        dropout=0.1,
        lr_init=1e-4,
        lr_peak=1e-3,
        lr_warmup_epochs=1,
        lr_decay_every=3,
        lr_decay_factor=0.5,
        val_split=0.2,
        seed=5,
    )

    assert ds.exists()
    assert meta.exists()
    assert model.exists()
    assert out["train_samples"] > 0
    assert out["val_samples"] > 0
    assert out["model_arch"] == "mlp_2layer"
    assert out["format_version"] == 3


def test_factorized_inference_loads_legacy_and_new(tmp_path: Path) -> None:
    head_dims = {
        "action_type": 4,
        "play_habitat": 4,
        "gain_food_primary": 7,
        "draw_mode": 4,
        "lay_eggs_bin": 11,
        "play_cost_bin": 7,
        "play_power_color": 7,
    }
    legacy_path = tmp_path / "legacy_model.npz"
    legacy_out = {
        "W1": np.zeros((10, 8), dtype=np.float32),
        "b1": np.zeros((8,), dtype=np.float32),
        "metadata_json": np.asarray(
            [__import__("json").dumps({"head_dims": head_dims, "value_prediction_mode": "none"})],
            dtype=object,
        ),
    }
    for hn, d in head_dims.items():
        legacy_out[f"W_{hn}"] = np.zeros((8, d), dtype=np.float32)
        legacy_out[f"b_{hn}"] = np.zeros((d,), dtype=np.float32)
    np.savez_compressed(legacy_path, **legacy_out)

    legacy_model = FactorizedPolicyModel(legacy_path)
    logits_old, value_old = legacy_model.forward(np.zeros((10,), dtype=np.float32))
    assert "action_type" in logits_old
    assert value_old is None

    ds = tmp_path / "bc.jsonl"
    meta = tmp_path / "bc.meta.json"
    new_model_path = tmp_path / "new_model.npz"
    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=13,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(new_model_path),
        epochs=1,
        batch_size=32,
        hidden1=64,
        hidden2=32,
        val_split=0.2,
        seed=13,
    )
    new_model = FactorizedPolicyModel(new_model_path)
    logits_new, _ = new_model.forward(np.zeros((new_model.W1.shape[0],), dtype=np.float32))
    assert "action_type" in logits_new


def test_train_bc_early_stopping_triggers(tmp_path: Path) -> None:
    ds = tmp_path / "toy.jsonl"
    meta = tmp_path / "toy.meta.json"
    model = tmp_path / "toy_model.npz"
    head_dims = {
        "action_type": 4,
        "play_habitat": 4,
        "gain_food_primary": 7,
        "draw_mode": 4,
        "lay_eggs_bin": 11,
        "play_cost_bin": 7,
        "play_power_color": 7,
    }
    meta.write_text(
        json.dumps(
            {
                "feature_dim": 4,
                "target_heads": head_dims,
                "value_target_config": {"score_scale": 160.0, "score_bias": 0.0},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for i in range(30):
        rows.append(
            {
                "state": [0.1, 0.2, 0.3, 0.4],
                "targets": {
                    "action_type": i % 4,
                    "play_habitat": i % 4,
                    "gain_food_primary": i % 7,
                    "draw_mode": i % 4,
                    "lay_eggs_bin": i % 11,
                    "play_cost_bin": i % 7,
                    "play_power_color": i % 7,
                },
                "value_target_score": float(50 + (i % 5)),
            }
        )
    ds.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=10,
        batch_size=8,
        hidden1=16,
        hidden2=8,
        val_split=0.2,
        seed=42,
        early_stop_enabled=True,
        early_stop_patience=1,
        early_stop_min_delta=10.0,
        early_stop_restore_best=True,
    )
    assert out["stopped_early"] is True
    assert out["epochs_completed"] < 10
    assert out["best_val_loss_epoch"] >= 1

    out_no_es = train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=3,
        batch_size=8,
        hidden1=16,
        hidden2=8,
        val_split=0.2,
        seed=42,
        early_stop_enabled=False,
    )
    assert out_no_es["stopped_early"] is False
    assert out_no_es["epochs_completed"] == 3
