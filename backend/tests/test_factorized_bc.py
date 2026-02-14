from pathlib import Path

import json
import random
import numpy as np

from backend.models.bird import Bird, FoodCost
from backend.models.enums import (
    ActionType, BeakDirection, BoardType, FoodType, GameSet, Habitat, NestType, PowerColor,
)
from backend.models.player import Player
from backend.solver.move_generator import Move
from backend.ml.factorized_policy import encode_factorized_targets
from backend.ml.factorized_inference import FactorizedPolicyModel, score_move_with_factorized_model
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.move_features import MOVE_FEATURE_DIM, encode_move_features
from backend.ml.train_factorized_bc import train_bc, _grad_softplus_neg_delta


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
    assert out["format_version"] == 5
    assert out["has_move_value_head"] is True
    assert out["batch_norm_enabled"] is True
    z = np.load(model, allow_pickle=True)
    assert "W_move_value" in z
    assert "b_move_value" in z
    assert "bn1_gamma" in z
    assert "bn1_beta" in z
    assert "bn1_running_mean" in z
    assert "bn1_running_var" in z
    assert "bn2_gamma" in z
    assert "bn2_beta" in z
    assert "bn2_running_mean" in z
    assert "bn2_running_var" in z


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
    assert legacy_model.has_move_value_head is False

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
    assert new_model.has_move_value_head is True

    p = Player(name="P1")
    mv = Move(action_type=ActionType.LAY_EGGS, description="lay")
    state = np.zeros((new_model.W1.shape[0],), dtype=np.float32)
    score = new_model.score_move(state, mv, p, logits=logits_new)
    assert np.isfinite(score)


def test_factorized_inference_raises_on_state_dim_mismatch(tmp_path: Path) -> None:
    head_dims = {
        "action_type": 4,
        "play_habitat": 4,
        "gain_food_primary": 7,
        "draw_mode": 4,
        "lay_eggs_bin": 11,
        "play_cost_bin": 7,
        "play_power_color": 7,
    }
    p = tmp_path / "mismatch_model.npz"
    meta = {"head_dims": head_dims, "value_prediction_mode": "none"}
    out = {
        "W1": np.zeros((10, 8), dtype=np.float32),
        "b1": np.zeros((8,), dtype=np.float32),
        "metadata_json": np.asarray([json.dumps(meta)], dtype=object),
    }
    for hn, d in head_dims.items():
        out[f"W_{hn}"] = np.zeros((8, d), dtype=np.float32)
        out[f"b_{hn}"] = np.zeros((d,), dtype=np.float32)
    np.savez_compressed(p, **out)

    model = FactorizedPolicyModel(p)
    try:
        model.forward(np.zeros((11,), dtype=np.float32))
        assert False, "expected ValueError for mismatched state dimension"
    except ValueError as exc:
        msg = str(exc)
        assert "expects 10" in msg
        assert "got 11" in msg


def test_encode_move_features_shape_and_blocks() -> None:
    bird = Bird(
        name="Test Bird",
        scientific_name="Testus birdus",
        game_set=GameSet.CORE,
        color=PowerColor.BROWN,
        power_text="",
        victory_points=7,
        nest_type=NestType.BOWL,
        egg_limit=4,
        wingspan_cm=25,
        habitats=frozenset({Habitat.FOREST}),
        food_cost=FoodCost(items=(FoodType.SEED, FoodType.FISH), total=2),
        beak_direction=BeakDirection.LEFT,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )
    p = Player(name="P1", hand=[bird])

    play_move = Move(
        action_type=ActionType.PLAY_BIRD,
        description="play",
        bird_name=bird.name,
        habitat=Habitat.FOREST,
    )
    f_play = encode_move_features(play_move, p)
    assert len(f_play) == MOVE_FEATURE_DIM
    assert f_play[0] == 1.0  # play action one-hot
    assert f_play[4] == 0.7  # vp / 10
    assert f_play[5] == (2.0 / 6.0)  # cost / 6
    assert f_play[6] == (4.0 / 6.0)  # egg cap / 6
    assert sum(f_play[7:14]) == 1.0  # power color one-hot
    assert sum(f_play[14:21]) == 0.0  # gain block off

    gain_move = Move(
        action_type=ActionType.GAIN_FOOD,
        description="gain",
        food_choices=[FoodType.FRUIT, FoodType.SEED],
    )
    f_gain = encode_move_features(gain_move, p)
    assert len(f_gain) == MOVE_FEATURE_DIM
    assert f_gain[1] == 1.0  # gain action one-hot
    assert sum(f_gain[4:14]) == 0.0  # play block off
    assert sum(f_gain[14:21]) == 1.0  # food one-hot on

    eggs_move = Move(action_type=ActionType.LAY_EGGS, description="eggs")
    draw_move = Move(action_type=ActionType.DRAW_CARDS, description="draw")
    f_eggs = encode_move_features(eggs_move, p)
    f_draw = encode_move_features(draw_move, p)
    assert sum(f_eggs[4:21]) == 0.0
    assert sum(f_draw[4:21]) == 0.0


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
        early_stop_min_delta=1e9,
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


def test_train_bc_batch_norm_disable_compatibility(tmp_path: Path) -> None:
    ds = tmp_path / "bn_off.jsonl"
    meta = tmp_path / "bn_off.meta.json"
    model = tmp_path / "bn_off_model.npz"

    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=101,
    )

    out = train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden1=32,
        hidden2=16,
        batch_norm_enabled=False,
        val_split=0.2,
        seed=101,
    )
    assert out["batch_norm_enabled"] is False

    z = np.load(model, allow_pickle=True)
    assert "bn1_gamma" not in z
    assert "bn2_gamma" not in z

    inf = FactorizedPolicyModel(model)
    logits, _ = inf.forward(np.zeros((inf.feature_dim,), dtype=np.float32))
    assert "action_type" in logits


def test_grad_softplus_neg_delta_is_stable_for_large_values() -> None:
    vals = [1e6, 1e4, 100.0, 0.0, -100.0, -1e4, -1e6]
    grads = [_grad_softplus_neg_delta(v) for v in vals]
    for g in grads:
        assert np.isfinite(g)
        assert -1.0 <= g <= 0.0
    assert grads[0] == 0.0
    assert grads[-1] == -1.0


def test_score_move_falls_back_to_logits_when_move_head_unreliable(tmp_path: Path) -> None:
    head_dims = {
        "action_type": 4,
        "play_habitat": 4,
        "gain_food_primary": 7,
        "draw_mode": 4,
        "lay_eggs_bin": 11,
        "play_cost_bin": 7,
        "play_power_color": 7,
    }
    p = tmp_path / "mv_unreliable.npz"
    meta = {
        "head_dims": head_dims,
        "best_val_loss_epoch": 1,
        "history": [
            {
                "epoch": 1,
                "val_move_pair_acc": 0.2,
                "val_move_rank_margin_mean": -0.1,
            }
        ],
    }
    out = {
        "W1": np.zeros((10, 8), dtype=np.float32),
        "b1": np.zeros((8,), dtype=np.float32),
        "W2": np.zeros((8, 8), dtype=np.float32),
        "b2": np.zeros((8,), dtype=np.float32),
        "W_move_value": np.ones((10 + MOVE_FEATURE_DIM,), dtype=np.float32),
        "b_move_value": np.asarray([3.0], dtype=np.float32),
        "metadata_json": np.asarray([json.dumps(meta)], dtype=object),
    }
    for hn, d in head_dims.items():
        out[f"W_{hn}"] = np.zeros((8, d), dtype=np.float32)
        out[f"b_{hn}"] = np.zeros((d,), dtype=np.float32)
    np.savez_compressed(p, **out)

    model = FactorizedPolicyModel(p)
    assert model.use_move_value_head is False
    player = Player(name="P1")
    move = Move(action_type=ActionType.LAY_EGGS, description="lay")
    state = np.zeros((10,), dtype=np.float32)
    logits, _ = model.forward(state)
    expected = score_move_with_factorized_model(logits, move, player)
    got = model.score_move(state, move, player, logits=logits)
    assert abs(got - expected) < 1e-6


def test_train_margin_mean_uses_only_rows_with_move_pairs(tmp_path: Path) -> None:
    ds = tmp_path / "toy_margin.jsonl"
    meta = tmp_path / "toy_margin.meta.json"
    model = tmp_path / "toy_margin_model.npz"
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

    ranked_rows = []
    plain_rows = []
    for _ in range(2):
        ranked_rows.append(
            {
                "state": [1.0, 0.0, 0.0, 0.0],
                "targets": {
                    "action_type": 0,
                    "play_habitat": 0,
                    "gain_food_primary": 0,
                    "draw_mode": 0,
                    "lay_eggs_bin": 0,
                    "play_cost_bin": 0,
                    "play_power_color": 0,
                },
                "value_target_score": 80.0,
                "move_pos": [1.0] * MOVE_FEATURE_DIM,
                "move_negs": [[0.0] * MOVE_FEATURE_DIM],
            }
        )
    for i in range(18):
        plain_rows.append(
            {
                "state": [0.0, 1.0, 0.0, 0.0],
                "targets": {
                    "action_type": i % 4,
                    "play_habitat": i % 4,
                    "gain_food_primary": i % 7,
                    "draw_mode": i % 4,
                    "lay_eggs_bin": i % 11,
                    "play_cost_bin": i % 7,
                    "play_power_color": i % 7,
                },
                "value_target_score": 60.0,
                "move_pos": [0.0] * MOVE_FEATURE_DIM,
                "move_negs": [],
            }
        )

    rows = ranked_rows + plain_rows
    ds.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=64,
        hidden1=16,
        hidden2=8,
        dropout=0.0,
        lr_init=0.0,
        lr_peak=0.0,
        lr_warmup_epochs=0,
        lr_decay_every=0,
        lr_decay_factor=1.0,
        val_split=0.05,
        seed=123,
        value_loss_weight=0.0,
        move_value_enabled=True,
        move_value_loss_weight=0.2,
    )
    assert out["history"]
    reported = float(out["history"][0]["train_move_rank_margin_mean"])

    shuffled = list(rows)
    random.Random(123).shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * 0.05))
    train_rows = shuffled[val_n:] if len(shuffled) > val_n else shuffled
    ranked_train = [r for r in train_rows if r.get("move_negs")]
    assert ranked_train, "expected ranked rows in train split"

    z = np.load(model, allow_pickle=True)
    wmv = z["W_move_value"]
    bmv = float(z["b_move_value"][0])
    margins = []
    for r in ranked_train:
        x = np.asarray(r["state"], dtype=np.float32)
        pos = np.asarray(r["move_pos"], dtype=np.float32)
        neg = np.asarray(r["move_negs"][0], dtype=np.float32)
        s_pos = float(np.concatenate([x, pos], axis=0) @ wmv + bmv)
        s_neg = float(np.concatenate([x, neg], axis=0) @ wmv + bmv)
        margins.append(s_pos - s_neg)
    expected = float(sum(margins) / len(margins))
    assert abs(reported - expected) < 1e-5
