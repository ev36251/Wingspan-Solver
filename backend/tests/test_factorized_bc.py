from pathlib import Path

from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.solver.move_generator import Move
from backend.ml.factorized_policy import encode_factorized_targets
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
        hidden=64,
        lr=1e-3,
        val_split=0.2,
        seed=5,
    )

    assert ds.exists()
    assert meta.exists()
    assert model.exists()
    assert out["train_samples"] > 0
    assert out["val_samples"] > 0
