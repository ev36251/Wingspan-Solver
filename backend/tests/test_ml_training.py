from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.self_play_dataset import generate_dataset
from backend.ml.train_policy_value import train


def test_train_policy_value_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "sp.jsonl"
    meta = tmp_path / "sp.meta.json"
    out_model = tmp_path / "model.npz"

    generate_dataset(
        output_jsonl=str(dataset),
        metadata_path=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=13,
    )

    result = train(
        dataset_jsonl=str(dataset),
        metadata_json=str(meta),
        out_model=str(out_model),
        epochs=1,
        batch_size=32,
        hidden_dim=64,
        learning_rate=1e-3,
        value_loss_weight=0.5,
        val_split=0.2,
        seed=13,
    )

    assert out_model.exists()
    assert result["train_samples"] > 0
    assert result["val_samples"] > 0
    assert len(result["history"]) == 1
