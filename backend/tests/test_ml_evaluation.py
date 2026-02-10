from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.self_play_dataset import generate_dataset
from backend.ml.train_policy_value import train
from backend.ml.evaluate_policy_value import evaluate_nn_vs_heuristic


def test_eval_nn_vs_heuristic_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    meta = tmp_path / "dataset.meta.json"
    model = tmp_path / "model.npz"

    generate_dataset(
        output_jsonl=str(dataset),
        metadata_path=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=17,
    )

    train(
        dataset_jsonl=str(dataset),
        metadata_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=16,
        hidden_dim=32,
        learning_rate=1e-3,
        value_loss_weight=0.5,
        val_split=0.2,
        seed=17,
    )

    res = evaluate_nn_vs_heuristic(
        model_path=str(model),
        dataset_meta_path=str(meta),
        games=2,
        board_type=BoardType.OCEANIA,
        max_turns=140,
        seed=17,
    )

    assert res.games == 2
    assert res.nn_wins + res.heuristic_wins + res.ties == 2
