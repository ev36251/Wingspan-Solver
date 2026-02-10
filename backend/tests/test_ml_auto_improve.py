from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.auto_improve import run_auto_improve


def test_auto_improve_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto"
    manifest = run_auto_improve(
        out_dir=str(out_dir),
        iterations=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        games_per_iter=2,
        train_epochs=1,
        train_batch=32,
        train_hidden=64,
        train_lr=1e-3,
        value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        seed=3,
    )

    assert (out_dir / "auto_improve_manifest.json").exists()
    assert (out_dir / "best_model.npz").exists()
    assert manifest["best"]["iteration"] == 1
    assert len(manifest["history"]) == 1
