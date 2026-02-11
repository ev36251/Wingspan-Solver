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
        train_hidden1=64,
        train_hidden2=32,
        train_dropout=0.1,
        train_lr_init=1e-4,
        train_lr_peak=1e-3,
        train_lr_warmup_epochs=1,
        train_lr_decay_every=3,
        train_lr_decay_factor=0.5,
        value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        strict_kpi_gate_enabled=False,
        seed=3,
    )

    assert (out_dir / "auto_improve_factorized_manifest.json").exists()
    assert (out_dir / "best_model.npz").exists()
    assert manifest["config"]["strict_rules_only"] is True
    assert manifest["config"]["reject_non_strict_powers"] is True
    assert manifest["config"]["strict_kpi_gate_enabled"] is False
    assert "train_early_stop_enabled" in manifest["config"]
    assert "train_early_stop_patience" in manifest["config"]
    assert "train_early_stop_min_delta" in manifest["config"]
    assert "train_early_stop_restore_best" in manifest["config"]
    assert manifest["best"]["iteration"] == 1
    assert len(manifest["history"]) == 1
