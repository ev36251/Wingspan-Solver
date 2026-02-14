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
        value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        strict_kpi_gate_enabled=False,
        seed=3,
    )

    assert (out_dir / "auto_improve_factorized_manifest.json").exists()
    best_info = manifest.get("best") or {}
    best_path = best_info.get("path")
    if best_path:
        assert Path(best_path).exists()
    else:
        assert not (out_dir / "best_model.npz").exists()
    assert manifest["config"]["strict_rules_only"] is True
    assert manifest["config"]["reject_non_strict_powers"] is True
    assert manifest["config"]["strict_kpi_gate_enabled"] is False
    assert manifest["config"]["promotion_primary_opponent"] == "champion"
    assert manifest["config"]["champion_self_play_enabled"] is True
    assert "train_early_stop_enabled" in manifest["config"]
    assert "train_early_stop_patience" in manifest["config"]
    assert "train_early_stop_min_delta" in manifest["config"]
    assert "train_early_stop_restore_best" in manifest["config"]
    assert manifest["config"]["train_hidden1"] == 384
    assert manifest["config"]["train_hidden2"] == 192
    assert manifest["config"]["train_dropout"] == 0.2
    assert manifest["config"]["train_lr_peak"] == 5e-4
    assert manifest["config"]["train_lr_warmup_epochs"] == 3
    assert manifest["config"]["train_lr_decay_every"] == 5
    assert manifest["config"]["train_lr_decay_factor"] == 0.7
    assert "data_accumulation_enabled" in manifest["config"]
    assert "data_accumulation_decay" in manifest["config"]
    assert "max_accumulated_samples" in manifest["config"]
    assert "promotion_primary_eval" in manifest["history"][0]
    assert len(manifest["history"]) == 1
