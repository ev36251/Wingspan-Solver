from pathlib import Path

from backend.models.enums import BoardType
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.train_factorized_bc import train_bc
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.auto_improve_factorized import run_auto_improve_factorized


def test_factorized_eval_smoke(tmp_path: Path) -> None:
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
        seed=8,
        lookahead_depth=1,
        n_step=1,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden=64,
        lr=1e-3,
        val_split=0.2,
        seed=8,
    )

    ev = evaluate_factorized_vs_heuristic(
        model_path=str(model),
        games=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=8,
    )
    assert ev.games == 2
    assert ev.nn_wins + ev.heuristic_wins + ev.ties == 2


def test_auto_improve_factorized_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        games_per_iter=2,
        proposal_top_k=3,
        lookahead_depth=1,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        train_epochs=1,
        train_batch=32,
        train_hidden=64,
        train_lr=1e-3,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        promotion_games=4,
        seed=9,
    )

    assert (out_dir / "auto_improve_factorized_manifest.json").exists()
    assert len(manifest["history"]) == 1
    # strict gating field exists
    assert "promotion_gate" in manifest["history"][0]
