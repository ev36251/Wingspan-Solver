"""Modal.com cloud runner for AlphaZero self-play shards.

Dispatches self-play data-generation workers to Modal containers so that
dozens of shards can run in parallel without being limited by local CPU count.
Training, eval, and the promotion gate continue to run locally (they're fast
or benefit from the MPS GPU).

Setup (one-time):
    pip install modal
    modal setup          # opens browser for authentication

Usage:
    Pass --use-modal to auto_improve_alphazero.py.  Example:

    python -m backend.ml.auto_improve_alphazero \\
        --out-dir reports/ml/alphazero_v10 \\
        --iterations 40 --games-per-iter 200 --mcts-sims 50 \\
        --dataset-workers 32 --use-modal \\
        --value-blend 1.0 --promotion-games 40 \\
        --use-per-slot-encoding --use-hand-habitat-features \\
        --train-init-model-path reports/ml/alphazero_v9/best_model.npz \\
        --train-lr-init 0.00002 --train-lr-peak 0.0001 \\
        --train-lr-warmup-epochs 5 --train-early-stop-patience 10 \\
        --gate-mode champion

Notes:
    - Each Modal worker runs one shard (N games). Use --dataset-workers to
      control how many shards (and thus Modal containers) run in parallel.
    - Modal bills per CPU-second.  At 2 vCPU / shard and ~120 s/game (50 sims,
      value-blend 1.0) ≈ $0.0001 per game.  200 games ≈ $0.02 / iteration.
    - The model weights (~9 MB) are serialised into each task dict; no Volume
      is required for inference weights.
    - Output JSONL files are returned as strings and written to local disk by
      the coordinator; only the local machine needs persistent storage.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate project files (needed for image build-time mounts)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent        # backend/ml/
_BACKEND = _HERE.parent              # backend/
_PROJECT = _BACKEND.parent           # project root (where xlsx lives)
_XLSX = _PROJECT / "wingspan-20260128.xlsx"

# ---------------------------------------------------------------------------
# Modal app definition (only evaluated when modal is installed)
# ---------------------------------------------------------------------------
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore


def _build_image():
    """Build the Modal container image with all dependencies.

    Modal auto-mounts the `backend` package to /root/backend/ (detected from
    the function file's location inside the package).  config.py therefore
    resolves PROJECT_ROOT = /root, so the xlsx must live at /root/wingspan-…

    Modal rule: add_local_* must be the LAST steps in the chain.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy",
            "openpyxl",
            "threadpoolctl",
        )
        # xlsx path that config.py expects: Path(__file__).parent.parent / xlsx
        # __file__ = /root/backend/config.py  →  parent.parent = /root
        .add_local_file(str(_XLSX), remote_path="/root/wingspan-20260128.xlsx")
    )


if _MODAL_AVAILABLE:
    _image = _build_image()
    app = modal.App("wingspan-selfplay", image=_image)

    @app.function(cpu=2, memory=4096, timeout=7200)
    def run_az_shard_remote(task: dict) -> dict:
        """Run one self-play shard inside a Modal container.

        The task dict mirrors _run_az_shard's contract, with one addition:
            task["model_bytes"]: bytes  — raw content of the .npz model file.

        Returns:
            {"jsonl_content": str, "meta": dict}
        """
        import os
        import sys
        import tempfile

        # Single-threaded BLAS to avoid overhead on tiny GEMV ops
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass

        from backend.ml.alphazero_self_play import generate_self_play_dataset
        from backend.models.enums import BoardType

        # Write model bytes to a temp file so existing code can load it
        model_bytes: bytes = task.pop("model_bytes")
        model_tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        model_tmp.write(model_bytes)
        model_tmp.close()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_jsonl = os.path.join(tmpdir, "shard.jsonl")
                out_meta_path = os.path.join(tmpdir, "shard.meta.json")

                meta = generate_self_play_dataset(
                    model_path=model_tmp.name,
                    out_jsonl=out_jsonl,
                    out_meta=out_meta_path,
                    games=task["games"],
                    players=task["players"],
                    board_type=BoardType(task["board_type"]),
                    mcts_sims=task["mcts_sims"],
                    c_puct=task["c_puct"],
                    value_blend=task["value_blend"],
                    rollout_policy=task["rollout_policy"],
                    temperature_cutoff=task["temperature_cutoff"],
                    seed=task["seed"],
                    max_turns=task["max_turns"],
                    strict_rules_mode=task["strict_rules_mode"],
                    value_target_score_scale=task["value_target_score_scale"],
                    value_target_score_bias=task["value_target_score_bias"],
                    enable_identity_features=task.get("enable_identity_features"),
                    identity_hash_dim=task.get("identity_hash_dim"),
                    use_per_slot_encoding=task.get("use_per_slot_encoding"),
                    use_hand_habitat_features=task.get("use_hand_habitat_features"),
                    use_tray_per_slot_encoding=task.get("use_tray_per_slot_encoding"),
                    use_opponent_board_encoding=task.get("use_opponent_board_encoding"),
                    use_power_features=task.get("use_power_features"),
                )

                with open(out_jsonl, "r", encoding="utf-8") as f:
                    jsonl_content = f.read()

        finally:
            os.unlink(model_tmp.name)

        return {"jsonl_content": jsonl_content, "meta": meta}


# ---------------------------------------------------------------------------
# Public dispatcher (called from auto_improve_alphazero.py)
# ---------------------------------------------------------------------------

def dispatch_shards_modal(
    shard_tasks: list[dict],
    model_path: str,
    cpu_per_worker: int = 2,
) -> list[dict]:
    """Dispatch shard tasks to Modal and return results in _run_az_shard format.

    Each returned dict:  {"jsonl": str, "meta_path": str, "meta": dict}

    The caller (_merge_az_shards) reads the local JSONL paths, so this
    function writes the returned content to the paths in each task.

    Args:
        shard_tasks:     List of task dicts (same format as _run_az_shard).
        model_path:      Local path to the .npz champion model.
        cpu_per_worker:  vCPUs per Modal container (passed via app.function
                         override; default 2 matches the decorator).
    """
    if not _MODAL_AVAILABLE:
        raise RuntimeError(
            "Modal is not installed. Run:\n"
            "    pip install modal\n"
            "    modal setup"
        )

    model_bytes = Path(model_path).read_bytes()

    # Embed model bytes into each task dict (Modal serialises with pickle)
    remote_tasks = [{**t, "model_bytes": model_bytes} for t in shard_tasks]

    if remote_tasks:
        print(
            f"  [modal] dispatching {len(remote_tasks)} shards "
            f"({remote_tasks[0].get('games', 0)} games each) …"
        )
    else:
        print("  [modal] no shards to dispatch")

    # Run all shards in parallel on Modal; .map() returns results in order.
    # app.run() registers the ephemeral app with Modal's servers so the
    # function can be hydrated and called from this local Python process.
    with modal.enable_output(), app.run():
        results = list(run_az_shard_remote.map(remote_tasks))

    # Write JSONL content to the local paths that _merge_az_shards expects
    out = []
    for task, result in zip(shard_tasks, results):
        jsonl_path = task["out_jsonl"]
        meta_path = task["out_meta"]
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        Path(jsonl_path).write_text(result["jsonl_content"], encoding="utf-8")
        Path(meta_path).write_text(
            json.dumps(result["meta"], indent=2), encoding="utf-8"
        )
        out.append({"jsonl": jsonl_path, "meta_path": meta_path, "meta": result["meta"]})

    return out
