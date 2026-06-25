"""Modal.com cloud runner for solo single-seed optimization.

The solo optimizer is embarrassingly parallel across seeds, so each Modal
container runs a shard of seeds and returns the best line found for each.
Aggregation (analytics + dataset JSONL) happens locally.

Setup (one-time):
    pip install modal
    modal setup          # browser auth

Usage:
    python -m backend.ml.solo_seed_optimizer multi \\
        --seeds 0-99 --games-per-seed 150 --use-modal --seeds-per-shard 1 \\
        --out reports/ml/solo_seed/best_lines.jsonl

Notes:
    - One Modal container per shard; `--seeds-per-shard` controls how many
      seeds each container handles (1 => maximum parallelism).
    - Results (best-line rows) are returned as gzip-compressed JSON; only the
      local machine needs persistent storage.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

_HERE = Path(__file__).parent        # backend/ml/
_BACKEND = _HERE.parent              # backend/
_PROJECT = _BACKEND.parent           # project root (xlsx lives here)
_XLSX = _PROJECT / "wingspan-20260128.xlsx"

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore


def _build_image():
    """Container image: deps + the xlsx at the path config.py expects.

    Modal auto-mounts the `backend` package to /root/backend/, so config.py
    resolves PROJECT_ROOT = /root and the xlsx must live at /root/wingspan-…
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .env({"PYTHONHASHSEED": "0"})  # deterministic hash order -> reproducible games
        .pip_install("numpy", "openpyxl", "threadpoolctl")
        .add_local_file(str(_XLSX), remote_path="/root/wingspan-20260128.xlsx")
    )


if _MODAL_AVAILABLE:
    _image = _build_image()
    app = modal.App("wingspan-solo-seed", image=_image)

    @app.function(cpu=2, memory=4096, timeout=7200)
    def run_solo_shard_remote(task: dict) -> dict:
        """Run a shard of seeds inside a Modal container.

        task = {"seeds": [int], "games_per_seed": int,
                "board_type": str, "draft_top_k": int}
        Returns {"rows_gz": bytes} — gzip-compressed JSON list of best-line rows.
        """
        import os
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass

        from backend.config import EXCEL_FILE
        from backend.data.registries import load_all
        from backend.models.enums import BoardType
        from backend.ml.solo_seed_optimizer import seed_rows

        load_all(EXCEL_FILE)
        board = BoardType(task["board_type"])
        gps = int(task["games_per_seed"])
        top_k = int(task.get("draft_top_k", 12))
        value_samples = int(task.get("value_samples", 3))

        rows = []
        for seed in task["seeds"]:
            rows.extend(seed_rows(int(seed), gps, board, draft_top_k=top_k,
                                  value_samples=value_samples))
        return {"rows_gz": gzip.compress(json.dumps(rows).encode("utf-8"), 6)}

    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_search_shard_remote(task: dict) -> dict:
        """Net-guided rollout SEARCH generation shard (flywheel iteration data).

        task = {"seeds": [int], "board_type": str, "model_bytes": bytes,
                "k_schedule": [int], "rollout": str, "n_drafts": int}
        Returns {"rows_gz": bytes} — best search line per seed (role=policy).
        """
        import os, tempfile
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass

        from backend.config import EXCEL_FILE
        from backend.data.registries import load_all
        from backend.models.enums import BoardType
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder
        from backend.ml.solo_search import search_seed_row

        load_all(EXCEL_FILE)
        mb = task.pop("model_bytes")
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.write(mb); tmp.close()
        try:
            model = FactorizedPolicyModel(tmp.name)
            encoder = StateEncoder.resolve_for_model(model.meta)
            board = BoardType(task["board_type"])
            k_schedule = list(task["k_schedule"])
            rollout = task.get("rollout", "net")
            n_drafts = int(task.get("n_drafts", 3))
            n_rollouts = int(task.get("n_rollouts", 1))
            rollout_temp = float(task.get("rollout_temp", 0.7))
            rows = []
            for seed in task["seeds"]:
                row = search_seed_row(int(seed), model, encoder, k_schedule,
                                      rollout, n_drafts, board,
                                      n_rollouts=n_rollouts, rollout_temp=rollout_temp)
                if row is not None:
                    rows.append(row)
        finally:
            os.unlink(tmp.name)
        return {"rows_gz": gzip.compress(json.dumps(rows).encode("utf-8"), 6)}


def dispatch_search_modal(seeds, model_path, board_type, k_schedule,
                          rollout="net", n_drafts=3, seeds_per_shard=10,
                          n_rollouts=1, rollout_temp=0.7) -> list[dict]:
    """Fan net-guided search generation across Modal containers."""
    if not _MODAL_AVAILABLE:
        raise RuntimeError("Modal is not installed. Run: pip install modal; modal setup")
    model_bytes = Path(model_path).read_bytes()
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [
        {
            "seeds": s, "board_type": board_type.value, "model_bytes": model_bytes,
            "k_schedule": list(k_schedule), "rollout": rollout, "n_drafts": int(n_drafts),
            "n_rollouts": int(n_rollouts), "rollout_temp": float(rollout_temp),
        }
        for s in shards
    ]
    print(f"  [modal] dispatching {len(tasks)} search shards "
          f"({len(seeds)} seeds, k_schedule={list(k_schedule)}, n_drafts={n_drafts}) …")
    all_rows: list[dict] = []
    with modal.enable_output(), app.run():
        for result in run_search_shard_remote.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                all_rows.extend(json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return all_rows


def dispatch_solo_modal(seeds, games_per_seed: int, board_type,
                        seeds_per_shard: int = 1, draft_top_k: int = 12,
                        value_samples: int = 3, cpu_per_worker: int = 2) -> list[dict]:
    """Fan seeds out across Modal containers; return all rows (best + value-only)."""
    if not _MODAL_AVAILABLE:
        raise RuntimeError(
            "Modal is not installed. Run:\n    pip install modal\n    modal setup"
        )

    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [
        {
            "seeds": s,
            "games_per_seed": int(games_per_seed),
            "board_type": board_type.value,
            "draft_top_k": int(draft_top_k),
            "value_samples": int(value_samples),
        }
        for s in shards
    ]
    print(f"  [modal] dispatching {len(tasks)} shards "
          f"({len(seeds)} seeds, {games_per_seed} games/seed) …")

    remote_fn = run_solo_shard_remote
    requested_cpu = max(1, int(cpu_per_worker))
    if requested_cpu != 2:
        remote_fn = remote_fn.with_options(cpu=requested_cpu)

    all_rows: list[dict] = []
    with modal.enable_output(), app.run():
        for result in remote_fn.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                all_rows.extend(json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return all_rows
