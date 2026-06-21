"""Modal.com cloud runner for the 2-player evaluation harness.

Each game is independent, so seeds fan out across containers. A shard runs a few
seeds for one eval mode (heuristic / ablation / selfplay) and returns per-game
result rows; the local side aggregates win rate + significance via
`two_player.summarize`.

Usage:
    python -m backend.ml.two_player --mode ablation --seeds 0-99 \\
        --use-modal --seeds-per-shard 2
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

_HERE = Path(__file__).parent
_BACKEND = _HERE.parent
_PROJECT = _BACKEND.parent
_XLSX = _PROJECT / "wingspan-20260128.xlsx"

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore


def _build_image():
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy", "openpyxl", "threadpoolctl")
        .add_local_file(str(_XLSX), remote_path="/root/wingspan-20260128.xlsx")
    )


if _MODAL_AVAILABLE:
    app = modal.App("wingspan-2p-eval", image=_build_image())

    @app.function(cpu=2, memory=4096, timeout=14400)
    def run_2p_eval_shard_remote(task: dict) -> dict:
        """Run a shard of seeds for one eval mode; return gzipped result rows."""
        import os
        import tempfile
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
        from backend.ml.two_player import play_one

        load_all(EXCEL_FILE)
        mb = task.pop("model_bytes")
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.write(mb); tmp.close()
        try:
            model = FactorizedPolicyModel(tmp.name)
            encoder = StateEncoder.resolve_for_model(model.meta)
            board = BoardType(task["board_type"])
            mode = task["mode"]
            cfg = task["cfg"]
            rows = [play_one(mode, int(s), model, encoder, board, cfg) for s in task["seeds"]]
        finally:
            os.unlink(tmp.name)
        return {"rows_gz": gzip.compress(json.dumps(rows).encode("utf-8"), 6)}


def dispatch_2p_eval_modal(mode, seeds, model_path, board, cfg, seeds_per_shard=2):
    """Fan 2-player eval games across Modal containers; return all result rows."""
    if not _MODAL_AVAILABLE:
        raise RuntimeError("Modal is not installed. Run: pip install modal; modal setup")
    model_bytes = Path(model_path).read_bytes()
    seeds = list(seeds)
    step = max(1, int(seeds_per_shard))
    shards = [seeds[i:i + step] for i in range(0, len(seeds), step)]
    tasks = [
        {"seeds": s, "mode": mode, "board_type": board.value,
         "model_bytes": model_bytes, "cfg": cfg}
        for s in shards
    ]
    print(f"  [modal] dispatching {len(tasks)} shards ({len(seeds)} games, mode={mode}) …")
    all_rows: list[dict] = []
    with modal.enable_output(), app.run():
        for result in run_2p_eval_shard_remote.map(tasks):
            payload = result.get("rows_gz")
            if isinstance(payload, (bytes, bytearray)):
                all_rows.extend(json.loads(gzip.decompress(bytes(payload)).decode("utf-8")))
    return all_rows
