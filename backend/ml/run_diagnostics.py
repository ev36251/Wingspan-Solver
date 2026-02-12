"""Diagnostics helpers for auto-improve ML run artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_dirs(run_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in run_dir.glob("iter_*"):
        if p.is_dir():
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def _iter_stage(iter_dir: Path) -> str:
    has_ds = (iter_dir / "bc_dataset.jsonl").exists()
    has_meta = (iter_dir / "bc_dataset.meta.json").exists()
    has_model = (iter_dir / "factorized_model.npz").exists()
    has_eval = (iter_dir / "eval.json").exists()
    has_gate = (iter_dir / "promotion_gate_eval.json").exists()
    has_pool = (iter_dir / "pool_eval.json").exists()
    has_kpi = (iter_dir / "kpi_gate.json").exists()

    if has_kpi and has_pool and has_gate and has_eval and has_model:
        return "complete"
    if has_eval and has_model:
        return "gating"
    if has_model:
        return "evaluation"
    if has_ds or has_meta:
        return "training"
    return "pending"


def _iter_summary(iter_dir: Path) -> dict:
    meta = _read_json(iter_dir / "bc_dataset.meta.json") or {}
    ev = _read_json(iter_dir / "eval.json") or {}
    gate = _read_json(iter_dir / "promotion_gate_eval.json") or {}

    return {
        "iteration": iter_dir.name,
        "stage": _iter_stage(iter_dir),
        "samples": int(meta.get("samples", 0)),
        "mean_player_score": float(meta.get("mean_player_score", 0.0)),
        "strict_game_fraction": float(meta.get("strict_game_fraction", 0.0)),
        "strict_games": int(meta.get("strict_games", 0)),
        "relaxed_games": int(meta.get("relaxed_games", 0)),
        "eval": {
            "nn_wins": int(ev.get("nn_wins", 0)),
            "games": int(ev.get("games", 0)),
            "nn_mean_margin": float(ev.get("nn_mean_margin", 0.0)),
            "nn_mean_score": float(ev.get("nn_mean_score", 0.0)),
        },
        "promotion_gate": {
            "nn_wins": int(gate.get("nn_wins", 0)),
            "games": int(gate.get("games", 0)),
            "nn_mean_margin": float(gate.get("nn_mean_margin", 0.0)),
            "primary_opponent": str(gate.get("primary_opponent", "heuristic")),
        },
    }


def summarize_run(run_dir: Path, *, include_iterations: bool = False) -> dict:
    manifest_path = run_dir / "auto_improve_factorized_manifest.json"
    manifest = _read_json(manifest_path)
    iters = _iter_dirs(run_dir)
    iter_summaries = [_iter_summary(p) for p in iters]

    completed_iterations = 0
    best_exists = False
    if manifest is not None:
        completed_iterations = len(manifest.get("history", []))
        best_exists = bool(manifest.get("best"))

    current_stage = "pending"
    if iter_summaries:
        # The first non-complete iteration is current; otherwise latest complete.
        current = next((x for x in iter_summaries if x["stage"] != "complete"), iter_summaries[-1])
        current_stage = current["stage"]

    status = "completed" if manifest is not None else ("in_progress" if iter_summaries else "empty")
    latest = iter_summaries[-1] if iter_summaries else None

    out = {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "status": status,
        "current_stage": current_stage,
        "iterations_detected": len(iters),
        "iterations_completed": completed_iterations,
        "best_exists": best_exists,
        "latest_iteration": latest,
        "updated_at_epoch": int(run_dir.stat().st_mtime),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_dir.stat().st_mtime)),
    }
    if include_iterations:
        out["iterations"] = iter_summaries
        if manifest is not None:
            out["manifest"] = {
                "config": manifest.get("config", {}),
                "best": manifest.get("best"),
            }
    return out


def build_dashboard(*, reports_ml_dir: str = "reports/ml", limit: int = 20, include_iterations: bool = False) -> dict:
    base = Path(reports_ml_dir)
    if not base.exists():
        return {
            "generated_at_epoch": int(time.time()),
            "reports_ml_dir": str(base),
            "runs": [],
            "active_runs": [],
        }

    run_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("auto_improve_factorized")]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    runs = [summarize_run(p, include_iterations=include_iterations) for p in run_dirs[: max(1, int(limit))]]
    active = [r for r in runs if r["status"] == "in_progress"]

    latest_completed = next((r for r in runs if r["status"] == "completed"), None)
    return {
        "generated_at_epoch": int(time.time()),
        "reports_ml_dir": str(base),
        "runs": runs,
        "active_runs": active,
        "latest_completed_run": latest_completed,
    }

