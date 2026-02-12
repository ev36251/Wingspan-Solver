"""ML diagnostics routes for auto-improve training runs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.ml.run_diagnostics import build_dashboard, summarize_run
from pathlib import Path

router = APIRouter()


@router.get("/ml/runs/dashboard")
async def get_ml_runs_dashboard(
    limit: int = Query(20, ge=1, le=200),
    include_iterations: bool = Query(False),
):
    return build_dashboard(limit=limit, include_iterations=include_iterations)


@router.get("/ml/runs/{run_name}/diagnostics")
async def get_ml_run_diagnostics(
    run_name: str,
    include_iterations: bool = Query(True),
):
    run_dir = Path("reports/ml") / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")
    return summarize_run(run_dir, include_iterations=include_iterations)

