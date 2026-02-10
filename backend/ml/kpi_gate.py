"""Promotion gate checks for model/engine KPI JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_summary(doc: dict) -> dict:
    # Supported formats:
    # - factorized pool eval: {"summary": {...}}
    # - factorized eval result: direct metric keys
    if "summary" in doc and isinstance(doc["summary"], dict):
        return doc["summary"]
    return doc


def run_gate(
    candidate_path: str,
    baseline_path: str | None,
    min_win_rate: float,
    min_mean_score: float,
    min_rate_ge_100: float,
    min_rate_ge_120: float,
    require_non_regression: bool,
) -> dict:
    cand = _extract_summary(_read_json(candidate_path))
    base = _extract_summary(_read_json(baseline_path)) if baseline_path else None

    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    c_win = float(cand.get("nn_win_rate", 0.0))
    c_mean = float(cand.get("nn_mean_score", 0.0))
    c_ge100 = float(cand.get("nn_rate_ge_100", 0.0))
    c_ge120 = float(cand.get("nn_rate_ge_120", 0.0))

    add_check("min_win_rate", c_win >= min_win_rate, f"{c_win:.4f} >= {min_win_rate:.4f}")
    add_check("min_mean_score", c_mean >= min_mean_score, f"{c_mean:.3f} >= {min_mean_score:.3f}")
    add_check("min_rate_ge_100", c_ge100 >= min_rate_ge_100, f"{c_ge100:.4f} >= {min_rate_ge_100:.4f}")
    add_check("min_rate_ge_120", c_ge120 >= min_rate_ge_120, f"{c_ge120:.4f} >= {min_rate_ge_120:.4f}")

    if require_non_regression and base is not None:
        b_win = float(base.get("nn_win_rate", 0.0))
        b_mean = float(base.get("nn_mean_score", 0.0))
        b_ge100 = float(base.get("nn_rate_ge_100", 0.0))
        b_ge120 = float(base.get("nn_rate_ge_120", 0.0))
        add_check("non_regression_win_rate", c_win >= b_win, f"{c_win:.4f} >= {b_win:.4f}")
        add_check("non_regression_mean_score", c_mean >= b_mean, f"{c_mean:.3f} >= {b_mean:.3f}")
        add_check("non_regression_rate_ge_100", c_ge100 >= b_ge100, f"{c_ge100:.4f} >= {b_ge100:.4f}")
        add_check("non_regression_rate_ge_120", c_ge120 >= b_ge120, f"{c_ge120:.4f} >= {b_ge120:.4f}")

    passed = all(c["passed"] for c in checks)
    return {
        "candidate_path": candidate_path,
        "baseline_path": baseline_path,
        "passed": passed,
        "checks": checks,
        "candidate_summary": {
            "nn_win_rate": c_win,
            "nn_mean_score": c_mean,
            "nn_rate_ge_100": c_ge100,
            "nn_rate_ge_120": c_ge120,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KPI promotion gate on eval JSON outputs")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--min-win-rate", type=float, default=0.0)
    parser.add_argument("--min-mean-score", type=float, default=0.0)
    parser.add_argument("--min-rate-ge-100", type=float, default=0.0)
    parser.add_argument("--min-rate-ge-120", type=float, default=0.0)
    parser.add_argument("--require-non-regression", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline else None
    result = run_gate(
        candidate_path=args.candidate,
        baseline_path=baseline_path,
        min_win_rate=args.min_win_rate,
        min_mean_score=args.min_mean_score,
        min_rate_ge_100=args.min_rate_ge_100,
        min_rate_ge_120=args.min_rate_ge_120,
        require_non_regression=args.require_non_regression,
    )

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {p}")
    else:
        print(json.dumps(result, indent=2))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
