"""Compare strict KPI candidate JSON to a baseline and enforce promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_compare(
    candidate_path: str,
    baseline_path: str,
    *,
    min_round1_pct_15_30: float = 0.20,
    max_round1_strict_rejected_games: int = 0,
    max_smoke_strict_rejected_games: int = 0,
    min_strict_certified_birds: int = 50,
    require_non_regression: bool = True,
    mean_score_tolerance: float = 0.5,
) -> dict:
    cand = _read(candidate_path)
    base = _read(baseline_path)

    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    c_r1 = cand["round1_benchmark"]["totals"]
    b_r1 = base["round1_benchmark"]["totals"]
    c_r1_rej = int(cand["round1_benchmark"].get("strict_rejected_games", 0))
    c_smoke_rej = int(cand["smoke_dataset_meta"].get("strict_rejected_games", 0))
    c_smoke_mean = float(cand["smoke_dataset_meta"].get("mean_player_score", 0.0))
    b_smoke_mean = float(base["smoke_dataset_meta"].get("mean_player_score", 0.0))
    c_strict_certified = int(cand.get("power_coverage", {}).get("strict_certified_birds", 0))

    add(
        "min_round1_pct_15_30",
        float(c_r1["pct_15_30"]) >= min_round1_pct_15_30,
        f"{c_r1['pct_15_30']:.4f} >= {min_round1_pct_15_30:.4f}",
    )
    add(
        "max_round1_strict_rejected_games",
        c_r1_rej <= int(max_round1_strict_rejected_games),
        f"{c_r1_rej} <= {max_round1_strict_rejected_games}",
    )
    add(
        "max_smoke_strict_rejected_games",
        c_smoke_rej <= int(max_smoke_strict_rejected_games),
        f"{c_smoke_rej} <= {max_smoke_strict_rejected_games}",
    )
    add(
        "min_strict_certified_birds",
        c_strict_certified >= int(min_strict_certified_birds),
        f"{c_strict_certified} >= {min_strict_certified_birds}",
    )

    if require_non_regression:
        add(
            "non_regression_round1_mean",
            float(c_r1["mean"]) >= float(b_r1["mean"]) - float(mean_score_tolerance),
            f"{c_r1['mean']:.3f} >= {float(b_r1['mean']) - float(mean_score_tolerance):.3f}",
        )
        add(
            "non_regression_smoke_mean_player_score",
            c_smoke_mean >= b_smoke_mean - float(mean_score_tolerance),
            f"{c_smoke_mean:.3f} >= {b_smoke_mean - float(mean_score_tolerance):.3f}",
        )

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "candidate_path": candidate_path,
        "baseline_path": baseline_path,
        "checks": checks,
        "candidate_summary": {
            "round1_mean": c_r1["mean"],
            "round1_pct_15_30": c_r1["pct_15_30"],
            "round1_strict_rejected_games": c_r1_rej,
            "smoke_mean_player_score": c_smoke_mean,
            "smoke_strict_rejected_games": c_smoke_rej,
            "strict_certified_birds": c_strict_certified,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict KPI candidate vs baseline")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-round1-pct-15-30", type=float, default=0.20)
    parser.add_argument("--max-round1-strict-rejected-games", type=int, default=0)
    parser.add_argument("--max-smoke-strict-rejected-games", type=int, default=0)
    parser.add_argument("--min-strict-certified-birds", type=int, default=50)
    parser.add_argument("--require-non-regression", action="store_true")
    parser.add_argument("--mean-score-tolerance", type=float, default=0.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = run_compare(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        min_round1_pct_15_30=args.min_round1_pct_15_30,
        max_round1_strict_rejected_games=args.max_round1_strict_rejected_games,
        max_smoke_strict_rejected_games=args.max_smoke_strict_rejected_games,
        min_strict_certified_birds=args.min_strict_certified_birds,
        require_non_regression=args.require_non_regression,
        mean_score_tolerance=args.mean_score_tolerance,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
