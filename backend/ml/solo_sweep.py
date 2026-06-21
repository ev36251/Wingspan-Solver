"""Search-budget sweep: score-vs-time frontier for the net-guided search.

Each config is fanned across Modal (gen records the searched game score), so we
get reliable means on 50 held-out seeds per config in minutes instead of hours.
Local per-game time is annotated from single-machine measurements.

    python -m backend.ml.solo_sweep
"""

from __future__ import annotations

import statistics as st
import time

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.ml.modal_solo import dispatch_search_modal

MODEL = "reports/ml/solo_seed/solo_net_spread.npz"   # iter-1 (== iter-2)
SEEDS = list(range(4000, 4050))                       # 50 held-out
BOARD = BoardType.OCEANIA

# (label, k_schedule, n_drafts, approx local time/game)
CONFIGS = [
    ("d1 nd3 k8",   [8],       3, "~20s"),
    ("d2 nd3 k8 *", [8, 3],    3, "~70s"),
    ("d2 nd5 k8",   [8, 3],    5, "~117s"),
    ("d2 nd3 k12",  [12, 3],   3, "~105s"),
    ("d3 nd3 k8",   [8, 3, 3], 3, "~210s"),
]


def main():
    load_all(EXCEL_FILE)
    print(f"search-budget sweep | {len(SEEDS)} held-out seeds (4000-4049) | model=iter-1/2")
    results = []
    for label, ksched, ndrafts, tnote in CONFIGS:
        t0 = time.time()
        rows = dispatch_search_modal(SEEDS, MODEL, BOARD, ksched,
                                     rollout="net", n_drafts=ndrafts, seeds_per_shard=5)
        scores = [r["score"] for r in rows]
        m = st.mean(scores) if scores else 0.0
        results.append((label, m, tnote, len(scores)))
        print(f"  {label:<12} mean={m:5.1f}  (n={len(scores)})  local {tnote:<6} "
              f"[modal {time.time()-t0:.0f}s]")

    print("\n==================== FRONTIER (score-sorted) ====================")
    print(f"  {'config':<12} {'mean':>5}   {'time/game':<8}")
    for label, m, tnote, n in sorted(results, key=lambda x: -x[1]):
        print(f"  {label:<12} {m:5.1f}   {tnote}")


if __name__ == "__main__":
    main()
