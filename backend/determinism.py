"""Process-level determinism guard.

The engine's move generation, power resolution and food-payment logic iterate
over `frozenset`/`set` collections (habitats, food types, board birds, …). Their
iteration order depends on Python's per-process hash randomization, which leaked
into argmax/`sorted` tie-breaks and made whole games non-reproducible across
runs (same seed -> different game). Hash randomization can only be disabled
*before* interpreter start, so the single robust fix is to pin ``PYTHONHASHSEED``
and re-exec the process if it isn't already set.

Call :func:`ensure_deterministic_hashing` at the very top of any entry point that
needs reproducible games (eval harnesses, dataset/tier generators). Remote Modal
functions set ``PYTHONHASHSEED`` via the image env instead.
"""

from __future__ import annotations

import os
import sys

_PINNED = "0"


def ensure_deterministic_hashing() -> None:
    """Pin PYTHONHASHSEED=0, re-exec'ing the process once if needed.

    No-op if already pinned (so the re-exec'd child returns immediately) or if
    explicitly disabled via ``WINGSPAN_ALLOW_HASH_RANDOMIZATION=1``.
    """
    if os.environ.get("WINGSPAN_ALLOW_HASH_RANDOMIZATION") == "1":
        return
    if os.environ.get("PYTHONHASHSEED") == _PINNED:
        return
    os.environ["PYTHONHASHSEED"] = _PINNED
    os.execv(sys.executable, [sys.executable, *sys.argv])
