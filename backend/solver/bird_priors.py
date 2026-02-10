"""Configurable bird strength priors for heuristic scoring.

This module intentionally keeps bird-tier knowledge out of code. Users can
edit a JSON file to change tier mappings and per-bird overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from backend.config import PROJECT_ROOT


DEFAULT_BIRD_PRIOR_PATH = PROJECT_ROOT / "reports" / "ml" / "config" / "bird_priors.json"


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass(frozen=True)
class BirdPriorConfig:
    enabled: bool
    tier_values: dict[str, float]
    phase_scale: dict[int, float]
    default_tier: str
    bird_tiers: dict[str, str]
    bird_values: dict[str, float]


_CACHE: BirdPriorConfig | None = None
_CACHE_MTIME_NS: int | None = None
_CACHE_PATH: Path | None = None


def _resolve_path() -> Path:
    path = os.getenv("WINGSPAN_BIRD_PRIORS_PATH", "").strip()
    if path:
        return Path(path)
    return DEFAULT_BIRD_PRIOR_PATH


def _default_config() -> BirdPriorConfig:
    return BirdPriorConfig(
        enabled=False,
        tier_values={},
        phase_scale={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        default_tier="",
        bird_tiers={},
        bird_values={},
    )


def _parse_config(raw: dict) -> BirdPriorConfig:
    enabled = bool(raw.get("enabled", True))
    tier_values_raw = raw.get("tier_values", {}) or {}
    tier_values = {
        _normalize_name(str(k)): float(v) for k, v in tier_values_raw.items()
    }

    phase_scale_raw = raw.get("phase_scale", {}) or {}
    phase_scale = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    for k, v in phase_scale_raw.items():
        try:
            rd = int(k)
            if 1 <= rd <= 4:
                phase_scale[rd] = float(v)
        except Exception:
            continue

    default_tier = _normalize_name(str(raw.get("default_tier", "")))

    bird_tiers_raw = raw.get("bird_tiers", {}) or {}
    bird_tiers = {
        _normalize_name(str(name)): _normalize_name(str(tier))
        for name, tier in bird_tiers_raw.items()
    }

    bird_values_raw = raw.get("bird_values", {}) or {}
    bird_values = {
        _normalize_name(str(name)): float(val)
        for name, val in bird_values_raw.items()
    }

    return BirdPriorConfig(
        enabled=enabled,
        tier_values=tier_values,
        phase_scale=phase_scale,
        default_tier=default_tier,
        bird_tiers=bird_tiers,
        bird_values=bird_values,
    )


def get_bird_prior_config() -> BirdPriorConfig:
    global _CACHE, _CACHE_MTIME_NS, _CACHE_PATH

    path = _resolve_path()
    mtime_ns: int | None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = None

    if _CACHE is not None and _CACHE_PATH == path and _CACHE_MTIME_NS == mtime_ns:
        return _CACHE

    if mtime_ns is None:
        cfg = _default_config()
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = _parse_config(raw)
        except Exception:
            cfg = _default_config()

    _CACHE = cfg
    _CACHE_MTIME_NS = mtime_ns
    _CACHE_PATH = path
    return cfg


def clear_bird_prior_cache() -> None:
    global _CACHE, _CACHE_MTIME_NS, _CACHE_PATH
    _CACHE = None
    _CACHE_MTIME_NS = None
    _CACHE_PATH = None


def bird_prior_value(bird_name: str, current_round: int) -> float:
    cfg = get_bird_prior_config()
    if not cfg.enabled:
        return 0.0

    nm = _normalize_name(bird_name)
    base = cfg.bird_values.get(nm)
    if base is None:
        tier = cfg.bird_tiers.get(nm, cfg.default_tier)
        base = cfg.tier_values.get(tier, 0.0)

    rd = min(4, max(1, int(current_round)))
    return float(base) * float(cfg.phase_scale.get(rd, 1.0))
