import json
import pytest

from backend.solver.bird_priors import (
    bird_prior_value,
    clear_bird_prior_cache,
    get_bird_prior_config,
)


def test_bird_prior_uses_tier_and_phase_scale(tmp_path, monkeypatch):
    cfg = {
        "enabled": True,
        "tier_values": {"god": 1.2, "tier1": 0.4},
        "phase_scale": {"1": 1.0, "2": 0.8, "3": 0.6, "4": 0.4},
        "default_tier": "tier1",
        "bird_tiers": {"Wood Duck": "god"},
        "bird_values": {},
    }
    p = tmp_path / "bird_priors.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("WINGSPAN_BIRD_PRIORS_PATH", str(p))
    clear_bird_prior_cache()

    assert bird_prior_value("Wood Duck", 1) == 1.2
    assert bird_prior_value("Wood Duck", 3) == 0.72
    # Unknown bird falls back to default_tier.
    assert bird_prior_value("Unknown Bird", 2) == pytest.approx(0.32)


def test_bird_prior_override_beats_tier(tmp_path, monkeypatch):
    cfg = {
        "enabled": True,
        "tier_values": {"god": 1.0},
        "phase_scale": {"1": 1.0},
        "default_tier": "",
        "bird_tiers": {"Wood Duck": "god"},
        "bird_values": {"Wood Duck": 0.25},
    }
    p = tmp_path / "bird_priors.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("WINGSPAN_BIRD_PRIORS_PATH", str(p))
    clear_bird_prior_cache()

    assert bird_prior_value("Wood Duck", 1) == 0.25


def test_missing_file_disables_priors(monkeypatch):
    monkeypatch.setenv("WINGSPAN_BIRD_PRIORS_PATH", "/tmp/does-not-exist-priors.json")
    clear_bird_prior_cache()

    cfg = get_bird_prior_config()
    assert cfg.enabled is False
    assert bird_prior_value("Wood Duck", 1) == 0.0
