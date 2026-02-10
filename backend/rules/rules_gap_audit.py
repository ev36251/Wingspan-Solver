"""Audit rule-coverage gaps that can suppress scores or distort outcomes."""

from __future__ import annotations

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.scoring import is_supported_round_goal_description


def run_rules_gap_audit() -> dict:
    """Return structured coverage info for known scoring-critical rule surfaces."""
    regs = load_all(EXCEL_FILE)
    goal_reg = regs[2]

    unsupported_goals = sorted(
        {
            g.description
            for g in goal_reg.all_goals
            if not is_supported_round_goal_description(g.description)
        }
    )

    return {
        "round_goals_total": len(goal_reg.all_goals),
        "round_goals_supported": len(goal_reg.all_goals) - len(unsupported_goals),
        "round_goals_unsupported": len(unsupported_goals),
        "unsupported_goal_descriptions": unsupported_goals,
    }
