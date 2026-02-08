from dataclasses import dataclass
from .enums import GameSet


@dataclass(frozen=True)
class BonusScoringTier:
    min_count: int
    max_count: int | None  # None = unlimited
    points: int


@dataclass(frozen=True)
class BonusCard:
    name: str
    game_sets: frozenset[GameSet]
    condition_text: str
    explanation_text: str | None
    scoring_tiers: tuple[BonusScoringTier, ...]
    is_per_bird: bool  # True = "N per bird", False = tiered scoring
    is_automa: bool
    draft_value_pct: float | None  # Pre-computed draft value from spreadsheet

    def score(self, qualifying_count: int) -> int:
        """Calculate VP for a given number of qualifying birds."""
        if self.is_per_bird and self.scoring_tiers:
            return qualifying_count * self.scoring_tiers[0].points

        best = 0
        for tier in self.scoring_tiers:
            if qualifying_count >= tier.min_count:
                if tier.max_count is None or qualifying_count <= tier.max_count:
                    best = max(best, tier.points)
                elif qualifying_count > tier.max_count:
                    best = max(best, tier.points)
        return best

    def __str__(self) -> str:
        return self.name
