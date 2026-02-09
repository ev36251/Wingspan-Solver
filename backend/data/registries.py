"""Singleton registries for birds, bonus cards, and goals.

Load once at startup, then provide fast lookup by name, set, habitat, etc.
"""

from pathlib import Path
import unicodedata

from backend.models.bird import Bird
from backend.models.bonus_card import BonusCard
from backend.models.goal import Goal
from backend.models.enums import GameSet, Habitat, PowerColor, NestType
from backend.data.loader import load_birds, load_bonus_cards, load_goals


class BirdRegistry:
    """Fast lookup for birds by various criteria."""

    def __init__(self, birds: list[Bird]):
        self._birds = birds
        self._by_name: dict[str, Bird] = {}
        self._by_name_norm: dict[str, Bird] = {}
        self._by_set: dict[GameSet, list[Bird]] = {}
        self._by_habitat: dict[Habitat, list[Bird]] = {}
        self._by_color: dict[PowerColor, list[Bird]] = {}
        self._by_nest: dict[NestType, list[Bird]] = {}

        for bird in birds:
            self._by_name[bird.name.lower()] = bird
            self._by_name_norm[self._normalize(bird.name)] = bird

            self._by_set.setdefault(bird.game_set, []).append(bird)

            for hab in bird.habitats:
                self._by_habitat.setdefault(hab, []).append(bird)

            self._by_color.setdefault(bird.color, []).append(bird)
            self._by_nest.setdefault(bird.nest_type, []).append(bird)

    @property
    def all_birds(self) -> list[Bird]:
        return self._birds

    def get(self, name: str) -> Bird | None:
        key = name.lower()
        bird = self._by_name.get(key)
        if bird:
            return bird
        return self._by_name_norm.get(self._normalize(name))

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for diacritic-insensitive matching."""
        normalized = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return stripped.lower()

    def search(self, query: str) -> list[Bird]:
        """Search birds by partial name match."""
        q = query.lower()
        q_norm = self._normalize(query)
        results = []
        for b in self._birds:
            name = b.name.lower()
            if q in name:
                results.append(b)
                continue
            if q_norm in self._normalize(b.name):
                results.append(b)
        return results

    def by_set(self, game_set: GameSet) -> list[Bird]:
        return self._by_set.get(game_set, [])

    def by_habitat(self, habitat: Habitat) -> list[Bird]:
        return self._by_habitat.get(habitat, [])

    def by_color(self, color: PowerColor) -> list[Bird]:
        return self._by_color.get(color, [])

    def by_nest(self, nest_type: NestType) -> list[Bird]:
        return self._by_nest.get(nest_type, [])

    def __len__(self) -> int:
        return len(self._birds)


class BonusCardRegistry:
    """Fast lookup for bonus cards."""

    def __init__(self, cards: list[BonusCard]):
        self._cards = cards
        self._by_name: dict[str, BonusCard] = {
            c.name.lower(): c for c in cards
        }

    @property
    def all_cards(self) -> list[BonusCard]:
        return self._cards

    def get(self, name: str) -> BonusCard | None:
        return self._by_name.get(name.lower())

    def by_set(self, game_set: GameSet) -> list[BonusCard]:
        return [c for c in self._cards if game_set in c.game_sets]

    def __len__(self) -> int:
        return len(self._cards)


class GoalRegistry:
    """Fast lookup for goals."""

    def __init__(self, goals: list[Goal]):
        self._goals = goals
        self._by_set: dict[GameSet, list[Goal]] = {}
        for goal in goals:
            self._by_set.setdefault(goal.game_set, []).append(goal)

    @property
    def all_goals(self) -> list[Goal]:
        return self._goals

    def by_set(self, game_set: GameSet) -> list[Goal]:
        return self._by_set.get(game_set, [])

    def __len__(self) -> int:
        return len(self._goals)


# Global singleton instances — initialized by load_all()
_bird_registry: BirdRegistry | None = None
_bonus_registry: BonusCardRegistry | None = None
_goal_registry: GoalRegistry | None = None


def load_all(filepath: Path) -> tuple[BirdRegistry, BonusCardRegistry, GoalRegistry]:
    """Load all data from Excel and initialize registries."""
    global _bird_registry, _bonus_registry, _goal_registry

    birds = load_birds(filepath)
    bonus_cards = load_bonus_cards(filepath)
    goals = load_goals(filepath)

    _bird_registry = BirdRegistry(birds)
    _bonus_registry = BonusCardRegistry(bonus_cards)
    _goal_registry = GoalRegistry(goals)

    return _bird_registry, _bonus_registry, _goal_registry


def get_bird_registry() -> BirdRegistry:
    if _bird_registry is None:
        raise RuntimeError("Registries not loaded. Call load_all() first.")
    return _bird_registry


def get_bonus_registry() -> BonusCardRegistry:
    if _bonus_registry is None:
        raise RuntimeError("Registries not loaded. Call load_all() first.")
    return _bonus_registry


def get_goal_registry() -> GoalRegistry:
    if _goal_registry is None:
        raise RuntimeError("Registries not loaded. Call load_all() first.")
    return _goal_registry
