"""Player board: 3 habitats, each with 5 bird slots."""

from dataclasses import dataclass, field
from .bird import Bird
from .enums import FoodType, Habitat, NestType


@dataclass
class BirdSlot:
    """A single slot on the board that can hold a bird and its tokens."""
    bird: Bird | None = None
    eggs: int = 0
    cached_food: dict[FoodType, int] = field(default_factory=dict)
    tucked_cards: int = 0
    counts_double: bool = False  # Teal power: counts double for round goals

    @property
    def is_empty(self) -> bool:
        return self.bird is None

    @property
    def total_cached_food(self) -> int:
        return sum(self.cached_food.values())

    def cache_food(self, food_type: FoodType, count: int = 1) -> None:
        self.cached_food[food_type] = self.cached_food.get(food_type, 0) + count

    def can_hold_more_eggs(self) -> bool:
        if self.bird is None:
            return False
        return self.eggs < self.bird.egg_limit

    def eggs_space(self) -> int:
        if self.bird is None:
            return 0
        return max(0, self.bird.egg_limit - self.eggs)


@dataclass
class HabitatRow:
    """One habitat row with 5 bird slots."""
    habitat: Habitat
    slots: list[BirdSlot] = field(default_factory=lambda: [BirdSlot() for _ in range(5)])
    nectar_spent: int = 0  # Oceania: track nectar used in this habitat

    @property
    def bird_count(self) -> int:
        return sum(1 for s in self.slots if not s.is_empty)

    @property
    def is_full(self) -> bool:
        return self.bird_count == 5

    def next_empty_slot(self) -> int | None:
        """Index of the leftmost empty slot, or None if full."""
        for i, slot in enumerate(self.slots):
            if slot.is_empty:
                return i
        return None

    def occupied_slots(self) -> list[tuple[int, BirdSlot]]:
        """Return (index, slot) pairs for all occupied slots."""
        return [(i, s) for i, s in enumerate(self.slots) if not s.is_empty]

    def total_eggs(self) -> int:
        return sum(s.eggs for s in self.slots)

    def birds(self) -> list[Bird]:
        """All birds in this row, left to right."""
        return [s.bird for s in self.slots if s.bird is not None]


@dataclass
class PlayerBoard:
    """A player's full board with 3 habitat rows."""
    forest: HabitatRow = field(default_factory=lambda: HabitatRow(Habitat.FOREST))
    grassland: HabitatRow = field(default_factory=lambda: HabitatRow(Habitat.GRASSLAND))
    wetland: HabitatRow = field(default_factory=lambda: HabitatRow(Habitat.WETLAND))

    def get_row(self, habitat: Habitat) -> HabitatRow:
        return {
            Habitat.FOREST: self.forest,
            Habitat.GRASSLAND: self.grassland,
            Habitat.WETLAND: self.wetland,
        }[habitat]

    def all_rows(self) -> list[HabitatRow]:
        return [self.forest, self.grassland, self.wetland]

    def total_birds(self) -> int:
        return sum(row.bird_count for row in self.all_rows())

    def total_eggs(self) -> int:
        return sum(row.total_eggs() for row in self.all_rows())

    def all_birds(self) -> list[Bird]:
        """All birds on the board across all habitats."""
        birds = []
        for row in self.all_rows():
            birds.extend(row.birds())
        return birds

    def all_slots(self) -> list[tuple[Habitat, int, BirdSlot]]:
        """All (habitat, index, slot) triples across the board."""
        result = []
        for row in self.all_rows():
            for i, slot in enumerate(row.slots):
                result.append((row.habitat, i, slot))
        return result

    def birds_with_nest(self, nest_type: NestType) -> list[Bird]:
        """All birds matching a nest type (wild matches everything)."""
        result = []
        for bird in self.all_birds():
            if bird.nest_type == nest_type or bird.nest_type == NestType.WILD:
                result.append(bird)
        return result

    def find_bird(self, bird_name: str) -> tuple[Habitat, int, BirdSlot] | None:
        """Find a bird by name on the board."""
        for habitat, idx, slot in self.all_slots():
            if slot.bird and slot.bird.name == bird_name:
                return (habitat, idx, slot)
        return None
