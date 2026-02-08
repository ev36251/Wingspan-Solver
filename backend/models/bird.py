from dataclasses import dataclass
from .enums import FoodType, Habitat, NestType, PowerColor, BeakDirection, GameSet


@dataclass(frozen=True)
class FoodCost:
    """A bird's food cost.

    items: tuple of FoodType required (e.g., (SEED, SEED, FISH))
    is_or: True = pay ANY one of the distinct types (slash cost in rules)
    total: total food tokens to pay
    """
    items: tuple[FoodType, ...]
    is_or: bool = False
    total: int = 0

    @property
    def distinct_types(self) -> frozenset[FoodType]:
        return frozenset(self.items)


@dataclass(frozen=True)
class Bird:
    name: str
    scientific_name: str
    game_set: GameSet
    color: PowerColor
    power_text: str
    victory_points: int
    nest_type: NestType
    egg_limit: int
    wingspan_cm: int | None  # None for flightless birds
    habitats: frozenset[Habitat]
    food_cost: FoodCost
    beak_direction: BeakDirection
    is_predator: bool
    is_flocking: bool
    is_bonus_card_bird: bool  # Has bonus card interaction
    bonus_eligibility: frozenset[str]  # Bonus card names this bird qualifies for

    def __str__(self) -> str:
        return f"{self.name} ({self.victory_points}VP, {self.color.value})"

    def can_live_in(self, habitat: Habitat) -> bool:
        return habitat in self.habitats
