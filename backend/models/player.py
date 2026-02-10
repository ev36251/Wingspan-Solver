"""Player state: board, resources, hand, bonus cards."""

from dataclasses import dataclass, field
from .bird import Bird
from .bonus_card import BonusCard
from .board import PlayerBoard
from .enums import FoodType


@dataclass
class FoodSupply:
    """A player's personal food token supply."""
    invertebrate: int = 0
    seed: int = 0
    fish: int = 0
    fruit: int = 0
    rodent: int = 0
    nectar: int = 0

    def get(self, food_type: FoodType) -> int:
        return {
            FoodType.INVERTEBRATE: self.invertebrate,
            FoodType.SEED: self.seed,
            FoodType.FISH: self.fish,
            FoodType.FRUIT: self.fruit,
            FoodType.RODENT: self.rodent,
            FoodType.NECTAR: self.nectar,
        }.get(food_type, 0)

    def add(self, food_type: FoodType, count: int = 1) -> None:
        if food_type == FoodType.INVERTEBRATE:
            self.invertebrate += count
        elif food_type == FoodType.SEED:
            self.seed += count
        elif food_type == FoodType.FISH:
            self.fish += count
        elif food_type == FoodType.FRUIT:
            self.fruit += count
        elif food_type == FoodType.RODENT:
            self.rodent += count
        elif food_type == FoodType.NECTAR:
            self.nectar += count

    def spend(self, food_type: FoodType, count: int = 1) -> bool:
        """Spend food. Returns False if not enough."""
        current = self.get(food_type)
        if current < count:
            return False
        self.add(food_type, -count)
        return True

    def total(self) -> int:
        """Total food tokens across all types (excluding nectar for some uses)."""
        return (self.invertebrate + self.seed + self.fish +
                self.fruit + self.rodent + self.nectar)

    def total_non_nectar(self) -> int:
        return self.invertebrate + self.seed + self.fish + self.fruit + self.rodent

    def has(self, food_type: FoodType, count: int = 1) -> bool:
        return self.get(food_type) >= count

    def as_dict(self) -> dict[FoodType, int]:
        """Return as a dict of food type -> count (non-zero only)."""
        result = {}
        for ft in (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                    FoodType.FRUIT, FoodType.RODENT, FoodType.NECTAR):
            val = self.get(ft)
            if val > 0:
                result[ft] = val
        return result


@dataclass
class Player:
    """Full state of a single player."""
    name: str
    board: PlayerBoard = field(default_factory=PlayerBoard)
    food_supply: FoodSupply = field(default_factory=FoodSupply)
    hand: list[Bird] = field(default_factory=list)
    bonus_cards: list[BonusCard] = field(default_factory=list)
    action_cubes_remaining: int = 0
    unknown_hand_count: int = 0  # Face-down cards (count only, identity unknown)
    unknown_bonus_count: int = 0  # Bonus cards held (count only, identity unknown)
    # Current-round count of times the player took the main "play a bird" action.
    play_bird_actions_this_round: int = 0

    @property
    def total_birds(self) -> int:
        return self.board.total_birds()

    @property
    def hand_size(self) -> int:
        return len(self.hand) + self.unknown_hand_count

    def has_bird_in_hand(self, bird_name: str) -> bool:
        return any(b.name == bird_name for b in self.hand)

    def remove_from_hand(self, bird_name: str) -> Bird | None:
        """Remove and return a bird from hand by name."""
        for i, bird in enumerate(self.hand):
            if bird.name == bird_name:
                return self.hand.pop(i)
        return None
