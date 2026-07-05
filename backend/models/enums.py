from enum import Enum

# Enum.__hash__ hashes the member *name string* on every call. These enums key
# hot dicts throughout the engine (food supplies, payments, cached food), where
# that shows up as millions of string hashes per solver call. Members are
# process-wide singletons (copy/deepcopy/pickle all resolve back to the same
# object) and Enum equality is identity, so the id-based object.__hash__ is
# consistent and much cheaper.


class FoodType(Enum):
    INVERTEBRATE = "invertebrate"
    SEED = "seed"
    FISH = "fish"
    FRUIT = "fruit"
    RODENT = "rodent"
    NECTAR = "nectar"
    WILD = "wild"

    __hash__ = object.__hash__


class Habitat(Enum):
    FOREST = "forest"
    GRASSLAND = "grassland"
    WETLAND = "wetland"

    __hash__ = object.__hash__


class NestType(Enum):
    BOWL = "bowl"
    CAVITY = "cavity"
    GROUND = "ground"
    PLATFORM = "platform"
    WILD = "wild"  # Star nest — counts as any type

    __hash__ = object.__hash__


class PowerColor(Enum):
    WHITE = "white"
    BROWN = "brown"
    PINK = "pink"
    TEAL = "teal"
    YELLOW = "yellow"
    NONE = "none"

    __hash__ = object.__hash__


class BeakDirection(Enum):
    LEFT = "L"
    RIGHT = "R"
    NONE = "N"


class GameSet(Enum):
    CORE = "core"
    EUROPEAN = "european"
    OCEANIA = "oceania"
    ASIA = "asia"
    PROMO_UK = "promo_uk"


# Sets we include in the solver
INCLUDED_SETS = {GameSet.CORE, GameSet.EUROPEAN, GameSet.OCEANIA, GameSet.ASIA,
                 GameSet.PROMO_UK}

# Map from spreadsheet string to GameSet
SET_NAME_MAP = {
    "core": GameSet.CORE,
    "european": GameSet.EUROPEAN,
    "oceania": GameSet.OCEANIA,
    "asia": GameSet.ASIA,
    "promoUK": GameSet.PROMO_UK,
}


class ActionType(Enum):
    PLAY_BIRD = "play_bird"
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"

    __hash__ = object.__hash__


class BoardType(Enum):
    BASE = "base"
    OCEANIA = "oceania"
