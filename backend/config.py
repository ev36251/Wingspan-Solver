from dataclasses import dataclass, field
from pathlib import Path
from backend.models.enums import GameSet, BoardType, Habitat

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
EXCEL_FILE = DATA_DIR / "wingspan-20260128.xlsx"

# Included expansions
INCLUDED_SETS = {GameSet.CORE, GameSet.EUROPEAN, GameSet.OCEANIA, GameSet.ASIA}

# Spreadsheet set names that map to our included sets
INCLUDED_SET_NAMES = {"core", "european", "oceania", "asia"}

# Bonus cards can belong to multiple sets (e.g., "core, asia")
# We include a bonus card if ANY of its sets are in our included sets

# Game constants
ROUNDS = 4
ACTIONS_PER_ROUND = (8, 7, 6, 5)  # Round 1-4
HABITAT_SLOTS = 5
BIRDFEEDER_DICE = 5

# Egg cost per column position (0-indexed, 5 columns per habitat, 15 birds max)
# Column 1: free, Columns 2-3: 1 egg, Columns 4-5: 2 eggs
EGG_COST_BY_COLUMN = (0, 1, 1, 2, 2)

# Legacy constants — still used as fallback when no board type is set
FOREST_FOOD_GAIN = (1, 1, 1, 2, 2)
GRASSLAND_EGG_GAIN = (2, 2, 2, 2, 3)
WETLAND_CARD_GAIN = (1, 1, 1, 2, 2)


# --- Board-specific action column definitions ---

@dataclass(frozen=True)
class ColumnBonus:
    """Optional bonus trade on a habitat action column.

    bonus_type:
      "extra"          — gain +1 per use of the base resource
      "reset_feeder"   — reset birdfeeder before gaining food
      "reset_tray"     — reset card tray before drawing
    cost_options: what can be discarded to activate, e.g. ("card",) or ("card", "food")
    max_uses: how many times the trade can be done (1 or 2)
    """
    bonus_type: str
    cost_options: tuple[str, ...] = ("card",)
    max_uses: int = 1


@dataclass(frozen=True)
class ActionColumn:
    """One column of a habitat action row.

    bonus: primary bonus (usually "extra" — gain +1 resource)
    reset_bonus: optional second bonus (reset feeder or tray)
    Some Oceania columns have both a bonus and a reset_bonus.
    """
    base_gain: int
    bonus: ColumnBonus | None = None
    reset_bonus: ColumnBonus | None = None


# Base Game Board (6 columns: 0-4 birds + col 5 = all filled)
# Cols 0-4 correspond to 0-4 birds in the habitat row.
# Col 5 (all 5 slots filled) adds a bonus trade for +1.
# Forest: 1, 1+card, 2, 2+card, 3, 3+card
# Grassland: 2, 2+food, 3, 3+food, 4, 4+food
# Wetland: 1, 1+egg, 2, 2+egg, 3, 3+egg
BASE_BOARD: dict[Habitat, tuple[ActionColumn, ...]] = {
    Habitat.FOREST: (
        ActionColumn(1),
        ActionColumn(1, ColumnBonus("extra", ("card",))),
        ActionColumn(2),
        ActionColumn(2, ColumnBonus("extra", ("card",))),
        ActionColumn(3),
        ActionColumn(3, ColumnBonus("extra", ("card",))),
    ),
    Habitat.GRASSLAND: (
        ActionColumn(2),
        ActionColumn(2, ColumnBonus("extra", ("food",))),
        ActionColumn(3),
        ActionColumn(3, ColumnBonus("extra", ("food",))),
        ActionColumn(4),
        ActionColumn(4, ColumnBonus("extra", ("food",))),
    ),
    Habitat.WETLAND: (
        ActionColumn(1),
        ActionColumn(1, ColumnBonus("extra", ("egg",))),
        ActionColumn(2),
        ActionColumn(2, ColumnBonus("extra", ("egg",))),
        ActionColumn(3),
        ActionColumn(3, ColumnBonus("extra", ("egg",))),
    ),
}

# Oceania Game Board (6 columns: 0-5 birds)
# Pattern: even columns have extra bonus, odd columns have reset bonus,
# column 3 has BOTH reset and extra, column 5 (all filled) is max gain.
#
# Forest (gain food):
#   0: 1 +card(extra)   1: 2 +reset_feeder(food)   2: 2 +card(extra)
#   3: 2 +reset_feeder(food) +card(extra)   4: 3 +card(extra)   5: 4
#
# Grassland (lay eggs):
#   0: 1 +(card|food)(extra)   1: 2   2: 2 +(card|food)(extra)
#   3: 2 +(card|food)(extra)x2   4: 3 +(card|food)(extra)   5: 4
#
# Wetland (draw cards):
#   0: 1 +(egg|nectar)(extra)   1: 2 +reset_tray(food)
#   2: 2 +(egg|nectar)(extra)
#   3: 2 +reset_tray(food) +(egg|nectar)(extra)
#   4: 3 +(egg|nectar)(extra)   5: 4
OCEANIA_BOARD: dict[Habitat, tuple[ActionColumn, ...]] = {
    Habitat.FOREST: (
        ActionColumn(1, ColumnBonus("extra", ("card",))),
        ActionColumn(2, reset_bonus=ColumnBonus("reset_feeder", ("food",))),
        ActionColumn(2, ColumnBonus("extra", ("card",))),
        ActionColumn(2, ColumnBonus("extra", ("card",)),
                     reset_bonus=ColumnBonus("reset_feeder", ("food",))),
        ActionColumn(3, ColumnBonus("extra", ("card",))),
        ActionColumn(4),
    ),
    Habitat.GRASSLAND: (
        ActionColumn(1, ColumnBonus("extra", ("card", "food"))),
        ActionColumn(2),
        ActionColumn(2, ColumnBonus("extra", ("card", "food"))),
        ActionColumn(2, ColumnBonus("extra", ("card", "food"), max_uses=2)),
        ActionColumn(3, ColumnBonus("extra", ("card", "food"))),
        ActionColumn(4),
    ),
    Habitat.WETLAND: (
        ActionColumn(1, ColumnBonus("extra", ("egg", "nectar"))),
        ActionColumn(2, reset_bonus=ColumnBonus("reset_tray", ("food",))),
        ActionColumn(2, ColumnBonus("extra", ("egg", "nectar"))),
        ActionColumn(2, ColumnBonus("extra", ("egg", "nectar")),
                     reset_bonus=ColumnBonus("reset_tray", ("food",))),
        ActionColumn(3, ColumnBonus("extra", ("egg", "nectar"))),
        ActionColumn(4),
    ),
}

BOARDS = {
    BoardType.BASE: BASE_BOARD,
    BoardType.OCEANIA: OCEANIA_BOARD,
}


def get_action_column(board_type: BoardType, habitat: Habitat,
                      bird_count: int) -> ActionColumn:
    """Get the action column for a given board, habitat, and bird count."""
    columns = BOARDS[board_type][habitat]
    idx = min(bird_count, len(columns) - 1)
    return columns[idx]

# Bonus card names from the spreadsheet (columns 39-64)
BONUS_CARD_COLUMNS = {
    39: "Anatomist",
    40: "Cartographer",
    41: "Historian",
    42: "Photographer",
    43: "Backyard Birder",
    44: "Bird Bander",
    45: "Bird Counter",
    46: "Bird Feeder",
    47: "Diet Specialist",
    48: "Enclosure Builder",
    49: "Endangered Species Protector",
    50: "Falconer",
    51: "Fishery Manager",
    52: "Food Web Expert",
    53: "Forester",
    54: "Large Bird Specialist",
    55: "Nest Box Builder",
    56: "Omnivore Expert",
    57: "Passerine Specialist",
    58: "Platform Builder",
    59: "Prairie Manager",
    60: "Rodentologist",
    61: "Small Clutch Specialist",
    62: "Viticulturalist",
    63: "Wetland Scientist",
    64: "Wildlife Gardener",
}
