"""Egg laying power templates — covers ~74 birds."""

from backend.models.enums import FoodType, Habitat, NestType
from backend.powers.base import PowerEffect, PowerContext, PowerResult


class LayEggs(PowerEffect):
    """Lay eggs on birds matching criteria.

    This single template covers many variants through parameters:
    - target="self": only on this bird
    - target="any": on any bird with space
    - target="nest:bowl": only on birds with bowl nests
    - target="habitat:forest": only on birds in a specific habitat
    - target="this_row": only on birds in this bird's row

    Also handles "all players" variants where each player may lay eggs.
    """

    def __init__(self, count: int = 1, target: str = "any",
                 nest_filter: NestType | None = None,
                 habitat_filter: Habitat | None = None,
                 all_players: bool = False):
        self.count = count
        self.target = target
        self.nest_filter = nest_filter
        self.habitat_filter = habitat_filter
        self.all_players = all_players

    def execute(self, ctx: PowerContext) -> PowerResult:
        total_laid = 0

        if self.all_players:
            for p in ctx.game_state.players:
                laid = self._lay_for_player(p, ctx)
                if p.name == ctx.player.name:
                    total_laid = laid
        else:
            total_laid = self._lay_for_player(ctx.player, ctx)

        return PowerResult(eggs_laid=total_laid,
                           description=f"Laid {total_laid} eggs")

    def _lay_for_player(self, player, ctx: PowerContext) -> int:
        """Lay eggs for a specific player, returning count laid."""
        remaining = self.count
        laid = 0

        if self.target == "self":
            slot = player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            while remaining > 0 and slot.can_hold_more_eggs():
                slot.eggs += 1
                remaining -= 1
                laid += 1
            return laid

        # Find all eligible slots
        candidates = []
        for row in player.board.all_rows():
            if self.habitat_filter and row.habitat != self.habitat_filter:
                continue
            if self.target == "this_row" and row.habitat != ctx.habitat:
                continue
            for i, slot in enumerate(row.slots):
                if slot.bird is None:
                    continue
                if not slot.can_hold_more_eggs():
                    continue
                if self.nest_filter:
                    if (slot.bird.nest_type != self.nest_filter
                            and slot.bird.nest_type != NestType.WILD):
                        continue
                candidates.append((slot.eggs_space(), slot))

        # Sort by most available space (spread eggs out)
        candidates.sort(key=lambda x: -x[0])

        for _, slot in candidates:
            while remaining > 0 and slot.can_hold_more_eggs():
                slot.eggs += 1
                remaining -= 1
                laid += 1

        return laid

    def estimate_value(self, ctx: PowerContext) -> float:
        # Each egg = 1 point
        if self.target == "self":
            slot = ctx.player.board.get_row(ctx.habitat).slots[ctx.slot_index]
            available = slot.eggs_space() if slot.bird else 0
            return min(self.count, available) * 0.8
        base = self.count * 0.8
        if self.all_players:
            base *= 0.7
        return base


class LayEggsEachBirdInRow(PowerEffect):
    """Lay 1 egg on each bird in this bird's row and/or column.

    Example: "Lay 1 [egg] on each bird in this bird's row."
    """

    def __init__(self, include_column: bool = False):
        self.include_column = include_column

    def execute(self, ctx: PowerContext) -> PowerResult:
        laid = 0
        row = ctx.player.board.get_row(ctx.habitat)

        for slot in row.slots:
            if slot.bird and slot.can_hold_more_eggs():
                slot.eggs += 1
                laid += 1

        if self.include_column:
            # Lay on birds in same column index across other habitats
            for other_row in ctx.player.board.all_rows():
                if other_row.habitat == ctx.habitat:
                    continue
                slot = other_row.slots[ctx.slot_index]
                if slot.bird and slot.can_hold_more_eggs():
                    slot.eggs += 1
                    laid += 1

        return PowerResult(eggs_laid=laid,
                           description=f"Laid {laid} eggs across row/column")

    def estimate_value(self, ctx: PowerContext) -> float:
        row = ctx.player.board.get_row(ctx.habitat)
        birds_in_row = row.bird_count
        return birds_in_row * 0.7
