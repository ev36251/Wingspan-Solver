"""Timed bird-power triggers (between turns, round end, game end)."""

from __future__ import annotations

from backend.models.enums import ActionType, PowerColor
from backend.powers.base import PowerContext, NoPower, FallbackPower
from backend.powers.registry import get_power


def _iter_occupied_slots(player):
    for habitat, slot_idx, slot in player.board.all_slots():
        if slot.bird is None:
            continue
        yield habitat, slot_idx, slot


def _pink_matches_action(power_text: str, trigger_action: ActionType | None) -> bool:
    if trigger_action is None:
        return False
    t = power_text.lower()
    # If no explicit action keyword is present, allow the trigger.
    has_specific = any(
        k in t
        for k in ("gain food", "lay eggs", "draw cards", "play a bird", "play bird")
    )
    if not has_specific:
        return True
    if trigger_action == ActionType.GAIN_FOOD:
        return "gain food" in t
    if trigger_action == ActionType.LAY_EGGS:
        return "lay eggs" in t
    if trigger_action == ActionType.DRAW_CARDS:
        return "draw cards" in t
    if trigger_action == ActionType.PLAY_BIRD:
        return ("play a bird" in t) or ("play bird" in t)
    return True


def trigger_between_turn_powers(game_state, trigger_player, trigger_action: ActionType | None) -> int:
    """Trigger pink powers on non-active players after an action resolves."""
    activated = 0
    for p in game_state.players:
        if p.name == trigger_player.name:
            continue
        for habitat, slot_idx, slot in _iter_occupied_slots(p):
            bird = slot.bird
            if bird.color != PowerColor.PINK:
                continue
            if not _pink_matches_action(bird.power_text or "", trigger_action):
                continue
            power = get_power(bird)
            if isinstance(power, (NoPower, FallbackPower)):
                continue
            ctx = PowerContext(
                game_state=game_state,
                player=p,
                bird=bird,
                slot_index=slot_idx,
                habitat=habitat,
                trigger_player=trigger_player,
                trigger_action=trigger_action,
            )
            if power.can_execute(ctx):
                result = power.execute(ctx)
                if result.executed:
                    activated += 1
    return activated


def trigger_end_of_round_powers(game_state, round_num: int) -> int:
    """Trigger round-end powers (teal and explicit end-of-round text)."""
    del round_num
    activated = 0
    for p in game_state.players:
        for habitat, slot_idx, slot in _iter_occupied_slots(p):
            bird = slot.bird
            text = (bird.power_text or "").lower()
            is_round_power = (bird.color == PowerColor.TEAL) or ("end of round" in text)
            if not is_round_power:
                continue
            power = get_power(bird)
            if isinstance(power, (NoPower, FallbackPower)):
                continue
            ctx = PowerContext(
                game_state=game_state,
                player=p,
                bird=bird,
                slot_index=slot_idx,
                habitat=habitat,
            )
            if power.can_execute(ctx):
                result = power.execute(ctx)
                if result.executed:
                    activated += 1
    return activated


def trigger_end_of_game_powers(game_state) -> int:
    """Trigger game-end powers once (yellow and explicit end-of-game text)."""
    if getattr(game_state, "_end_game_powers_resolved", False):
        return 0

    activated = 0
    for p in game_state.players:
        for habitat, slot_idx, slot in _iter_occupied_slots(p):
            bird = slot.bird
            text = (bird.power_text or "").lower()
            is_game_power = (bird.color == PowerColor.YELLOW) or ("end of game" in text)
            if not is_game_power:
                continue
            power = get_power(bird)
            if isinstance(power, (NoPower, FallbackPower)):
                continue
            ctx = PowerContext(
                game_state=game_state,
                player=p,
                bird=bird,
                slot_index=slot_idx,
                habitat=habitat,
            )
            if power.can_execute(ctx):
                result = power.execute(ctx)
                if result.executed:
                    activated += 1

    setattr(game_state, "_end_game_powers_resolved", True)
    return activated

