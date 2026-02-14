"""Helpers for explicit, deterministic power decision payloads."""

from __future__ import annotations

from copy import deepcopy


def _queue_key(player_name: str, bird_name: str) -> str:
    return f"{player_name}::{bird_name}"


_ALLOWED_CHOICE_KEYS_BY_BIRD: dict[str, set[str]] = {
    "Forster's Tern": {"discard_names"},
    "Wood Duck": {"discard_names"},
    "Great Hornbill": {"tuck_name", "cache"},
    "Mandarin Duck": {"keep_name", "tuck_name", "give_name"},
    "Snowy Owl": {"prefer", "egg_target_bird"},
    "Common Nightingale": {"food_type"},
    "Red-Bellied Woodpecker": {"cache"},
    "Willie-Wagtail": {"reset_tray", "draw_name"},
    "White Stork": {"draw_name"},
    "Pink-Eared Duck": {"keep_name", "give_player"},
    "Black Drongo": {"discard_names"},
}


def validate_power_choice_payload(bird_name: str, choice: dict) -> None:
    """Validate payload keys for strict mapped birds.

    Unknown keys are rejected to avoid silent no-op behavior.
    Non-strict birds are currently accepted without key validation.
    """
    allowed = _ALLOWED_CHOICE_KEYS_BY_BIRD.get(bird_name)
    if allowed is None:
        return
    unknown = set(choice.keys()) - allowed - {"activate"}
    if unknown:
        raise ValueError(
            f"Unknown choice key(s) for {bird_name}: {sorted(unknown)}; allowed={sorted(allowed)}"
        )


def queue_power_choice(game_state, player_name: str, bird_name: str, choice: dict) -> None:
    """Append one choice payload for a specific player's bird power."""
    validate_power_choice_payload(bird_name, choice)
    key = _queue_key(player_name, bird_name)
    if key not in game_state.power_choice_queues:
        game_state.power_choice_queues[key] = []
    game_state.power_choice_queues[key].append(choice)


def consume_power_choice(game_state, player_name: str, bird_name: str) -> dict | None:
    """Pop the next queued choice payload for a specific player's bird power."""
    key = _queue_key(player_name, bird_name)
    queue = game_state.power_choice_queues.get(key)
    if not queue:
        return None
    choice = queue.pop(0)
    if not queue:
        game_state.power_choice_queues.pop(key, None)
    return choice


def consume_power_activation_decision(game_state, player_name: str, bird_name: str) -> bool | None:
    """Consume an optional queued activation decision for a power.

    Returns:
    - True: explicitly activate
    - False: explicitly skip
    - None: no explicit activation decision queued
    """
    key = _queue_key(player_name, bird_name)
    queue = game_state.power_choice_queues.get(key)
    if not queue:
        return None

    first = queue[0]
    if "activate" not in first:
        return None

    activate = bool(first.get("activate"))
    rest = {k: v for k, v in first.items() if k != "activate"}

    # Consume this queue item.
    queue.pop(0)
    if rest:
        # Preserve remaining keys as next choice payload for the power itself.
        queue.insert(0, rest)
    if not queue:
        game_state.power_choice_queues.pop(key, None)
    return activate


def queue_many_power_choices(game_state, items: list[dict]) -> int:
    """Queue many choices. Item shape: {player_name, bird_name, choice}."""
    count = 0
    for item in items:
        queue_power_choice(
            game_state=game_state,
            player_name=item["player_name"],
            bird_name=item["bird_name"],
            choice=item.get("choice", {}),
        )
        count += 1
    return count


def power_choice_queue_summary(game_state) -> dict[str, int]:
    """Return queue sizes keyed by '<player>::<bird>'."""
    return {k: len(v) for k, v in game_state.power_choice_queues.items() if v}


def power_choice_queue_snapshot(game_state) -> dict[str, list[dict]]:
    """Return a copy of the full queued payloads for debugging."""
    return {k: deepcopy(v) for k, v in game_state.power_choice_queues.items()}
