"""Hidden-information determinization helpers for search."""

from __future__ import annotations

import copy
import random

from backend.data.registries import get_bird_registry, get_bonus_registry
from backend.models.game_state import GameState


def _sample_without_replacement(pool: list, count: int, rng: random.Random) -> list:
    """Sample up to count items from pool, removing sampled entries."""
    out = []
    for _ in range(min(max(0, count), len(pool))):
        idx = rng.randrange(len(pool))
        out.append(pool.pop(idx))
    return out


def sample_hidden_state(game: GameState, rng: random.Random) -> GameState:
    """Sample one plausible hidden state.

    Current model:
    - Randomizes remaining deck order.
    - Samples concrete identities for unknown opponent hand/bonus counts.
    """
    sim = copy.deepcopy(game)
    deck_cards = getattr(sim, "_deck_cards", None)

    # Keep the concrete deck list in sync with deck_remaining before shuffling.
    if isinstance(deck_cards, list):
        if sim.deck_remaining < len(deck_cards):
            del deck_cards[max(0, sim.deck_remaining):]
        elif sim.deck_remaining > len(deck_cards):
            sim.deck_remaining = len(deck_cards)
        rng.shuffle(deck_cards)
        sim.deck_remaining = len(deck_cards)

    # Sample unknown hand cards.
    if any(p.unknown_hand_count > 0 for p in sim.players):
        if isinstance(deck_cards, list):
            for p in sim.players:
                take = min(p.unknown_hand_count, len(deck_cards))
                sampled = _sample_without_replacement(deck_cards, take, rng)
                if sampled:
                    p.hand.extend(sampled)
                p.unknown_hand_count = max(0, p.unknown_hand_count - len(sampled))
            sim.deck_remaining = len(deck_cards)
        else:
            known_birds = {
                b.name
                for pl in sim.players
                for b in pl.hand
            }
            for pl in sim.players:
                for b in pl.board.all_birds():
                    known_birds.add(b.name)
            known_birds.update(b.name for b in sim.card_tray.face_up)

            bird_pool = [b for b in get_bird_registry().all_birds if b.name not in known_birds]
            rng.shuffle(bird_pool)
            for p in sim.players:
                sampled = _sample_without_replacement(bird_pool, p.unknown_hand_count, rng)
                if sampled:
                    p.hand.extend(sampled)
                p.unknown_hand_count = max(0, p.unknown_hand_count - len(sampled))

    # Sample unknown bonus cards.
    if any(p.unknown_bonus_count > 0 for p in sim.players):
        known_bonus = {
            bc.name
            for pl in sim.players
            for bc in pl.bonus_cards
        }
        bonus_pool = [bc for bc in get_bonus_registry().all_cards if bc.name not in known_bonus]
        rng.shuffle(bonus_pool)
        for p in sim.players:
            sampled = _sample_without_replacement(bonus_pool, p.unknown_bonus_count, rng)
            if sampled:
                p.bonus_cards.extend(sampled)
            p.unknown_bonus_count = max(0, p.unknown_bonus_count - len(sampled))

    # Keep deck-composition tracker aligned with determinized deck identities.
    if isinstance(deck_cards, list):
        sim.deck_remaining = len(deck_cards)
        if sim.deck_tracker is not None:
            sim.deck_tracker.reset_from_cards(deck_cards)
    else:
        sim.deck_tracker = None

    return sim
