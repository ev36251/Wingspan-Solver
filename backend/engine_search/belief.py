"""Hidden-information determinization helpers for search."""

from __future__ import annotations

import copy
import random

from backend.models.game_state import GameState


def sample_hidden_state(game: GameState, rng: random.Random) -> GameState:
    """Sample one plausible hidden state.

    Current model:
    - Randomizes remaining deck order.
    - Keeps all visible/public information fixed.
    """
    sim = copy.deepcopy(game)
    deck_cards = getattr(sim, "_deck_cards", None)
    if isinstance(deck_cards, list):
        rng.shuffle(deck_cards)
        sim.deck_remaining = len(deck_cards)
    return sim
