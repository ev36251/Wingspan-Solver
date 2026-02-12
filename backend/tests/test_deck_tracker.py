"""Tests for composition-aware deck tracking and consumption."""

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, Habitat
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.templates.draw_cards import DrawCards
from backend.powers.templates.predator import PredatorLookAt
from backend.solver.deck_tracker import DeckTracker
from backend.solver.heuristics import _evaluate_draw_cards, HeuristicWeights
from backend.solver.move_generator import Move


def _pick_birds(reg, pred, n: int):
    picked = []
    for b in reg.all_birds:
        if pred(b):
            picked.append(b)
            if len(picked) >= n:
                break
    if len(picked) < n:
        raise AssertionError(f"Need at least {n} birds matching predicate")
    return picked


def test_deck_tracker_stats_and_updates():
    bird_reg, _, _ = load_all(EXCEL_FILE)
    low_wing = _pick_birds(bird_reg, lambda b: b.wingspan_cm is not None and b.wingspan_cm < 60, 1)[0]
    high_wing = _pick_birds(bird_reg, lambda b: b.wingspan_cm is not None and b.wingspan_cm >= 100, 1)[0]
    tracker = DeckTracker()
    tracker.reset_from_cards([low_wing, high_wing])

    assert tracker.remaining_count == 2
    assert tracker.avg_remaining_vp() >= 0.0
    assert tracker.predator_success_rate(75) == 0.5

    assert tracker.mark_drawn(low_wing.name)
    assert tracker.remaining_count == 1
    assert tracker.predator_success_rate(75) == 0.0


def test_draw_power_consumes_deck_tracker():
    bird_reg, _, _ = load_all(EXCEL_FILE)
    deck_cards = _pick_birds(bird_reg, lambda _b: True, 3)

    game = create_new_game(["Alice", "Bob"])
    game._deck_cards = list(deck_cards)  # type: ignore[attr-defined]
    game.deck_remaining = len(deck_cards)
    assert game.deck_tracker is not None
    game.deck_tracker.reset_from_cards(deck_cards)
    player = game.players[0]

    power = DrawCards(draw=2, keep=2)
    ctx = PowerContext(game_state=game, player=player, bird=deck_cards[0], slot_index=0, habitat=Habitat.WETLAND)
    result = power.execute(ctx)

    assert result.cards_drawn == 2
    assert game.deck_remaining == 1
    assert game.deck_tracker.remaining_count == 1


def test_predator_and_draw_eval_use_tracker_composition():
    bird_reg, _, _ = load_all(EXCEL_FILE)
    high_vp_cards = _pick_birds(bird_reg, lambda b: b.victory_points >= 7, 8)
    low_vp_cards = _pick_birds(bird_reg, lambda b: b.victory_points <= 2, 8)
    low_wingspan_cards = _pick_birds(
        bird_reg, lambda b: b.wingspan_cm is not None and b.wingspan_cm < 75, 8
    )
    high_wingspan_cards = _pick_birds(
        bird_reg, lambda b: b.wingspan_cm is not None and b.wingspan_cm >= 100, 8
    )

    game = create_new_game(["Alice", "Bob"])
    player = game.players[0]
    assert game.deck_tracker is not None

    # Draw-card heuristic should prefer high-VP remaining decks.
    draw_move = Move(action_type=ActionType.DRAW_CARDS, description="draw", deck_draws=1)
    game.deck_tracker.reset_from_cards(high_vp_cards)
    game.deck_remaining = len(high_vp_cards)
    high_draw_val = _evaluate_draw_cards(game, player, draw_move, HeuristicWeights())

    game.deck_tracker.reset_from_cards(low_vp_cards)
    game.deck_remaining = len(low_vp_cards)
    low_draw_val = _evaluate_draw_cards(game, player, draw_move, HeuristicWeights())
    assert high_draw_val > low_draw_val

    # Predator estimate should be higher when remaining deck has smaller wingspans.
    pred = PredatorLookAt(wingspan_threshold=75)
    ctx = PowerContext(game_state=game, player=player, bird=high_vp_cards[0], slot_index=0, habitat=Habitat.FOREST)
    game.deck_tracker.reset_from_cards(low_wingspan_cards)
    game.deck_remaining = len(low_wingspan_cards)
    low_ws_value = pred.estimate_value(ctx)

    game.deck_tracker.reset_from_cards(high_wingspan_cards)
    game.deck_remaining = len(high_wingspan_cards)
    high_ws_value = pred.estimate_value(ctx)
    assert low_ws_value > high_ws_value
