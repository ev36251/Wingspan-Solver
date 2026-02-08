"""Lightweight game simulation for Monte Carlo playouts.

Plays games to completion using heuristic-weighted random moves.
Designed for speed — avoids deep copies where possible and uses
simplified move execution.
"""

import copy
import random
from backend.config import ACTIONS_PER_ROUND, ROUNDS
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.scoring import calculate_score
from backend.engine.actions import (
    execute_play_bird, execute_gain_food, execute_lay_eggs, execute_draw_cards,
)
from backend.solver.move_generator import Move, generate_all_moves


def deep_copy_game(game: GameState) -> GameState:
    """Deep copy a game state for simulation."""
    return copy.deepcopy(game)


def pick_weighted_random_move(moves: list[Move], game: GameState,
                               player: Player) -> Move:
    """Pick a move using epsilon-greedy heuristic selection.

    70% of the time: pick the best move according to the full heuristic evaluator.
    30% of the time: pick from top moves weighted by heuristic score.
    This produces more realistic playouts than uniform random while preserving
    some exploration.
    """
    if not moves:
        raise ValueError("No moves to pick from")
    if len(moves) == 1:
        return moves[0]

    from backend.solver.heuristics import _estimate_move_value, dynamic_weights

    weights = dynamic_weights(game)

    # Score all moves with the full heuristic
    scored = [(m, _estimate_move_value(game, player, m, weights)) for m in moves]
    scored.sort(key=lambda x: -x[1])

    # Epsilon-greedy: 70% best move, 30% weighted random from top candidates
    if random.random() < 0.7:
        return scored[0][0]

    # Weighted random from top 5 candidates (softmax-style)
    top_n = scored[:min(5, len(scored))]
    min_score = min(s for _, s in top_n)
    selection_weights = [max(s - min_score + 0.5, 0.1) for _, s in top_n]
    return random.choices([m for m, _ in top_n], weights=selection_weights, k=1)[0]


def execute_move_on_sim(game: GameState, player: Player, move: Move) -> bool:
    """Execute a move on a simulation game state. Returns True if successful."""
    if move.action_type == ActionType.PLAY_BIRD:
        from backend.data.registries import get_bird_registry
        bird = get_bird_registry().get(move.bird_name)
        if not bird:
            return False
        result = execute_play_bird(
            game, player, bird, move.habitat, move.food_payment
        )
        return result.success

    elif move.action_type == ActionType.GAIN_FOOD:
        food_choices = move.food_choices
        if not food_choices:
            # Feeder was empty — just take whatever after reroll
            if game.birdfeeder.should_reroll():
                game.birdfeeder.reroll()
            available = list(game.birdfeeder.available_food_types())
            if available:
                food_choices = [random.choice(available)]
        result = execute_gain_food(
            game, player, food_choices, move.bonus_count, move.reset_bonus
        )
        return result.success

    elif move.action_type == ActionType.LAY_EGGS:
        result = execute_lay_eggs(game, player, move.egg_distribution, move.bonus_count)
        return result.success

    elif move.action_type == ActionType.DRAW_CARDS:
        result = execute_draw_cards(
            game, player, move.tray_indices, move.deck_draws,
            move.bonus_count, move.reset_bonus
        )
        return result.success

    return False


def simulate_playout(game: GameState, max_turns: int = 200) -> dict[str, int]:
    """Run a single random playout from the current state to game end.

    Returns a dict of {player_name: final_score}.
    """
    sim = deep_copy_game(game)
    turns = 0

    while not sim.is_game_over and turns < max_turns:
        player = sim.current_player
        if player.action_cubes_remaining <= 0:
            # All players might be exhausted — advance round
            if all(p.action_cubes_remaining <= 0 for p in sim.players):
                sim.advance_round()
                continue
            sim.current_player_idx = (sim.current_player_idx + 1) % sim.num_players
            continue

        moves = generate_all_moves(sim, player)
        if not moves:
            # No legal moves — skip this player
            player.action_cubes_remaining = 0
            sim.current_player_idx = (sim.current_player_idx + 1) % sim.num_players
            continue

        move = pick_weighted_random_move(moves, sim, player)
        success = execute_move_on_sim(sim, player, move)

        if success:
            sim.advance_turn()
        else:
            # Move failed — try a simple fallback (gain food or lay eggs)
            fallback_executed = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    if execute_move_on_sim(sim, player, m):
                        sim.advance_turn()
                        fallback_executed = True
                        break
            if not fallback_executed:
                # Force skip
                player.action_cubes_remaining -= 1
                sim.advance_turn()

        turns += 1

    # Calculate final scores
    return {
        p.name: calculate_score(sim, p).total
        for p in sim.players
    }
