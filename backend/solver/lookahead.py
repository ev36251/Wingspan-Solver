"""Lookahead search solver: evaluates moves by simulating future turns.

Instead of static evaluation, this solver:
1. Takes the top-K candidate moves (beam search)
2. Simulates executing each move
3. Fast-forwards through opponent turns (using their best heuristic move)
4. Recursively evaluates our best next move at the resulting state
5. Returns moves ranked by the best achievable future position

This catches multi-turn combos like "gain food now → play 9-VP bird next turn"
that the single-move heuristic would miss.
"""

from dataclasses import dataclass, field

from backend.models.game_state import GameState
from backend.models.player import Player
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.heuristics import (
    evaluate_position, rank_moves, HeuristicWeights, DEFAULT_WEIGHTS,
    dynamic_weights,
)
from backend.solver.simulation import execute_move_on_sim, deep_copy_game


@dataclass
class LookaheadResult:
    """Result of lookahead evaluation for a single root move."""
    move: Move
    score: float  # Position value at the deepest evaluated node
    rank: int = 0
    depth_reached: int = 0
    best_sequence: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0  # Original heuristic score for comparison


def _advance_to_player(game: GameState, player_name: str,
                       max_steps: int = 30) -> bool:
    """Advance the game until it's player_name's turn.

    Simulates opponent turns by executing their top heuristic move.
    Returns True if we reached the player's turn with cubes remaining.
    """
    steps = 0
    while steps < max_steps:
        if game.is_game_over:
            return False
        current = game.current_player
        if current.name == player_name:
            return current.action_cubes_remaining > 0
        if current.action_cubes_remaining <= 0:
            # Shouldn't happen after advance_turn, but guard against it
            return False

        # Simulate opponent's best move
        moves = generate_all_moves(game, current)
        if moves:
            ranked = rank_moves(game, current)
            if ranked:
                execute_move_on_sim(game, current, ranked[0].move)
        game.advance_turn()
        steps += 1

    return False


def lookahead_search(
    game: GameState,
    player: Player | None = None,
    depth: int = 2,
    beam_width: int = 6,
    weights: HeuristicWeights | None = None,
) -> list[LookaheadResult]:
    """Rank moves using depth-limited lookahead with beam search.

    Args:
        game: Current game state
        player: Player to solve for (defaults to current player)
        depth: How many of our own turns to look ahead (1-3)
        beam_width: Number of top candidates to evaluate at each level
        weights: Heuristic weights for position evaluation

    Returns:
        Moves ranked by the best future position they lead to.
    """
    if player is None:
        player = game.current_player
    player_name = player.name

    # Use dynamic trained weights if none provided
    if weights is None:
        weights = dynamic_weights(game)

    # Get heuristic-ranked candidates, pruned by beam width
    candidates = rank_moves(game, player, weights)
    if not candidates:
        return []
    candidates = candidates[:beam_width]

    results = []
    for rm in candidates:
        sim = deep_copy_game(game)
        sim_player = sim.get_player(player_name)

        # Execute the candidate move on the simulation
        success = execute_move_on_sim(sim, sim_player, rm.move)
        if not success:
            # Move failed — fall back to heuristic score
            results.append(LookaheadResult(
                move=rm.move, score=rm.score, depth_reached=0,
                best_sequence=[rm.move.description],
                heuristic_score=rm.score,
            ))
            continue

        sim.advance_turn()

        # Depth 1: evaluate position immediately after our move
        if depth <= 1 or sim.is_game_over:
            sim_player = sim.get_player(player_name)
            score = evaluate_position(sim, sim_player, weights)
            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=1,
                best_sequence=[rm.move.description],
                heuristic_score=rm.score,
            ))
            continue

        # Depth 2+: advance through opponent turns, then recurse
        reached = _advance_to_player(sim, player_name)
        sim_player = sim.get_player(player_name)

        if not reached or sim.is_game_over or not sim_player:
            # Game ended or couldn't reach our turn
            eval_player = sim_player or player
            score = evaluate_position(sim, eval_player, weights)
            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=1,
                best_sequence=[rm.move.description],
                heuristic_score=rm.score,
            ))
        else:
            # Recurse with reduced depth
            inner = lookahead_search(
                sim, sim_player, depth - 1, beam_width, weights
            )
            if inner:
                best_inner = inner[0]
                score = best_inner.score
                seq = [rm.move.description] + best_inner.best_sequence
                inner_depth = best_inner.depth_reached + 1
            else:
                score = evaluate_position(sim, sim_player, weights)
                seq = [rm.move.description]
                inner_depth = 1

            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=inner_depth,
                best_sequence=seq,
                heuristic_score=rm.score,
            ))

    # Sort by lookahead score descending
    results.sort(key=lambda r: -r.score)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
