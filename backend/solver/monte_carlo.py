"""Monte Carlo solver: evaluate moves by running many random playouts.

For each legal move, the solver:
1. Makes the move on a copy of the game state
2. Runs N random playouts to completion
3. Averages the resulting scores
4. Ranks moves by average score for the current player
"""

import time
import copy
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from backend.models.enums import ActionType, FoodType
from backend.models.game_state import GameState
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import (
    deep_copy_game, execute_move_on_sim, simulate_playout,
)


@dataclass
class MCResult:
    """Result of Monte Carlo evaluation for a single move."""
    move: Move
    avg_score: float = 0.0
    win_rate: float = 0.0
    simulations: int = 0
    rank: int = 0


@dataclass
class MCConfig:
    """Configuration for Monte Carlo search."""
    simulations_per_move: int = 50
    max_workers: int = 1  # 1 = single process (safer for typical use)
    time_limit_seconds: float = 30.0
    max_moves_to_evaluate: int = 20  # Limit moves to keep time reasonable


def _evaluate_move_worker(args: tuple) -> tuple[int, list[int]]:
    """Worker function for parallel evaluation.

    Args: (move_index, game_state_bytes, move, player_name, num_sims)
    Returns: (move_index, list_of_scores)

    Note: This function is designed to be picklable for multiprocessing.
    We pass the game state and reconstruct inside the worker.
    """
    move_idx, game, move, player_name, num_sims = args
    scores = []
    for _ in range(num_sims):
        # Make a copy, apply the move, then simulate
        sim = deep_copy_game(game)
        player = sim.get_player(player_name)
        if not player:
            break

        success = execute_move_on_sim(sim, player, move)
        if success:
            sim.advance_turn()

        result = simulate_playout(sim)
        scores.append(result.get(player_name, 0))

    return move_idx, scores


def monte_carlo_evaluate(
    game: GameState,
    config: MCConfig | None = None,
) -> list[MCResult]:
    """Run Monte Carlo evaluation on all legal moves.

    Returns MCResult list sorted by average score (highest first).
    """
    if config is None:
        config = MCConfig()

    player = game.current_player
    moves = generate_all_moves(game, player)

    if not moves:
        return []

    # Deduplicate play-bird moves (keep first payment option per bird+habitat)
    seen = set()
    deduped_moves = []
    for m in moves:
        if m.action_type == ActionType.PLAY_BIRD:
            key = (m.bird_name, m.habitat)
            if key in seen:
                continue
            seen.add(key)
        deduped_moves.append(m)

    # Limit number of moves to evaluate
    moves_to_eval = deduped_moves[:config.max_moves_to_evaluate]

    start_time = time.perf_counter()
    results: list[MCResult] = [MCResult(move=m) for m in moves_to_eval]

    if config.max_workers <= 1:
        # Single-process evaluation
        for i, move in enumerate(moves_to_eval):
            elapsed = time.perf_counter() - start_time
            if elapsed > config.time_limit_seconds:
                break

            move_scores = []
            for _ in range(config.simulations_per_move):
                if time.perf_counter() - start_time > config.time_limit_seconds:
                    break

                sim = deep_copy_game(game)
                sim_player = sim.get_player(player.name)
                if not sim_player:
                    break

                success = execute_move_on_sim(sim, sim_player, move)
                if success:
                    sim.advance_turn()

                playout_scores = simulate_playout(sim)
                move_scores.append(playout_scores.get(player.name, 0))

            if move_scores:
                results[i].avg_score = sum(move_scores) / len(move_scores)
                results[i].simulations = len(move_scores)
    else:
        # Parallel evaluation with ProcessPoolExecutor
        tasks = [
            (i, game, move, player.name, config.simulations_per_move)
            for i, move in enumerate(moves_to_eval)
        ]

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(_evaluate_move_worker, t): t[0] for t in tasks}

            for future in as_completed(futures):
                elapsed = time.perf_counter() - start_time
                if elapsed > config.time_limit_seconds:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                try:
                    move_idx, scores = future.result(timeout=config.time_limit_seconds)
                    if scores:
                        results[move_idx].avg_score = sum(scores) / len(scores)
                        results[move_idx].simulations = len(scores)
                except Exception:
                    pass

    # Calculate win rates (how often this move leads to highest score)
    for r in results:
        if r.simulations > 0:
            # Simple approximation: compare avg_score to other players' likely scores
            # In practice, this would need per-simulation tracking
            r.win_rate = 0.0  # Placeholder — would need more data

    # Sort by average score descending
    results.sort(key=lambda r: -r.avg_score)

    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


def monte_carlo_best_move(
    game: GameState,
    simulations: int = 50,
    time_limit: float = 10.0,
) -> MCResult | None:
    """Quick interface: get the best move via MC evaluation."""
    config = MCConfig(
        simulations_per_move=simulations,
        time_limit_seconds=time_limit,
    )
    results = monte_carlo_evaluate(game, config)
    return results[0] if results else None
