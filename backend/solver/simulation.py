"""Lightweight game simulation for Monte Carlo playouts.

Plays games to completion using heuristic-weighted random moves.
Designed for speed — avoids deep copies where possible and uses
simplified move execution.
"""

import copy
import random
from backend.config import ACTIONS_PER_ROUND, ROUNDS, get_action_column
from backend.models.enums import ActionType, FoodType, Habitat, PowerColor
from backend.models.game_state import GameState
from backend.models.player import Player
from backend.engine.scoring import calculate_score
from backend.engine.actions import (
    execute_play_bird, execute_gain_food, execute_lay_eggs, execute_draw_cards,
    ActionResult,
)
from backend.solver.move_generator import Move, generate_all_moves


_SIM_BIRD_POOL: list | None = None


def _get_sim_bird_pool():
    """Lazily load a pool of birds for simulated deck draws."""
    global _SIM_BIRD_POOL
    if _SIM_BIRD_POOL is None:
        from backend.data.registries import get_bird_registry
        _SIM_BIRD_POOL = list(get_bird_registry().all_birds)
    return _SIM_BIRD_POOL


def _draw_sim_deck_card(game: GameState):
    """Draw one card from finite deck identities if present, else fallback random."""
    deck_cards = getattr(game, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        game.deck_remaining = max(0, len(deck_cards))
        if game.deck_tracker is not None:
            game.deck_tracker.mark_drawn(card.name)
        return card
    if game.deck_remaining <= 0:
        return None
    pool = _get_sim_bird_pool()
    if not pool:
        return None
    game.deck_remaining = max(0, game.deck_remaining - 1)
    card = random.choice(pool)
    if game.deck_tracker is not None:
        game.deck_tracker.mark_drawn(card.name)
    return card


def _add_random_deck_draws(player: Player, count: int) -> None:
    """Add deck draws to hand, preferring finite deck identities when available."""
    game = getattr(player, "_sim_game_ref", None)
    for _ in range(count):
        if isinstance(game, GameState):
            card = _draw_sim_deck_card(game)
            if card is None:
                break
            player.hand.append(card)
            continue
        pool = _get_sim_bird_pool()
        if not pool:
            return
        player.hand.append(random.choice(pool))


def _refill_tray(game: GameState) -> None:
    """Refill the face-up tray to 3 cards using deck draws."""
    needed = game.card_tray.needs_refill()
    if needed <= 0 or game.deck_remaining <= 0:
        return
    for _ in range(min(needed, game.deck_remaining)):
        card = _draw_sim_deck_card(game)
        if card is None:
            break
        game.card_tray.add_card(card)


def _score_round_goal(game: GameState, round_num: int) -> None:
    """Compute and store end-of-round goal scores for the given round."""
    if round_num < 1 or round_num > len(game.round_goals):
        return
    goal = game.round_goals[round_num - 1]
    if goal is None or goal.description.lower() == "no goal":
        return

    from backend.engine.scoring import compute_round_goal_scores
    scores = compute_round_goal_scores(game, round_num)

    if round_num not in game.round_goal_scores:
        game.round_goal_scores[round_num] = {}
    for name, pts in scores.items():
        game.round_goal_scores[round_num][name] = pts


def deep_copy_game(game: GameState) -> GameState:
    """Deep copy a game state for simulation."""
    return copy.deepcopy(game)


def pick_weighted_random_move(moves: list[Move], game: GameState,
                               player: Player,
                               base_weights=None) -> Move:
    """Pick a move using epsilon-greedy heuristic selection.

    80% of the time: pick the best move according to the full heuristic evaluator.
    20% of the time: pick from top moves weighted by heuristic score.
    This produces more realistic playouts than uniform random while preserving
    some exploration.

    If base_weights is provided, it's passed to dynamic_weights() for training.
    """
    if not moves:
        raise ValueError("No moves to pick from")
    if len(moves) == 1:
        return moves[0]

    from backend.solver.heuristics import _estimate_move_value, dynamic_weights

    weights = dynamic_weights(game, base_weights)

    # Score all moves with the full heuristic
    scored = [(m, _estimate_move_value(game, player, m, weights)) for m in moves]
    scored.sort(key=lambda x: -x[1])

    # Epsilon-greedy: 80% best move, 20% weighted random from top candidates
    if random.random() < 0.8:
        return scored[0][0]

    # Weighted random from top 3 candidates (softmax-style)
    top_n = scored[:min(3, len(scored))]
    min_score = min(s for _, s in top_n)
    selection_weights = [max(s - min_score + 0.5, 0.1) for _, s in top_n]
    return random.choices([m for m, _ in top_n], weights=selection_weights, k=1)[0]


def _fast_rollout_score(move: Move, game: GameState, player: Player) -> float:
    """Lightweight move scoring used for high-throughput rollouts."""
    if move.action_type == ActionType.PLAY_BIRD:
        bird = next((b for b in player.hand if b.name == move.bird_name), None)
        if not bird:
            return 0.5
        return float(bird.victory_points) + float(bird.egg_limit) * 0.3

    if move.action_type == ActionType.GAIN_FOOD:
        bird_count = player.board.forest.bird_count
        column = get_action_column(game.board_type, Habitat.FOREST, bird_count)
        food_count = len(move.food_choices) if move.food_choices else (column.base_gain + move.bonus_count)
        base = float(food_count) * 0.5
        available_before = player.food_supply.total_non_nectar() + player.food_supply.get(FoodType.NECTAR)
        available_after = available_before + food_count
        unlocks = any(
            b.food_cost.total > available_before and b.food_cost.total <= available_after
            for b in player.hand
        )
        return base + (1.0 if unlocks else 0.0)

    if move.action_type == ActionType.LAY_EGGS:
        if move.egg_distribution:
            egg_count = sum(move.egg_distribution.values())
        else:
            bird_count = player.board.grassland.bird_count
            column = get_action_column(game.board_type, Habitat.GRASSLAND, bird_count)
            egg_count = column.base_gain + move.bonus_count
        return float(egg_count)

    if move.action_type == ActionType.DRAW_CARDS:
        card_count = move.deck_draws + len(move.tray_indices)
        if card_count <= 0:
            bird_count = player.board.wetland.bird_count
            column = get_action_column(game.board_type, Habitat.WETLAND, bird_count)
            card_count = column.base_gain + move.bonus_count
        return float(card_count) * 0.3

    return 0.0


def fast_rollout_move(moves: list[Move], game: GameState, player: Player) -> Move:
    """Choose a move with a lightweight scoring policy for faster MCTS rollouts."""
    if not moves:
        raise ValueError("No moves to pick from")
    if len(moves) == 1:
        return moves[0]

    scored = [(m, _fast_rollout_score(m, game, player)) for m in moves]
    scored.sort(key=lambda x: -x[1])
    if random.random() < 0.85:
        return scored[0][0]

    top_n = scored[:min(3, len(scored))]
    min_score = min(s for _, s in top_n)
    selection_weights = [max(s - min_score + 0.2, 0.05) for _, s in top_n]
    return random.choices([m for m, _ in top_n], weights=selection_weights, k=1)[0]


def execute_move_on_sim_result(
    game: GameState,
    player: Player,
    move: Move,
) -> tuple[bool, ActionResult | None]:
    """Execute a move on a simulation game state, returning success and result."""
    setattr(player, "_sim_game_ref", game)

    if move.action_type == ActionType.PLAY_BIRD:
        from backend.data.registries import get_bird_registry
        bird = get_bird_registry().get(move.bird_name)
        if not bird:
            return False, None
        result = execute_play_bird(
            game, player, bird, move.habitat, move.food_payment
        )
        return result.success, result

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
        return result.success, result

    elif move.action_type == ActionType.LAY_EGGS:
        result = execute_lay_eggs(game, player, move.egg_distribution, move.bonus_count)
        return result.success, result

    elif move.action_type == ActionType.DRAW_CARDS:
        deck_draws = move.deck_draws
        result = execute_draw_cards(
            game, player, move.tray_indices, deck_draws,
            move.bonus_count, move.reset_bonus
        )
        return result.success, result

    return False, None


def execute_move_on_sim(game: GameState, player: Player, move: Move) -> bool:
    """Execute a move on a simulation game state. Returns True if successful."""
    success, _ = execute_move_on_sim_result(game, player, move)
    return success


def simulate_playout(
    game: GameState,
    max_turns: int = 200,
    base_weights=None,
    rollout_policy: str = "heuristic",
) -> dict[str, int]:
    """Run a single random playout from the current state to game end.

    Returns a dict of {player_name: final_score}.
    If base_weights is provided, it's used for move evaluation during playout.
    """
    sim = deep_copy_game(game)
    strict_mode = getattr(sim, "strict_rules_mode", False)
    turns = 0

    while not sim.is_game_over and turns < max_turns:
        player = sim.current_player
        if strict_mode:
            from backend.powers.registry import get_power_source, is_strict_power_source_allowed
            for p in sim.players:
                for b in p.board.all_birds():
                    src = get_power_source(b)
                    if not is_strict_power_source_allowed(src):
                        raise RuntimeError(
                            f"Strict rules mode rejected simulation due to non-strict power mapping: {b.name} ({src})"
                        )
        if player.action_cubes_remaining <= 0:
            # All players might be exhausted — advance round
            if all(p.action_cubes_remaining <= 0 for p in sim.players):
                _score_round_goal(sim, sim.current_round)
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

        if rollout_policy == "fast":
            move = fast_rollout_move(moves, sim, player)
        else:
            move = pick_weighted_random_move(moves, sim, player, base_weights)
        success = execute_move_on_sim(sim, player, move)

        if success:
            sim.advance_turn()
            _refill_tray(sim)
        else:
            # Move failed — try a simple fallback (gain food or lay eggs)
            fallback_executed = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    if execute_move_on_sim(sim, player, m):
                        sim.advance_turn()
                        _refill_tray(sim)
                        fallback_executed = True
                        break
            if not fallback_executed:
                # Force skip
                player.action_cubes_remaining -= 1
                sim.advance_turn()
                _refill_tray(sim)

        turns += 1

    # Calculate final scores
    return {
        p.name: calculate_score(sim, p).total
        for p in sim.players
    }
