"""Deterministic solo single-seed score optimizer.

This is the first building block for the "play the same game many times and
optimize for the highest possible score" training idea.

For one fixed seed it builds a *fully deterministic* single-player Wingspan
game:
  - fixed pre-shuffled deck stack (the Nth draw is always the same card),
  - fixed starting hand / bonus card / round goals,
  - a seeded birdfeeder whose Kth reroll is always the same dice
    (robust to the agent changing its moves between replays).

It then replays that exact game many times with a randomized heuristic policy
(simulated-annealing temperature schedule) and keeps the best-scoring line.
Because the whole game is scored to the end, the search can "see" engines
(food gathering, round-end teal birds like Sri Lanka Blue-Magpie, cached food)
pay off — something a one-step greedy evaluator cannot.

Usage:
    python -m backend.ml.solo_seed_optimizer --seed 42 --iterations 300
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.models.birdfeeder import Birdfeeder, NUM_DICE
from backend.solver.self_play import create_training_game
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, deep_copy_game, _refill_tray
from backend.solver.heuristics import dynamic_weights, _estimate_move_value
from backend.engine.scoring import calculate_score
from backend.models.enums import ActionType


class SeededBirdfeeder(Birdfeeder):
    """Birdfeeder whose rerolls come from a private seeded RNG.

    This makes the dice stream a deterministic function of the seed, so two
    replays of the same game see identical dice up to the point their reroll
    *counts* diverge — exactly the behavior we want when optimizing one seed.
    The private RNG pickles/deep-copies with the object, so every replay
    starts from the same dice stream.
    """

    def __init__(self, board_type: BoardType, seed: int):
        self._rng = random.Random(seed)
        super().__init__(board_type=board_type)  # __post_init__ calls reroll()

    def reroll(self) -> None:
        faces = self.dice_faces
        self.dice = [self._rng.choice(faces) for _ in range(NUM_DICE)]


def build_solo_game(seed: int, board_type: BoardType = BoardType.OCEANIA):
    """Build a deterministic single-player game for the given seed."""
    # Seed global RNG so create_training_game's deck shuffle / hand / goals /
    # bonus deal are reproducible for this seed.
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    game = create_training_game(
        num_players=1,
        board_type=board_type,
        setup_mode="legacy_fixed5",
    )
    # Swap in a deterministic birdfeeder stream (independent of move order).
    game.birdfeeder = SeededBirdfeeder(board_type, seed ^ 0x5EED)
    game.birdfeeder.reroll()
    return game


def _sample_move(scored, rng: random.Random, temperature: float):
    """Pick a move from (move, score) pairs.

    temperature == 0 -> greedy (argmax). Otherwise softmax over scores.
    """
    if temperature <= 0.0 or len(scored) == 1:
        return max(scored, key=lambda x: x[1])[0]
    smax = max(s for _, s in scored)
    weights = [math.exp((s - smax) / temperature) for _, s in scored]
    total = sum(weights)
    if total <= 0.0:
        return max(scored, key=lambda x: x[1])[0]
    r = rng.random() * total
    upto = 0.0
    for (move, _), w in zip(scored, weights):
        upto += w
        if upto >= r:
            return move
    return scored[-1][0]


@dataclass
class SoloResult:
    score: int
    breakdown: dict
    action_counts: dict
    trajectory: list


def play_solo(base_game, rng: random.Random, temperature: float,
              max_turns: int = 400) -> SoloResult:
    """Play one full deterministic solo game with a randomized heuristic policy."""
    game = deep_copy_game(base_game)
    player = game.players[0]
    trajectory: list[str] = []
    action_counts: Counter = Counter()
    turns = 0

    while not game.is_game_over and turns < max_turns:
        player = game.current_player
        if player.action_cubes_remaining <= 0:
            game.advance_round()
            continue

        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.advance_turn()
            continue

        weights = dynamic_weights(game)
        scored = [(m, _estimate_move_value(game, player, m, weights)) for m in moves]
        move = _sample_move(scored, rng, temperature)

        success = execute_move_on_sim(game, player, move)
        if success:
            action_counts[move.action_type.value] += 1
            trajectory.append(
                f"R{game.current_round} {move.action_type.value}: {move.description}"
            )
            game.advance_turn()
            _refill_tray(game)
        else:
            # Fallback to keep the game progressing (mirrors simulate_playout).
            fallback = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    if execute_move_on_sim(game, player, m):
                        action_counts[m.action_type.value] += 1
                        trajectory.append(
                            f"R{game.current_round} {m.action_type.value}: {m.description}"
                        )
                        game.advance_turn()
                        _refill_tray(game)
                        fallback = True
                        break
            if not fallback:
                player.action_cubes_remaining -= 1
                game.advance_turn()
        turns += 1

    breakdown = calculate_score(game, game.players[0])
    return SoloResult(
        score=breakdown.total,
        breakdown=breakdown.as_dict(),
        action_counts=dict(action_counts),
        trajectory=trajectory,
    )


def _anneal_temperature(it: int, iterations: int,
                        t_start: float = 6.0, t_end: float = 0.4) -> float:
    if iterations <= 1:
        return t_end
    frac = it / (iterations - 1)
    return t_start * ((t_end / t_start) ** frac)


def optimize_seed(seed: int, iterations: int = 300,
                  board_type: BoardType = BoardType.OCEANIA):
    """Search for the highest-scoring solo line on one fixed seed."""
    base = build_solo_game(seed, board_type)

    # Greedy heuristic baseline (temperature 0) for comparison.
    greedy = play_solo(base, random.Random(0), temperature=0.0)

    best = greedy
    history = [greedy.score]
    for it in range(iterations):
        temp = _anneal_temperature(it, iterations)
        prng = random.Random((seed * 1_000_003) ^ (it * 2_654_435_761))
        result = play_solo(base, prng, temp)
        if result.score > best.score:
            best = result
        history.append(result.score)
    return greedy, best, history


def _fmt_breakdown(bd: dict) -> str:
    keys = ["bird_vp", "eggs", "cached_food", "tucked_cards",
            "bonus_cards", "round_goals", "nectar"]
    return "  ".join(f"{k}={bd[k]}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--show-trajectory", action="store_true")
    args = ap.parse_args()

    load_all(EXCEL_FILE)
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE

    # Determinism self-check: same seed + same policy RNG => identical score.
    base = build_solo_game(args.seed, board)
    a = play_solo(base, random.Random(123), 1.5)
    b = play_solo(build_solo_game(args.seed, board), random.Random(123), 1.5)
    determinism_ok = (a.score == b.score and a.trajectory == b.trajectory)

    greedy, best, history = optimize_seed(args.seed, args.iterations, board)

    print(f"=== Solo seed optimizer  (seed={args.seed}, board={args.board}, "
          f"iterations={args.iterations}) ===")
    print(f"determinism self-check (replay identical): {determinism_ok}")
    print()
    print(f"greedy-heuristic baseline : {greedy.score}")
    print(f"   breakdown : {_fmt_breakdown(greedy.breakdown)}")
    print(f"   actions   : {greedy.action_counts}")
    print()
    print(f"BEST found over {args.iterations} replays : {best.score}  "
          f"(+{best.score - greedy.score} vs greedy)")
    print(f"   breakdown : {_fmt_breakdown(best.breakdown)}")
    print(f"   actions   : {best.action_counts}")
    print()
    # Running best curve, sampled.
    running = []
    cur = history[0]
    for s in history:
        cur = max(cur, s)
        running.append(cur)
    marks = [0, len(running) // 4, len(running) // 2, 3 * len(running) // 4, len(running) - 1]
    print("running-best by iteration: "
          + "  ".join(f"it{ m}={running[m]}" for m in marks))

    if args.show_trajectory:
        print("\n=== best trajectory ===")
        for line in best.trajectory:
            print("  " + line)


if __name__ == "__main__":
    main()
