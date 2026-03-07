"""
Bird valuation analysis: standalone rankings and pairwise synergy.

Usage:
    python -m backend.ml.bird_analysis \
        --model-path reports/ml/alphazero_v15/best_model.npz \
        --out reports/ml/bird_analysis/v15_iter015.json \
        --top-n 30 --n-states 80 --synergy-top 20

Standalone ranking:
    For each bird, create N random early-game board states with generous food,
    inject the bird into the player's hand, and score the "play this bird" move
    using the policy head. Average across states for a stable ranking.

Synergy measurement:
    For the top-K standalone birds, measure how much each bird's policy score
    increases when a companion bird is already placed on the board.
    synergy(A, B) = avg_score(play A | B on board) - avg_score(play A | empty board)
    A large positive value means the model has learned these two work well together.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry
from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.models.player import FoodSupply
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import deep_copy_game
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder


def _load_model(model_path: str):
    model = FactorizedPolicyModel(model_path)
    enc = StateEncoder.resolve_for_model(model.meta)
    return model, enc


def _make_state(seed: int, board_type: BoardType = BoardType.OCEANIA):
    """Create a random early-game state with generous food supply."""
    random.seed(seed)
    np.random.seed(seed % (2**31))
    game = create_training_game(num_players=2, board_type=board_type)
    player = game.players[0]
    # Give plenty of every food type so any bird is affordable
    for food_type in FoodType:
        if food_type != FoodType.WILD:
            player.food_supply.add(food_type, 8)
    # Clear the hand so we control exactly what's in it
    player.hand.clear()
    return game


def _score_bird_in_state(model, enc, game, bird, player_idx: int = 0) -> float | None:
    """
    Inject bird into player's hand, score the best 'play this bird' move.
    Returns None if the bird is not legally playable in this state.
    """
    sim = deep_copy_game(game)
    player = sim.players[player_idx]
    player.hand.append(bird)

    moves = generate_all_moves(sim, player)
    play_moves = [m for m in moves
                  if m.action_type == ActionType.PLAY_BIRD
                  and getattr(m, "bird_name", None) == bird.name]

    if not play_moves:
        return None

    state_vec = np.asarray(enc.encode(sim, player_idx), dtype=np.float32)
    logits, _ = model.forward(state_vec)
    scores = [model.score_move(state_vec, m, player, logits=logits) for m in play_moves]
    return float(max(scores))


def _place_bird_on_board(sim_game, player_idx: int, bird, habitat: Habitat, slot_idx: int = 0):
    """Directly place a bird in a board slot without running full action logic."""
    row = sim_game.players[player_idx].board.get_row(habitat)
    if slot_idx < len(row.slots) and row.slots[slot_idx].is_empty:
        row.slots[slot_idx].bird = bird


def rank_birds_standalone(
    model,
    enc,
    birds: list,
    n_states: int = 80,
    seed: int = 42,
    board_type: BoardType = BoardType.OCEANIA,
) -> list[dict]:
    """
    Rank all provided birds by average policy score across N random early-game states.
    Returns list of dicts sorted by avg_score descending.
    """
    rng = random.Random(seed)
    states = [_make_state(rng.randint(0, 999999), board_type) for _ in range(n_states)]

    results = []
    for bird in birds:
        scores = []
        for game in states:
            s = _score_bird_in_state(model, enc, game, bird)
            if s is not None:
                scores.append(s)
        if scores:
            results.append({
                "bird": bird.name,
                "avg_score": round(float(np.mean(scores)), 4),
                "max_score": round(float(np.max(scores)), 4),
                "playable_rate": round(len(scores) / n_states, 3),
                "point_value": bird.victory_points,
                "habitats": [h.value for h in bird.habitats],
                "power_color": bird.color.value if bird.color else None,
                "n_scored": len(scores),
            })

    results.sort(key=lambda x: x["avg_score"], reverse=True)
    return results


def measure_pairwise_synergy(
    model,
    enc,
    top_birds: list,
    n_states: int = 40,
    seed: int = 99,
    board_type: BoardType = BoardType.OCEANIA,
) -> list[dict]:
    """
    For every pair in top_birds, measure synergy(A, B):
    how much the model's score for playing A increases when B is already on the board.

    B is placed in slot 0 of its first natural habitat (consistent baseline).
    """
    rng = random.Random(seed)
    states = [_make_state(rng.randint(0, 999999), board_type) for _ in range(n_states)]

    # First compute baseline scores for each bird (no companion)
    baseline: dict[str, float] = {}
    for bird in top_birds:
        scores = []
        for game in states:
            s = _score_bird_in_state(model, enc, game, bird)
            if s is not None:
                scores.append(s)
        baseline[bird.name] = float(np.mean(scores)) if scores else 0.0

    synergies = []
    bird_names = {b.name for b in top_birds}

    for b_bird in top_birds:
        b_habitat = next(iter(b_bird.habitats)) if b_bird.habitats else None
        if b_habitat is None:
            continue

        # Build states with B already on the board
        b_states = []
        for game in states:
            sim = deep_copy_game(game)
            _place_bird_on_board(sim, 0, b_bird, b_habitat, slot_idx=0)
            b_states.append(sim)

        for a_bird in top_birds:
            if a_bird.name == b_bird.name:
                continue

            scores_with_b = []
            for game in b_states:
                s = _score_bird_in_state(model, enc, game, a_bird)
                if s is not None:
                    scores_with_b.append(s)

            if not scores_with_b:
                continue

            avg_with_b = float(np.mean(scores_with_b))
            syn = avg_with_b - baseline[a_bird.name]

            synergies.append({
                "bird_a": a_bird.name,
                "bird_b": b_bird.name,
                "synergy": round(syn, 4),
                "score_with_b": round(avg_with_b, 4),
                "score_baseline": round(baseline[a_bird.name], 4),
                "a_habitats": [h.value for h in a_bird.habitats],
                "b_habitats": [h.value for h in b_bird.habitats],
            })

    synergies.sort(key=lambda x: x["synergy"], reverse=True)
    return synergies


def run_analysis(
    model_path: str,
    out_path: str,
    top_n: int = 30,
    n_states: int = 80,
    synergy_top: int = 20,
    board_type: BoardType = BoardType.OCEANIA,
    seed: int = 42,
):
    load_all(EXCEL_FILE)
    model, enc = _load_model(model_path)
    registry = get_bird_registry()
    all_birds = registry.all_birds

    print(f"Loaded {len(all_birds)} birds. Running standalone ranking ({n_states} states)...")
    standings = rank_birds_standalone(model, enc, all_birds, n_states=n_states,
                                      seed=seed, board_type=board_type)

    print(f"\nTop {top_n} standalone birds:")
    for i, r in enumerate(standings[:top_n], 1):
        habitats = "/".join(r["habitats"])
        print(f"  {i:2d}. {r['bird']:<30s} pts={r['point_value']:2d}  "
              f"avg={r['avg_score']:.3f}  habitats={habitats}  "
              f"power={r['power_color'] or 'none'}")

    synergy_birds_names = {r["bird"] for r in standings[:synergy_top]}
    synergy_birds = [b for b in all_birds if b.name in synergy_birds_names]

    print(f"\nMeasuring pairwise synergy for top {synergy_top} birds ({n_states//2} states)...")
    synergies = measure_pairwise_synergy(model, enc, synergy_birds,
                                         n_states=max(20, n_states // 2),
                                         seed=seed + 1, board_type=board_type)

    print(f"\nTop 20 synergistic pairs:")
    for r in synergies[:20]:
        print(f"  {r['bird_a']:<28s} + {r['bird_b']:<28s}  synergy={r['synergy']:+.3f}")

    result = {
        "model_path": model_path,
        "n_states_standalone": n_states,
        "n_states_synergy": max(20, n_states // 2),
        "top_n": top_n,
        "synergy_top_k": synergy_top,
        "standalone_rankings": standings,
        "top_synergies": synergies[:50],
        "bottom_synergies": synergies[-20:],  # most anti-synergistic pairs too
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")
    return result


def main():
    p = argparse.ArgumentParser(description="Analyze bird valuations from a trained model")
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-n", type=int, default=30, help="Birds to show in standalone ranking")
    p.add_argument("--n-states", type=int, default=80, help="Random board states per bird")
    p.add_argument("--synergy-top", type=int, default=20, help="Top-N birds to use for synergy analysis")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_analysis(
        model_path=args.model_path,
        out_path=args.out,
        top_n=args.top_n,
        n_states=args.n_states,
        synergy_top=args.synergy_top,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
