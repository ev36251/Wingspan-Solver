"""
Champion bird-outcome analysis.

Unlike `bird_analysis.py` (which ranks birds by the *policy head's preference*
for playing them in synthetic states), this harness plays full champion-vs-
champion MCTS games and attributes the *game's final score* back to every bird
that was actually played.  It answers two questions the policy-preference
ranking cannot:

    1. Which birds does the champion actually play, and how often / where?
    2. Which birds consistently show up in higher-scoring games?

For each bird we report:
    - times_played          total plays across all player-games
    - games_in              number of player-games it appeared in (support)
    - mean_score            mean final score of player-games it appeared in
    - delta                 mean_score - overall_mean_score  (the ranking key)
    - win_rate              win rate of player-games it appeared in
    - habitats              where it was played (forest/grassland/wetland counts)

We also report the action-tempo mix (how many turns go to lay-eggs vs the
other actions) so the "too many egg turns in grassland" hypothesis is
measurable rather than anecdotal.

Usage:
    python -m backend.ml.champion_bird_outcomes \
        --model-path models/champions/v20_iter3_champion.npz \
        --out reports/ml/champion_birds/v20_iter3.json \
        --games 60 --mcts-sims 60 --min-support 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry
from backend.engine.scoring import calculate_score
from backend.ml.factorized_inference import load_policy_model
from backend.ml.mcts import MCTS
from backend.ml.state_encoder import StateEncoder
from backend.models.enums import ActionType, BoardType
from backend.solver.move_generator import generate_all_moves
from backend.solver.self_play import create_training_game
from backend.solver.simulation import (
    _refill_tray,
    _score_round_goal,
    execute_move_on_sim,
)


ACTION_LABEL = {
    ActionType.PLAY_BIRD: "play_bird",
    ActionType.GAIN_FOOD: "gain_food(forest)",
    ActionType.LAY_EGGS: "lay_eggs(grass)",
    ActionType.DRAW_CARDS: "draw_cards(wet)",
}


def play_game(
    mcts: MCTS,
    encoder: StateEncoder,
    *,
    players: int = 2,
    board_type: BoardType = BoardType.OCEANIA,
    temperature_cutoff: int = 8,
    max_turns: int = 240,
):
    """Play one champion-vs-champion game.

    Returns (final_scores, breakdowns, birds_by_player, action_counter) where
    birds_by_player maps player_name -> list[(bird_name, habitat_value)].
    """
    game = create_training_game(num_players=players, board_type=board_type)
    player_turns: dict[str, int] = {p.name: 0 for p in game.players}
    birds_by_player: dict[str, list] = {p.name: [] for p in game.players}
    actions = Counter()
    turns_total = 0

    while not game.is_game_over and turns_total < max_turns:
        player = game.current_player
        player_idx = game.current_player_idx

        if player.action_cubes_remaining <= 0:
            if all(p.action_cubes_remaining <= 0 for p in game.players):
                _score_round_goal(game, game.current_round)
                game.advance_round()
            else:
                game.current_player_idx = (player_idx + 1) % game.num_players
            turns_total += 1
            continue

        moves = generate_all_moves(game, player)
        if not moves:
            player.action_cubes_remaining = 0
            game.current_player_idx = (player_idx + 1) % game.num_players
            turns_total += 1
            continue

        tau = 1.0 if player_turns[player.name] < temperature_cutoff else 0.0
        mcts_moves, mcts_probs = mcts.get_policy(game, player_idx, temperature=tau)
        if mcts_moves:
            chosen = random.choices(mcts_moves, weights=mcts_probs, k=1)[0]
        else:
            from backend.solver.heuristics import _estimate_move_value, dynamic_weights
            w = dynamic_weights(game)
            chosen = max(moves, key=lambda m: _estimate_move_value(game, player, m, w))

        actions[chosen.action_type] += 1
        if chosen.action_type == ActionType.PLAY_BIRD and chosen.bird_name:
            hab = chosen.habitat.value if chosen.habitat is not None else "?"
            birds_by_player[player.name].append((chosen.bird_name, hab))

        success = execute_move_on_sim(game, player, chosen)
        if not success:
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
        else:
            _refill_tray(game)
            game.advance_turn()

        player_turns[player.name] = player_turns.get(player.name, 0) + 1
        turns_total += 1

    breakdowns = {p.name: calculate_score(game, p) for p in game.players}
    scores = {n: float(b.total) for n, b in breakdowns.items()}
    return scores, breakdowns, birds_by_player, actions


def run(model_path: str, games: int, mcts_sims: int, board_type: BoardType, seed: int):
    load_all(EXCEL_FILE)
    reg = get_bird_registry()
    model = load_policy_model(model_path)
    enc = StateEncoder.resolve_for_model(model.meta)
    mcts = MCTS(model=model, encoder=enc, num_sims=mcts_sims)

    random.seed(seed)
    np.random.seed(seed % (2**31))

    # Each player-game is one outcome sample.
    samples = []                     # list of {score, win, birds:[(name,hab)]}
    actions_total = Counter()
    t0 = time.time()
    for g in range(games):
        scores, _bd, birds, actions = play_game(
            mcts, enc, board_type=board_type
        )
        actions_total += actions
        best = max(scores.values())
        for name, score in scores.items():
            others = [v for n, v in scores.items() if n != name]
            win = 1.0 if score > max(others) else (0.5 if score == max(others) else 0.0)
            samples.append({"score": score, "win": win, "birds": birds[name]})
        if (g + 1) % 5 == 0 or g == 0:
            elapsed = time.time() - t0
            print(f"  game {g+1}/{games} | {elapsed:.0f}s "
                  f"({elapsed/(g+1):.1f}s/game) | last best={best:.0f}")

    overall_mean = float(np.mean([s["score"] for s in samples]))
    overall_win = float(np.mean([s["win"] for s in samples]))

    # Aggregate per bird across player-games.
    per_bird = defaultdict(lambda: {"plays": 0, "scores": [], "wins": [], "hab": Counter()})
    for s in samples:
        seen_this_game = set()
        for name, hab in s["birds"]:
            pb = per_bird[name]
            pb["plays"] += 1
            pb["hab"][hab] += 1
            if name not in seen_this_game:        # one outcome sample per player-game
                pb["scores"].append(s["score"])
                pb["wins"].append(s["win"])
                seen_this_game.add(name)

    rows = []
    for name, pb in per_bird.items():
        b = reg.get(name)
        rows.append({
            "bird": name,
            "color": b.color.value if b else "?",
            "game_set": str(b.game_set).split(".")[-1] if b else "?",
            "vp": b.victory_points if b else None,
            "power": (b.power_text[:70] + "…") if b and len(b.power_text) > 70 else (b.power_text if b else ""),
            "plays": pb["plays"],
            "games_in": len(pb["scores"]),
            "mean_score": round(float(np.mean(pb["scores"])), 1),
            "delta": round(float(np.mean(pb["scores"]) - overall_mean), 1),
            "win_rate": round(float(np.mean(pb["wins"])), 3),
            "habitats": dict(pb["hab"]),
        })

    action_mix = {ACTION_LABEL[a]: round(c / sum(actions_total.values()), 3)
                  for a, c in sorted(actions_total.items(), key=lambda x: -x[1])}

    return {
        "model_path": model_path,
        "games": games,
        "player_games": len(samples),
        "mcts_sims": mcts_sims,
        "board_type": board_type.value,
        "overall_mean_score": round(overall_mean, 1),
        "overall_win_rate": round(overall_win, 3),
        "action_mix": action_mix,
        "distinct_birds_played": len(rows),
        "birds": rows,
    }


def _print_report(result: dict, min_support: int, top_n: int):
    print(f"\n{'='*70}")
    print(f"CHAMPION BIRD OUTCOMES — {result['model_path']}")
    print(f"{result['games']} games ({result['player_games']} player-games), "
          f"{result['mcts_sims']}-sim MCTS, {result['board_type']} board")
    print(f"overall mean score: {result['overall_mean_score']} | "
          f"distinct birds played: {result['distinct_birds_played']}")
    print(f"{'='*70}")

    print("\nACTION TEMPO (share of all turns):")
    for a, p in result["action_mix"].items():
        print(f"   {a:20s} {p*100:5.1f}%")

    ranked = [r for r in result["birds"] if r["games_in"] >= min_support]
    ranked.sort(key=lambda r: r["delta"], reverse=True)

    print(f"\nTOP {top_n} BIRDS by score-delta (min {min_support} games):")
    print(f"   {'bird':28s} {'set':8s} {'col':6s} {'n':>3s} {'mean':>5s} {'Δ':>6s} {'win':>5s}")
    for r in ranked[:top_n]:
        print(f"   {r['bird'][:27]:28s} {r['game_set'][:7]:8s} {r['color'][:5]:6s} "
              f"{r['games_in']:>3d} {r['mean_score']:>5.1f} {r['delta']:>+6.1f} {r['win_rate']:>5.2f}")

    print(f"\nBOTTOM 10 BIRDS by score-delta (min {min_support} games):")
    for r in ranked[-10:]:
        print(f"   {r['bird'][:27]:28s} {r['game_set'][:7]:8s} {r['color'][:5]:6s} "
              f"{r['games_in']:>3d} {r['mean_score']:>5.1f} {r['delta']:>+6.1f} {r['win_rate']:>5.2f}")

    most = sorted(result["birds"], key=lambda r: r["plays"], reverse=True)[:15]
    print("\nMOST-PLAYED 15 BIRDS:")
    for r in most:
        print(f"   {r['bird'][:27]:28s} plays={r['plays']:>3d} "
              f"games_in={r['games_in']:>3d} Δ={r['delta']:>+5.1f} hab={r['habitats']}")


def main():
    p = argparse.ArgumentParser(description="Rank birds by champion game outcomes")
    p.add_argument("--model-path", default="models/champions/v20_iter3_champion.npz")
    p.add_argument("--out", default="reports/ml/champion_birds/v20_iter3.json")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--mcts-sims", type=int, default=60)
    p.add_argument("--min-support", type=int, default=4)
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    result = run(args.model_path, args.games, args.mcts_sims, BoardType.OCEANIA, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _print_report(result, args.min_support, args.top_n)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
