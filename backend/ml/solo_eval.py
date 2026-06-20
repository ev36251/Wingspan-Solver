"""Evaluate a trained solo policy net by actually playing games.

Plays full deterministic solo games with the greedy net policy and compares to
the greedy rule-based heuristic, head-to-head from the *same* opening draft, on
**fresh seeds the net never trained on** (default 4000+). This tests whether
behavioral cloning of the best-of-N search produced a policy that (a) beats the
heuristic and (b) generalizes beyond the training seeds.

Usage:
    python -m backend.ml.solo_eval --model reports/ml/solo_seed/solo_net.npz \
        --seeds 4000-4099
"""

from __future__ import annotations

import argparse
import statistics as st

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, _refill_tray
from backend.solver.heuristics import dynamic_weights, _estimate_move_value
from backend.engine.scoring import calculate_score
from backend.ml.solo_seed_optimizer import (
    deal_seed, draft_candidates, build_game_from_draft,
    _food_dict_from_rec, _bonus_by_name,
)


def heuristic_chooser(game, player, moves):
    w = dynamic_weights(game)
    return max(moves, key=lambda m: _estimate_move_value(game, player, m, w))


def make_net_chooser(model: FactorizedPolicyModel, encoder: StateEncoder):
    def choose(game, player, moves):
        state = np.asarray(encoder.encode(game, game.current_player_idx), dtype=np.float32)
        logits, _ = model.forward(state)
        return max(moves, key=lambda m: model.score_move(state, m, player, logits=logits))
    return choose


def play_with(game, choose, max_turns: int = 400) -> int:
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
        mv = choose(game, player, moves)
        if execute_move_on_sim(game, player, mv):
            game.advance_turn()
            _refill_tray(game)
        else:
            player.action_cubes_remaining = max(0, player.action_cubes_remaining - 1)
            game.advance_turn()
        turns += 1
    return calculate_score(game, game.players[0]).total


def evaluate(model_path: str, seeds, board=BoardType.OCEANIA):
    load_all(EXCEL_FILE)
    model = FactorizedPolicyModel(model_path)
    encoder = StateEncoder.resolve_for_model(model.meta)
    net_choose = make_net_chooser(model, encoder)

    net_scores, heur_scores, wins = [], [], 0
    for seed in seeds:
        deal = deal_seed(seed, board)
        rec = draft_candidates(deal, top_k=1)[0]          # same opening for both
        food = _food_dict_from_rec(rec)
        bonus = _bonus_by_name(deal, rec.bonus_card)

        h = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), heuristic_chooser)
        n = play_with(build_game_from_draft(deal, rec.birds_to_keep, food, bonus), net_choose)
        heur_scores.append(h)
        net_scores.append(n)
        if n > h:
            wins += 1
        print(f"seed {seed:>4}:  net={n:>3}  heuristic={h:>3}  {'NET' if n>h else ('tie' if n==h else 'heur')}")

    nN = len(net_scores)
    print("\n==================== SUMMARY ====================")
    print(f"fresh seeds (held out): {seeds[0]}–{seeds[-1]}  (n={nN})")
    print(f"net   greedy : mean={st.mean(net_scores):.1f}  median={st.median(net_scores)}  "
          f"min={min(net_scores)}  max={max(net_scores)}")
    print(f"heur  greedy : mean={st.mean(heur_scores):.1f}  median={st.median(heur_scores)}  "
          f"min={min(heur_scores)}  max={max(heur_scores)}")
    diff = st.mean(net_scores) - st.mean(heur_scores)
    print(f"net - heuristic: {diff:+.1f} mean pts   |   net wins {wins}/{nN} ({100*wins/nN:.0f}%)")


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net.npz")
    ap.add_argument("--seeds", default="4000-4099", help="held-out seeds (>=4000)")
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    evaluate(args.model, _parse_seeds(args.seeds), board)


if __name__ == "__main__":
    main()
