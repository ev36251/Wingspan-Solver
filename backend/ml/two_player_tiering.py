"""2-player bird tier list.

The solo optimizer structurally penalizes opponent-reactive (pink) birds because
there is no opponent. This plays real 2-player games with the rollout-search
agent on both seats, records the birds each player ends with and that player's
final score, and ranks birds by appearances x average score -- the same metric
as the solo tier list, but in a competitive setting.

    python -m backend.ml.two_player_tiering --seeds 0-999 --use-modal --seeds-per-shard 4
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry
from backend.models.enums import BoardType, GameSet
from backend.ml.factorized_inference import FactorizedPolicyModel
from backend.ml.state_encoder import StateEncoder
from backend.ml.solo_eval import make_net_chooser
from backend.ml.two_player import build_2p_game, play_multi, make_search_chooser


def tiering_game_rows(seed, model, encoder, board, top_k=6):
    """Play one 2-player selfish-search game; return per-player {score, birds}."""
    net_choose = make_net_chooser(model, encoder)
    agents = [make_search_chooser(model, encoder, top_k, i, net_choose, "selfish")
              for i in range(2)]
    game = build_2p_game(seed, board)
    scores = play_multi(game, agents)
    rows = []
    for i, p in enumerate(game.players):
        rows.append({
            "seed": seed, "player": i, "score": int(scores[i]),
            "birds": [b.name for b in p.board.all_birds()],
        })
    return rows


def seed_rows(seed, model, encoder, board, top_k=6):
    return tiering_game_rows(seed, model, encoder, board, top_k)


def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def summarize(rows):
    by_bird = defaultdict(list)
    for r in rows:
        for nm in set(r["birds"]):
            by_bird[nm].append(r["score"])
    load_all(EXCEL_FILE)
    br = get_bird_registry()
    promo = {b.name for b in br.all_birds if getattr(b, "game_set", None) == GameSet.PROMO_UK}

    n_games = len({(r["seed"], r["player"]) for r in rows})
    print(f"player-games: {n_games} | mean player score: {st.mean(r['score'] for r in rows):.1f}")

    def tier(names, title):
        print(f"\n=== {title} (appearances x avg score) ===")
        data = [(nm, len(by_bird.get(nm, [])), st.mean(by_bird[nm]) if by_bird.get(nm) else 0)
                for nm in names]
        data.sort(key=lambda x: (-x[1], -x[2]))
        for nm, cnt, avg in data:
            b = br.get(nm)
            col = b.color.value if b else "?"
            print(f"  {nm:26s} {col:6s} {cnt:4d}x  avg={avg:5.1f}")

    tier(promo, "PROMO UK birds, 2-PLAYER")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reports/ml/solo_seed/solo_net_spread.npz")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--board", choices=["oceania", "base"], default="oceania")
    ap.add_argument("--out", default="reports/ml/two_player/tiering_2p.jsonl")
    ap.add_argument("--use-modal", action="store_true")
    ap.add_argument("--seeds-per-shard", type=int, default=4)
    args = ap.parse_args()
    board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
    seeds = _parse_seeds(args.seeds)

    if args.use_modal:
        from backend.ml.modal_two_player import dispatch_2p_tiering_modal
        rows = dispatch_2p_tiering_modal(seeds, args.model, board, args.top_k,
                                         args.seeds_per_shard)
    else:
        load_all(EXCEL_FILE)
        model = FactorizedPolicyModel(args.model)
        encoder = StateEncoder.resolve_for_model(model.meta)
        rows = []
        for s in seeds:
            rows.extend(seed_rows(s, model, encoder, board, args.top_k))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarize(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
