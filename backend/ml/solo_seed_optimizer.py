"""Deterministic solo single-seed score optimizer + multi-seed driver.

Building block for the "play the same game many times and optimize for the
highest possible score" idea.

Determinism per seed:
  - a fixed DEAL: 5 dealt birds, 2 bonus cards, 3 tray birds, a fixed deck
    stack (Nth draw always the same card), 4 round goals, and a seeded
    birdfeeder reroll stream.
The DEAL is the random part (fixed per seed); the player's choices -- the
opening DRAFT and every in-game move -- are what we optimize.

The draft obeys the real Wingspan rule: keep a total of 5 (birds + food),
at most 1 food token of each of the 5 non-nectar types, plus 1 of the 2 bonus
cards, plus the free starting nectar (Oceania). The draft is a *search
dimension*: each replay samples a candidate draft (guided by analyze_setup),
so the best line tells us good draft choices too.

Single seed:
    python -m backend.ml.solo_seed_optimizer single --seed 42 --games 150 --show-trajectory
Many seeds (dataset + analytics, no model trained -> no overfit):
    python -m backend.ml.solo_seed_optimizer multi --seeds 0-29 --games-per-seed 150 \
        --out reports/ml/solo_seed/best_lines.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.config import EXCEL_FILE
from backend.data.registries import (
    load_all, get_bird_registry, get_bonus_registry, get_goal_registry,
)
from backend.models.enums import BoardType, FoodType, ActionType
from backend.models.birdfeeder import Birdfeeder, NUM_DICE
from backend.models.game_state import create_new_game
from backend.solver.deck_tracker import DeckTracker
from backend.solver.setup_advisor import analyze_setup, STARTING_FOOD
from backend.solver.move_generator import generate_all_moves
from backend.solver.simulation import execute_move_on_sim, deep_copy_game, _refill_tray
from backend.solver.heuristics import dynamic_weights, _estimate_move_value
from backend.engine.scoring import calculate_score


class SeededBirdfeeder(Birdfeeder):
    """Birdfeeder whose rerolls come from a private seeded RNG (pickles with
    the object, so every replay starts from the same dice stream)."""

    def __init__(self, board_type: BoardType, seed: int):
        self._rng = random.Random(seed)
        super().__init__(board_type=board_type)

    def reroll(self) -> None:
        faces = self.dice_faces
        self.dice = [self._rng.choice(faces) for _ in range(NUM_DICE)]


# ---------------------------------------------------------------------------
# Deal: the fixed random part of one seed
# ---------------------------------------------------------------------------

@dataclass
class Deal:
    seed: int
    board_type: BoardType
    dealt_birds: list           # 5 bird cards
    dealt_bonus: list           # 2 bonus cards
    tray: list                  # 3 face-up birds
    deck: list                  # remaining deck stack (top = end of list)
    goals: list                 # 4 round goals
    feeder_seed: int


def deal_seed(seed: int, board_type: BoardType = BoardType.OCEANIA) -> Deal:
    rng = random.Random(seed)
    birds = list(get_bird_registry().all_birds)
    rng.shuffle(birds)
    bonus = list(get_bonus_registry().all_cards)
    rng.shuffle(bonus)
    goals = rng.sample(list(get_goal_registry().all_goals), 4)

    dealt_birds = birds[0:5]
    tray = birds[5:8]
    deck = birds[8:]
    rng.shuffle(deck)  # deck order independent of hand/tray identity
    return Deal(
        seed=seed, board_type=board_type,
        dealt_birds=dealt_birds, dealt_bonus=bonus[0:2],
        tray=tray, deck=deck, goals=goals,
        feeder_seed=seed ^ 0x5EED,
    )


def build_game_from_draft(deal: Deal, keep_bird_names, keep_food: dict, bonus):
    """Construct a deterministic solo game from a deal + a draft choice."""
    game = create_new_game(["Solo"], deal.goals, deal.board_type)  # +1 nectar (Oceania)
    p = game.players[0]

    by_name = {b.name: b for b in deal.dealt_birds}
    p.hand = [by_name[n] for n in keep_bird_names if n in by_name]
    p.bonus_cards = [bonus]
    for ft, cnt in keep_food.items():
        if cnt > 0:
            p.food_supply.add(ft, int(cnt))

    for b in deal.tray:
        game.card_tray.add_card(b)

    game._deck_cards = list(deal.deck)            # type: ignore[attr-defined]
    game.deck_remaining = len(deal.deck)
    game.deck_tracker = DeckTracker()
    game.deck_tracker.reset_from_cards(deal.deck)

    game.birdfeeder = SeededBirdfeeder(deal.board_type, deal.feeder_seed)
    game.birdfeeder.reroll()
    return game


def draft_candidates(deal: Deal, top_k: int = 12):
    """Real-rule draft options (keep 5 total, <=1 per food type, 1 bonus),
    ranked by the setup advisor. Returns SetupRecommendation list."""
    return analyze_setup(
        birds=deal.dealt_birds,
        bonus_cards=deal.dealt_bonus,
        round_goals=deal.goals,
        top_n=top_k,
        tray_birds=deal.tray,
        turn_order=1,
        num_players=1,
    )


def _food_dict_from_rec(rec) -> dict:
    out: dict = {}
    for k, v in rec.food_to_keep.items():
        try:
            out[FoodType(k)] = int(v)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Play one full solo game with a randomized heuristic move policy
# ---------------------------------------------------------------------------

@dataclass
class SoloResult:
    score: int = 0
    breakdown: dict = field(default_factory=dict)
    action_counts: dict = field(default_factory=dict)
    board_birds: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    draft: dict = field(default_factory=dict)


def _sample_move(scored, rng: random.Random, temperature: float):
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


def play_solo(base_game, rng: random.Random, temperature: float,
              max_turns: int = 400) -> SoloResult:
    game = deep_copy_game(base_game)
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
        if execute_move_on_sim(game, player, move):
            action_counts[move.action_type.value] += 1
            trajectory.append(f"R{game.current_round} {move.action_type.value}: {move.description}")
            game.advance_turn()
            _refill_tray(game)
        else:
            fallback = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    if execute_move_on_sim(game, player, m):
                        action_counts[m.action_type.value] += 1
                        game.advance_turn()
                        _refill_tray(game)
                        fallback = True
                        break
            if not fallback:
                player.action_cubes_remaining -= 1
                game.advance_turn()
        turns += 1

    p = game.players[0]
    bd = calculate_score(game, p)
    return SoloResult(
        score=bd.total, breakdown=bd.as_dict(), action_counts=dict(action_counts),
        board_birds=[b.name for b in p.board.all_birds()], trajectory=trajectory,
    )


def _anneal(it: int, iters: int, t0: float = 6.0, t1: float = 0.4) -> float:
    if iters <= 1:
        return t1
    return t0 * ((t1 / t0) ** (it / (iters - 1)))


def optimize_seed(seed: int, games: int = 150,
                  board_type: BoardType = BoardType.OCEANIA,
                  draft_top_k: int = 12) -> SoloResult:
    """Search for the best solo line on one seed, optimizing draft + play."""
    deal = deal_seed(seed, board_type)
    recs = draft_candidates(deal, top_k=draft_top_k)
    if not recs:
        raise RuntimeError(f"no draft candidates for seed {seed}")

    # Pre-build a base game per candidate draft (reused across iterations).
    bases = []
    for rec in recs:
        g = build_game_from_draft(deal, rec.birds_to_keep, _food_dict_from_rec(rec),
                                  _bonus_by_name(deal, rec.bonus_card))
        bases.append((rec, g))

    best = SoloResult(score=-1)
    for it in range(games):
        temp = _anneal(it, games)
        # Explore drafts more early, exploit the advisor's top pick later.
        draft_idx = random.Random(seed * 31 + it).randrange(len(bases)) if temp > 1.5 else 0
        rec, base = bases[draft_idx]
        prng = random.Random((seed * 1_000_003) ^ (it * 2_654_435_761))
        res = play_solo(base, prng, temp)
        if res.score > best.score:
            res.draft = {
                "birds_kept": rec.birds_to_keep,
                "food_kept": rec.food_to_keep,
                "bonus": rec.bonus_card,
                "num_birds_kept": len(rec.birds_to_keep),
            }
            best = res
    return best


def _bonus_by_name(deal: Deal, name: str):
    for b in deal.dealt_bonus:
        if b.name == name:
            return b
    return deal.dealt_bonus[0]


# ---------------------------------------------------------------------------
# Multi-seed driver: dataset + analytics
# ---------------------------------------------------------------------------

def run_multi_seed(seeds, games_per_seed: int, out_path: str | None,
                   board_type: BoardType = BoardType.OCEANIA):
    bird_appear: Counter = Counter()
    bird_score_sum: defaultdict = defaultdict(int)
    bonus_appear: Counter = Counter()
    draft_birds_kept: list[int] = []
    scores: list[int] = []
    breakdown_sum: Counter = Counter()
    rows = []

    fout = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fout = open(out_path, "w", encoding="utf-8")

    for seed in seeds:
        best = optimize_seed(seed, games_per_seed, board_type)
        scores.append(best.score)
        draft_birds_kept.append(best.draft.get("num_birds_kept", 0))
        bonus_appear[best.draft.get("bonus", "?")] += 1
        for b in best.board_birds:
            bird_appear[b] += 1
            bird_score_sum[b] += best.score
        for k, v in best.breakdown.items():
            if k != "total":
                breakdown_sum[k] += v
        row = {
            "seed": seed, "score": best.score, "breakdown": best.breakdown,
            "actions": best.action_counts, "draft": best.draft,
            "board_birds": best.board_birds, "trajectory": best.trajectory,
        }
        rows.append(row)
        if fout:
            fout.write(json.dumps(row) + "\n")
        print(f"seed {seed:>4}: best={best.score:>3}  "
              f"draft_birds={best.draft.get('num_birds_kept')}  "
              f"bonus={best.draft.get('bonus')!r:<28} "
              f"{ {k: best.breakdown[k] for k in ('bird_vp','cached_food','tucked_cards','eggs','round_goals')} }")

    if fout:
        fout.close()

    n = len(scores)
    print("\n==================== AGGREGATE ====================")
    print(f"seeds: {n}   games/seed: {games_per_seed}")
    print(f"score: mean={sum(scores)/n:.1f}  min={min(scores)}  max={max(scores)}")
    print(f"avg draft birds kept: {sum(draft_birds_kept)/n:.2f} / 5")
    print(f"avg score composition: "
          + "  ".join(f"{k}={breakdown_sum[k]/n:.1f}"
                      for k in ('bird_vp','eggs','cached_food','tucked_cards',
                                'bonus_cards','round_goals','nectar')))
    print("\nmost valuable birds (by appearance in best lines, with avg game score):")
    for name, cnt in bird_appear.most_common(20):
        print(f"  {cnt:>2}x  avg_score={bird_score_sum[name]/cnt:5.1f}   {name}")
    print("\nbonus cards chosen in best drafts:")
    for name, cnt in bonus_appear.most_common(10):
        print(f"  {cnt:>2}x  {name}")
    return rows


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def _fmt_breakdown(bd: dict) -> str:
    keys = ["bird_vp", "eggs", "cached_food", "tucked_cards", "bonus_cards", "round_goals", "nectar"]
    return "  ".join(f"{k}={bd[k]}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--games", type=int, default=150)
    s.add_argument("--board", choices=["oceania", "base"], default="oceania")
    s.add_argument("--show-trajectory", action="store_true")

    m = sub.add_parser("multi")
    m.add_argument("--seeds", type=str, default="0-9")
    m.add_argument("--games-per-seed", type=int, default=150)
    m.add_argument("--board", choices=["oceania", "base"], default="oceania")
    m.add_argument("--out", type=str, default="reports/ml/solo_seed/best_lines.jsonl")

    args = ap.parse_args()
    load_all(EXCEL_FILE)

    if args.cmd == "single":
        board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
        best = optimize_seed(args.seed, args.games, board)
        print(f"=== seed {args.seed}  ({args.games} replays, draft-aware) ===")
        print(f"BEST score: {best.score}")
        print(f"  draft     : {best.draft}")
        print(f"  breakdown : {_fmt_breakdown(best.breakdown)}")
        print(f"  actions   : {best.action_counts}")
        print(f"  board     : {best.board_birds}")
        if args.show_trajectory:
            print("\n=== best trajectory ===")
            for line in best.trajectory:
                print("  " + line)
    else:
        board = BoardType.OCEANIA if args.board == "oceania" else BoardType.BASE
        run_multi_seed(_parse_seeds(args.seeds), args.games_per_seed, args.out, board)


if __name__ == "__main__":
    main()
