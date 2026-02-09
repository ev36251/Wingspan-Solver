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
from backend.models.enums import FoodType, ActionType, Habitat
from backend.solver.heuristics import (
    evaluate_position, rank_moves, HeuristicWeights, DEFAULT_WEIGHTS,
    dynamic_weights, _estimate_goal_progress,
)
from backend.solver.simulation import execute_move_on_sim_result, execute_move_on_sim, deep_copy_game
from backend.engine.actions import execute_play_bird_discounted
from backend.engine.rules import find_food_payment_options
from backend.config import EGG_COST_BY_COLUMN
from backend.powers.registry import get_power
from backend.powers.templates.play_bird import PlayAdditionalBird
from backend.solver.move_generator import Move


@dataclass
class LookaheadResult:
    """Result of lookahead evaluation for a single root move."""
    move: Move
    score: float  # Position value at the deepest evaluated node
    rank: int = 0
    depth_reached: int = 0
    best_sequence: list[str] = field(default_factory=list)
    plan_details: list[dict] = field(default_factory=list)
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


def _snapshot_player(game: GameState, player: Player) -> dict:
    """Snapshot player resources for delta reporting."""
    food = {
        FoodType.INVERTEBRATE.value: player.food_supply.get(FoodType.INVERTEBRATE),
        FoodType.SEED.value: player.food_supply.get(FoodType.SEED),
        FoodType.FISH.value: player.food_supply.get(FoodType.FISH),
        FoodType.FRUIT.value: player.food_supply.get(FoodType.FRUIT),
        FoodType.RODENT.value: player.food_supply.get(FoodType.RODENT),
        FoodType.NECTAR.value: player.food_supply.get(FoodType.NECTAR),
    }
    cached = 0
    tucked = 0
    for row in player.board.all_rows():
        for slot in row.slots:
            if slot.bird:
                cached += slot.total_cached_food
                tucked += slot.tucked_cards
    nectar_spent = {
        "forest": player.board.forest.nectar_spent,
        "grassland": player.board.grassland.nectar_spent,
        "wetland": player.board.wetland.nectar_spent,
    }
    return {
        "food": food,
        "eggs": player.board.total_eggs(),
        "hand": player.hand_size,
        "cached": cached,
        "tucked": tucked,
        "nectar_spent": nectar_spent,
    }


def _state_signature(game: GameState, player_name: str) -> tuple:
    """Build a hashable signature of the game state for caching."""
    players_sig = []
    for p in game.players:
        board_sig = []
        for row in p.board.all_rows():
            row_sig = []
            for slot in row.slots:
                if slot.bird:
                    row_sig.append((
                        slot.bird.name, slot.eggs, slot.total_cached_food,
                        slot.tucked_cards, row.nectar_spent,
                    ))
                else:
                    row_sig.append((None, 0, 0, 0, row.nectar_spent))
            board_sig.append(tuple(row_sig))
        players_sig.append((
            p.name,
            p.action_cubes_remaining,
            p.food_supply.invertebrate, p.food_supply.seed, p.food_supply.fish,
            p.food_supply.fruit, p.food_supply.rodent, p.food_supply.nectar,
            p.hand_size,
            tuple(board_sig),
        ))

    tray = tuple(b.name for b in game.card_tray.face_up)
    return (
        player_name,
        game.current_round,
        game.current_player_idx,
        tray,
        tuple(players_sig),
    )


def _delta_snapshot(before: dict, after: dict) -> dict:
    """Compute delta between two snapshots."""
    food_delta = {}
    for k, v in after["food"].items():
        diff = v - before["food"].get(k, 0)
        if diff != 0:
            food_delta[k] = diff
    nectar_delta = {}
    for k, v in after["nectar_spent"].items():
        diff = v - before["nectar_spent"].get(k, 0)
        if diff != 0:
            nectar_delta[k] = diff
    delta = {}
    if food_delta:
        delta["food"] = food_delta
    eggs = after["eggs"] - before["eggs"]
    cards = after["hand"] - before["hand"]
    cached = after["cached"] - before["cached"]
    tucked = after["tucked"] - before["tucked"]
    if eggs:
        delta["eggs"] = eggs
    if cards:
        delta["cards"] = cards
    if cached:
        delta["cached"] = cached
    if tucked:
        delta["tucked"] = tucked
    if nectar_delta:
        delta["nectar_spent"] = nectar_delta
    return delta


def _summarize_power_activation(pa) -> str:
    """Render a single power activation into a compact string."""
    parts = []
    if pa.result.description:
        parts.append(pa.result.description)
    if pa.result.food_gained:
        for ft, c in pa.result.food_gained.items():
            parts.append(f"+{c} {ft.value}")
    if pa.result.eggs_laid:
        parts.append(f"+{pa.result.eggs_laid} egg{'s' if pa.result.eggs_laid != 1 else ''}")
    if pa.result.cards_drawn:
        parts.append(f"+{pa.result.cards_drawn} card{'s' if pa.result.cards_drawn != 1 else ''}")
    if pa.result.cards_tucked:
        parts.append(f"tuck {pa.result.cards_tucked}")
    if pa.result.food_cached:
        for ft, c in pa.result.food_cached.items():
            parts.append(f"cache {c} {ft.value}")
    detail = "; ".join(parts) if parts else "power activated"
    return f"{pa.bird_name}: {detail}"


def _goal_gap(game: GameState, player: Player) -> dict | None:
    """Compute current round goal gap vs best opponent."""
    goal = game.current_round_goal()
    if not goal or goal.description.lower() == "no goal":
        return None
    opponents = [p for p in game.players if p.name != player.name]
    my_progress = _estimate_goal_progress(player, goal)
    best_opp = max((_estimate_goal_progress(opp, goal) for opp in opponents), default=0)
    return {
        "goal": goal.description,
        "my_progress": my_progress,
        "best_opponent": best_opp,
        "gap": my_progress - best_opp,
    }


def _apply_food_discount(payment: dict[FoodType, int], discount: int, player: Player) -> dict[FoodType, int]:
    """Reduce a food payment by up to `discount` total items."""
    if discount <= 0:
        return dict(payment)
    remaining = discount
    reduced = dict(payment)
    # First: remove any tokens the player can't actually pay (covers discount)
    for ft, count in list(reduced.items()):
        if remaining <= 0:
            break
        have = player.food_supply.get(ft)
        if have < count:
            short = count - have
            use = min(short, remaining)
            new_count = count - use
            if new_count > 0:
                reduced[ft] = new_count
            else:
                reduced.pop(ft, None)
            remaining -= use
    # Then: prefer to discount non-nectar first, keep nectar for majority scoring
    order = [
        FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
        FoodType.FRUIT, FoodType.RODENT, FoodType.NECTAR,
    ]
    for ft in order:
        if remaining <= 0:
            break
        count = reduced.get(ft, 0)
        if count <= 0:
            continue
        use = min(count, remaining)
        new_count = count - use
        if new_count > 0:
            reduced[ft] = new_count
        else:
            reduced.pop(ft, None)
        remaining -= use
    return reduced


def _best_bonus_play(
    game: GameState,
    player: Player,
    power: PlayAdditionalBird,
    weights: HeuristicWeights,
    exclude: set[str] | None = None,
) -> tuple[str, Habitat, dict[FoodType, int], float] | None:
    """Pick the best additional bird to play for a PlayAdditionalBird power."""
    best: tuple[str, Habitat, dict[FoodType, int], float] | None = None
    exclude = exclude or set()
    habitats = [power.habitat_filter] if power.habitat_filter else [
        Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND
    ]

    for bird in list(player.hand):
        if bird.name in exclude:
            continue
        for hab in habitats:
            if hab is None:
                continue
            if not power.ignore_habitat and not bird.can_live_in(hab):
                continue

            row = player.board.get_row(hab)
            slot_idx = row.next_empty_slot()
            if slot_idx is None:
                continue

            egg_cost = max(0, EGG_COST_BY_COLUMN[slot_idx] - power.egg_discount)
            if player.board.total_eggs() < egg_cost:
                continue

            payment_opts = find_food_payment_options(player, bird.food_cost)
            if not payment_opts and power.food_discount > 0:
                # Try generating options with virtual flexible food, then apply discount
                temp = type("Tmp", (), {})()
                temp.food_supply = type("FS", (), {"get": player.food_supply.get,
                                                   "has": player.food_supply.has,
                                                   "total": player.food_supply.total,
                                                   "total_non_nectar": player.food_supply.total_non_nectar})()
                # Monkey-patch nectar count
                orig_get = temp.food_supply.get
                def _get(ft):
                    if ft == FoodType.NECTAR:
                        return orig_get(ft) + power.food_discount
                    return orig_get(ft)
                temp.food_supply.get = _get
                temp.food_supply.has = lambda ft, c=1: _get(ft) >= c
                payment_opts = find_food_payment_options(temp, bird.food_cost)
            if not payment_opts:
                payment_opts = [{}]
            for payment in payment_opts:
                reduced = _apply_food_discount(payment, power.food_discount, player)
                if any(player.food_supply.get(ft) < cnt for ft, cnt in reduced.items()):
                    continue

                # Quick delta estimate: use move value as a proxy
                from backend.solver.heuristics import _estimate_move_value
                move = Move(
                    action_type=ActionType.PLAY_BIRD,
                    description=f"Bonus play {bird.name} in {hab.value}",
                    bird_name=bird.name,
                    habitat=hab,
                    food_payment=reduced,
                )
                move_score = _estimate_move_value(game, player, move, weights)

                if best is None or move_score > best[3]:
                    best = (bird.name, hab, reduced, move_score)

    return best


def lookahead_search(
    game: GameState,
    player: Player | None = None,
    depth: int = 2,
    beam_width: int = 6,
    weights: HeuristicWeights | None = None,
    _cache: dict | None = None,
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

    if _cache is None:
        _cache = {"rank": {}, "eval": {}}

    # Get heuristic-ranked candidates, pruned by beam width
    sig = _state_signature(game, player_name)
    if sig in _cache["rank"]:
        candidates = _cache["rank"][sig]
    else:
        candidates = rank_moves(game, player, weights)
        _cache["rank"][sig] = candidates
    if not candidates:
        return []
    candidates = candidates[:beam_width]

    results = []
    for rm in candidates:
        sim = deep_copy_game(game)
        sim_player = sim.get_player(player_name)

        # Execute the candidate move on the simulation
        before = _snapshot_player(sim, sim_player)
        goal_before = _goal_gap(sim, sim_player)
        success, result = execute_move_on_sim_result(sim, sim_player, rm.move)
        if not success:
            # Move failed — fall back to heuristic score
            results.append(LookaheadResult(
                move=rm.move, score=rm.score, depth_reached=0,
                best_sequence=[rm.move.description],
                plan_details=[{
                    "description": rm.move.description,
                    "delta": {},
                }],
                heuristic_score=rm.score,
            ))
            continue
        bonus_power_notes: list[str] = []
        # Auto-execute play-another-bird powers after playing a bird
        if rm.move.action_type == ActionType.PLAY_BIRD and rm.move.bird_name:
            bird_reg = None
            try:
                from backend.data.registries import get_bird_registry
                bird_reg = get_bird_registry()
            except Exception:
                bird_reg = None
            if bird_reg:
                played_bird = bird_reg.get(rm.move.bird_name)
                if played_bird:
                    power = get_power(played_bird)
                    chain_count = 0
                    excluded = {played_bird.name}
                    while isinstance(power, PlayAdditionalBird) and chain_count < 2:
                        pick = _best_bonus_play(sim, sim_player, power, weights, exclude=excluded)
                        if not pick:
                            break
                        bname, hab, payment, _ = pick
                        bonus_bird = bird_reg.get(bname)
                        if not bonus_bird:
                            break
                        bonus_result = execute_play_bird_discounted(
                            sim, sim_player, bonus_bird, hab, payment,
                            egg_discount=power.egg_discount,
                        )
                        if not bonus_result.success:
                            break
                        bonus_power_notes.append(
                            f"Bonus play: {bname} in {hab.value}"
                        )
                        excluded.add(bname)
                        power = get_power(bonus_bird)
                        chain_count += 1

        after = _snapshot_player(sim, sim_player)
        goal_after = _goal_gap(sim, sim_player)
        delta = _delta_snapshot(before, after)
        power_notes = []
        if result and result.power_activations:
            power_notes = [_summarize_power_activation(pa) for pa in result.power_activations]
        if bonus_power_notes:
            power_notes.extend(bonus_power_notes)
        detail = {
            "description": rm.move.description,
            "delta": delta,
        }
        if power_notes:
            detail["power"] = power_notes
        if goal_before and goal_after:
            detail["goal"] = {
                "description": goal_before["goal"],
                "gap_before": goal_before["gap"],
                "gap_after": goal_after["gap"],
            }

        sim.advance_turn()

        # Depth 1: evaluate position immediately after our move
        if depth <= 1 or sim.is_game_over:
            sim_player = sim.get_player(player_name)
            eval_sig = _state_signature(sim, player_name)
            if eval_sig in _cache["eval"]:
                score = _cache["eval"][eval_sig]
            else:
                score = evaluate_position(sim, sim_player, weights)
                _cache["eval"][eval_sig] = score
            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=1,
                best_sequence=[rm.move.description],
                plan_details=[detail],
                heuristic_score=rm.score,
            ))
            continue

        # Depth 2+: advance through opponent turns, then recurse
        reached = _advance_to_player(sim, player_name)
        sim_player = sim.get_player(player_name)

        if not reached or sim.is_game_over or not sim_player:
            # Game ended or couldn't reach our turn
            eval_player = sim_player or player
            eval_sig = _state_signature(sim, player_name)
            if eval_sig in _cache["eval"]:
                score = _cache["eval"][eval_sig]
            else:
                score = evaluate_position(sim, eval_player, weights)
                _cache["eval"][eval_sig] = score
            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=1,
                best_sequence=[rm.move.description],
                plan_details=[detail],
                heuristic_score=rm.score,
            ))
        else:
            # Recurse with reduced depth
            inner = lookahead_search(
                sim, sim_player, depth - 1, beam_width, weights, _cache
            )
            if inner:
                best_inner = inner[0]
                score = best_inner.score
                seq = [rm.move.description] + best_inner.best_sequence
                plan = [detail] + best_inner.plan_details
                inner_depth = best_inner.depth_reached + 1
            else:
                eval_sig = _state_signature(sim, player_name)
                if eval_sig in _cache["eval"]:
                    score = _cache["eval"][eval_sig]
                else:
                    score = evaluate_position(sim, sim_player, weights)
                    _cache["eval"][eval_sig] = score
                seq = [rm.move.description]
                plan = [detail]
                inner_depth = 1

            results.append(LookaheadResult(
                move=rm.move, score=score, depth_reached=inner_depth,
                best_sequence=seq,
                plan_details=plan,
                heuristic_score=rm.score,
            ))

    # Sort by lookahead score descending
    results.sort(key=lambda r: -r.score)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
