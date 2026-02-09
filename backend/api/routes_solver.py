"""Solver endpoint routes — heuristic, Monte Carlo, max-score, and analysis."""

import copy
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.enums import FoodType, ActionType, Habitat
from backend.models.player import Player
from backend.config import EGG_COST_BY_COLUMN, get_action_column
from backend.data.registries import get_bird_registry
from backend.solver.heuristics import (
    rank_moves, _activation_advice, _player_after_bonus,
    estimate_move_breakdown, dynamic_weights,
)
from backend.solver.monte_carlo import monte_carlo_evaluate, MCConfig
from backend.solver.max_score import calculate_max_score
from backend.solver.analysis import analyze_game
from backend.solver.lookahead import lookahead_search
from backend.solver.move_generator import Move
from backend.solver.simulation import deep_copy_game, execute_move_on_sim, simulate_playout

router = APIRouter()

# Import game store from routes_game
from backend.api.routes_game import _get_game


class SolverMoveRecommendation(BaseModel):
    rank: int
    action_type: str
    description: str
    score: float
    reasoning: str = ""
    details: dict = Field(default_factory=dict)


class HeuristicResponse(BaseModel):
    recommendations: list[SolverMoveRecommendation]
    evaluation_time_ms: float = 0
    feeder_reroll_available: bool = False
    player_name: str = ""


class MonteCarloRequest(BaseModel):
    simulations_per_move: int = Field(default=50, ge=5, le=500)
    time_limit_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


class MonteCarloResponse(BaseModel):
    recommendations: list[SolverMoveRecommendation]
    simulations_run: int = 0
    evaluation_time_ms: float = 0


class MaxScoreResponse(BaseModel):
    max_possible_score: int = 0
    current_score: int = 0
    efficiency_pct: float = 0
    breakdown: dict = Field(default_factory=dict)


class LookaheadRequest(BaseModel):
    depth: int = Field(default=2, ge=1, le=3)
    beam_width: int = Field(default=6, ge=2, le=10)


class LookaheadMoveRec(BaseModel):
    rank: int
    action_type: str
    description: str
    score: float
    heuristic_score: float
    depth_reached: int
    best_sequence: list[str]
    details: dict = Field(default_factory=dict)


class LookaheadResponse(BaseModel):
    recommendations: list[LookaheadMoveRec]
    evaluation_time_ms: float = 0
    depth: int = 2
    beam_width: int = 6


class AnalysisResponse(BaseModel):
    actual_score: int = 0
    max_possible_score: int = 0
    efficiency_pct: float = 0
    deviations: list[dict] = Field(default_factory=list)


def _project_final_scores(
    game,
    player_name: str,
    move: Move,
    simulations: int,
) -> list[int]:
    """Estimate final score distribution for a move via rollout playouts."""
    scores: list[int] = []
    for _ in range(max(1, simulations)):
        sim = deep_copy_game(game)
        sim_player = sim.get_player(player_name)
        if not sim_player:
            continue
        success = execute_move_on_sim(sim, sim_player, move)
        if success:
            sim.advance_turn()
        result = simulate_playout(sim, max_turns=260)
        if player_name in result:
            scores.append(result[player_name])
    return scores


@router.post("/{game_id}/solve/heuristic", response_model=HeuristicResponse)
async def solve_heuristic(game_id: str, player_idx: int | None = None) -> HeuristicResponse:
    """Quick move ranking using static evaluation.

    Pass player_idx to evaluate for a specific player (0-based).
    Defaults to the game's current_player_idx.
    """
    game = _get_game(game_id)

    if game.is_game_over:
        raise HTTPException(400, "Game is already over")

    # Resolve target player
    if player_idx is not None:
        if player_idx < 0 or player_idx >= len(game.players):
            raise HTTPException(400, f"Invalid player_idx: {player_idx}")
        player = game.players[player_idx]
    else:
        player = game.current_player

    start = time.perf_counter()

    # Adaptive lookahead: deeper search in late game when tree is small
    cubes = player.action_cubes_remaining
    if game.current_round >= 3 and cubes <= 4:
        depth = min(3, cubes)
    else:
        depth = 2
    la_results = lookahead_search(game, player=player, depth=depth, beam_width=5)
    # Also get heuristic reasoning for display
    weights = dynamic_weights(game)
    heuristic_ranked = rank_moves(game, player=player, weights=weights)
    reasoning_map = {rm.move.description: rm.reasoning for rm in heuristic_ranked}

    # End-score mode: rerank top candidates by projected final game score.
    # Target ~3s total compute for stronger long-horizon quality.
    target_compute_seconds = 2.8
    actions_left = game.total_actions_remaining
    if actions_left >= 50:
        sims_per_move = 12
    elif actions_left >= 30:
        sims_per_move = 16
    elif actions_left >= 16:
        sims_per_move = 22
    else:
        sims_per_move = 30

    projections: dict[str, dict[str, float]] = {}
    if la_results:
        eval_count = min(8, len(la_results))
        projected = []
        for idx, la in enumerate(la_results[:eval_count]):
            if time.perf_counter() - start > target_compute_seconds and idx >= 3:
                # Ensure we project at least a few top moves, then respect budget.
                projected.append((la.score, la))
                continue
            rollout_scores = _project_final_scores(
                game, player.name, la.move, simulations=sims_per_move
            )
            if rollout_scores:
                avg = sum(rollout_scores) / len(rollout_scores)
                projections[la.move.description] = {
                    "expected_final_score": round(avg, 2),
                    "p100": round(sum(1 for s in rollout_scores if s >= 100) / len(rollout_scores), 3),
                    "p110": round(sum(1 for s in rollout_scores if s >= 110) / len(rollout_scores), 3),
                    "p120": round(sum(1 for s in rollout_scores if s >= 120) / len(rollout_scores), 3),
                    "rollout_samples": float(len(rollout_scores)),
                }
                projected.append((avg, la))
            else:
                projected.append((la.score, la))

        # Keep unevaluated candidates but prioritize projected final score.
        projected.sort(key=lambda x: -x[0])
        ordered = [la for _, la in projected]
        if len(la_results) > eval_count:
            ordered.extend(la_results[eval_count:])
        for i, la in enumerate(ordered):
            la.rank = i + 1
            if la.move.description in projections:
                la.score = projections[la.move.description]["expected_final_score"]
        la_results = ordered

    elapsed_ms = (time.perf_counter() - start) * 1000

    bird_reg = get_bird_registry()

    recommendations = []
    def _activation_player_after_move(base_player: Player, move: Move, column_bonus) -> Player:
        """Prepare a player snapshot for activation advice after the move's gains."""
        advice_player = copy.deepcopy(base_player)
        advice_player = _player_after_bonus(advice_player, move, column_bonus)

        if move.action_type == ActionType.GAIN_FOOD:
            for ft in move.food_choices:
                advice_player.food_supply.add(ft)
        elif move.action_type == ActionType.DRAW_CARDS:
            gained = move.deck_draws + len(move.tray_indices)
            if gained > 0:
                advice_player.unknown_hand_count += gained
        elif move.action_type == ActionType.LAY_EGGS:
            for (hab, slot_idx), count in move.egg_distribution.items():
                row = advice_player.board.get_row(hab)
                if slot_idx < 0 or slot_idx >= len(row.slots):
                    continue
                slot = row.slots[slot_idx]
                if slot.bird is None:
                    continue
                for _ in range(count):
                    if slot.can_hold_more_eggs():
                        slot.eggs += 1

        return advice_player

    for la in la_results:
        details = {}
        if la.move.bird_name:
            details["bird_name"] = la.move.bird_name
            bird = bird_reg.get(la.move.bird_name)
            if bird:
                details["bird_vp"] = bird.victory_points
                details["bird_nest_type"] = bird.nest_type.value if bird.nest_type else None
                details["bird_egg_limit"] = bird.egg_limit
        if la.move.habitat:
            details["habitat"] = la.move.habitat.value
            # Include egg column cost for play-bird moves
            if la.move.action_type.value == "play_bird":
                row = player.board.get_row(la.move.habitat)
                col = row.bird_count
                egg_cost = EGG_COST_BY_COLUMN[col] if col < len(EGG_COST_BY_COLUMN) else 0
                details["egg_cost"] = egg_cost
        if la.move.food_payment:
            details["food_payment"] = {
                ft.value: c for ft, c in la.move.food_payment.items()
            }
        if la.move.food_choices:
            details["food_choices"] = [ft.value for ft in la.move.food_choices]
        if la.move.egg_distribution:
            details["egg_distribution"] = {
                hab.value: {str(slot_idx): count}
                for (hab, slot_idx), count in la.move.egg_distribution.items()
            }
        if la.move.tray_indices:
            details["tray_indices"] = la.move.tray_indices
        details["deck_draws"] = la.move.deck_draws
        details["bonus_count"] = la.move.bonus_count
        details["reset_bonus"] = la.move.reset_bonus
        details["score_mode"] = "projected_final_score"
        details["score_target"] = "maximize_end_game_points"
        if la.move.description in projections:
            details["projected_final_score"] = projections[la.move.description]["expected_final_score"]
            details["p100"] = projections[la.move.description]["p100"]
            details["p110"] = projections[la.move.description]["p110"]
            details["p120"] = projections[la.move.description]["p120"]
            details["rollout_samples"] = int(projections[la.move.description]["rollout_samples"])
        if la.best_sequence and len(la.best_sequence) > 1:
            details["best_sequence"] = la.best_sequence
            details["plan"] = la.best_sequence[:3]
            details["plan_depth"] = la.depth_reached
        if la.plan_details:
            details["plan_details"] = la.plan_details[:3]
        details["breakdown"] = estimate_move_breakdown(game, player, la.move, weights=weights)

        # Provide explicit activation advice for habitat actions
        if la.move.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS, ActionType.DRAW_CARDS):
            hab = la.move.habitat
            if hab is None:
                if la.move.action_type == ActionType.GAIN_FOOD:
                    hab = Habitat.FOREST
                elif la.move.action_type == ActionType.LAY_EGGS:
                    hab = Habitat.GRASSLAND
                else:
                    hab = Habitat.WETLAND
            column = get_action_column(game.board_type, hab, player.board.get_row(hab).bird_count)
            advice_player = _activation_player_after_move(player, la.move, column.bonus)
            advice = _activation_advice(game, advice_player, hab)
            if advice:
                details["activation_advice"] = advice

        recommendations.append(SolverMoveRecommendation(
            rank=la.rank,
            action_type=la.move.action_type.value,
            description=la.move.description,
            score=round(la.score, 2),
            reasoning=reasoning_map.get(la.move.description, ""),
            details=details,
        ))

    # Explainability: compare top picks
    if recommendations:
        recommendations.sort(key=lambda r: r.rank)
        top = recommendations[0]
        if len(recommendations) > 1:
            second = recommendations[1]
            margin = round(top.score - second.score, 2)
            if margin != 0:
                top.details["margin_vs_next"] = margin
        # Goal gap change (if present in plan details)
        if isinstance(top.details.get("plan_details"), list):
            for step in top.details["plan_details"]:
                goal = step.get("goal")
                if goal:
                    top.details.setdefault("why", []).append(
                        f"Goal gap {goal.get('gap_before')} → {goal.get('gap_after')}"
                    )
                    break
        if "margin_vs_next" in top.details:
            top.details.setdefault("why", []).insert(0, f"+{top.details['margin_vs_next']} pts vs #2")

    # Check if the feeder has all-same-face (free reroll available)
    feeder = game.birdfeeder
    reroll_available = feeder.all_same_face() and feeder.count > 0

    return HeuristicResponse(
        recommendations=recommendations[:3],
        evaluation_time_ms=round(elapsed_ms, 1),
        feeder_reroll_available=reroll_available,
        player_name=player.name,
    )


@router.post("/{game_id}/solve/monte-carlo", response_model=MonteCarloResponse)
async def solve_monte_carlo(
    game_id: str,
    req: MonteCarloRequest | None = None,
) -> MonteCarloResponse:
    """Deep analysis using Monte Carlo simulation."""
    game = _get_game(game_id)

    if game.is_game_over:
        raise HTTPException(400, "Game is already over")

    if req is None:
        req = MonteCarloRequest()

    config = MCConfig(
        simulations_per_move=req.simulations_per_move,
        time_limit_seconds=req.time_limit_seconds,
    )

    start = time.perf_counter()
    mc_results = monte_carlo_evaluate(game, config)
    elapsed_ms = (time.perf_counter() - start) * 1000

    total_sims = sum(r.simulations for r in mc_results)

    recommendations = []
    for r in mc_results:
        details = {}
        if r.move.bird_name:
            details["bird_name"] = r.move.bird_name
        if r.move.habitat:
            details["habitat"] = r.move.habitat.value
        if r.move.food_payment:
            details["food_payment"] = {
                ft.value: c for ft, c in r.move.food_payment.items()
            }
        if r.move.food_choices:
            details["food_choices"] = [ft.value for ft in r.move.food_choices]
        details["simulations"] = r.simulations

        recommendations.append(SolverMoveRecommendation(
            rank=r.rank,
            action_type=r.move.action_type.value,
            description=r.move.description,
            score=round(r.avg_score, 1),
            details=details,
        ))

    return MonteCarloResponse(
        recommendations=recommendations,
        simulations_run=total_sims,
        evaluation_time_ms=round(elapsed_ms, 1),
    )


@router.post("/{game_id}/solve/lookahead", response_model=LookaheadResponse)
async def solve_lookahead(
    game_id: str,
    req: LookaheadRequest | None = None,
) -> LookaheadResponse:
    """Move ranking using depth-limited lookahead search.

    Simulates executing each top candidate move, fast-forwards through
    opponent turns, and recursively evaluates future positions. Finds
    multi-turn combos the heuristic alone would miss.
    """
    game = _get_game(game_id)

    if game.is_game_over:
        raise HTTPException(400, "Game is already over")

    if req is None:
        req = LookaheadRequest()

    start = time.perf_counter()
    results = lookahead_search(
        game, depth=req.depth, beam_width=req.beam_width,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    recommendations = []
    for r in results:
        details = {}
        if r.move.bird_name:
            details["bird_name"] = r.move.bird_name
        if r.move.habitat:
            details["habitat"] = r.move.habitat.value
        if r.move.food_payment:
            details["food_payment"] = {
                ft.value: c for ft, c in r.move.food_payment.items()
            }
        if r.move.food_choices:
            details["food_choices"] = [ft.value for ft in r.move.food_choices]

        recommendations.append(LookaheadMoveRec(
            rank=r.rank,
            action_type=r.move.action_type.value,
            description=r.move.description,
            score=round(r.score, 2),
            heuristic_score=round(r.heuristic_score, 2),
            depth_reached=r.depth_reached,
            best_sequence=r.best_sequence,
            details=details,
        ))

    return LookaheadResponse(
        recommendations=recommendations,
        evaluation_time_ms=round(elapsed_ms, 1),
        depth=req.depth,
        beam_width=req.beam_width,
    )


@router.post("/{game_id}/solve/max-score", response_model=MaxScoreResponse)
async def solve_max_score(game_id: str, player_name: str | None = None) -> MaxScoreResponse:
    """Calculate theoretical maximum score from current state."""
    game = _get_game(game_id)

    # Default to current player if not specified
    if player_name:
        player = game.get_player(player_name)
        if not player:
            raise HTTPException(400, f"Player '{player_name}' not found")
    else:
        player = game.current_player

    from backend.engine.scoring import calculate_score
    current_score = calculate_score(game, player).total
    max_bd = calculate_max_score(game, player)

    efficiency = (current_score / max_bd.total * 100) if max_bd.total > 0 else 0.0

    return MaxScoreResponse(
        max_possible_score=max_bd.total,
        current_score=current_score,
        efficiency_pct=round(efficiency, 1),
        breakdown=max_bd.as_dict(),
    )


class AfterResetRequest(BaseModel):
    """Input for after-reset follow-up recommendation.

    After the solver recommends a move with a reset bonus, the user
    physically resets the feeder or tray, inputs the new state, and
    gets a follow-up recommendation for which food/cards to take.
    """
    reset_type: str  # "feeder" or "tray"
    new_feeder_dice: list[str | list[str]] = Field(default_factory=list)
    new_tray_cards: list[str] = Field(default_factory=list)
    total_to_gain: int = Field(default=0, ge=0, le=6)


class AfterResetRecommendation(BaseModel):
    rank: int
    description: str
    score: float
    reasoning: str = ""
    details: dict = Field(default_factory=dict)


class AfterResetResponse(BaseModel):
    recommendations: list[AfterResetRecommendation]
    reset_type: str
    total_to_gain: int


@router.post("/{game_id}/solve/after-reset", response_model=AfterResetResponse)
async def solve_after_reset(game_id: str, req: AfterResetRequest) -> AfterResetResponse:
    """Follow-up recommendation after feeder/tray reset.

    When the solver recommends "reset feeder" or "reset tray", the user
    physically resets it, inputs the new dice/cards here, and gets a
    specific recommendation for which food to take or which cards to draw.
    """
    game = _get_game(game_id)

    if game.is_game_over:
        raise HTTPException(400, "Game is already over")

    if req.reset_type not in ("feeder", "tray"):
        raise HTTPException(400, "reset_type must be 'feeder' or 'tray'")

    # Create a temporary game copy with updated feeder/tray
    temp_game = copy.deepcopy(game)
    player = temp_game.current_player

    if req.reset_type == "feeder":
        if not req.new_feeder_dice:
            raise HTTPException(400, "new_feeder_dice required for feeder reset")

        # Parse dice faces
        from backend.models.birdfeeder import Birdfeeder
        dice = []
        for d in req.new_feeder_dice:
            if isinstance(d, list):
                dice.append(tuple(FoodType(f) for f in d))
            else:
                dice.append(FoodType(d))
        temp_game.birdfeeder.set_dice(dice)

        # Determine how many food to gain
        if req.total_to_gain > 0:
            food_count = req.total_to_gain
        else:
            from backend.config import get_action_column
            from backend.models.enums import Habitat
            bird_count = player.board.forest.bird_count
            column = get_action_column(temp_game.board_type, Habitat.FOREST, bird_count)
            food_count = column.base_gain

        # Generate food moves for the given count
        from backend.solver.move_generator import (
            _feeder_type_counts, _generate_food_combos, _food_combo_description,
            Move, generate_gain_food_moves,
        )
        from backend.models.enums import ActionType

        type_counts = _feeder_type_counts(temp_game.birdfeeder)
        combos = _generate_food_combos(type_counts, food_count)

        # Filter out combos that exceed actual dice availability
        # (the generator includes single-type fallbacks for mid-action rerolls,
        # but after a reset the dice are fixed)
        from collections import Counter
        valid_combos = []
        for combo in combos:
            combo_counts = Counter(combo)
            if all(combo_counts[ft] <= type_counts.get(ft, 0) for ft in combo_counts):
                valid_combos.append(combo)
        combos = valid_combos if valid_combos else combos[:1]

        # Score each combo using the heuristic evaluator
        from backend.solver.heuristics import _evaluate_gain_food, dynamic_weights
        weights = dynamic_weights(temp_game)

        scored_combos = []
        for combo in combos:
            move = Move(
                action_type=ActionType.GAIN_FOOD,
                description=f"Take {_food_combo_description(combo)}",
                food_choices=combo,
            )
            score = _evaluate_gain_food(temp_game, player, move, weights)
            scored_combos.append((combo, move, score))

        # Sort by score descending
        scored_combos.sort(key=lambda x: -x[2])

        recommendations = []
        for i, (combo, move, score) in enumerate(scored_combos[:5]):
            # Generate reasoning for the top food pick
            reasoning_parts = []
            # Check if this combo fills food needs for hand birds
            if player.hand:
                for bird in player.hand:
                    if not bird.food_cost.is_or:
                        needed = {}
                        for ft in bird.food_cost.items:
                            needed[ft] = needed.get(ft, 0) + 1
                        for ft in list(needed.keys()):
                            have = player.food_supply.get(ft)
                            needed[ft] = max(0, needed[ft] - have)
                            if needed[ft] == 0:
                                del needed[ft]
                        if needed:
                            filled = sum(1 for ft in combo if ft in needed)
                            if filled > 0:
                                reasoning_parts.append(
                                    f"fills {filled} food need{'s' if filled > 1 else ''} for {bird.name}"
                                )
                                break
                    else:
                        # OR cost: any matching food works
                        if any(ft in bird.food_cost.distinct_types for ft in combo):
                            reasoning_parts.append(f"covers cost for {bird.name}")
                            break

            # Nectar value
            if FoodType.NECTAR in combo:
                reasoning_parts.append("gains nectar for majority")

            # Food diversity
            unique_types = len(set(combo))
            if unique_types >= 3:
                reasoning_parts.append("diverse food types")

            recommendations.append(AfterResetRecommendation(
                rank=i + 1,
                description=_food_combo_description(combo),
                score=round(score, 2),
                reasoning="; ".join(reasoning_parts) if reasoning_parts else "general food gain",
                details={"food_choices": [ft.value for ft in combo]},
            ))

        return AfterResetResponse(
            recommendations=recommendations,
            reset_type="feeder",
            total_to_gain=food_count,
        )

    else:  # tray reset
        if not req.new_tray_cards:
            raise HTTPException(400, "new_tray_cards required for tray reset")

        # Replace tray cards
        from backend.data.registries import get_bird_registry
        bird_reg = get_bird_registry()
        temp_game.card_tray.clear()
        for name in req.new_tray_cards:
            bird = bird_reg.get(name)
            if not bird:
                raise HTTPException(400, f"Bird not found: '{name}'")
            temp_game.card_tray.add_card(bird)

        # Determine how many cards to draw
        if req.total_to_gain > 0:
            card_count = req.total_to_gain
        else:
            from backend.config import get_action_column
            from backend.models.enums import Habitat
            bird_count = player.board.wetland.bird_count
            column = get_action_column(temp_game.board_type, Habitat.WETLAND, bird_count)
            card_count = column.base_gain

        # Evaluate each tray card as a pick option
        from backend.solver.heuristics import (
            _goal_alignment_value, dynamic_weights,
        )
        from backend.models.enums import ActionType
        weights = dynamic_weights(temp_game)

        card_options = []
        for i, bird in enumerate(temp_game.card_tray.face_up):
            score = 0.0
            reasoning_parts = []

            # VP value
            score += bird.victory_points * 0.5
            if bird.victory_points >= 5:
                reasoning_parts.append(f"high VP ({bird.victory_points})")
            elif bird.victory_points > 0:
                reasoning_parts.append(f"{bird.victory_points}VP")

            # Power value
            from backend.models.enums import PowerColor
            from backend.powers.registry import get_power
            from backend.powers.base import NoPower
            rounds_remaining = max(0, 4 - temp_game.current_round)
            if bird.color == PowerColor.BROWN and rounds_remaining > 0:
                power = get_power(bird)
                if not isinstance(power, NoPower):
                    score += rounds_remaining * 0.8
                    reasoning_parts.append("brown engine power")
            elif bird.color == PowerColor.WHITE:
                reasoning_parts.append("when-played power")
                score += 0.5

            # Bonus card synergy
            for bc in player.bonus_cards:
                if bc.name in bird.bonus_eligibility:
                    score += 1.2
                    reasoning_parts.append(f"bonus: {bc.name}")
                    break

            # Goal alignment
            best_goal_val = 0.0
            for hab in bird.habitats:
                gv = _goal_alignment_value(temp_game, bird, hab, weights)
                best_goal_val = max(best_goal_val, gv)
            if best_goal_val > 0.3:
                score += best_goal_val
                reasoning_parts.append("helps round goal")

            # Food affordability
            food_available = (player.food_supply.total_non_nectar() +
                              player.food_supply.get(FoodType.NECTAR))
            if food_available >= bird.food_cost.total:
                score += 0.5
                reasoning_parts.append("affordable now")
            elif food_available + 2 >= bird.food_cost.total:
                reasoning_parts.append("nearly affordable")

            # Egg capacity
            if bird.egg_limit >= 4:
                score += 0.4
                reasoning_parts.append(f"holds {bird.egg_limit} eggs")

            card_options.append((bird, score, reasoning_parts, i))

        # Sort by score descending
        card_options.sort(key=lambda x: -x[1])

        recommendations = []
        for rank, (bird, score, reasoning_parts, tray_idx) in enumerate(card_options):
            deck_draws = max(0, card_count - 1)
            if card_count > 1:
                desc = f"Take {bird.name} from tray + {deck_draws} from deck"
            else:
                desc = f"Take {bird.name} from tray"

            recommendations.append(AfterResetRecommendation(
                rank=rank + 1,
                description=desc,
                score=round(score, 2),
                reasoning="; ".join(reasoning_parts) if reasoning_parts else "general card draw",
                details={
                    "bird_name": bird.name,
                    "tray_index": tray_idx,
                    "deck_draws": deck_draws,
                },
            ))

        # Also add "all from deck" option
        if temp_game.deck_remaining >= card_count:
            recommendations.append(AfterResetRecommendation(
                rank=len(recommendations) + 1,
                description=f"Draw {card_count} from deck",
                score=round(card_count * 0.3, 2),
                reasoning="unknown cards, no tray picks",
                details={"bird_name": None, "tray_index": -1, "deck_draws": card_count},
            ))

        return AfterResetResponse(
            recommendations=recommendations,
            reset_type="tray",
            total_to_gain=card_count,
        )


@router.post("/{game_id}/analyze", response_model=AnalysisResponse)
async def analyze_game_endpoint(game_id: str, player_name: str | None = None) -> AnalysisResponse:
    """Post-game deviation analysis."""
    game = _get_game(game_id)

    # Default to current player if not specified
    if player_name:
        player = game.get_player(player_name)
        if not player:
            raise HTTPException(400, f"Player '{player_name}' not found")
    else:
        player = game.current_player

    results = analyze_game(game, player.name)
    result = results.get(player.name)
    if not result:
        raise HTTPException(500, "Analysis failed")

    return AnalysisResponse(
        actual_score=result.actual_score,
        max_possible_score=result.max_possible_score,
        efficiency_pct=result.efficiency_pct,
        deviations=[d.as_dict() for d in result.deviations],
    )
