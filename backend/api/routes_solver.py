"""Solver endpoint routes — heuristic, Monte Carlo, max-score, and analysis."""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.enums import FoodType
from backend.solver.heuristics import rank_moves
from backend.solver.monte_carlo import monte_carlo_evaluate, MCConfig
from backend.solver.max_score import calculate_max_score
from backend.solver.analysis import analyze_game
from backend.solver.lookahead import lookahead_search

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


@router.post("/{game_id}/solve/heuristic", response_model=HeuristicResponse)
async def solve_heuristic(game_id: str) -> HeuristicResponse:
    """Quick move ranking using static evaluation."""
    game = _get_game(game_id)

    if game.is_game_over:
        raise HTTPException(400, "Game is already over")

    start = time.perf_counter()
    ranked = rank_moves(game)
    elapsed_ms = (time.perf_counter() - start) * 1000

    recommendations = []
    for rm in ranked:
        details = {}
        if rm.move.bird_name:
            details["bird_name"] = rm.move.bird_name
        if rm.move.habitat:
            details["habitat"] = rm.move.habitat.value
        if rm.move.food_payment:
            details["food_payment"] = {
                ft.value: c for ft, c in rm.move.food_payment.items()
            }
        if rm.move.food_choices:
            details["food_choices"] = [ft.value for ft in rm.move.food_choices]

        recommendations.append(SolverMoveRecommendation(
            rank=rm.rank,
            action_type=rm.move.action_type.value,
            description=rm.move.description,
            score=round(rm.score, 2),
            reasoning=rm.reasoning,
            details=details,
        ))

    return HeuristicResponse(
        recommendations=recommendations[:3],
        evaluation_time_ms=round(elapsed_ms, 1),
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
