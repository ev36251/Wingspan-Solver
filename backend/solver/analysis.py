"""Post-game deviation analysis.

Compares actual moves vs solver recommendations to identify
where the player deviated from optimal play.
"""

from dataclasses import dataclass, field

from backend.models.game_state import GameState, MoveRecord
from backend.models.player import Player
from backend.engine.scoring import calculate_score
from backend.solver.max_score import calculate_max_score, MaxScoreBreakdown


@dataclass
class Deviation:
    """A decision point where the player deviated from the solver's top pick."""
    round: int
    turn: int
    player_name: str
    chosen_action: str
    chosen_description: str
    recommended_description: str
    rank_chosen: int
    total_moves: int
    impact: str  # "high", "medium", "low"

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "turn": self.turn,
            "player_name": self.player_name,
            "chosen_action": self.chosen_action,
            "chosen_description": self.chosen_description,
            "recommended_description": self.recommended_description,
            "rank_chosen": self.rank_chosen,
            "total_moves": self.total_moves,
            "impact": self.impact,
        }


@dataclass
class AnalysisResult:
    """Complete analysis of a player's game."""
    actual_score: int
    max_possible_score: int
    efficiency_pct: float
    score_breakdown: dict = field(default_factory=dict)
    max_breakdown: dict = field(default_factory=dict)
    deviations: list[Deviation] = field(default_factory=list)
    moves_analyzed: int = 0
    suboptimal_moves: int = 0


def _classify_impact(solver_rank: int, total_moves: int) -> str:
    """Classify the impact of a deviation based on rank and alternatives."""
    if total_moves <= 2:
        return "low"
    if solver_rank >= 5:
        return "high"
    if solver_rank >= 3:
        return "medium"
    return "low"


def _extract_deviations(
    move_history: list[MoveRecord],
    player_name: str,
    max_deviations: int = 5,
) -> tuple[list[Deviation], int, int]:
    """Extract deviations from move history for a specific player.

    Returns (deviations, total_moves_analyzed, suboptimal_count).
    """
    deviations = []
    moves_analyzed = 0
    suboptimal = 0

    for record in move_history:
        if record.player_name != player_name:
            continue
        if record.solver_rank is None:
            continue

        moves_analyzed += 1

        if record.solver_rank > 1:
            suboptimal += 1
            impact = _classify_impact(record.solver_rank, record.total_moves)
            deviations.append(Deviation(
                round=record.round,
                turn=record.turn,
                player_name=record.player_name,
                chosen_action=record.action_type,
                chosen_description=record.description,
                recommended_description=record.best_move_description,
                rank_chosen=record.solver_rank,
                total_moves=record.total_moves,
                impact=impact,
            ))

    # Sort by impact severity (high first), then chronologically
    impact_order = {"high": 0, "medium": 1, "low": 2}
    deviations.sort(key=lambda d: (impact_order.get(d.impact, 3), d.round, d.turn))

    return deviations[:max_deviations], moves_analyzed, suboptimal


def analyze_player(game: GameState, player: Player) -> AnalysisResult:
    """Analyze a single player's performance in the game."""
    # Current score
    score = calculate_score(game, player)
    actual = score.total

    # Max possible score
    max_bd = calculate_max_score(game, player)

    # Efficiency
    efficiency = (actual / max_bd.total * 100) if max_bd.total > 0 else 0.0

    # Deviations from move history
    deviations, moves_analyzed, suboptimal = _extract_deviations(
        game.move_history, player.name
    )

    return AnalysisResult(
        actual_score=actual,
        max_possible_score=max_bd.total,
        efficiency_pct=round(efficiency, 1),
        score_breakdown=score.as_dict(),
        max_breakdown=max_bd.as_dict(),
        deviations=deviations,
        moves_analyzed=moves_analyzed,
        suboptimal_moves=suboptimal,
    )


def analyze_game(game: GameState, player_name: str | None = None) -> dict[str, AnalysisResult]:
    """Analyze one or all players in the game.

    If player_name is given, only analyze that player.
    Otherwise, analyze all players.
    """
    results = {}
    for player in game.players:
        if player_name and player.name != player_name:
            continue
        results[player.name] = analyze_player(game, player)
    return results
