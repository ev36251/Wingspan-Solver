"""Setup draft analysis routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.data.registries import get_bird_registry, get_bonus_registry, get_goal_registry
from backend.solver.setup_advisor import analyze_setup

router = APIRouter()


class SetupAnalyzeRequest(BaseModel):
    bird_names: list[str] = Field(..., min_length=1, max_length=5)
    bonus_card_names: list[str] = Field(..., min_length=1, max_length=2)
    round_goals: list[str] = Field(default_factory=list)


class SetupRecommendationSchema(BaseModel):
    rank: int
    score: float
    birds_to_keep: list[str]
    food_to_keep: dict[str, int]
    bonus_card: str
    reasoning: str


class SetupAnalyzeResponse(BaseModel):
    recommendations: list[SetupRecommendationSchema]
    total_combinations: int


@router.post("/setup/analyze", response_model=SetupAnalyzeResponse)
async def analyze_draft(req: SetupAnalyzeRequest) -> SetupAnalyzeResponse:
    """Analyze starting draft options and recommend the best combination."""
    bird_reg = get_bird_registry()
    bonus_reg = get_bonus_registry()
    goal_reg = get_goal_registry()

    birds = []
    for name in req.bird_names:
        bird = bird_reg.get(name)
        if not bird:
            raise HTTPException(400, f"Bird not found: '{name}'")
        birds.append(bird)

    bonus_cards = []
    for name in req.bonus_card_names:
        bc = bonus_reg.get(name)
        if not bc:
            raise HTTPException(400, f"Bonus card not found: '{name}'")
        bonus_cards.append(bc)

    from backend.models.goal import NO_GOAL
    round_goals = []
    for desc in req.round_goals:
        if desc.lower() == "no goal":
            round_goals.append(NO_GOAL)
            continue
        found = None
        for g in goal_reg.all_goals:
            if g.description.lower() == desc.lower():
                found = g
                break
        if not found:
            raise HTTPException(400, f"Goal not found: '{desc}'")
        round_goals.append(found)

    recommendations = analyze_setup(birds, bonus_cards, round_goals)

    # Calculate total combinations
    from math import comb
    n_birds = len(birds)
    total = sum(
        comb(n_birds, k) * comb(5, 5 - k) * len(bonus_cards)
        for k in range(n_birds + 1)
        if 5 - k >= 0 and 5 - k <= 5
    )

    return SetupAnalyzeResponse(
        recommendations=[
            SetupRecommendationSchema(
                rank=r.rank,
                score=r.score,
                birds_to_keep=r.birds_to_keep,
                food_to_keep=r.food_to_keep,
                bonus_card=r.bonus_card,
                reasoning=r.reasoning,
            )
            for r in recommendations
        ],
        total_combinations=total,
    )
