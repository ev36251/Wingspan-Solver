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
    tray_cards: list[str] = Field(default_factory=list, max_length=3)
    turn_order: int = Field(default=1, ge=1, le=5)
    num_players: int = Field(default=2, ge=2, le=5)
    rollout_top_k: int = Field(default=5, ge=0, le=20)
    rollout_simulations: int = Field(default=15, ge=0, le=60)
    rollout_max_turns: int = Field(default=220, ge=60, le=300)
    # Use the net-guided rollout-search engine to evaluate each opening (the real
    # ~92 agent plays the draft out). Strongest, but slower (~35s per opening).
    use_engine: bool = Field(default=True)
    engine_openings: int = Field(default=4, ge=2, le=6)  # how many openings the engine plays out
    engine_sims: int = Field(default=2, ge=1, le=6)
    engine_rollouts: int = Field(default=1, ge=1, le=4)


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
def analyze_draft(req: SetupAnalyzeRequest) -> SetupAnalyzeResponse:
    """Analyze starting draft options and recommend the best combination.

    Plain `def` (not `async`): this runs many full-game engine playouts (CPU
    bound, tens of seconds). An `async` handler would run it on the event loop
    and freeze every other request until it finished; `def` runs it in the
    threadpool so the rest of the app stays responsive.
    """
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

    tray_birds = []
    for name in req.tray_cards:
        bird = bird_reg.get(name)
        if not bird:
            raise HTTPException(400, f"Bird not found: '{name}'")
        tray_birds.append(bird)

    # Optionally load the deployed engine so it plays each opening out itself.
    model = encoder = None
    student_model = student_encoder = None
    sims = req.rollout_simulations
    top_k = req.rollout_top_k
    if req.use_engine:
        from backend.api.routes_solver import _get_policy_components, _get_rollout_student
        model, encoder = _get_policy_components()
        if model is not None:
            # Engine playouts are expensive; the distilled student makes each
            # rollout ~3x cheaper (same one the live "Recommend" advisor uses),
            # so the draft analysis finishes in a fraction of the time.
            student_model, student_encoder = _get_rollout_student()
            sims = req.engine_sims
            top_k = req.engine_openings

    recommendations = analyze_setup(
        birds, bonus_cards, round_goals,
        tray_birds=tray_birds,
        turn_order=req.turn_order,
        num_players=req.num_players,
        rollout_top_k=top_k,
        rollout_simulations=sims,
        rollout_max_turns=req.rollout_max_turns,
        model=model,
        encoder=encoder,
        eng_rollouts=req.engine_rollouts,
        rollout_model=student_model,
        rollout_encoder=student_encoder,
    )

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
