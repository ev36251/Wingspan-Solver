"""Data lookup routes: birds, bonus cards, goals."""

from fastapi import APIRouter, Query

from backend.data.registries import get_bird_registry, get_bonus_registry, get_goal_registry
from backend.models.enums import GameSet, Habitat, PowerColor, NestType
from backend.api.schemas import BirdListResponse, BonusCardListResponse, GoalListResponse
from backend.api.serializers import bird_to_schema, bonus_card_to_schema, goal_to_schema

router = APIRouter()


@router.get("/birds", response_model=BirdListResponse)
async def list_birds(
    search: str | None = Query(None, description="Search birds by name"),
    game_set: str | None = Query(None, description="Filter by expansion set"),
    habitat: str | None = Query(None, description="Filter by habitat"),
    color: str | None = Query(None, description="Filter by power color"),
    nest_type: str | None = Query(None, description="Filter by nest type"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    reg = get_bird_registry()

    if search:
        birds = reg.search(search)
    elif game_set:
        birds = reg.by_set(GameSet(game_set))
    elif habitat:
        birds = reg.by_habitat(Habitat(habitat))
    elif color:
        birds = reg.by_color(PowerColor(color))
    elif nest_type:
        birds = reg.by_nest(NestType(nest_type))
    else:
        birds = reg.all_birds

    total = len(birds)
    birds = birds[offset:offset + limit]
    return BirdListResponse(
        birds=[bird_to_schema(b) for b in birds],
        total=total,
    )


@router.get("/birds/{bird_name}")
async def get_bird(bird_name: str):
    reg = get_bird_registry()
    bird = reg.get(bird_name)
    if not bird:
        from fastapi import HTTPException
        raise HTTPException(404, f"Bird '{bird_name}' not found")
    return bird_to_schema(bird)


@router.get("/bonus-cards", response_model=BonusCardListResponse)
async def list_bonus_cards(
    game_set: str | None = Query(None, description="Filter by expansion set"),
):
    reg = get_bonus_registry()
    if game_set:
        cards = reg.by_set(GameSet(game_set))
    else:
        cards = reg.all_cards
    return BonusCardListResponse(
        bonus_cards=[bonus_card_to_schema(c) for c in cards],
        total=len(cards),
    )


@router.get("/goals", response_model=GoalListResponse)
async def list_goals(
    game_set: str | None = Query(None, description="Filter by expansion set"),
):
    reg = get_goal_registry()
    if game_set:
        goals = reg.by_set(GameSet(game_set))
    else:
        goals = reg.all_goals
    return GoalListResponse(
        goals=[goal_to_schema(g) for g in goals],
        total=len(goals),
    )
