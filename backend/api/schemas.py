"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


# --- Bird schemas ---

class FoodCostSchema(BaseModel):
    items: list[str]  # FoodType values
    is_or: bool
    total: int


class BirdSchema(BaseModel):
    name: str
    scientific_name: str
    game_set: str
    color: str
    power_text: str
    victory_points: int
    nest_type: str
    egg_limit: int
    wingspan_cm: int | None
    habitats: list[str]
    food_cost: FoodCostSchema
    beak_direction: str
    is_predator: bool
    is_flocking: bool

    model_config = {"from_attributes": True}


class BirdListResponse(BaseModel):
    birds: list[BirdSchema]
    total: int


# --- Bonus card schemas ---

class BonusScoringTierSchema(BaseModel):
    min_count: int
    max_count: int | None
    points: int


class BonusCardSchema(BaseModel):
    name: str
    game_sets: list[str]
    condition_text: str
    explanation_text: str | None
    scoring_tiers: list[BonusScoringTierSchema]
    is_per_bird: bool

    model_config = {"from_attributes": True}


class BonusCardListResponse(BaseModel):
    bonus_cards: list[BonusCardSchema]
    total: int


# --- Goal schemas ---

class GoalSchema(BaseModel):
    description: str
    game_set: str
    scoring: list[float]  # [4th, 3rd, 2nd, 1st]
    reverse_description: str

    model_config = {"from_attributes": True}


class GoalListResponse(BaseModel):
    goals: list[GoalSchema]
    total: int


# --- Game state schemas ---

class BirdSlotSchema(BaseModel):
    bird_name: str | None = None
    egg_limit: int = 0
    eggs: int = 0
    cached_food: dict[str, int] = Field(default_factory=dict)
    tucked_cards: int = 0


class HabitatRowSchema(BaseModel):
    habitat: str
    slots: list[BirdSlotSchema]
    nectar_spent: int = 0


class FoodSupplySchema(BaseModel):
    invertebrate: int = 0
    seed: int = 0
    fish: int = 0
    fruit: int = 0
    rodent: int = 0
    nectar: int = 0


class PlayerSchema(BaseModel):
    name: str
    board: list[HabitatRowSchema]  # 3 rows: forest, grassland, wetland
    food_supply: FoodSupplySchema
    hand: list[str] = Field(default_factory=list)  # Bird names in hand
    bonus_cards: list[str] = Field(default_factory=list)  # Bonus card names
    action_cubes_remaining: int = 0
    unknown_hand_count: int = 0  # Face-down cards (count only)
    unknown_bonus_count: int = 0  # Unknown bonus cards held (count only)


class BirdfeederSchema(BaseModel):
    dice: list[str | list[str]] = Field(default_factory=list)  # FoodType values or [choice1, choice2]


class CardTraySchema(BaseModel):
    face_up: list[str] = Field(default_factory=list)  # Bird names


class GameStateSchema(BaseModel):
    players: list[PlayerSchema]
    board_type: str = "base"  # "base" or "oceania"
    current_player_idx: int = 0
    current_round: int = 1
    turn_in_round: int = 1
    birdfeeder: BirdfeederSchema = Field(default_factory=BirdfeederSchema)
    card_tray: CardTraySchema = Field(default_factory=CardTraySchema)
    round_goals: list[str] = Field(default_factory=list)  # Goal descriptions
    round_goal_scores: dict[int, dict[str, int]] = Field(default_factory=dict)
    deck_remaining: int = 0


# --- Request schemas ---

class CreateGameRequest(BaseModel):
    player_names: list[str] = Field(..., min_length=1, max_length=5)
    board_type: str = "base"  # "base" or "oceania"
    round_goals: list[str] | None = None  # Goal descriptions


class PlayBirdRequest(BaseModel):
    bird_name: str
    habitat: str  # forest, grassland, wetland
    food_payment: dict[str, int] = Field(default_factory=dict)  # {food_type: count}
    egg_payment_slots: list[list] | None = None  # [[habitat, slot_idx], ...]


class GainFoodRequest(BaseModel):
    food_choices: list[str]  # FoodType values to take from feeder
    bonus_count: int = 0  # Number of "extra" bonus trades to activate
    reset_bonus: bool = False  # Whether to activate the reset_feeder bonus


class LayEggsRequest(BaseModel):
    egg_distribution: dict[str, int] = Field(default_factory=dict)
    # Key format: "habitat:slot_idx", value = count
    bonus_count: int = 0  # Number of bonus trades to activate


class DrawCardsRequest(BaseModel):
    from_tray_indices: list[int] = Field(default_factory=list)
    from_deck_count: int = 0
    bonus_count: int = 0  # Number of "extra" bonus trades to activate
    reset_bonus: bool = False  # Whether to activate the reset_tray bonus


# --- Score schemas ---

class ScoreBreakdownSchema(BaseModel):
    bird_vp: int = 0
    eggs: int = 0
    cached_food: int = 0
    tucked_cards: int = 0
    bonus_cards: int = 0
    round_goals: int = 0
    nectar: int = 0
    total: int = 0


class AllScoresResponse(BaseModel):
    scores: dict[str, ScoreBreakdownSchema]


# --- Legal moves ---

class LegalMoveSchema(BaseModel):
    action_type: str
    description: str
    details: dict = Field(default_factory=dict)


class LegalMovesResponse(BaseModel):
    moves: list[LegalMoveSchema]
    total: int


# --- Action result ---

class ActionResultSchema(BaseModel):
    success: bool
    action_type: str
    message: str = ""
    food_gained: dict[str, int] = Field(default_factory=dict)
    eggs_laid: int = 0
    cards_drawn: int = 0
    bird_played: str | None = None
