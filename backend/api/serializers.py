"""Convert between domain models and API schemas."""

from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard, BonusScoringTier
from backend.models.goal import Goal
from backend.models.board import BirdSlot, HabitatRow, PlayerBoard
from backend.models.player import Player, FoodSupply
from backend.models.birdfeeder import Birdfeeder
from backend.models.card_tray import CardTray
from backend.models.game_state import GameState
from backend.models.enums import FoodType, Habitat
from backend.engine.scoring import ScoreBreakdown
from backend.engine.actions import ActionResult
from backend.api.schemas import (
    BirdSchema, FoodCostSchema, BonusCardSchema, BonusScoringTierSchema,
    GoalSchema, BirdSlotSchema, HabitatRowSchema, FoodSupplySchema,
    PlayerSchema, BirdfeederSchema, CardTraySchema, GameStateSchema,
    ScoreBreakdownSchema, ActionResultSchema,
)


def bird_to_schema(bird: Bird) -> BirdSchema:
    return BirdSchema(
        name=bird.name,
        scientific_name=bird.scientific_name,
        game_set=bird.game_set.value,
        color=bird.color.value,
        power_text=bird.power_text,
        victory_points=bird.victory_points,
        nest_type=bird.nest_type.value,
        egg_limit=bird.egg_limit,
        wingspan_cm=bird.wingspan_cm,
        habitats=[h.value for h in bird.habitats],
        food_cost=FoodCostSchema(
            items=[ft.value for ft in bird.food_cost.items],
            is_or=bird.food_cost.is_or,
            total=bird.food_cost.total,
        ),
        beak_direction=bird.beak_direction.value,
        is_predator=bird.is_predator,
        is_flocking=bird.is_flocking,
    )


def bonus_card_to_schema(card: BonusCard) -> BonusCardSchema:
    return BonusCardSchema(
        name=card.name,
        game_sets=[gs.value for gs in card.game_sets],
        condition_text=card.condition_text,
        explanation_text=card.explanation_text,
        scoring_tiers=[
            BonusScoringTierSchema(
                min_count=t.min_count,
                max_count=t.max_count,
                points=t.points,
            )
            for t in card.scoring_tiers
        ],
        is_per_bird=card.is_per_bird,
    )


def goal_to_schema(goal: Goal) -> GoalSchema:
    return GoalSchema(
        description=goal.description,
        game_set=goal.game_set.value,
        scoring=list(goal.scoring),
        reverse_description=goal.reverse_description,
    )


def bird_slot_to_schema(slot: BirdSlot) -> BirdSlotSchema:
    return BirdSlotSchema(
        bird_name=slot.bird.name if slot.bird else None,
        egg_limit=slot.bird.egg_limit if slot.bird else 0,
        eggs=slot.eggs,
        cached_food={ft.value: c for ft, c in slot.cached_food.items()},
        tucked_cards=slot.tucked_cards,
    )


def habitat_row_to_schema(row: HabitatRow) -> HabitatRowSchema:
    return HabitatRowSchema(
        habitat=row.habitat.value,
        slots=[bird_slot_to_schema(s) for s in row.slots],
        nectar_spent=row.nectar_spent,
    )


def player_to_schema(player: Player) -> PlayerSchema:
    return PlayerSchema(
        name=player.name,
        board=[habitat_row_to_schema(r) for r in player.board.all_rows()],
        food_supply=FoodSupplySchema(
            invertebrate=player.food_supply.invertebrate,
            seed=player.food_supply.seed,
            fish=player.food_supply.fish,
            fruit=player.food_supply.fruit,
            rodent=player.food_supply.rodent,
            nectar=player.food_supply.nectar,
        ),
        hand=[b.name for b in player.hand],
        bonus_cards=[bc.name for bc in player.bonus_cards],
        action_cubes_remaining=player.action_cubes_remaining,
        unknown_hand_count=player.unknown_hand_count,
        unknown_bonus_count=player.unknown_bonus_count,
    )


def birdfeeder_to_schema(feeder: Birdfeeder) -> BirdfeederSchema:
    dice = []
    for die in feeder.dice:
        if isinstance(die, tuple):
            dice.append([ft.value for ft in die])
        else:
            dice.append(die.value)
    return BirdfeederSchema(dice=dice)


def card_tray_to_schema(tray: CardTray) -> CardTraySchema:
    return CardTraySchema(face_up=[b.name for b in tray.face_up])


def game_state_to_schema(gs: GameState) -> GameStateSchema:
    return GameStateSchema(
        players=[player_to_schema(p) for p in gs.players],
        board_type=gs.board_type.value,
        current_player_idx=gs.current_player_idx,
        current_round=gs.current_round,
        turn_in_round=gs.turn_in_round,
        birdfeeder=birdfeeder_to_schema(gs.birdfeeder),
        card_tray=card_tray_to_schema(gs.card_tray),
        round_goals=[g.description for g in gs.round_goals],
        round_goal_scores=gs.round_goal_scores,
        deck_remaining=gs.deck_remaining,
    )


def score_breakdown_to_schema(sb: ScoreBreakdown) -> ScoreBreakdownSchema:
    return ScoreBreakdownSchema(
        bird_vp=sb.bird_vp,
        eggs=sb.eggs,
        cached_food=sb.cached_food,
        tucked_cards=sb.tucked_cards,
        bonus_cards=sb.bonus_cards,
        round_goals=sb.round_goals,
        nectar=sb.nectar,
        total=sb.total,
    )


def action_result_to_schema(ar: ActionResult) -> ActionResultSchema:
    return ActionResultSchema(
        success=ar.success,
        action_type=ar.action_type.value,
        message=ar.message,
        food_gained={ft.value: c for ft, c in ar.food_gained.items()},
        eggs_laid=ar.eggs_laid,
        cards_drawn=ar.cards_drawn,
        bird_played=ar.bird_played,
    )


# --- Deserializers: schema -> domain ---

def _parse_food_type(s: str) -> FoodType:
    return FoodType(s)


def _parse_habitat(s: str) -> Habitat:
    return Habitat(s)


def schema_to_food_payment(payment: dict[str, int]) -> dict[FoodType, int]:
    return {FoodType(k): v for k, v in payment.items()}


def schema_to_egg_distribution(dist: dict[str, int]) -> dict[tuple[Habitat, int], int]:
    result = {}
    for key, count in dist.items():
        parts = key.split(":")
        habitat = Habitat(parts[0])
        slot_idx = int(parts[1])
        result[(habitat, slot_idx)] = count
    return result
