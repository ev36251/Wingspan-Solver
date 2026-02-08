"""Game state management routes."""

import uuid
from fastapi import APIRouter, HTTPException

from backend.data.registries import get_bird_registry, get_bonus_registry, get_goal_registry
from backend.models.enums import BoardType, FoodType, Habitat, ActionType
from backend.models.game_state import GameState, MoveRecord, create_new_game
from backend.models.player import Player, FoodSupply
from backend.models.board import PlayerBoard, HabitatRow, BirdSlot
from backend.models.birdfeeder import Birdfeeder
from backend.models.card_tray import CardTray
from backend.engine.rules import (
    can_play_bird, can_gain_food, can_lay_eggs, can_draw_cards,
    find_food_payment_options,
)
from backend.engine.actions import (
    execute_play_bird, execute_gain_food, execute_lay_eggs, execute_draw_cards,
)
from backend.engine.scoring import calculate_score, calculate_all_scores
from backend.api.schemas import (
    CreateGameRequest, GameStateSchema, PlayBirdRequest, GainFoodRequest,
    LayEggsRequest, DrawCardsRequest, AllScoresResponse, LegalMovesResponse,
    LegalMoveSchema, ActionResultSchema,
)
from backend.api.serializers import (
    game_state_to_schema, score_breakdown_to_schema, action_result_to_schema,
    schema_to_food_payment, schema_to_egg_distribution,
)

router = APIRouter()

# In-memory game store
_games: dict[str, GameState] = {}


def _get_game(game_id: str) -> GameState:
    if game_id not in _games:
        raise HTTPException(404, f"Game '{game_id}' not found")
    return _games[game_id]


def _record_move(game: GameState, action_type: str, description: str) -> None:
    """Record a move in the game history with heuristic solver ranking."""
    from backend.solver.heuristics import rank_moves

    player = game.current_player
    solver_rank = None
    total_moves = 0
    best_desc = ""

    try:
        ranked = rank_moves(game, player)
        total_moves = len(ranked)
        if ranked:
            best_desc = ranked[0].move.description
            # Find the rank of the chosen move by matching action type + key details
            for rm in ranked:
                if rm.move.action_type.value == action_type and _move_matches(
                    rm.move.description, description
                ):
                    solver_rank = rm.rank
                    break
            # If no exact match, use a high rank to indicate deviation
            if solver_rank is None and ranked:
                solver_rank = total_moves
    except Exception:
        pass

    game.move_history.append(MoveRecord(
        round=game.current_round,
        turn=game.turn_in_round,
        player_name=player.name,
        action_type=action_type,
        description=description,
        solver_rank=solver_rank,
        total_moves=total_moves,
        best_move_description=best_desc,
    ))


def _move_matches(solver_desc: str, action_desc: str) -> bool:
    """Check if a solver move description matches the action taken.

    Uses key substring matching since exact descriptions may differ
    between the move generator and the API action.
    """
    # Normalize for comparison
    s = solver_desc.lower()
    a = action_desc.lower()
    # Direct containment
    if a in s or s in a:
        return True
    # Match on bird name for play-bird moves
    if "play " in s and "play " in a:
        s_bird = s.split("play ", 1)[1].split(" in ")[0]
        a_bird = a.split("play ", 1)[1].split(" in ")[0]
        return s_bird == a_bird
    return False


@router.post("", status_code=201)
async def create_game(req: CreateGameRequest) -> dict:
    bird_reg = get_bird_registry()
    goal_reg = get_goal_registry()

    round_goals = None
    if req.round_goals:
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

    board_type = BoardType(req.board_type)
    game = create_new_game(req.player_names, round_goals, board_type)
    # Set initial deck size (all birds minus tray and hands)
    game.deck_remaining = len(bird_reg) - game.card_tray.count

    game_id = uuid.uuid4().hex[:8]
    _games[game_id] = game

    return {
        "game_id": game_id,
        "state": game_state_to_schema(game).model_dump(),
    }


@router.get("/{game_id}")
async def get_game(game_id: str) -> dict:
    game = _get_game(game_id)
    return game_state_to_schema(game).model_dump()


@router.put("/{game_id}/state")
async def update_game_state(game_id: str, state: GameStateSchema) -> dict:
    """Replace the full game state (for manual state input from UI)."""
    game = _get_game(game_id)
    bird_reg = get_bird_registry()
    bonus_reg = get_bonus_registry()
    goal_reg = get_goal_registry()

    # Rebuild game state from schema
    players = []
    for ps in state.players:
        food = FoodSupply(
            invertebrate=ps.food_supply.invertebrate,
            seed=ps.food_supply.seed,
            fish=ps.food_supply.fish,
            fruit=ps.food_supply.fruit,
            rodent=ps.food_supply.rodent,
            nectar=ps.food_supply.nectar,
        )

        board = PlayerBoard()
        for i, row_schema in enumerate(ps.board):
            habitat = Habitat(row_schema.habitat)
            row = board.get_row(habitat)
            row.nectar_spent = row_schema.nectar_spent
            for j, slot_schema in enumerate(row_schema.slots):
                if slot_schema.bird_name:
                    bird = bird_reg.get(slot_schema.bird_name)
                    if not bird:
                        raise HTTPException(400, f"Bird not found: '{slot_schema.bird_name}'")
                    row.slots[j].bird = bird
                row.slots[j].eggs = slot_schema.eggs
                row.slots[j].cached_food = {
                    FoodType(k): v for k, v in slot_schema.cached_food.items()
                }
                row.slots[j].tucked_cards = slot_schema.tucked_cards

        hand = []
        for name in ps.hand:
            bird = bird_reg.get(name)
            if not bird:
                raise HTTPException(400, f"Bird not found: '{name}'")
            hand.append(bird)

        bonus_cards = []
        for name in ps.bonus_cards:
            bc = bonus_reg.get(name)
            if not bc:
                raise HTTPException(400, f"Bonus card not found: '{name}'")
            bonus_cards.append(bc)

        player = Player(
            name=ps.name,
            board=board,
            food_supply=food,
            hand=hand,
            bonus_cards=bonus_cards,
            action_cubes_remaining=ps.action_cubes_remaining,
            unknown_hand_count=ps.unknown_hand_count,
            unknown_bonus_count=ps.unknown_bonus_count,
        )
        players.append(player)

    # Rebuild birdfeeder
    board_type = BoardType(state.board_type)
    feeder = Birdfeeder.__new__(Birdfeeder)
    feeder.board_type = board_type
    dice = []
    for d in state.birdfeeder.dice:
        if isinstance(d, list):
            dice.append(tuple(FoodType(f) for f in d))
        else:
            dice.append(FoodType(d))
    feeder.dice = dice

    # Rebuild card tray
    tray = CardTray()
    for name in state.card_tray.face_up:
        bird = bird_reg.get(name)
        if bird:
            tray.add_card(bird)

    # Rebuild round goals
    from backend.models.goal import NO_GOAL
    round_goals = []
    for desc in state.round_goals:
        if desc.lower() == "no goal":
            round_goals.append(NO_GOAL)
        else:
            for g in goal_reg.all_goals:
                if g.description.lower() == desc.lower():
                    round_goals.append(g)
                    break

    # Update in place
    game.players = players
    game.board_type = board_type
    game.current_player_idx = state.current_player_idx
    game.current_round = state.current_round
    game.turn_in_round = state.turn_in_round
    game.birdfeeder = feeder
    game.card_tray = tray
    game.round_goals = round_goals
    game.round_goal_scores = state.round_goal_scores
    game.deck_remaining = state.deck_remaining

    return game_state_to_schema(game).model_dump()


@router.get("/{game_id}/legal-moves", response_model=LegalMovesResponse)
async def get_legal_moves(game_id: str) -> LegalMovesResponse:
    game = _get_game(game_id)
    player = game.current_player
    moves: list[LegalMoveSchema] = []

    # Play bird moves
    for bird in player.hand:
        for habitat in bird.habitats:
            legal, reason = can_play_bird(player, bird, habitat, game)
            if legal:
                payment_options = find_food_payment_options(player, bird.food_cost)
                moves.append(LegalMoveSchema(
                    action_type=ActionType.PLAY_BIRD.value,
                    description=f"Play {bird.name} in {habitat.value}",
                    details={
                        "bird_name": bird.name,
                        "habitat": habitat.value,
                        "food_payment_options": [
                            {ft.value: c for ft, c in opt.items()}
                            for opt in payment_options
                        ],
                    },
                ))

    # Gain food
    legal, _ = can_gain_food(player, game)
    if legal:
        available = game.birdfeeder.available_food_types()
        moves.append(LegalMoveSchema(
            action_type=ActionType.GAIN_FOOD.value,
            description="Gain food from birdfeeder",
            details={"available_food": [ft.value for ft in available]},
        ))

    # Lay eggs
    legal, _ = can_lay_eggs(player)
    if legal:
        eligible_slots = []
        for row in player.board.all_rows():
            for i, slot in enumerate(row.slots):
                if slot.bird and slot.can_hold_more_eggs():
                    eligible_slots.append({
                        "habitat": row.habitat.value,
                        "slot_index": i,
                        "bird_name": slot.bird.name,
                        "space": slot.eggs_space(),
                    })
        moves.append(LegalMoveSchema(
            action_type=ActionType.LAY_EGGS.value,
            description="Lay eggs",
            details={"eligible_slots": eligible_slots},
        ))

    # Draw cards
    legal, _ = can_draw_cards(player)
    if legal:
        tray_birds = [b.name for b in game.card_tray.face_up]
        moves.append(LegalMoveSchema(
            action_type=ActionType.DRAW_CARDS.value,
            description="Draw cards",
            details={
                "tray_cards": tray_birds,
                "deck_remaining": game.deck_remaining,
            },
        ))

    return LegalMovesResponse(moves=moves, total=len(moves))


@router.post("/{game_id}/play-bird", response_model=ActionResultSchema)
async def play_bird(game_id: str, req: PlayBirdRequest) -> ActionResultSchema:
    game = _get_game(game_id)
    player = game.current_player
    bird_reg = get_bird_registry()

    bird = bird_reg.get(req.bird_name)
    if not bird:
        raise HTTPException(400, f"Bird not found: '{req.bird_name}'")

    habitat = Habitat(req.habitat)
    food_payment = schema_to_food_payment(req.food_payment)

    egg_slots = None
    if req.egg_payment_slots:
        egg_slots = [(Habitat(s[0]), s[1]) for s in req.egg_payment_slots]

    result = execute_play_bird(game, player, bird, habitat, food_payment, egg_slots)
    if result.success:
        _record_move(game, ActionType.PLAY_BIRD.value,
                     f"Play {bird.name} in {habitat.value}")
        game.advance_turn()
    return action_result_to_schema(result)


@router.post("/{game_id}/gain-food", response_model=ActionResultSchema)
async def gain_food(game_id: str, req: GainFoodRequest) -> ActionResultSchema:
    game = _get_game(game_id)
    player = game.current_player
    food_choices = [FoodType(f) for f in req.food_choices]
    result = execute_gain_food(game, player, food_choices, req.bonus_count, req.reset_bonus)
    if result.success:
        _record_move(game, ActionType.GAIN_FOOD.value, "Gain food from birdfeeder")
        game.advance_turn()
    return action_result_to_schema(result)


@router.post("/{game_id}/lay-eggs", response_model=ActionResultSchema)
async def lay_eggs(game_id: str, req: LayEggsRequest) -> ActionResultSchema:
    game = _get_game(game_id)
    player = game.current_player
    distribution = schema_to_egg_distribution(req.egg_distribution)
    result = execute_lay_eggs(game, player, distribution, req.bonus_count)
    if result.success:
        _record_move(game, ActionType.LAY_EGGS.value, "Lay eggs")
        game.advance_turn()
    return action_result_to_schema(result)


@router.post("/{game_id}/draw-cards", response_model=ActionResultSchema)
async def draw_cards(game_id: str, req: DrawCardsRequest) -> ActionResultSchema:
    game = _get_game(game_id)
    player = game.current_player
    result = execute_draw_cards(game, player, req.from_tray_indices, req.from_deck_count, req.bonus_count, req.reset_bonus)
    if result.success:
        _record_move(game, ActionType.DRAW_CARDS.value, "Draw cards")
        game.advance_turn()
    return action_result_to_schema(result)


@router.get("/{game_id}/score", response_model=AllScoresResponse)
async def get_scores(game_id: str) -> AllScoresResponse:
    game = _get_game(game_id)
    scores = calculate_all_scores(game)
    return AllScoresResponse(
        scores={
            name: score_breakdown_to_schema(sb)
            for name, sb in scores.items()
        }
    )
