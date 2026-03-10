from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, Habitat
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.registry import clear_cache, get_power


def test_all_bird_powers_smoke_execute_without_exceptions():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    all_birds = list(birds.all_birds)
    assert len(all_birds) == 446

    for bird in all_birds:
        game = create_new_game(["A", "B"])
        player = game.players[0]
        # Keep a small deterministic deck available for draw/tuck powers.
        game._deck_cards = list(all_birds[:20])
        game.deck_remaining = len(game._deck_cards)

        habitat = next((h for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND) if bird.can_live_in(h)), None)
        if habitat is None:
            continue
        player.board.get_row(habitat).slots[0].bird = bird

        power = get_power(bird)
        ctx = PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=habitat,
            trigger_player=game.players[1],
            trigger_action=ActionType.GAIN_FOOD,
            trigger_meta={"food_gained": {}, "cards_tucked": 1},
        )

        # Smoke each public path a solver might use.
        power.estimate_value(ctx)
        power.describe_activation(ctx)
        if power.can_execute(ctx):
            power.execute(ctx)

