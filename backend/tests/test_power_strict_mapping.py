from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType, Habitat, PowerColor
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.choices import queue_power_choice
from backend.powers.registry import get_power, get_power_source, clear_cache
from backend.powers.templates.strict import (
    DrawThenDiscardFromHand,
    AllPlayersLayEggsSelfBonus,
    EndRoundMandarinDuck,
    TuckThenCacheFromSupply,
    CommonNightingaleChooseFoodAllPlayers,
    PinkEaredDuckDrawKeepGive,
)


def test_strict_mapping_uses_explicit_power_classes():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    assert isinstance(get_power(birds.get("Forster's Tern")), DrawThenDiscardFromHand)
    assert isinstance(get_power(birds.get("Wood Duck")), DrawThenDiscardFromHand)
    assert isinstance(get_power(birds.get("Golden Pheasant")), AllPlayersLayEggsSelfBonus)
    assert isinstance(get_power(birds.get("Mandarin Duck")), EndRoundMandarinDuck)
    assert isinstance(get_power(birds.get("Great Hornbill")), TuckThenCacheFromSupply)
    assert isinstance(get_power(birds.get("Common Nightingale")), CommonNightingaleChooseFoodAllPlayers)
    assert isinstance(get_power(birds.get("Pink-Eared Duck")), PinkEaredDuckDrawKeepGive)


def test_forsters_tern_draw_then_discard_preserves_hand_count():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a = game.players[0]
    tern = birds.get("Forster's Tern")
    spare = birds.get("Spotted Dove")
    # Put tern on board and one discardable card in hand.
    a.board.wetland.slots[0].bird = tern
    a.hand = [spare]
    game._deck_cards = [birds.get("Twite")]
    game.deck_remaining = 1

    p = get_power(tern)
    res = p.execute(PowerContext(game_state=game, player=a, bird=tern, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert len(a.hand) == 1
    assert game.deck_remaining == 0


def test_golden_pheasant_lays_for_all_players_with_self_bonus():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    pheasant = birds.get("Golden Pheasant")
    # Give both players at least one bird with room for eggs.
    a.board.grassland.slots[0].bird = pheasant
    b.board.grassland.slots[0].bird = birds.get("Twite")

    p = get_power(pheasant)
    res = p.execute(PowerContext(game_state=game, player=a, bird=pheasant, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert res.eggs_laid >= 2
    assert a.board.total_eggs() > b.board.total_eggs()


def test_mandarin_duck_respects_explicit_keep_tuck_give_choice():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    duck = birds.get("Mandarin Duck")
    a.board.wetland.slots[0].bird = duck
    game._deck_cards = [
        birds.get("Spotted Dove"),
        birds.get("Twite"),
        birds.get("Golden Pheasant"),
        birds.get("Wood Duck"),
        birds.get("Violet-Green Swallow"),
    ]
    game.deck_remaining = 5
    queue_power_choice(
        game,
        "A",
        "Mandarin Duck",
        {"keep_name": "Golden Pheasant", "tuck_name": "Twite", "give_name": "Wood Duck"},
    )
    p = get_power(duck)
    res = p.execute(PowerContext(game_state=game, player=a, bird=duck, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Golden Pheasant" for c in a.hand)
    assert any(c.name == "Wood Duck" for c in b.hand)
    assert a.board.wetland.slots[0].tucked_cards == 1


def test_pink_eared_duck_keeps_one_and_gives_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    duck = birds.get("Pink-Eared Duck")
    a.board.wetland.slots[0].bird = duck
    game._deck_cards = [birds.get("Twite"), birds.get("Golden Pheasant")]
    game.deck_remaining = 2

    queue_power_choice(
        game,
        "A",
        "Pink-Eared Duck",
        {"keep_name": "Golden Pheasant", "give_player": "B"},
    )
    p = get_power(duck)
    res = p.execute(PowerContext(game_state=game, player=a, bird=duck, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Golden Pheasant" for c in a.hand)
    assert len(b.hand) == 1


def test_common_nightingale_choice_applies_to_all_players():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Common Nightingale")
    a.board.forest.slots[0].bird = bird

    queue_power_choice(game, "A", "Common Nightingale", {"food_type": "fish"})
    p = get_power(bird)
    res = p.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.FOREST))
    assert res.executed
    assert a.food_supply.fish == 1
    assert b.food_supply.fish == 1


def test_all_teal_birds_are_explicitly_strict_mapped():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    teal_birds = [b for b in birds.all_birds if b.color == PowerColor.TEAL]
    assert teal_birds, "Expected teal birds in registry"

    non_strict = [
        (b.name, get_power_source(b))
        for b in teal_birds
        if get_power_source(b) != "strict"
    ]
    assert non_strict == []


def test_all_yellow_birds_are_explicitly_strict_mapped():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    yellow_birds = [b for b in birds.all_birds if b.color == PowerColor.YELLOW]
    assert yellow_birds, "Expected yellow birds in registry"

    non_strict = [
        (b.name, get_power_source(b))
        for b in yellow_birds
        if get_power_source(b) != "strict"
    ]
    assert non_strict == []


def test_all_pink_birds_are_explicitly_strict_mapped():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    pink_birds = [b for b in birds.all_birds if b.color == PowerColor.PINK]
    assert pink_birds, "Expected pink birds in registry"

    non_strict = [
        (b.name, get_power_source(b))
        for b in pink_birds
        if get_power_source(b) != "strict"
    ]
    assert non_strict == []


def test_all_brown_birds_are_strict_certified():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    brown_birds = [b for b in birds.all_birds if b.color == PowerColor.BROWN]
    assert brown_birds, "Expected brown birds in registry"

    non_strict = [
        (b.name, get_power_source(b))
        for b in brown_birds
        if get_power_source(b) != "strict"
    ]
    assert non_strict == []


def test_all_white_birds_are_strict_certified():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    white_birds = [b for b in birds.all_birds if b.color == PowerColor.WHITE]
    assert white_birds, "Expected white birds in registry"

    non_strict = [
        (b.name, get_power_source(b))
        for b in white_birds
        if get_power_source(b) != "strict"
    ]
    assert non_strict == []
