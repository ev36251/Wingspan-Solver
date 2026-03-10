import random

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import (
    BoardType,
    Habitat,
    PowerColor,
    FoodType,
    GameSet,
    NestType,
    BeakDirection,
)
from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard, BonusScoringTier
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.choices import queue_power_choice
from backend.engine.actions import execute_draw_cards
from backend.powers.registry import (
    get_power,
    get_power_source,
    clear_cache,
    get_all_explicit_power_names,
)
from backend.powers.templates.special import DiscardEggForBenefit
from backend.powers.templates.draw_cards import DrawBonusCards
from backend.powers.templates.tuck_cards import DrawDiscardCacheByFoodIcon
from backend.powers.templates.unique import TuckToPayCost
from backend.powers.templates.strict import (
    AmericanOystercatcherDraftPlayersPlusOneClockwise,
    BlackNoddyResetFeederGainFishOptionalDiscardToTuck,
    DrawPerColumnBirdWithEggKeepOne,
    DiscardAllEggsFromOneNestThenTuckDouble,
    GreatCormorantMoveFishThenOptionalRoll,
    KeaDrawBonusDiscardFoodForMoreKeepOne,
    ResetFeederGainAllMayCacheAny,
    ResetFeederGainOneFromOptions,
    StealSpecificFoodCacheThenTargetGainDieFromFeeder,
    DrawThenEndTurnDiscardFromHand,
    DrawKeepTuckDiscard,
    ChooseBirdsInHabitatTuckFromHandEach,
    ChooseBirdsInHabitatCacheFoodEach,
    LayEggsOnEachOtherBirdInColumn,
    EndGameLayEggOnMatchingNest,
    SnowyOwlBonusThenChoice,
    AllPlayersLayEggsSelfBonus,
    EndRoundMandarinDuck,
    TuckHandThenDrawEqualPerOpponentWetlandCubes,
    CommonNightingaleChooseFoodAllPlayers,
    PinkEaredDuckDrawKeepGive,
    DiscardFoodToTuckFromDeck,
    AllPlayersMayTuckAndOrCacheInHabitat,
    AllPlayersMayCacheFoodInHabitat,
    AllPlayersGainFoodThenSelfFoodOrEggBonus,
    AllPlayersDrawCardsWithSelfBonus,
    TuckFromHandThenAllPlayersGainFood,
    AllPlayersMayDiscardEggFromHabitatForWild,
    RollDiceCacheThenAllPlayersMayDiscardCardGainFood,
    AllPlayersMayDiscardFoodToLayEgg,
    SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo,
    EachPlayerGainDieFromFeederStartingChoice,
    GainSeedFromSupplyOrTuckFromDeck,
    NoisyMinerTuckLayOthersLayEgg,
    NorthIslandBrownKiwiDiscardBonusDrawKeep,
    LayEggOnSelfIfTotalEggsBelow,
    PushYourLuckPredatorDiceCache,
    PushYourLuckDrawByWingspanTotalThenTuck,
    WillowTitCacheOneFromFeederOptionalResetOnAllSame,
    ResetFeederGainRodentMayGiveLayUpToThree,
)
from backend.powers.templates.strict_more import EmuGainAllSeedKeepHalfDistributeRemainder


def _tray_bird_fixture(
    *,
    name: str,
    nest_type: NestType = NestType.BOWL,
    vp: int = 1,
    wingspan_cm: int = 30,
    habitats: frozenset[Habitat] | None = None,
    food_items: tuple[FoodType, ...] = (FoodType.SEED,),
    is_predator: bool = False,
) -> Bird:
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.CORE,
        color=PowerColor.BROWN,
        power_text="",
        victory_points=vp,
        nest_type=nest_type,
        egg_limit=2,
        wingspan_cm=wingspan_cm,
        habitats=habitats or frozenset({Habitat.FOREST}),
        food_cost=FoodCost(items=food_items, is_or=False, total=len(food_items)),
        beak_direction=BeakDirection.NONE,
        is_predator=is_predator,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )


def test_strict_mapping_uses_explicit_power_classes():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    assert isinstance(get_power(birds.get("Forster's Tern")), DrawThenEndTurnDiscardFromHand)
    assert isinstance(get_power(birds.get("Wood Duck")), DrawThenEndTurnDiscardFromHand)
    assert isinstance(get_power(birds.get("Golden Pheasant")), AllPlayersLayEggsSelfBonus)
    assert isinstance(get_power(birds.get("Lazuli Bunting")), AllPlayersLayEggsSelfBonus)
    assert isinstance(get_power(birds.get("Pileated Woodpecker")), AllPlayersLayEggsSelfBonus)
    assert isinstance(get_power(birds.get("Western Meadowlark")), AllPlayersLayEggsSelfBonus)
    assert isinstance(get_power(birds.get("Mandarin Duck")), EndRoundMandarinDuck)
    assert isinstance(get_power(birds.get("Great Hornbill")), AllPlayersMayTuckAndOrCacheInHabitat)
    assert isinstance(get_power(birds.get("Brown Shrike")), AllPlayersMayCacheFoodInHabitat)
    assert isinstance(get_power(birds.get("Common Nightingale")), CommonNightingaleChooseFoodAllPlayers)
    assert isinstance(get_power(birds.get("Bluethroat")), CommonNightingaleChooseFoodAllPlayers)
    assert isinstance(get_power(birds.get("Blyth's Hornbill")), DiscardAllEggsFromOneNestThenTuckDouble)
    assert isinstance(get_power(birds.get("Eastern Rosella")), AllPlayersGainFoodThenSelfFoodOrEggBonus)
    assert isinstance(get_power(birds.get("Himalayan Monal")), AllPlayersGainFoodThenSelfFoodOrEggBonus)
    assert isinstance(get_power(birds.get("Ibisbill")), AllPlayersDrawCardsWithSelfBonus)
    assert isinstance(get_power(birds.get("Indian Peafowl")), AllPlayersDrawCardsWithSelfBonus)
    assert isinstance(get_power(birds.get("Major Mitchell's Cockatoo")), TuckFromHandThenAllPlayersGainFood)
    assert isinstance(get_power(birds.get("Sulphur-Crested Cockatoo")), TuckFromHandThenAllPlayersGainFood)
    assert isinstance(get_power(birds.get("Lesser Frigatebird")), AllPlayersMayDiscardEggFromHabitatForWild)
    assert isinstance(get_power(birds.get("Rhinoceros Auklet")), RollDiceCacheThenAllPlayersMayDiscardCardGainFood)
    assert isinstance(get_power(birds.get("Sri Lanka Frogmouth")), RollDiceCacheThenAllPlayersMayDiscardCardGainFood)
    assert isinstance(get_power(birds.get("Zebra Dove")), AllPlayersMayDiscardFoodToLayEgg)
    assert isinstance(get_power(birds.get("Spoon-Billed Sandpiper")), SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo)
    assert isinstance(get_power(birds.get("Crested Ibis")), SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo)
    assert isinstance(get_power(birds.get("Noisy Miner")), NoisyMinerTuckLayOthersLayEgg)
    assert isinstance(get_power(birds.get("North Island Brown Kiwi")), NorthIslandBrownKiwiDiscardBonusDrawKeep)
    assert isinstance(get_power(birds.get("Kea")), KeaDrawBonusDiscardFoodForMoreKeepOne)
    assert isinstance(get_power(birds.get("Red Junglefowl")), LayEggOnSelfIfTotalEggsBelow)
    assert isinstance(get_power(birds.get("Anna's Hummingbird")), EachPlayerGainDieFromFeederStartingChoice)
    assert isinstance(get_power(birds.get("Ruby-Throated Hummingbird")), EachPlayerGainDieFromFeederStartingChoice)
    assert isinstance(get_power(birds.get("Asian Emerald Dove")), LayEggsOnEachOtherBirdInColumn)
    assert isinstance(get_power(birds.get("Scaly-Breasted Munia")), GainSeedFromSupplyOrTuckFromDeck)
    assert isinstance(get_power(birds.get("Ash-Throated Flycatcher")), EndGameLayEggOnMatchingNest)
    assert isinstance(get_power(birds.get("Bobolink")), EndGameLayEggOnMatchingNest)
    assert isinstance(get_power(birds.get("Inca Dove")), EndGameLayEggOnMatchingNest)
    assert isinstance(get_power(birds.get("Say's Phoebe")), EndGameLayEggOnMatchingNest)
    assert isinstance(get_power(birds.get("Great Cormorant")), GreatCormorantMoveFishThenOptionalRoll)
    assert isinstance(get_power(birds.get("Black Noddy")), BlackNoddyResetFeederGainFishOptionalDiscardToTuck)
    assert isinstance(get_power(birds.get("American Oystercatcher")), AmericanOystercatcherDraftPlayersPlusOneClockwise)
    assert isinstance(get_power(birds.get("Eurasian Nutcracker")), ChooseBirdsInHabitatCacheFoodEach)
    assert isinstance(get_power(birds.get("Brahminy Kite")), PushYourLuckPredatorDiceCache)
    assert isinstance(get_power(birds.get("Eurasian Eagle-Owl")), PushYourLuckDrawByWingspanTotalThenTuck)
    assert isinstance(get_power(birds.get("Eurasian Marsh-Harrier")), PushYourLuckDrawByWingspanTotalThenTuck)
    assert isinstance(get_power(birds.get("Forest Owlet")), PushYourLuckPredatorDiceCache)
    assert isinstance(get_power(birds.get("Purple Heron")), PushYourLuckPredatorDiceCache)
    assert isinstance(get_power(birds.get("White-Throated Kingfisher")), PushYourLuckPredatorDiceCache)
    assert isinstance(get_power(birds.get("Grey Shrikethrush")), ResetFeederGainAllMayCacheAny)
    assert isinstance(get_power(birds.get("White-Faced Heron")), ResetFeederGainAllMayCacheAny)
    assert isinstance(get_power(birds.get("Emu")), EmuGainAllSeedKeepHalfDistributeRemainder)
    assert isinstance(get_power(birds.get("Laughing Kookaburra")), ResetFeederGainOneFromOptions)
    assert isinstance(get_power(birds.get("Black-Shouldered Kite")), ResetFeederGainRodentMayGiveLayUpToThree)
    assert isinstance(get_power(birds.get("Willow Tit")), WillowTitCacheOneFromFeederOptionalResetOnAllSame)
    assert isinstance(get_power(birds.get("Little Grebe")), DrawPerColumnBirdWithEggKeepOne)
    assert isinstance(get_power(birds.get("Common Kingfisher")), StealSpecificFoodCacheThenTargetGainDieFromFeeder)
    assert isinstance(get_power(birds.get("Eurasian Jay")), StealSpecificFoodCacheThenTargetGainDieFromFeeder)
    assert isinstance(get_power(birds.get("Little Owl")), StealSpecificFoodCacheThenTargetGainDieFromFeeder)
    assert isinstance(get_power(birds.get("Red-Backed Shrike")), StealSpecificFoodCacheThenTargetGainDieFromFeeder)
    assert isinstance(get_power(birds.get("Audouin's Gull")), DrawKeepTuckDiscard)
    assert isinstance(get_power(birds.get("Common Chaffinch")), ChooseBirdsInHabitatTuckFromHandEach)
    assert isinstance(get_power(birds.get("Common Chiffchaff")), ChooseBirdsInHabitatTuckFromHandEach)
    assert isinstance(get_power(birds.get("Mute Swan")), ChooseBirdsInHabitatTuckFromHandEach)
    assert isinstance(get_power(birds.get("Green Pygmy-Goose")), PinkEaredDuckDrawKeepGive)
    assert isinstance(get_power(birds.get("Little Bustard")), SnowyOwlBonusThenChoice)
    assert isinstance(get_power(birds.get("Bonelli's Eagle")), TuckToPayCost)
    assert isinstance(get_power(birds.get("Eurasian Sparrowhawk")), TuckToPayCost)
    assert isinstance(get_power(birds.get("Northern Goshawk")), TuckToPayCost)
    assert isinstance(get_power(birds.get("Greater Flamingo")), TuckHandThenDrawEqualPerOpponentWetlandCubes)
    assert isinstance(get_power(birds.get("Franklin's Gull")), DiscardEggForBenefit)
    assert isinstance(get_power(birds.get("Killdeer")), DiscardEggForBenefit)
    assert isinstance(get_power(birds.get("American Crow")), DiscardEggForBenefit)
    assert isinstance(get_power(birds.get("Black-Crowned Night-Heron")), DiscardEggForBenefit)
    assert isinstance(get_power(birds.get("Fish Crow")), DiscardEggForBenefit)
    assert isinstance(get_power(birds.get("Pink-Eared Duck")), PinkEaredDuckDrawKeepGive)
    assert isinstance(get_power(birds.get("American White Pelican")), DiscardFoodToTuckFromDeck)
    assert isinstance(get_power(birds.get("Canada Goose")), DiscardFoodToTuckFromDeck)
    assert isinstance(get_power(birds.get("Little Penguin")), DrawDiscardCacheByFoodIcon)
    assert bool(getattr(get_power(birds.get("Coal Tit")), "spendable", False))
    assert bool(getattr(get_power(birds.get("Eurasian Nuthatch")), "spendable", False))
    assert not bool(getattr(get_power(birds.get("Carolina Chickadee")), "spendable", False))
    assert bool(getattr(get_power(birds.get("Lazuli Bunting")), "self_bonus_must_be_distinct_bird", False))
    assert bool(getattr(get_power(birds.get("Pileated Woodpecker")), "self_bonus_must_be_distinct_bird", False))
    assert bool(getattr(get_power(birds.get("Western Meadowlark")), "self_bonus_must_be_distinct_bird", False))
    assert not bool(getattr(get_power(birds.get("Golden Pheasant")), "self_bonus_must_be_distinct_bird", False))


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
    # End-of-turn discard is deferred, not immediate.
    assert len(a.hand) == 2
    assert game.deck_remaining == 0
    assert game.pending_end_turn_hand_discards.get(a.name, 0) == 1


def test_american_oystercatcher_drafts_clockwise_and_keeps_extra_in_1v1():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    oystercatcher = birds.get("American Oystercatcher")
    assert oystercatcher is not None
    habitat = next(iter(oystercatcher.habitats))
    a.board.get_row(habitat).slots[0].bird = oystercatcher

    twite = birds.get("Twite")
    wood_duck = birds.get("Wood Duck")
    golden = birds.get("Golden Pheasant")
    game._deck_cards = [twite, wood_duck, golden]
    game.deck_remaining = 3

    queue_power_choice(game, "A", "American Oystercatcher", {"pick_name": "Golden Pheasant"})
    queue_power_choice(game, "B", "American Oystercatcher", {"pick_name": "Twite"})

    power = get_power(oystercatcher)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=a,
            bird=oystercatcher,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    assert res.cards_drawn == 2
    assert sorted(card.name for card in a.hand) == ["Golden Pheasant", "Wood Duck"]
    assert [card.name for card in b.hand] == ["Twite"]
    assert game.deck_remaining == 0


def test_mute_swan_tucks_up_to_three_in_wetland_then_draws_one_if_any_tucked():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    mute_swan = birds.get("Mute Swan")
    assert mute_swan is not None
    habitat = Habitat.WETLAND
    row = player.board.get_row(habitat)
    row.slots[0].bird = mute_swan
    row.slots[1].bird = birds.get("Twite")
    row.slots[2].bird = birds.get("Wood Duck")
    player.hand = [birds.get("Golden Pheasant"), birds.get("Spotted Dove"), birds.get("Violet-Green Swallow")]
    game._deck_cards = [birds.get("Wedge-Tailed Eagle")]
    game.deck_remaining = 1

    queue_power_choice(game, "A", "Mute Swan", {"target_slots": [0, 2]})
    power = get_power(mute_swan)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=mute_swan,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    assert res.cards_tucked == 2
    assert res.cards_drawn == 1
    assert row.slots[0].tucked_cards == 1
    assert row.slots[1].tucked_cards == 0
    assert row.slots[2].tucked_cards == 1
    assert len(player.hand) == 2  # 3 start -2 tucked +1 drawn


def test_eurasian_nutcracker_can_cache_on_up_to_five_chosen_forest_birds():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    nutcracker = birds.get("Eurasian Nutcracker")
    assert nutcracker is not None
    habitat = Habitat.FOREST
    row = player.board.get_row(habitat)
    row.slots[0].bird = nutcracker
    row.slots[1].bird = birds.get("Twite")
    row.slots[2].bird = birds.get("Wood Duck")
    row.slots[3].bird = birds.get("Golden Pheasant")

    queue_power_choice(game, "A", "Eurasian Nutcracker", {"target_slots": [0, 2, 3]})
    power = get_power(nutcracker)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=nutcracker,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    assert res.food_cached.get(FoodType.SEED, 0) == 3
    assert row.slots[0].cached_food.get(FoodType.SEED, 0) == 1
    assert row.slots[1].cached_food.get(FoodType.SEED, 0) == 0
    assert row.slots[2].cached_food.get(FoodType.SEED, 0) == 1
    assert row.slots[3].cached_food.get(FoodType.SEED, 0) == 1


def test_end_turn_discard_is_resolved_after_draw_action():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a = game.players[0]
    tern = birds.get("Forster's Tern")
    spare = birds.get("Spotted Dove")
    assert tern is not None
    assert spare is not None

    a.board.wetland.slots[0].bird = tern
    a.hand = [spare]
    game._deck_cards = [birds.get("Twite")]
    game.deck_remaining = 1

    result = execute_draw_cards(
        game_state=game,
        player=a,
        from_tray_indices=[],
        from_deck_count=0,
        bonus_count=0,
        reset_bonus=False,
    )
    assert result.success
    assert game.pending_end_turn_hand_discards.get(a.name, 0) == 0
    # Net hand count preserved after deferred discard resolves.
    assert len(a.hand) == 1


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


def test_little_penguin_caches_fish_icons_from_drawn_cards():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    penguin = birds.get("Little Penguin")
    player.board.wetland.slots[0].bird = penguin

    def _fake_bird(name: str, items: tuple[FoodType, ...], is_or: bool = False, total: int = 0) -> Bird:
        return Bird(
            name=name,
            scientific_name=name,
            game_set=GameSet.CORE,
            color=PowerColor.BROWN,
            power_text="",
            victory_points=1,
            nest_type=NestType.BOWL,
            egg_limit=2,
            wingspan_cm=20,
            habitats=frozenset({Habitat.FOREST}),
            food_cost=FoodCost(items=items, is_or=is_or, total=total),
            beak_direction=BeakDirection.NONE,
            is_predator=False,
            is_flocking=False,
            is_bonus_card_bird=False,
            bonus_eligibility=frozenset(),
        )

    deck_cards = [
        _fake_bird("FishDouble", (FoodType.FISH, FoodType.FISH)),
        _fake_bird("SeedOnly", (FoodType.SEED,)),
        _fake_bird("FishSingle", (FoodType.FISH, FoodType.SEED)),
        _fake_bird("SlashFishRodent", (FoodType.FISH, FoodType.RODENT), is_or=True, total=1),
        _fake_bird("NoFish", (FoodType.FRUIT,)),
    ]
    expected_fish_icons = sum(
        sum(1 for ft in c.food_cost.items if ft == FoodType.FISH)
        for c in deck_cards
    )
    game._deck_cards = list(deck_cards)
    game.deck_remaining = len(deck_cards)

    power = get_power(penguin)
    res = power.execute(PowerContext(game_state=game, player=player, bird=penguin, slot_index=0, habitat=Habitat.WETLAND))

    assert res.executed
    assert res.food_cached.get(FoodType.FISH, 0) == expected_fish_icons
    assert player.board.wetland.slots[0].cached_food.get(FoodType.FISH, 0) == expected_fish_icons
    assert game.deck_remaining == 0
    assert game.discard_pile_count == 5


def test_grey_butcherbird_tucks_and_caches_rodent_on_wingspan_lt_40():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Grey Butcherbird")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Matching target: 30cm (<40), so this should tuck + cache rodent.
    target = _tray_bird_fixture(name="Small Target", wingspan_cm=30, vp=2)
    game._deck_cards = [target]  # type: ignore[attr-defined]
    game.deck_remaining = 1

    power = get_power(bird)
    assert type(power).__name__ == "PredatorLookAt"
    assert int(getattr(power, "wingspan_threshold", -1)) == 40
    assert str(getattr(power, "wingspan_cmp", "")) == "lt"
    assert getattr(power, "cache_food_type", None) == FoodType.RODENT
    assert int(getattr(power, "cache_count", 0)) == 1

    tucked_before = slot.tucked_cards
    cached_before = slot.cached_food.get(FoodType.RODENT, 0)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    assert res.cards_tucked == 1
    assert res.food_cached.get(FoodType.RODENT, 0) == 1
    assert slot.tucked_cards == tucked_before + 1
    assert slot.cached_food.get(FoodType.RODENT, 0) == cached_before + 1


def test_bells_vireo_draws_two_bonus_keeps_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Bell's Vireo")
    power = get_power(bird)
    assert isinstance(power, DrawBonusCards)
    assert getattr(power, "draw", None) == 2
    assert getattr(power, "keep", None) == 1


def test_draw_bonus_cards_reshuffles_bonus_discard_when_deck_is_empty():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Bell's Vireo")
    assert bird is not None
    player.board.grassland.slots[0].bird = bird

    game._bonus_cards = []  # type: ignore[attr-defined]
    game._bonus_discard_cards = [  # type: ignore[attr-defined]
        _bonus_fixture("Discard Low", 1),
        _bonus_fixture("Discard High", 9),
    ]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND)
    )

    assert res.executed
    assert len(player.bonus_cards) == 1
    assert player.bonus_cards[0].name == "Discard High"
    assert len(game._bonus_cards) == 0  # type: ignore[attr-defined]
    assert len(game._bonus_discard_cards) == 1  # type: ignore[attr-defined]


def test_hummingbirds_each_player_gains_one_die_from_feeder():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    for bird_name in ("Anna's Hummingbird", "Ruby-Throated Hummingbird"):
        game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        a, b = game.players
        bird = birds.get(bird_name)
        a.board.forest.slots[0].bird = bird

        # Deterministic feeder: no reroll, enough dice for both players.
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE])
        before_a = a.food_supply.total()
        before_b = b.food_supply.total()

        power = get_power(bird)
        res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.FOREST))

        assert res.executed
        assert a.food_supply.total() == before_a + 1
        assert b.food_supply.total() == before_b + 1


def test_scaly_breasted_munia_can_gain_seed_or_tuck_from_deck():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Scaly-Breasted Munia")
    assert bird is not None

    # Gain mode
    game_gain = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_gain = game_gain.players[0]
    player_gain.board.grassland.slots[0].bird = bird
    seed_before = player_gain.food_supply.get(FoodType.SEED)
    queue_power_choice(game_gain, player_gain.name, bird.name, {"mode": "gain"})
    power = get_power(bird)
    res_gain = power.execute(
        PowerContext(
            game_state=game_gain,
            player=player_gain,
            bird=bird,
            slot_index=0,
            habitat=Habitat.GRASSLAND,
        )
    )
    assert res_gain.executed
    assert player_gain.food_supply.get(FoodType.SEED) == seed_before + 1
    assert player_gain.board.grassland.slots[0].tucked_cards == 0

    # Tuck mode
    game_tuck = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_tuck = game_tuck.players[0]
    player_tuck.board.grassland.slots[0].bird = bird
    game_tuck._deck_cards = [birds.get("Twite")]  # type: ignore[attr-defined]
    game_tuck.deck_remaining = 1
    seed_before_tuck = player_tuck.food_supply.get(FoodType.SEED)
    queue_power_choice(game_tuck, player_tuck.name, bird.name, {"mode": "tuck"})
    res_tuck = power.execute(
        PowerContext(
            game_state=game_tuck,
            player=player_tuck,
            bird=bird,
            slot_index=0,
            habitat=Habitat.GRASSLAND,
        )
    )
    assert res_tuck.executed
    assert player_tuck.board.grassland.slots[0].tucked_cards == 1
    assert player_tuck.food_supply.get(FoodType.SEED) == seed_before_tuck


def test_black_noddy_can_keep_or_discard_gained_fish_to_tuck():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Black Noddy")
    assert bird is not None

    def _rigged_reroll(game):
        game.birdfeeder.set_dice(
            [FoodType.FISH, FoodType.FISH, FoodType.FISH, FoodType.SEED, FoodType.FRUIT]
        )

    # Keep all fish path
    game_keep = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_keep = game_keep.players[0]
    player_keep.board.wetland.slots[0].bird = bird
    game_keep._deck_cards = [birds.get("Twite"), birds.get("Golden Pheasant")]  # type: ignore[attr-defined]
    game_keep.deck_remaining = 2
    game_keep.birdfeeder.reroll = lambda: _rigged_reroll(game_keep)  # type: ignore[assignment]
    fish_before_keep = player_keep.food_supply.get(FoodType.FISH)

    power = get_power(bird)
    queue_power_choice(game_keep, player_keep.name, bird.name, {"discard_fish_for_tuck": 0})
    res_keep = power.execute(
        PowerContext(
            game_state=game_keep,
            player=player_keep,
            bird=bird,
            slot_index=0,
            habitat=Habitat.WETLAND,
        )
    )
    assert res_keep.executed
    assert player_keep.board.wetland.slots[0].tucked_cards == 0
    assert player_keep.food_supply.get(FoodType.FISH) == fish_before_keep + 3

    # Discard some fish to tuck path
    game_tuck = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_tuck = game_tuck.players[0]
    player_tuck.board.wetland.slots[0].bird = bird
    game_tuck._deck_cards = [birds.get("Twite"), birds.get("Golden Pheasant")]  # type: ignore[attr-defined]
    game_tuck.deck_remaining = 2
    game_tuck.birdfeeder.reroll = lambda: _rigged_reroll(game_tuck)  # type: ignore[assignment]
    fish_before_tuck = player_tuck.food_supply.get(FoodType.FISH)

    queue_power_choice(game_tuck, player_tuck.name, bird.name, {"discard_fish_for_tuck": 2})
    res_tuck = power.execute(
        PowerContext(
            game_state=game_tuck,
            player=player_tuck,
            bird=bird,
            slot_index=0,
            habitat=Habitat.WETLAND,
        )
    )
    assert res_tuck.executed
    assert player_tuck.board.wetland.slots[0].tucked_cards == 2
    assert player_tuck.food_supply.get(FoodType.FISH) == fish_before_tuck + 1


def test_grey_shrikethrush_can_split_rodents_between_cache_and_supply():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Grey Shrikethrush")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    def _rigged_reroll():
        game.birdfeeder.set_dice(
            [FoodType.RODENT, FoodType.RODENT, FoodType.RODENT, FoodType.SEED, FoodType.FRUIT]
        )

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]
    queue_power_choice(game, player.name, bird.name, {"cache_count": 2})
    supply_before = player.food_supply.get(FoodType.RODENT)
    cached_before = slot.cached_food.get(FoodType.RODENT, 0)

    power = get_power(bird)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.RODENT, 0) == cached_before + 2
    assert player.food_supply.get(FoodType.RODENT) == supply_before + 1


def test_great_cormorant_move_and_roll_steps_are_independent_optional(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Great Cormorant")
    assert bird is not None

    # Case 1: Skip move, do roll, hit fish => cache 1 fish.
    game_roll = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_roll = game_roll.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot_roll = player_roll.board.get_row(habitat).slots[0]
    slot_roll.bird = bird
    slot_roll.cache_food(FoodType.FISH, 2)
    fish_supply_before_roll = player_roll.food_supply.get(FoodType.FISH)
    fish_cache_before_roll = slot_roll.cached_food.get(FoodType.FISH, 0)
    game_roll.birdfeeder.set_dice(
        [FoodType.SEED, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE, FoodType.NECTAR]
    )

    rolls = iter([FoodType.SEED, FoodType.FISH])
    original_choice = random.choice
    monkeypatch.setattr("random.choice", lambda _seq: next(rolls))
    queue_power_choice(
        game_roll,
        player_roll.name,
        bird.name,
        {"move_fish_to_supply": False, "roll_dice": True, "roll_indices": [0, 1]},
    )

    power = get_power(bird)
    res_roll = power.execute(
        PowerContext(
            game_state=game_roll,
            player=player_roll,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res_roll.executed
    assert player_roll.food_supply.get(FoodType.FISH) == fish_supply_before_roll
    assert slot_roll.cached_food.get(FoodType.FISH, 0) == fish_cache_before_roll + 1
    assert res_roll.food_cached.get(FoodType.FISH, 0) == 1
    monkeypatch.setattr("random.choice", original_choice)

    # Case 2: Move fish, skip roll => gain 1 fish in supply, no cache from roll.
    game_move = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_move = game_move.players[0]
    slot_move = player_move.board.get_row(habitat).slots[0]
    slot_move.bird = bird
    slot_move.cache_food(FoodType.FISH, 1)
    fish_supply_before_move = player_move.food_supply.get(FoodType.FISH)
    fish_cache_before_move = slot_move.cached_food.get(FoodType.FISH, 0)

    queue_power_choice(
        game_move,
        player_move.name,
        bird.name,
        {"move_fish_to_supply": True, "roll_dice": False},
    )
    res_move = power.execute(
        PowerContext(
            game_state=game_move,
            player=player_move,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res_move.executed
    assert player_move.food_supply.get(FoodType.FISH) == fish_supply_before_move + 1
    assert slot_move.cached_food.get(FoodType.FISH, 0) == fish_cache_before_move - 1
    assert res_move.food_cached.get(FoodType.FISH, 0) == 0

    # Case 3: No cached fish and roll skipped => no-op (executed False).
    game_none = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_none = game_none.players[0]
    slot_none = player_none.board.get_row(habitat).slots[0]
    slot_none.bird = bird
    queue_power_choice(
        game_none,
        player_none.name,
        bird.name,
        {"move_fish_to_supply": False, "roll_dice": False},
    )
    res_none = power.execute(
        PowerContext(
            game_state=game_none,
            player=player_none,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert not res_none.executed


def test_kea_discards_food_for_additional_bonus_draws_and_tracks_spent_nectar():
    birds, bonus_reg, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Kea")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird

    player.food_supply.add(FoodType.NECTAR, 2)
    player.food_supply.add(FoodType.SEED, 1)
    game._bonus_cards = list(bonus_reg.all_cards[:10])  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    queue_power_choice(
        game,
        player.name,
        bird.name,
        {
            "discard_food_types": ["nectar", "seed"],
            "discard_food_count": 2,
        },
    )

    bonus_before = len(player.bonus_cards)
    discard_before = len(game._bonus_discard_cards)  # type: ignore[attr-defined]
    nectar_before = player.food_supply.get(FoodType.NECTAR)
    seed_before = player.food_supply.get(FoodType.SEED)
    nectar_spent_before = row.nectar_spent

    power = get_power(bird)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    # Draw 1 + 2 extra; keep 1, discard 2.
    assert len(player.bonus_cards) == bonus_before + 1
    assert len(game._bonus_discard_cards) == discard_before + 2  # type: ignore[attr-defined]
    assert player.food_supply.get(FoodType.NECTAR) == nectar_before - 1
    assert player.food_supply.get(FoodType.SEED) == seed_before - 1
    assert row.nectar_spent == nectar_spent_before + 1


def test_laughing_kookaburra_chooses_one_of_inv_fish_rodent_if_available():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Laughing Kookaburra")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird
    power = get_power(bird)

    def _set_and_run(dice, choice_food: str):
        game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(dice)  # type: ignore[assignment]
        queue_power_choice(game, player.name, bird.name, {"food_type": choice_food})
        return power.execute(
            PowerContext(
                game_state=game,
                player=player,
                bird=bird,
                slot_index=0,
                habitat=habitat,
            )
        )

    # All options available: explicit choice should be honored.
    fish_before = player.food_supply.get(FoodType.FISH)
    res_fish = _set_and_run(
        [FoodType.INVERTEBRATE, FoodType.FISH, FoodType.RODENT, FoodType.SEED, FoodType.FRUIT],
        "fish",
    )
    assert res_fish.executed
    assert player.food_supply.get(FoodType.FISH) == fish_before + 1

    # If chosen food is unavailable but one eligible type exists, gain the eligible one.
    rodent_before = player.food_supply.get(FoodType.RODENT)
    res_rodent = _set_and_run(
        [FoodType.RODENT, FoodType.SEED, FoodType.FRUIT, FoodType.SEED, FoodType.FRUIT],
        "fish",
    )
    assert res_rodent.executed
    assert player.food_supply.get(FoodType.RODENT) == rodent_before + 1

    # If none of inv/fish/rodent are present, gain nothing.
    total_before = player.food_supply.total()
    res_none = _set_and_run(
        [FoodType.SEED, FoodType.FRUIT, FoodType.SEED, FoodType.FRUIT, FoodType.NECTAR],
        "rodent",
    )
    assert not res_none.executed
    assert player.food_supply.total() == total_before


def test_black_shouldered_kite_takes_rodent_die_and_can_keep_it():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Black-Shouldered Kite")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    power = get_power(bird)

    # Feeder after reset includes one rodent.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.RODENT, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE]
    )

    rodent_before = a.food_supply.get(FoodType.RODENT)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert a.food_supply.get(FoodType.RODENT) == rodent_before + 1
    # Rodent die should be removed from feeder when gained.
    assert game.birdfeeder.count == 4
    assert FoodType.RODENT not in game.birdfeeder.available_food_types()


def test_black_shouldered_kite_can_give_rodent_to_other_player_and_lay_up_to_three_eggs():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Black-Shouldered Kite")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    power = get_power(bird)

    # Ensure feeder reset presents a rodent.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.RODENT, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE]
    )

    a_rodent_before = a.food_supply.get(FoodType.RODENT)
    b_rodent_before = b.food_supply.get(FoodType.RODENT)
    eggs_before = slot.eggs
    queue_power_choice(game, a.name, bird.name, {"give_player": b.name, "lay_eggs": 3})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    # Actor gains rodent then gives it; net unchanged.
    assert a.food_supply.get(FoodType.RODENT) == a_rodent_before
    assert b.food_supply.get(FoodType.RODENT) == b_rodent_before + 1
    assert slot.eggs == eggs_before + min(3, slot.bird.egg_limit - eggs_before)
    # Rodent die still removed from feeder due to gain step.
    assert game.birdfeeder.count == 4


def test_willow_tit_caches_one_of_inv_seed_fruit_and_removes_die():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Willow Tit")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    power = get_power(bird)

    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.RODENT, FoodType.FRUIT, FoodType.INVERTEBRATE])
    before_seed = slot.cached_food.get(FoodType.SEED, 0)
    queue_power_choice(game, player.name, bird.name, {"food_type": "seed"})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.SEED, 0) == before_seed + 1
    assert game.birdfeeder.count == 4


def test_willow_tit_reset_only_when_all_same_and_can_gain_nothing():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Willow Tit")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    power = get_power(bird)

    # All-same allows reset before gain.
    game.birdfeeder.set_dice([FoodType.RODENT] * 5)
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.FRUIT, FoodType.SEED, FoodType.FISH, FoodType.RODENT, FoodType.INVERTEBRATE]
    )
    before_fruit = slot.cached_food.get(FoodType.FRUIT, 0)
    queue_power_choice(game, player.name, bird.name, {"reset_feeder": True, "food_type": "fruit"})
    res_reset = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_reset.executed
    assert slot.cached_food.get(FoodType.FRUIT, 0) == before_fruit + 1

    # Not all-same: cannot reset; if no eligible food exists, gain nothing.
    game.birdfeeder.set_dice([FoodType.RODENT, FoodType.FISH, FoodType.RODENT, FoodType.FISH, FoodType.RODENT])
    before_total = sum(slot.cached_food.values())
    queue_power_choice(game, player.name, bird.name, {"reset_feeder": True, "food_type": "seed"})
    res_none = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_none.executed
    assert sum(slot.cached_food.values()) == before_total


def test_little_grebe_draws_per_egged_bird_in_column_and_keeps_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Little Grebe")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot_idx = 0
    player.board.get_row(habitat).slots[slot_idx].bird = bird
    player.board.get_row(habitat).slots[slot_idx].eggs = 1

    # Ensure all 3 habitats have an egged bird in this same column.
    for row in player.board.all_rows():
        if row.habitat == habitat:
            continue
        support = _tray_bird_fixture(
            name=f"{row.habitat.value.title()} Support",
            habitats=frozenset({row.habitat}),
        )
        row.slots[slot_idx].bird = support
        row.slots[slot_idx].eggs = 1

    # Draw 3 cards and keep one chosen card.
    drawn = [
        _tray_bird_fixture(name="Low", vp=1),
        _tray_bird_fixture(name="High", vp=6),
        _tray_bird_fixture(name="Mid", vp=3),
    ]
    game._deck_cards = list(drawn)  # type: ignore[attr-defined]
    game.deck_remaining = len(drawn)
    queue_power_choice(game, player.name, bird.name, {"keep_name": "High"})

    hand_before = len(player.hand)
    discard_before = game.discard_pile_count
    power = get_power(bird)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=slot_idx,
            habitat=habitat,
        )
    )
    assert res.executed
    assert res.cards_drawn == 1
    assert len(player.hand) == hand_before + 1
    assert any(c.name == "High" for c in player.hand)
    assert game.discard_pile_count == discard_before + 2


def test_little_owl_steals_rodent_then_target_rerolls_and_gains_one_die():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Little Owl")
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    b.food_supply.add(FoodType.RODENT, 1)

    # One die left (all same) -> target can reroll before choosing.
    game.birdfeeder.set_dice([FoodType.SEED])
    reroll_calls = {"n": 0}

    def _rigged_reroll():
        reroll_calls["n"] += 1
        game.birdfeeder.set_dice(
            [FoodType.FISH, FoodType.SEED, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE]
        )

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]
    queue_power_choice(game, a.name, bird.name, {"target_player": b.name})
    queue_power_choice(game, b.name, bird.name, {"food_type": "fish"})

    rodent_before = b.food_supply.get(FoodType.RODENT)
    fish_before = b.food_supply.get(FoodType.FISH)
    cached_before = slot.cached_food.get(FoodType.RODENT, 0)

    power = get_power(bird)
    res = power.execute(
        PowerContext(
            game_state=game,
            player=a,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert res.executed
    assert reroll_calls["n"] == 1
    assert slot.cached_food.get(FoodType.RODENT, 0) == cached_before + 1
    assert b.food_supply.get(FoodType.RODENT) == rodent_before - 1
    assert b.food_supply.get(FoodType.FISH) == fish_before + 1


def test_maned_duck_can_tuck_up_to_three_and_seed_only_if_tucked():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Maned Duck")
    assert bird is not None
    power = get_power(bird)
    assert type(power).__name__ == "TuckFromHand"

    # Tuck 1 path => gain 1 seed.
    game_one = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_one = game_one.players[0]
    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]
    slot_one = player_one.board.get_row(habitat).slots[0]
    slot_one.bird = bird
    player_one.hand = list(birds.all_birds[:5])
    seed_before_one = player_one.food_supply.get(FoodType.SEED)
    tucked_before_one = slot_one.tucked_cards
    queue_power_choice(game_one, player_one.name, bird.name, {"tuck_count": 1})

    res_one = power.execute(
        PowerContext(
            game_state=game_one,
            player=player_one,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res_one.executed
    assert res_one.cards_tucked == 1
    assert slot_one.tucked_cards == tucked_before_one + 1
    assert player_one.food_supply.get(FoodType.SEED) == seed_before_one + 1

    # Tuck 0 path => no tuck, no seed.
    game_zero = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_zero = game_zero.players[0]
    slot_zero = player_zero.board.get_row(habitat).slots[0]
    slot_zero.bird = bird
    player_zero.hand = list(birds.all_birds[:5])
    seed_before_zero = player_zero.food_supply.get(FoodType.SEED)
    tucked_before_zero = slot_zero.tucked_cards
    queue_power_choice(game_zero, player_zero.name, bird.name, {"tuck_count": 0})

    res_zero = power.execute(
        PowerContext(
            game_state=game_zero,
            player=player_zero,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert not res_zero.executed
    assert slot_zero.tucked_cards == tucked_before_zero
    assert player_zero.food_supply.get(FoodType.SEED) == seed_before_zero


def test_baya_weaver_can_tuck_one_or_two_and_lay_one_egg_if_any_tucked():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    bird = birds.get("Baya Weaver")
    assert bird is not None
    power = get_power(bird)
    assert type(power).__name__ == "TuckFromHand"

    habitat = sorted(list(bird.habitats), key=lambda h: h.value)[0]

    # Tuck 1 -> lays exactly 1 egg.
    game_one = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_one = game_one.players[0]
    slot_one = player_one.board.get_row(habitat).slots[0]
    slot_one.bird = bird
    player_one.hand = list(birds.all_birds[:3])
    eggs_before_one = slot_one.eggs
    queue_power_choice(game_one, player_one.name, bird.name, {"tuck_count": 1})
    res_one = power.execute(
        PowerContext(
            game_state=game_one,
            player=player_one,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res_one.executed
    assert res_one.cards_tucked == 1
    assert res_one.eggs_laid == 1
    assert slot_one.eggs == eggs_before_one + 1

    # Tuck 2 -> still lays only 1 egg.
    game_two = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_two = game_two.players[0]
    slot_two = player_two.board.get_row(habitat).slots[0]
    slot_two.bird = bird
    player_two.hand = list(birds.all_birds[:4])
    eggs_before_two = slot_two.eggs
    queue_power_choice(game_two, player_two.name, bird.name, {"tuck_count": 2})
    res_two = power.execute(
        PowerContext(
            game_state=game_two,
            player=player_two,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )
    assert res_two.executed
    assert res_two.cards_tucked == 2
    assert res_two.eggs_laid == 1
    assert slot_two.eggs == eggs_before_two + 1


def test_greater_flamingo_uses_chosen_other_player_wetland_cubes():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B", "C"], board_type=BoardType.OCEANIA)
    a, b, c = game.players
    bird = birds.get("Greater Flamingo")
    assert bird is not None
    habitat = Habitat.WETLAND if bird.can_live_in(Habitat.WETLAND) else list(bird.habitats)[0]
    a.board.get_row(habitat).slots[0].bird = bird
    a.hand = list(birds.all_birds[:6])
    game._deck_cards = list(birds.all_birds[6:30])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    # B has more wetland cubes than C, but explicit choice should target C.
    b.draw_cards_actions_this_round = 3
    c.draw_cards_actions_this_round = 1
    queue_power_choice(game, "A", "Greater Flamingo", {"target_player": "C"})

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat)
    )

    assert res.executed
    assert res.cards_tucked == 1
    assert res.cards_drawn == 1
    assert a.board.get_row(habitat).slots[0].tucked_cards == 1


def test_greater_flamingo_defaults_to_opponent_with_most_wetland_cubes():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B", "C"], board_type=BoardType.OCEANIA)
    a, b, c = game.players
    bird = birds.get("Greater Flamingo")
    assert bird is not None
    habitat = Habitat.WETLAND if bird.can_live_in(Habitat.WETLAND) else list(bird.habitats)[0]
    a.board.get_row(habitat).slots[0].bird = bird
    a.hand = list(birds.all_birds[:6])
    game._deck_cards = list(birds.all_birds[6:30])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    b.draw_cards_actions_this_round = 2
    c.draw_cards_actions_this_round = 1

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat)
    )

    assert res.executed
    assert res.cards_tucked == 2
    assert res.cards_drawn == 2
    assert a.board.get_row(habitat).slots[0].tucked_cards == 2


def test_common_chiffchaff_choose_birds_in_habitat_tucks_one_behind_each():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a = game.players[0]
    bird = birds.get("Common Chiffchaff")
    support1 = birds.get("Twite")
    support2 = birds.get("Wood Duck")
    assert bird is not None
    assert support1 is not None
    assert support2 is not None

    habitat = Habitat.FOREST if bird.can_live_in(Habitat.FOREST) else list(bird.habitats)[0]
    row = a.board.get_row(habitat)
    row.slots[0].bird = bird
    row.slots[1].bird = support1
    row.slots[2].bird = support2
    a.hand = list(birds.all_birds[:5])

    queue_power_choice(game, "A", "Common Chiffchaff", {"target_slots": [0, 2]})
    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat)
    )

    assert res.executed
    assert res.cards_tucked == 2
    assert row.slots[0].tucked_cards == 1
    assert row.slots[1].tucked_cards == 0
    assert row.slots[2].tucked_cards == 1
    assert len(a.hand) == 3


def test_ravens_discard_egg_from_other_bird_to_gain_two_food():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    for raven_name in ("Chihuahuan Raven", "Common Raven"):
        game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        player = game.players[0]
        raven = birds.get(raven_name)
        support = birds.get("Twite")
        assert raven is not None
        assert support is not None
        player.board.forest.slots[0].bird = raven
        player.board.grassland.slots[0].bird = support
        # Raven has eggs, but cost must come from another bird.
        player.board.forest.slots[0].eggs = 2
        player.board.grassland.slots[0].eggs = 1

        power = get_power(raven)
        assert isinstance(power, DiscardEggForBenefit)
        assert getattr(power, "require_other_bird", False) is True
        before_total_food = player.food_supply.total()
        before_raven_eggs = player.board.forest.slots[0].eggs
        before_support_eggs = player.board.grassland.slots[0].eggs

        res = power.execute(
            PowerContext(game_state=game, player=player, bird=raven, slot_index=0, habitat=Habitat.FOREST)
        )
        assert res.executed
        assert player.food_supply.total() - before_total_food == 2
        assert player.board.forest.slots[0].eggs == before_raven_eggs
        assert player.board.grassland.slots[0].eggs == before_support_eggs - 1

        # If no other birds have eggs, the power cannot execute.
        game2 = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        p2 = game2.players[0]
        p2.board.forest.slots[0].bird = raven
        p2.board.forest.slots[0].eggs = 1
        power2 = get_power(raven)
        before_total_food_2 = p2.food_supply.total()
        res2 = power2.execute(
            PowerContext(game_state=game2, player=p2, bird=raven, slot_index=0, habitat=Habitat.FOREST)
        )
        assert not res2.executed
        assert p2.food_supply.total() == before_total_food_2


def test_franklins_gull_and_killdeer_discard_egg_to_draw_two_cards():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    for bird_name in ("Franklin's Gull", "Killdeer"):
        game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        player = game.players[0]
        bird = birds.get(bird_name)
        support = birds.get("Twite")
        assert bird is not None
        assert support is not None
        player.board.wetland.slots[0].bird = bird
        player.board.forest.slots[0].bird = support
        player.board.forest.slots[0].eggs = 1
        game._deck_cards = [birds.get("Wood Duck"), birds.get("Golden Pheasant")]  # type: ignore[attr-defined]
        game.deck_remaining = 2

        power = get_power(bird)
        assert isinstance(power, DiscardEggForBenefit)
        assert getattr(power, "egg_cost", 0) == 1
        assert getattr(power, "card_gain", 0) == 2

        before_eggs = player.board.total_eggs()
        before_hand = len(player.hand)
        res = power.execute(
            PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND)
        )
        assert res.executed
        assert player.board.total_eggs() == before_eggs - 1
        assert len(player.hand) == before_hand + 2

        # No eggs available -> cannot execute.
        game2 = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        p2 = game2.players[0]
        p2.board.wetland.slots[0].bird = bird
        game2._deck_cards = [birds.get("Wood Duck"), birds.get("Golden Pheasant")]  # type: ignore[attr-defined]
        game2.deck_remaining = 2
        res2 = power.execute(
            PowerContext(game_state=game2, player=p2, bird=bird, slot_index=0, habitat=Habitat.WETLAND)
        )
        assert not res2.executed


def test_noisy_miner_tucks_lays_two_on_self_and_other_players_may_lay_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Noisy Miner")
    support_a = birds.get("Twite")
    support_b = birds.get("Spotted Dove")
    assert bird is not None
    assert support_a is not None
    assert support_b is not None
    a.board.grassland.slots[0].bird = bird
    a.board.forest.slots[0].bird = support_a
    b.board.forest.slots[0].bird = support_b
    a.hand = [birds.get("Wood Duck")]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND)
    )

    assert res.executed
    assert res.cards_tucked == 1
    assert a.board.grassland.slots[0].tucked_cards == 1
    assert a.board.grassland.slots[0].eggs == 2
    assert b.board.total_eggs() == 1


def test_north_island_brown_kiwi_discards_bonus_then_draws_four_keeps_two():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("North Island Brown Kiwi")
    assert bird is not None
    player.board.forest.slots[0].bird = bird

    old_keep = _bonus_fixture("Old Keep", 8)
    old_drop = _bonus_fixture("Old Drop", 0)
    player.bonus_cards = [old_keep, old_drop]
    # pop() draws from end: low, mid, high, top
    game._bonus_cards = [  # type: ignore[attr-defined]
        _bonus_fixture("Draw Filler", 1),
        _bonus_fixture("Draw High", 9),
        _bonus_fixture("Draw Mid", 6),
        _bonus_fixture("Draw Low", 2),
    ]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]
    queue_power_choice(
        game,
        player.name,
        "North Island Brown Kiwi",
        {"discard_bonus_name": "Old Drop"},
    )

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.FOREST)
    )

    assert res.executed
    assert len(player.bonus_cards) == 3  # start 2, discard 1, keep 2
    assert sorted(bc.name for bc in player.bonus_cards) == sorted(["Old Keep", "Draw High", "Draw Mid"])
    assert any(bc.name == "Old Drop" for bc in game._bonus_discard_cards)  # type: ignore[attr-defined]


def test_north_island_brown_kiwi_requires_bonus_card_to_discard():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("North Island Brown Kiwi")
    assert bird is not None
    player.board.forest.slots[0].bird = bird
    player.bonus_cards = []

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.FOREST)
    )

    assert not res.executed


def test_budgerigar_tuck_smallest_tray_bird_handles_flightless_cards():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Budgerigar")
    assert bird is not None
    player.board.grassland.slots[0].bird = bird

    # Include a flightless card (wingspan=None) to ensure no None-vs-int comparison crash.
    game.card_tray.face_up = [
        _tray_bird_fixture(name="Flightless Tray", wingspan_cm=None, vp=3),
        _tray_bird_fixture(name="Small Wing", wingspan_cm=19, vp=2),
        _tray_bird_fixture(name="Big Wing", wingspan_cm=75, vp=1),
    ]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND)
    )
    assert res.executed
    assert player.board.grassland.slots[0].tucked_cards == 1
    # Smallest numeric wingspan should be tucked first.
    assert all(c.name != "Small Wing" for c in game.card_tray.face_up)


def test_rufous_owl_tuck_tray_by_max_wingspan_ignores_flightless_cards():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Rufous Owl")
    assert bird is not None
    player.board.forest.slots[0].bird = bird

    game.card_tray.face_up = [
        _tray_bird_fixture(name="Flightless Tray", wingspan_cm=None, vp=8),
        _tray_bird_fixture(name="Eligible Small", wingspan_cm=35, vp=2),
        _tray_bird_fixture(name="Too Large", wingspan_cm=90, vp=6),
    ]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.FOREST)
    )
    assert res.executed
    assert player.board.forest.slots[0].tucked_cards == 1
    # Eligible small card should be removed; flightless card should be ignored safely.
    assert all(c.name != "Eligible Small" for c in game.card_tray.face_up)
    assert any(c.name == "Flightless Tray" for c in game.card_tray.face_up)


def test_discard_one_fish_tucks_two_cards_for_american_white_pelican():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("American White Pelican")
    player.board.wetland.slots[0].bird = bird
    player.food_supply.add(FoodType.FISH, 1)
    game._deck_cards = [birds.get("Twite"), birds.get("Golden Pheasant")]
    game.deck_remaining = 2

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert res.cards_tucked == 2
    assert player.board.wetland.slots[0].tucked_cards == 2
    assert player.food_supply.get(FoodType.FISH) == 0


def test_discard_one_seed_tucks_two_cards_for_canada_goose():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Canada Goose")
    player.board.grassland.slots[0].bird = bird
    player.food_supply.add(FoodType.SEED, 1)
    game._deck_cards = [birds.get("Twite"), birds.get("Golden Pheasant")]
    game.deck_remaining = 2

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert res.cards_tucked == 2
    assert player.board.grassland.slots[0].tucked_cards == 2
    assert player.food_supply.get(FoodType.SEED) == 0


def test_abbotts_booby_discards_any_two_bonus_cards_not_just_drawn():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    booby = birds.get("Abbott's Booby")
    assert booby is not None
    player.board.wetland.slots[0].bird = booby

    def _bonus(name: str, points: int) -> BonusCard:
        return BonusCard(
            name=name,
            game_sets=frozenset({GameSet.CORE}),
            condition_text=f"Fixture {name}",
            explanation_text=None,
            scoring_tiers=(BonusScoringTier(min_count=0, max_count=None, points=points),),
            is_per_bird=False,
            is_automa=False,
            draft_value_pct=float(points),
        )

    old_a = _bonus("Old A", 0)
    old_b = _bonus("Old B", 0)
    high = _bonus("New High", 8)
    mid = _bonus("New Mid", 5)
    low = _bonus("New Low", 2)
    filler = _bonus("Filler", 1)

    player.bonus_cards = [old_a, old_b]
    # DrawBonusCards pops from end => drawn cards are low, mid, high.
    game._bonus_cards = [filler, high, mid, low]  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    power = get_power(booby)
    res = power.execute(PowerContext(game_state=game, player=player, bird=booby, slot_index=0, habitat=Habitat.WETLAND))

    assert res.executed
    final_names = sorted(bc.name for bc in player.bonus_cards)
    assert final_names == sorted(["New High", "New Mid", "New Low"])
    discarded_names = sorted(bc.name for bc in game._bonus_discard_cards)  # type: ignore[attr-defined]
    assert discarded_names == sorted(["Old A", "Old B"])


def test_royal_spoonbill_can_refill_then_draw_platform_or_wild_from_tray():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Royal Spoonbill")
    assert bird is not None
    player.board.wetland.slots[0].bird = bird

    def _tray_bird(name: str, nest: NestType, vp: int) -> Bird:
        return Bird(
            name=name,
            scientific_name=name,
            game_set=GameSet.CORE,
            color=PowerColor.BROWN,
            power_text="",
            victory_points=vp,
            nest_type=nest,
            egg_limit=2,
            wingspan_cm=30,
            habitats=frozenset({Habitat.WETLAND}),
            food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
            beak_direction=BeakDirection.NONE,
            is_predator=False,
            is_flocking=False,
            is_bonus_card_bird=False,
            bonus_eligibility=frozenset(),
        )

    old_bowl = _tray_bird("Old Bowl", NestType.BOWL, 4)
    old_platform = _tray_bird("Old Platform", NestType.PLATFORM, 2)
    refill_wild = _tray_bird("Refill Wild", NestType.WILD, 3)
    filler = _tray_bird("Refill Filler", NestType.CAVITY, 1)
    game.card_tray.face_up = [old_bowl, old_platform]
    game._deck_cards = [filler, refill_wild]  # type: ignore[attr-defined]
    game.deck_remaining = 2

    queue_power_choice(
        game,
        "A",
        "Royal Spoonbill",
        {"tray_action": "refill", "draw_name": "Refill Wild"},
    )
    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))

    assert res.executed
    assert any(c.name == "Refill Wild" for c in player.hand)
    assert game.deck_remaining == 1
    assert len(game.card_tray.face_up) == 2
    assert game.discard_pile_count == 0


def test_royal_spoonbill_can_reset_then_draw_platform_or_wild_from_tray():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Royal Spoonbill")
    assert bird is not None
    player.board.wetland.slots[0].bird = bird

    def _tray_bird(name: str, nest: NestType, vp: int) -> Bird:
        return Bird(
            name=name,
            scientific_name=name,
            game_set=GameSet.CORE,
            color=PowerColor.BROWN,
            power_text="",
            victory_points=vp,
            nest_type=nest,
            egg_limit=2,
            wingspan_cm=30,
            habitats=frozenset({Habitat.WETLAND}),
            food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
            beak_direction=BeakDirection.NONE,
            is_predator=False,
            is_flocking=False,
            is_bonus_card_bird=False,
            bonus_eligibility=frozenset(),
        )

    game.card_tray.face_up = [
        _tray_bird("Old 1", NestType.BOWL, 1),
        _tray_bird("Old 2", NestType.GROUND, 2),
        _tray_bird("Old 3", NestType.CAVITY, 3),
    ]
    reset_target = _tray_bird("Reset Platform", NestType.PLATFORM, 6)
    reset_other_1 = _tray_bird("Reset Other 1", NestType.BOWL, 2)
    reset_other_2 = _tray_bird("Reset Other 2", NestType.CAVITY, 1)
    game._deck_cards = [reset_other_2, reset_other_1, reset_target]  # type: ignore[attr-defined]
    game.deck_remaining = 3

    queue_power_choice(
        game,
        "A",
        "Royal Spoonbill",
        {"tray_action": "reset", "draw_name": "Reset Platform"},
    )
    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))

    assert res.executed
    assert any(c.name == "Reset Platform" for c in player.hand)
    assert game.discard_pile_count == 3
    assert game.deck_remaining == 0
    assert len(game.card_tray.face_up) == 2


def test_australian_shelduck_can_refill_and_draw_cavity_or_wild():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Australian Shelduck")
    player.board.wetland.slots[0].bird = bird

    cavity = _tray_bird_fixture(name="Cavity Card", nest_type=NestType.CAVITY, vp=2)
    bowl = _tray_bird_fixture(name="Bowl Card", nest_type=NestType.BOWL, vp=8)
    wild = _tray_bird_fixture(name="Wild Card", nest_type=NestType.WILD, vp=4)
    filler = _tray_bird_fixture(name="Filler", nest_type=NestType.GROUND, vp=1)
    game.card_tray.face_up = [bowl, cavity]
    game._deck_cards = [filler, wild]  # type: ignore[attr-defined]
    game.deck_remaining = 2

    queue_power_choice(game, "A", "Australian Shelduck", {"tray_action": "refill", "draw_name": "Wild Card"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Wild Card" for c in player.hand)


def test_musk_duck_can_reset_and_draw_ground_or_wild():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Musk Duck")
    player.board.wetland.slots[0].bird = bird

    old = _tray_bird_fixture(name="Old", nest_type=NestType.CAVITY, vp=1)
    ground = _tray_bird_fixture(name="Ground Target", nest_type=NestType.GROUND, vp=5)
    other = _tray_bird_fixture(name="Other", nest_type=NestType.BOWL, vp=2)
    game.card_tray.face_up = [old]
    game._deck_cards = [other, ground]  # type: ignore[attr-defined]
    game.deck_remaining = 2

    queue_power_choice(game, "A", "Musk Duck", {"tray_action": "reset", "draw_name": "Ground Target"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Ground Target" for c in player.hand)
    assert game.discard_pile_count == 1


def test_willie_wagtail_can_refill_and_draw_bowl_or_wild():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Willie-Wagtail")
    assert bird is not None
    player.board.wetland.slots[0].bird = bird

    bowl = _tray_bird_fixture(name="Bowl Card", nest_type=NestType.BOWL, vp=2)
    cavity = _tray_bird_fixture(name="Cavity Card", nest_type=NestType.CAVITY, vp=8)
    wild = _tray_bird_fixture(name="Wild Card", nest_type=NestType.WILD, vp=4)
    filler = _tray_bird_fixture(name="Filler", nest_type=NestType.GROUND, vp=1)
    game.card_tray.face_up = [cavity]
    game._deck_cards = [filler, wild, bowl]  # type: ignore[attr-defined]
    game.deck_remaining = 3

    queue_power_choice(game, "A", "Willie-Wagtail", {"tray_action": "refill", "draw_name": "Wild Card"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Wild Card" for c in player.hand)


def test_brahminy_kite_can_stop_early_after_one_success(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Brahminy Kite")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    queue_power_choice(game, "A", "Brahminy Kite", {"roll_attempts": 1})
    monkeypatch.setattr("random.choice", lambda seq: FoodType.FISH if FoodType.FISH in seq else seq[0])
    before = slot.cached_food.get(FoodType.FISH, 0)
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.FISH, 0) == before + 1
    assert res.food_cached.get(FoodType.FISH, 0) == 1


def test_rhinoceros_auklet_caches_only_one_fish_even_if_rolls_would_hit_multiple():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Rhinoceros Auklet")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = [_tray_bird_fixture(name="A Card", vp=1)]
    b.hand = [_tray_bird_fixture(name="B Card", vp=1)]

    # Decline second-half discard for both players so cache behavior is isolated.
    queue_power_choice(game, "A", "Rhinoceros Auklet", {"activate": False})
    queue_power_choice(game, "B", "Rhinoceros Auklet", {"activate": False})

    power = get_power(bird)
    # Force the roll to always be fish: still only 1 fish should be cached.
    original_choice = random.choice
    try:
        random.choice = lambda seq: FoodType.FISH if FoodType.FISH in seq else seq[0]
        before_cached = slot.cached_food.get(FoodType.FISH, 0)
        res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    finally:
        random.choice = original_choice

    assert res.executed
    assert slot.cached_food.get(FoodType.FISH, 0) == before_cached + 1
    assert len(a.hand) == 1
    assert len(b.hand) == 1


def test_rhinoceros_auklet_all_players_discard_gain_fish_even_when_roll_misses():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Rhinoceros Auklet")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = [_tray_bird_fixture(name="A Low", vp=1), _tray_bird_fixture(name="A High", vp=6)]
    b.hand = [_tray_bird_fixture(name="B Low", vp=1)]

    fish_a_before = a.food_supply.get(FoodType.FISH)
    fish_b_before = b.food_supply.get(FoodType.FISH)
    cached_before = slot.cached_food.get(FoodType.FISH, 0)
    power = get_power(bird)

    # Force roll miss.
    original_choice = random.choice
    try:
        random.choice = lambda seq: FoodType.SEED if FoodType.SEED in seq else seq[0]
        res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    finally:
        random.choice = original_choice

    assert res.executed
    assert slot.cached_food.get(FoodType.FISH, 0) == cached_before
    assert a.food_supply.get(FoodType.FISH) == fish_a_before + 1
    assert b.food_supply.get(FoodType.FISH) == fish_b_before + 1
    assert len(a.hand) == 1
    assert len(b.hand) == 0


def test_sri_lanka_frogmouth_respects_per_player_optional_discard():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Sri Lanka Frogmouth")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = [_tray_bird_fixture(name="A Card", vp=1)]
    b.hand = [_tray_bird_fixture(name="B Card", vp=1)]

    # Actor chooses to discard/gain; opponent declines.
    queue_power_choice(game, "A", "Sri Lanka Frogmouth", {"activate": True})
    queue_power_choice(game, "B", "Sri Lanka Frogmouth", {"activate": False})

    inv_a_before = a.food_supply.get(FoodType.INVERTEBRATE)
    inv_b_before = b.food_supply.get(FoodType.INVERTEBRATE)
    power = get_power(bird)

    # Force roll miss so we only exercise the optional all-players branch.
    original_choice = random.choice
    try:
        random.choice = lambda seq: FoodType.SEED if FoodType.SEED in seq else seq[0]
        res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    finally:
        random.choice = original_choice

    assert res.executed
    assert a.food_supply.get(FoodType.INVERTEBRATE) == inv_a_before + 1
    assert b.food_supply.get(FoodType.INVERTEBRATE) == inv_b_before
    assert len(a.hand) == 0
    assert len(b.hand) == 1


def test_brahminy_kite_busts_and_loses_activation_cache_on_fail(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Brahminy Kite")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Attempt 1 hit, attempt 2 hit, attempt 3 miss -> bust (no net cache from this activation).
    rolls = iter([FoodType.FISH, FoodType.FISH, FoodType.SEED, FoodType.SEED, FoodType.SEED])
    monkeypatch.setattr("random.choice", lambda seq: next(rolls))
    queue_power_choice(game, "A", "Brahminy Kite", {"roll_attempts": 3})

    before = slot.cached_food.get(FoodType.FISH, 0)
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.FISH, 0) == before
    assert res.food_cached.get(FoodType.FISH, 0) == 0


def test_forest_owlet_can_stop_early_after_one_success(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Forest Owlet")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    queue_power_choice(game, "A", "Forest Owlet", {"roll_attempts": 1})
    monkeypatch.setattr(
        "random.choice",
        lambda seq: FoodType.INVERTEBRATE if FoodType.INVERTEBRATE in seq else seq[0],
    )
    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before + 1
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 1


def test_forest_owlet_busts_and_loses_activation_cache_on_fail(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Forest Owlet")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Attempt 1 hit, attempt 2 hit, attempt 3 miss -> bust (no net cache from this activation).
    queue_power_choice(game, "A", "Forest Owlet", {"roll_attempts": 3})
    power = get_power(bird)
    outcomes = iter([True, True, False])
    monkeypatch.setattr(power, "_roll_hits", lambda _faces: next(outcomes))

    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 0


def test_purple_heron_can_stop_early_after_one_success(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Purple Heron")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    queue_power_choice(game, "A", "Purple Heron", {"roll_attempts": 1})
    monkeypatch.setattr(
        "random.choice",
        lambda seq: FoodType.INVERTEBRATE if FoodType.INVERTEBRATE in seq else seq[0],
    )
    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before + 1
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 1


def test_purple_heron_busts_and_loses_activation_cache_on_fail(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Purple Heron")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Attempt 1 hit, attempt 2 hit, attempt 3 miss -> bust (no net cache from this activation).
    queue_power_choice(game, "A", "Purple Heron", {"roll_attempts": 3})
    power = get_power(bird)
    outcomes = iter([True, True, False])
    monkeypatch.setattr(power, "_roll_hits", lambda _faces: next(outcomes))

    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 0


def test_white_throated_kingfisher_can_stop_early_after_one_success(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("White-Throated Kingfisher")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    queue_power_choice(game, "A", "White-Throated Kingfisher", {"roll_attempts": 1})
    monkeypatch.setattr(
        "random.choice",
        lambda seq: FoodType.INVERTEBRATE if FoodType.INVERTEBRATE in seq else seq[0],
    )
    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before + 1
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 1


def test_white_throated_kingfisher_busts_and_loses_activation_cache_on_fail(monkeypatch):
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("White-Throated Kingfisher")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Attempt 1 hit, attempt 2 hit, attempt 3 miss -> bust (no net cache from this activation).
    queue_power_choice(game, "A", "White-Throated Kingfisher", {"roll_attempts": 3})
    power = get_power(bird)
    outcomes = iter([True, True, False])
    monkeypatch.setattr(power, "_roll_hits", lambda _faces: next(outcomes))

    before = slot.cached_food.get(FoodType.INVERTEBRATE, 0)
    res = power.execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert slot.cached_food.get(FoodType.INVERTEBRATE, 0) == before
    assert res.food_cached.get(FoodType.INVERTEBRATE, 0) == 0


def test_eurasian_marsh_harrier_can_stop_early_and_tuck_when_under_threshold():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Eurasian Marsh-Harrier")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    low_a = _tray_bird_fixture(name="Low A", wingspan_cm=20)
    low_b = _tray_bird_fixture(name="Low B", wingspan_cm=30)
    game._deck_cards = [low_b, low_a]  # type: ignore[attr-defined]
    game.deck_remaining = 2
    before_tucked = slot.tucked_cards
    before_discard = game.discard_pile_count

    queue_power_choice(game, "A", "Eurasian Marsh-Harrier", {"draw_attempts": 2})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 2
    assert slot.tucked_cards == before_tucked + 2
    assert game.discard_pile_count == before_discard
    assert game.deck_remaining == 0


def test_eurasian_marsh_harrier_discards_drawn_cards_when_total_wingspan_too_high():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Eurasian Marsh-Harrier")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    high_a = _tray_bird_fixture(name="High A", wingspan_cm=60)
    high_b = _tray_bird_fixture(name="High B", wingspan_cm=70)
    game._deck_cards = [high_b, high_a]  # type: ignore[attr-defined]
    game.deck_remaining = 2
    before_tucked = slot.tucked_cards
    before_discard = game.discard_pile_count

    queue_power_choice(game, "A", "Eurasian Marsh-Harrier", {"draw_attempts": 2})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 0
    assert slot.tucked_cards == before_tucked
    assert game.discard_pile_count == before_discard + 2
    assert game.deck_remaining == 0


def test_eurasian_marsh_harrier_stops_immediately_when_busts_before_max_draws():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Eurasian Marsh-Harrier")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    high_a = _tray_bird_fixture(name="High A", wingspan_cm=60)
    high_b = _tray_bird_fixture(name="High B", wingspan_cm=60)
    tail = _tray_bird_fixture(name="Tail", wingspan_cm=10)
    game._deck_cards = [tail, high_b, high_a]  # type: ignore[attr-defined]
    game.deck_remaining = 3
    before_discard = game.discard_pile_count

    queue_power_choice(game, "A", "Eurasian Marsh-Harrier", {"draw_attempts": 3})
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert res.cards_tucked == 0
    assert game.discard_pile_count == before_discard + 2
    # Must stop after bust on second draw; third draw should not occur.
    assert game.deck_remaining == 1
    assert len(getattr(game, "_deck_cards")) == 1
    assert getattr(game, "_deck_cards")[0].name == "Tail"


def test_eurasian_marsh_harrier_wild_wingspan_defaults_to_one_and_can_be_set():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    # Default wild wingspan selection should use the smallest positive value (1).
    game_ok = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_ok = game_ok.players[0]
    bird = birds.get("Eurasian Marsh-Harrier")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot_ok = player_ok.board.get_row(habitat).slots[0]
    slot_ok.bird = bird
    wild = _tray_bird_fixture(name="Wild Wingspan", wingspan_cm=None)  # type: ignore[arg-type]
    low = _tray_bird_fixture(name="Low", wingspan_cm=20)
    game_ok._deck_cards = [low, wild]  # type: ignore[attr-defined]
    game_ok.deck_remaining = 2
    before_tucked = slot_ok.tucked_cards
    res_ok = get_power(bird).execute(
        PowerContext(game_state=game_ok, player=player_ok, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res_ok.executed
    assert res_ok.cards_tucked == 2
    assert slot_ok.tucked_cards == before_tucked + 2

    # Explicit wild-wingspan choice can force a bust.
    game_fail = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_fail = game_fail.players[0]
    slot_fail = player_fail.board.get_row(habitat).slots[0]
    slot_fail.bird = bird
    wild_fail = _tray_bird_fixture(name="Wild Wingspan Fail", wingspan_cm=None)  # type: ignore[arg-type]
    game_fail._deck_cards = [wild_fail]  # type: ignore[attr-defined]
    game_fail.deck_remaining = 1
    discard_before = game_fail.discard_pile_count
    queue_power_choice(
        game_fail,
        "A",
        "Eurasian Marsh-Harrier",
        {"draw_attempts": 1, "wild_wingspans": [110]},
    )
    res_fail = get_power(bird).execute(
        PowerContext(game_state=game_fail, player=player_fail, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res_fail.executed
    assert res_fail.cards_tucked == 0
    assert slot_fail.tucked_cards == 0
    assert game_fail.discard_pile_count == discard_before + 1


def test_philippine_eagle_reroll_subset_can_hit_bonus_then_resets_feeder(monkeypatch):
    birds, bonus_reg, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Philippine Eagle")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    # Deterministic bonus deck so bonus draw can resolve without registry randomness.
    game._bonus_cards = [bonus_reg.all_cards[1], bonus_reg.all_cards[0]]  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    initial = [FoodType.RODENT, FoodType.SEED, FoodType.FRUIT, FoodType.FISH, FoodType.INVERTEBRATE]
    reset = [FoodType.SEED, FoodType.SEED, FoodType.SEED, FoodType.SEED, FoodType.SEED]
    reroll_calls = {"n": 0}

    def _rigged_reroll():
        reroll_calls["n"] += 1
        game.birdfeeder.set_dice(initial if reroll_calls["n"] == 1 else reset)

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]

    # Reroll only dice indices 1 and 2 once; both become rodents -> success.
    queue_power_choice(game, "A", "Philippine Eagle", {"reroll_sets": [[1, 2]]})
    picks = iter([FoodType.RODENT, FoodType.RODENT])
    monkeypatch.setattr("random.choice", lambda seq: next(picks))

    before_bonus = len(player.bonus_cards)
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert res.cards_drawn == 1
    assert len(player.bonus_cards) == before_bonus + 1
    # Always resets feeder after resolution.
    assert reroll_calls["n"] == 2
    assert game.birdfeeder.dice == reset


def test_philippine_eagle_can_stop_without_reroll_and_still_resets_feeder():
    birds, bonus_reg, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Philippine Eagle")
    assert bird is not None
    habitat = next(iter(bird.habitats))
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    game._bonus_cards = [bonus_reg.all_cards[1], bonus_reg.all_cards[0]]  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    initial = [FoodType.SEED, FoodType.SEED, FoodType.FRUIT, FoodType.FISH, FoodType.INVERTEBRATE]
    reset = [FoodType.RODENT, FoodType.RODENT, FoodType.SEED, FoodType.FRUIT, FoodType.FISH]
    reroll_calls = {"n": 0}

    def _rigged_reroll():
        reroll_calls["n"] += 1
        game.birdfeeder.set_dice(initial if reroll_calls["n"] == 1 else reset)

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]

    # Empty selection means "stop" (use 0 rerolls).
    queue_power_choice(game, "A", "Philippine Eagle", {"reroll_sets": [[]]})
    before_bonus = len(player.bonus_cards)
    res = get_power(bird).execute(
        PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat)
    )
    assert res.executed
    assert res.cards_drawn == 0
    assert len(player.bonus_cards) == before_bonus
    # Still resets feeder at end even on fail.
    assert reroll_calls["n"] == 2
    assert game.birdfeeder.dice == reset


def test_common_little_bittern_draws_tray_card_that_can_live_in_grassland():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Common Little Bittern")
    player.board.wetland.slots[0].bird = bird

    no = _tray_bird_fixture(name="Forest Only", habitats=frozenset({Habitat.FOREST}), vp=7)
    yes = _tray_bird_fixture(name="Grassland Bird", habitats=frozenset({Habitat.GRASSLAND}), vp=2)
    game.card_tray.face_up = [no, yes]

    queue_power_choice(game, "A", "Common Little Bittern", {"draw_name": "Grassland Bird"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Grassland Bird" for c in player.hand)


def test_squacco_heron_draws_tray_card_that_can_live_in_wetland():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Squacco Heron")
    player.board.wetland.slots[0].bird = bird

    no = _tray_bird_fixture(name="Forest Only", habitats=frozenset({Habitat.FOREST}), vp=7)
    yes = _tray_bird_fixture(name="Wetland Bird", habitats=frozenset({Habitat.WETLAND}), vp=2)
    game.card_tray.face_up = [no, yes]

    queue_power_choice(game, "A", "Squacco Heron", {"draw_name": "Wetland Bird"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Wetland Bird" for c in player.hand)


def test_brant_draws_all_face_up_tray_cards():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Brant")
    player.board.wetland.slots[0].bird = bird

    game.card_tray.face_up = [
        _tray_bird_fixture(name="T1"),
        _tray_bird_fixture(name="T2"),
        _tray_bird_fixture(name="T3"),
    ]
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert res.cards_drawn == 3
    assert len(game.card_tray.face_up) == 0


def test_yellow_bittern_draws_middle_tray_card():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Yellow Bittern")
    player.board.wetland.slots[0].bird = bird

    left = _tray_bird_fixture(name="Left")
    mid = _tray_bird_fixture(name="Middle")
    right = _tray_bird_fixture(name="Right")
    game.card_tray.face_up = [left, mid, right]
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "Middle" for c in player.hand)


def test_budgerigar_tucks_smallest_wingspan_from_tray():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Budgerigar")
    player.board.wetland.slots[0].bird = bird
    game.card_tray.face_up = [
        _tray_bird_fixture(name="Big", wingspan_cm=80),
        _tray_bird_fixture(name="Small", wingspan_cm=20),
    ]
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert player.board.wetland.slots[0].tucked_cards == 1
    assert all(c.name != "Small" for c in game.card_tray.face_up)


def test_cockatiel_discards_seed_to_tuck_chosen_tray_card():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Cockatiel")
    player.board.wetland.slots[0].bird = bird
    player.food_supply.add(FoodType.SEED, 1)
    game.card_tray.face_up = [_tray_bird_fixture(name="A"), _tray_bird_fixture(name="B")]

    queue_power_choice(game, "A", "Cockatiel", {"draw_name": "B"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert player.board.wetland.slots[0].tucked_cards == 1
    assert player.food_supply.get(FoodType.SEED) == 0
    assert all(c.name != "B" for c in game.card_tray.face_up)


def test_green_bee_eater_tucks_tray_card_with_invertebrate_cost():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Green Bee-Eater")
    player.board.wetland.slots[0].bird = bird
    game.card_tray.face_up = [
        _tray_bird_fixture(name="Seed Cost", food_items=(FoodType.SEED,)),
        _tray_bird_fixture(name="Invert Cost", food_items=(FoodType.INVERTEBRATE,)),
    ]

    queue_power_choice(game, "A", "Green Bee-Eater", {"draw_name": "Invert Cost"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert player.board.wetland.slots[0].tucked_cards == 1
    assert all(c.name != "Invert Cost" for c in game.card_tray.face_up)


def test_rufous_owl_tucks_tray_card_with_wingspan_below_threshold():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Rufous Owl")
    player.board.wetland.slots[0].bird = bird
    game.card_tray.face_up = [
        _tray_bird_fixture(name="Large", wingspan_cm=90),
        _tray_bird_fixture(name="Small", wingspan_cm=60),
    ]

    queue_power_choice(game, "A", "Rufous Owl", {"draw_name": "Small"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert player.board.wetland.slots[0].tucked_cards == 1
    assert all(c.name != "Small" for c in game.card_tray.face_up)


def test_black_throated_diver_discards_tray_refills_then_draws_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Black-Throated Diver")
    player.board.wetland.slots[0].bird = bird
    game.card_tray.face_up = [_tray_bird_fixture(name="Old1"), _tray_bird_fixture(name="Old2")]
    game._deck_cards = [_tray_bird_fixture(name="New1"), _tray_bird_fixture(name="New2")]  # type: ignore[attr-defined]
    game.deck_remaining = 2

    queue_power_choice(game, "A", "Black-Throated Diver", {"draw_name": "New2"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "New2" for c in player.hand)
    assert game.discard_pile_count == 2


def test_white_throated_dipper_discards_tray_refills_then_draws_one():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("White-Throated Dipper")
    player.board.wetland.slots[0].bird = bird
    game.card_tray.face_up = [_tray_bird_fixture(name="Old1"), _tray_bird_fixture(name="Old2")]
    game._deck_cards = [_tray_bird_fixture(name="New1"), _tray_bird_fixture(name="New2")]  # type: ignore[attr-defined]
    game.deck_remaining = 2

    queue_power_choice(game, "A", "White-Throated Dipper", {"draw_name": "New2"})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert any(c.name == "New2" for c in player.hand)
    assert game.discard_pile_count == 2


def test_red_wattled_lapwing_lays_egg_if_discarded_tray_includes_predator():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get("Red-Wattled Lapwing")
    player.board.wetland.slots[0].bird = bird

    predator = _tray_bird_fixture(name="Pred", is_predator=True)
    non_pred = _tray_bird_fixture(name="NonPred", is_predator=False)
    game.card_tray.face_up = [non_pred, predator]
    game._deck_cards = [_tray_bird_fixture(name="Refill")]  # type: ignore[attr-defined]
    game.deck_remaining = 1

    queue_power_choice(game, "A", "Red-Wattled Lapwing", {"discard_names": ["Pred"]})
    res = get_power(bird).execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=Habitat.WETLAND))
    assert res.executed
    assert res.eggs_laid == 1
    assert player.board.wetland.slots[0].eggs == 1


def test_eastern_rosella_all_players_nectar_with_self_seed_bonus():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Eastern Rosella")
    a.board.forest.slots[0].bird = bird

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.FOREST))

    assert res.executed
    assert a.food_supply.get(FoodType.NECTAR) >= 1
    assert b.food_supply.get(FoodType.NECTAR) >= 1
    assert a.food_supply.get(FoodType.SEED) == 1
    assert b.food_supply.get(FoodType.SEED) == 0


def test_indian_peafowl_all_players_draw_two_self_draws_one_extra():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Indian Peafowl")
    a.board.grassland.slots[0].bird = bird
    game._deck_cards = list(birds.all_birds[:30])
    game.deck_remaining = 30

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert len(a.hand) == 3
    assert len(b.hand) == 2


def test_major_mitchells_cockatoo_tuck_then_all_players_gain_seed():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Major Mitchell's Cockatoo")
    a.board.grassland.slots[0].bird = bird
    a.hand = [birds.get("Twite")]

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert res.cards_tucked == 1
    assert a.board.grassland.slots[0].tucked_cards == 1
    assert a.food_supply.get(FoodType.SEED) == 1
    assert b.food_supply.get(FoodType.SEED) == 1


def test_brown_shrike_all_players_may_cache_invertebrate_in_grassland():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Brown Shrike")
    a.board.grassland.slots[0].bird = bird
    b.board.grassland.slots[0].bird = birds.get("Twite")
    a.food_supply.add(FoodType.INVERTEBRATE, 1)
    b.food_supply.add(FoodType.INVERTEBRATE, 1)

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert a.board.grassland.slots[0].cached_food.get(FoodType.INVERTEBRATE, 0) == 1
    assert b.board.grassland.slots[0].cached_food.get(FoodType.INVERTEBRATE, 0) == 1
    assert a.food_supply.get(FoodType.INVERTEBRATE) == 0
    assert b.food_supply.get(FoodType.INVERTEBRATE) == 0


def test_zebra_dove_all_players_may_discard_seed_to_lay_egg():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Zebra Dove")
    a.board.grassland.slots[0].bird = bird
    b.board.grassland.slots[0].bird = birds.get("Twite")
    a.food_supply.add(FoodType.SEED, 1)
    b.food_supply.add(FoodType.SEED, 1)

    power = get_power(bird)
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.GRASSLAND))
    assert res.executed
    assert a.food_supply.get(FoodType.SEED) == 0
    assert b.food_supply.get(FoodType.SEED) == 0
    assert a.board.total_eggs() >= 1
    assert b.board.total_eggs() >= 1


def _bonus_fixture(name: str, points: int) -> BonusCard:
    return BonusCard(
        name=name,
        game_sets=frozenset({GameSet.CORE}),
        condition_text=f"Fixture {name}",
        explanation_text=None,
        scoring_tiers=(BonusScoringTier(min_count=0, max_count=None, points=points),),
        is_per_bird=False,
        is_automa=False,
        draft_value_pct=float(points),
    )


def test_spoon_billed_sandpiper_other_players_may_pay_two_resources_to_draw_bonus():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Spoon-Billed Sandpiper")
    assert bird is not None
    a.board.wetland.slots[0].bird = bird
    b.food_supply.add(FoodType.SEED, 2)

    # pop() order from end: active then opponent
    game._bonus_cards = [  # type: ignore[attr-defined]
        _bonus_fixture("Opp Low", 1),
        _bonus_fixture("Opp High", 8),
        _bonus_fixture("Active Low", 2),
        _bonus_fixture("Active High", 9),
    ]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.WETLAND)
    )

    assert res.executed
    assert len(a.bonus_cards) == 1
    assert len(b.bonus_cards) == 1
    assert a.bonus_cards[0].name == "Active High"
    assert b.bonus_cards[0].name == "Opp High"
    assert b.food_supply.get(FoodType.SEED) == 0
    assert sorted(bc.name for bc in game._bonus_discard_cards) == sorted(  # type: ignore[attr-defined]
        ["Active Low", "Opp Low"]
    )


def test_spoon_billed_sandpiper_nectar_payment_counts_as_spent_nectar():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Spoon-Billed Sandpiper")
    assert bird is not None
    a.board.wetland.slots[0].bird = bird
    for ft in (
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
        FoodType.NECTAR,
    ):
        have = b.food_supply.get(ft)
        if have > 0:
            b.food_supply.spend(ft, have)
    b.food_supply.add(FoodType.NECTAR, 2)

    game._bonus_cards = [  # type: ignore[attr-defined]
        _bonus_fixture("Opp Low", 1),
        _bonus_fixture("Opp High", 8),
        _bonus_fixture("Active Low", 2),
        _bonus_fixture("Active High", 9),
    ]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.WETLAND)
    )

    assert res.executed
    assert b.food_supply.get(FoodType.NECTAR) == 0
    assert b.board.wetland.nectar_spent == 2
    assert len(b.bonus_cards) == 1


def test_spoon_billed_sandpiper_opponent_may_decline():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    bird = birds.get("Spoon-Billed Sandpiper")
    assert bird is not None
    a.board.wetland.slots[0].bird = bird
    b.food_supply.add(FoodType.SEED, 2)

    game._bonus_cards = [  # type: ignore[attr-defined]
        _bonus_fixture("Opp Low", 1),
        _bonus_fixture("Opp High", 8),
        _bonus_fixture("Active Low", 2),
        _bonus_fixture("Active High", 9),
    ]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]
    queue_power_choice(game, "B", "Spoon-Billed Sandpiper", {"activate": False})

    power = get_power(bird)
    res = power.execute(
        PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=Habitat.WETLAND)
    )

    assert res.executed
    assert b.food_supply.get(FoodType.SEED) == 2
    assert len(b.bonus_cards) == 0


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


def test_all_446_birds_have_explicit_per_card_mapping():
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    all_names = {b.name for b in birds.all_birds}
    explicit = get_all_explicit_power_names()
    missing = sorted(all_names - explicit)
    assert missing == []
