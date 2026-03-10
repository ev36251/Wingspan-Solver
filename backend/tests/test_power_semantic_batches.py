import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.engine.rules import can_pay_food_cost
from backend.models.bird import Bird, FoodCost
from backend.models.enums import BeakDirection, BoardType, FoodType, GameSet, Habitat, NestType, PowerColor
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.choices import queue_power_choice
from backend.powers.registry import clear_cache, get_power


DRAW_CARDS_BIRDS = [
    "American Bittern", "Australasian Shoveler", "Australian Ibis",
    "Black-Necked Stilt", "Blue Rock-Thrush", "Canvasback", "Carolina Wren",
    "Common Loon", "Common Sandpiper",
    "Kelp Gull", "Little Egret",
    "Mallard", "Northern Shoveler", "Plumbeous Redstart",
    "Purple Gallinule", "Red Avadavat", "Sarus Crane", "Savi's Warbler", "Smew",
    "Spotted Sandpiper", "Wilson's Snipe",
]

COLUMN_EGG_DRAW_KEEP_BIRDS = [
    "Little Grebe",
]

DRAW_END_TURN_DISCARD_BIRDS = [
    "Black Tern",
    "Clark's Grebe",
    "Common Yellowthroat",
    "Forster's Tern",
    "Pied-Billed Grebe",
    "Red-Breasted Merganser",
    "Ruddy Duck",
    "Wood Duck",
]

DRAW_PER_EMPTY_KEEP_ONE_END_TURN_BIRDS = [
    "Great Crested Grebe",
    "Wilson's Storm Petrel",
]

GAIN_SUPPLY_BIRDS = [
    "American Goldfinch", "Azure Tit", "Baltimore Oriole", "Black-Chinned Hummingbird",
    "Blue-Gray Gnatcatcher", "Brown Pelican",
    "Eastern Phoebe", "Eurasian Hoopoe", "Northern Cardinal", "Olive-Backed Sunbird", "Osprey",
    "Painted Whitestart", "Red Crossbill", "Scissor-Tailed Flycatcher",
    "Silvereye", "Spotted Towhee", "Yellow-Bellied Sapsucker",
]

CHOOSE_OTHER_BOTH_GAIN_FOOD_BIRDS = [
    "Count Raggi's Bird-of-Paradise",
    "Eastern Whipbird",
    "Lewin's Honeyeater",
    "Regent Bowerbird",
]

CONDITIONAL_NEIGHBOR_HAS_FOOD_GAIN_BIRDS = [
    "Kererū",
    "Pesquet's Parrot",
    "Red-Capped Robin",
    "Red-Necked Avocet",
]

DISCARD_FOOD_GAIN_OTHER_BIRDS = [
    "Korimako",
    "Rufous-Banded Honeyeater",
]

MISTLETOEBIRD_MODE_BIRDS = [
    "Mistletoebird",
]

WINGSPAN_HABITAT_GAIN_BIRDS = [
    "Red Wattlebird",
]

PER_EGGED_BIRD_ROLL_GAIN_BIRDS = [
    "Desert Wheatear",
    "White-Browed Tit-Warbler",
]

DISCARD_EGG_FOR_FIXED_SUPPLY_BIRDS = [
    "Graceful Prinia",
]

DISCARD_OTHER_EGG_FOR_ONE_WILD_BIRDS = [
    "American Crow",
    "Black-Crowned Night-Heron",
    "Fish Crow",
]

RAVEN_EGG_FOR_FOOD_BIRDS = [
    "Chihuahuan Raven",
    "Common Raven",
]

DISCARD_EGG_DRAW_BIRDS = [
    "Franklin's Gull",
    "Killdeer",
]

TUCK_FROM_HAND_BIRDS = [
    "American Coot", "American Robin", "Barn Swallow", "Baya Weaver",
    "Brewer's Blackbird", "Bushtit", "Cedar Waxwing",
    "Common Grackle", "Dark-Eyed Junco", "Dickcissel", "Eurasian Coot",
    "House Finch", "Large-Billed Crow", "Maned Duck",
    "Pine Siskin", "Purple Martin", "Pygmy Nuthatch", "Red-Winged Blackbird",
    "Ring-Billed Gull", "Rosy Starling", "Tree Swallow", "Vaux's Swift",
    "White-Crested Laughingthrush", "White-Throated Swift", "Yellow-Headed Blackbird",
    "Yellow-Rumped Warbler",
]

TUCK_TO_PAY_COST_BIRDS = [
    "Bonelli's Eagle",
    "Eurasian Sparrowhawk",
    "Northern Goshawk",
]

CHOOSE_BIRDS_TUCK_FROM_HAND_BIRDS = [
    "Common Chaffinch",
    "Common Chiffchaff",
    "Mute Swan",
]

PER_OPPONENT_WETLAND_CUBES_BIRDS = [
    "Greater Flamingo",
]

LAY_EGGS_BIRDS = [
    "Baird's Sparrow",
    "California Quail", "Cassin's Sparrow", "Chipping Sparrow", "Common Iora",
    "Grasshopper Sparrow", "Green Pheasant",
    "Mourning Dove", "Northern Bobwhite", "Pūkeko",
    "Scaled Quail",
]

LAY_EGG_ON_SELF_PER_OTHER_HABITAT_BIRDS = [
    "Desert Finch",
]

DISCARD_FOOD_LAY_EGG_ON_SELF_BIRDS = [
    "Peaceful Dove",
    "Stubble Quail",
]

DISCARD_SPECIFIC_FOOD_THEN_LAY_UP_TO_BIRDS = [
    "Horsfield's Bushlark",
    "Thekla's Lark",
]

DISCARD_CARD_THEN_LAY_ON_SELF_BIRDS = [
    "Little Ringed Plover",
]

CHOOSE_OTHER_BOTH_LAY_EGG_BIRDS = [
    "Princess Stephanie's Astrapia",
]

CHOOSE_OTHER_LAY_THEN_DRAW_BIRDS = [
    "Brolga",
]

IF_HAS_FOOD_LAY_EGG_ON_SELF_BIRDS = [
    "Red-Vented Bulbul",
]

GIVE_FOOD_THEN_EGGS_OR_DIE_BIRDS = [
    "Red-Winged Parrot",
]

GIVE_CARD_THEN_LAY_EGGS_BIRDS = [
    "Satyr Tragopan",
]

LAY_EGG_ON_EACH_BIRD_IN_HABITAT_BIRDS = [
    "White-Breasted Woodswallow",
]

LAY_EGGS_ON_EACH_OTHER_BIRD_IN_COLUMN_BIRDS = [
    "Asian Emerald Dove",
]

LAY_EGG_ON_EACH_MATCHING_NEST_BIRDS = [
    "Ash-Throated Flycatcher",
    "Bobolink",
    "Inca Dove",
    "Say's Phoebe",
]

ALL_PLAYERS_LAY_ONE_PLUS_SELF_ADDITIONAL_ON_DISTINCT_NEST_BIRDS = [
    "Lazuli Bunting",
    "Pileated Woodpecker",
    "Western Meadowlark",
]

TOTAL_EGGS_THRESHOLD_SELF_LAY_BIRDS = [
    "Red Junglefowl",
]

DRAW_BONUS_BIRDS = [
    "Atlantic Puffin", "Bell's Vireo", "Black-Tailed Godwit", "California Condor",
    "Cassin's Finch", "Cerulean Warbler", "Chestnut-Collared Longspur", "Corsican Nuthatch",
    "Crested Ibis", "European Turtle Dove", "Greater Prairie-Chicken", "Kea", "King Rail",
    "North Island Brown Kiwi", "Painted Bunting", "Plains-Wanderer",
    "Red Knot", "Red-Cockaded Woodpecker", "Red-Crowned Crane", "Roseate Spoonbill",
    "Spoon-Billed Sandpiper", "Spotted Owl", "Sprague's Pipit", "White-Headed Duck",
    "Whooping Crane", "Wood Stork",
]

KEEP_GIVE_BIRDS = [
    "Green Pygmy-Goose",
]

BONUS_THEN_CARD_OR_EGG_BIRDS = [
    "Little Bustard",
]

PREDATOR_DICE_BIRDS = [
    "Anhinga", "Barn Owl", "Black Skimmer", "Broad-Winged Hawk",
    "Burrowing Owl", "Common Merganser", "Eastern Screech-Owl", "Eleonora's Falcon",
    "Eurasian Kestrel", "Ferruginous Hawk", "Mississippi Kite",
    "Northern Gannet", "Snowy Egret", "White-Faced Ibis", "Willet",
]

PUSH_YOUR_LUCK_PREDATOR_BIRDS = [
    "Brahminy Kite",
    "Forest Owlet",
    "Purple Heron",
    "White-Throated Kingfisher",
]

PUSH_YOUR_LUCK_WINGSPAN_DRAW_BIRDS = [
    "Eurasian Eagle-Owl",
    "Eurasian Marsh-Harrier",
]

PLAY_ADDITIONAL_BIRD_BIRDS = [
    "Australian Reed Warbler", "Common Moorhen", "Downy Woodpecker", "Eastern Bluebird",
    "Goldcrest", "Golden-Headed Cisticola", "Great Blue Heron", "Great Egret", "Grey Warbler",
    "House Wren", "Mountain Bluebird", "Red-Eyed Vireo", "Ruby-Crowned Kinglet",
    "Savannah Sparrow", "Short-Toed Treecreeper", "Small Minivet", "Trumpeter Finch",
    "Tufted Titmouse",
]

RESET_FEEDER_BIRDS = [
    "Bald Eagle", "Black Woodpecker", "Bullfinch",
    "European Bee-Eater", "European Honey Buzzard", "Great Tit",
    "Hawfinch", "Masked Lapwing", "Northern Flicker",
    "Tawny Frogmouth", "White-Bellied Sea-Eagle",
]

EMU_SPLIT_SEED_BIRDS = [
    "Emu",
]

RESET_FEEDER_GIVE_RODENT_BIRDS = [
    "Black-Shouldered Kite",
]

RESET_FEEDER_CHOOSE_ONE_BIRDS = [
    "Laughing Kookaburra",
]

RESET_GAIN_OPTIONAL_CACHE_BIRDS = [
    "Grey Shrikethrush",
    "White-Faced Heron",
]

GAIN_FEEDER_BIRDS = [
    "American Redstart", "Coppersmith Barbet", "Great Crested Flycatcher",
    "Great Spotted Woodpecker", "Indigo Bunting", "New Holland Honeyeater", "Parrot Crossbill",
    "Rainbow Lorikeet", "Rose-Breasted Grosbeak",
    "Verditer Flycatcher", "Western Tanager", "White-Backed Woodpecker",
]

EACH_PLAYER_FEEDER_BIRDS = [
    "Anna's Hummingbird",
    "Ruby-Throated Hummingbird",
]

FLOCKING_BIRDS = [
    # No birds currently use generic FlockingPower mappings.
]

CONDITIONAL_NEIGHBOR_FOOD_TUCK_BIRDS = [
    "Australian Zebra Finch",
]

CHOOSE_OTHER_RESET_GAIN_THEN_TUCK_BIRDS = [
    "Galah",
]

IF_EGG_LAID_THIS_TURN_TUCK_BIRDS = [
    "Grandala",
]

LOOK_KEEP_HABITAT_HAND_OR_TUCK_BIRDS = [
    "Grey Teal",
]

CACHE_OR_TUCK_THEN_TUCK_BIRDS = [
    "Rook",
]

TUCK_BEHIND_EACH_BIRD_IN_HABITAT_BIRDS = [
    "Welcome Swallow",
]

DISCARD_ALL_EGGS_NEST_TUCK_DOUBLE_BIRDS = [
    "Blyth's Hornbill",
]

RESET_GAIN_FISH_OPTIONAL_TUCK_BIRDS = [
    "Black Noddy",
]

DRAW_KEEP_TUCK_DISCARD_BIRDS = [
    "Audouin's Gull",
    "Ruddy Shelduck",
]

GAIN_OR_TUCK_BIRDS = [
    "Scaly-Breasted Munia",
]

PREDATOR_LOOK_BIRDS = [
    "Barred Owl", "Cooper's Hawk", "Golden Eagle", "Great Horned Owl", "Greater Roadrunner",
    "Grey Butcherbird", "Northern Harrier", "Peregrine Falcon", "Red-Shouldered Hawk",
    "Red-Tailed Hawk", "Swainson's Hawk",
]

CACHE_FEEDER_BIRDS = [
    "Blue Jay", "Clark's Nutcracker", "Common Green Magpie",
    "Red-Headed Woodpecker", "Steller's Jay",
]

WILLOW_TIT_BIRDS = [
    "Willow Tit",
]

STEAL_CACHE_TARGET_GAIN_DIE_BIRDS = [
    "Common Kingfisher",
    "Eurasian Jay",
    "Little Owl",
    "Red-Backed Shrike",
]

CACHE_SUPPLY_BIRDS = [
    "Carolina Chickadee", "Coal Tit", "Eurasian Nuthatch",
    "Juniper Titmouse", "Mountain Chickadee", "Red-Breasted Nuthatch",
    "White-Breasted Nuthatch",
]

SPENDABLE_CACHE_SUPPLY_BIRDS = [
    "Coal Tit",
    "Eurasian Nuthatch",
]

ROLL_PER_BIRD_IF_ANY_HIT_GAIN_CACHE_BIRDS = [
    "Stork-Billed Kingfisher",
]

MOVE_FISH_OPTIONAL_ROLL_CACHE_BIRDS = [
    "Great Cormorant",
]

CHOOSE_BIRDS_CACHE_SUPPLY_BIRDS = [
    "Eurasian Nutcracker",
]

PINK_LAY_EGG_TRIGGER_BIRDS = [
    "American Avocet", "Asian Koel", "Barrow's Goldeneye", "Bronzed Cowbird", "Brown-Headed Cowbird",
    "Horsfield's Bronze-Cuckoo", "Pheasant Coucal", "Violet Cuckoo", "Yellow-Billed Cuckoo",
]

MOVE_BIRD_BIRDS = [
    "Bewick's Wren", "Blue Grosbeak", "Chimney Swift", "Common Nighthawk", "Lincoln's Sparrow",
    "Song Sparrow", "White-Crowned Sparrow", "Yellow-Breasted Chat",
]

DISCARD_FOOD_TUCK_BIRDS = [
    "Black-Bellied Whistling-Duck", "Common Starling", "Common Swift", "Crimson Chat", "Double-Crested Cormorant",
    "Eurasian Collared-Dove", "Sandhill Crane",
]

NO_POWER_BIRDS = [
    "American Woodcock", "Blue-Winged Warbler", "Hooded Warbler", "Prothonotary Warbler", "Wild Turkey",
]


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


def _first_habitat(bird) -> Habitat:
    for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
        if bird.can_live_in(h):
            return h
    return Habitat.FOREST


def _make_support_bird(name: str, nest_type: NestType, habitat: Habitat) -> Bird:
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.CORE,
        color=PowerColor.NONE,
        power_text="",
        victory_points=1,
        nest_type=nest_type,
        egg_limit=6,
        wingspan_cm=30,
        habitats=frozenset({habitat}),
        food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
        beak_direction=BeakDirection.NONE,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )


def _make_support_bird_multi(
    *,
    name: str,
    nest_type: NestType = NestType.BOWL,
    habitats: frozenset[Habitat] | None = None,
    food_items: tuple[FoodType, ...] = (FoodType.SEED,),
    wingspan_cm: int = 30,
) -> Bird:
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.CORE,
        color=PowerColor.NONE,
        power_text="",
        victory_points=2,
        nest_type=nest_type,
        egg_limit=6,
        wingspan_cm=wingspan_cm,
        habitats=habitats or frozenset({Habitat.FOREST}),
        food_cost=FoodCost(items=food_items, is_or=False, total=len(food_items)),
        beak_direction=BeakDirection.NONE,
        is_predator=False,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )


@pytest.mark.parametrize("bird_name", DRAW_CARDS_BIRDS)
def test_semantic_batch_draw_cards(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    game._deck_cards = list(birds.all_birds[:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    power = get_power(bird)
    before_hand = len(player.hand)
    before_discard = game.discard_pile_count
    draw = int(getattr(power, "draw", 1))
    keep = int(getattr(power, "keep", draw))
    deck_before = game.deck_remaining

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    seen = min(draw, deck_before)
    expected_keep = min(keep, seen)
    assert len(player.hand) - before_hand == expected_keep
    assert res.cards_drawn == expected_keep
    assert game.discard_pile_count - before_discard == max(0, seen - expected_keep)


@pytest.mark.parametrize("bird_name", DRAW_END_TURN_DISCARD_BIRDS)
def test_semantic_batch_draw_then_end_turn_discard(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "DrawThenEndTurnDiscardFromHand"
    draw = int(getattr(power, "draw", 1))
    discard = int(getattr(power, "discard", 1))
    player.hand = list(birds.all_birds[:4])
    before_hand = len(player.hand)
    game._deck_cards = list(birds.all_birds[4:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    deck_before = game.deck_remaining

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    seen = min(draw, deck_before)
    assert res.executed
    assert res.cards_drawn == seen
    assert len(player.hand) - before_hand == seen
    assert game.pending_end_turn_hand_discards.get(player.name, 0) == (discard if seen > 0 else 0)


@pytest.mark.parametrize("bird_name", DRAW_PER_EMPTY_KEEP_ONE_END_TURN_BIRDS)
def test_semantic_batch_draw_per_empty_keep_one_at_end_turn(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    row.slots[1].bird = bird
    row.slots[0].bird = _make_support_bird_multi(name="Filled 0", habitats=frozenset({habitat}))
    row.slots[2].bird = _make_support_bird_multi(name="Filled 2", habitats=frozenset({habitat}))
    # Empty available slots should be 2 (indices 3 and 4).
    expected_draw = 2

    power = get_power(bird)
    assert type(power).__name__ == "DrawPerEmptySlotThenKeepOneAtEndTurn"
    player.hand = list(birds.all_birds[:3])
    before_hand = len(player.hand)
    game._deck_cards = list(birds.all_birds[3:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=1, habitat=habitat))
    assert res.executed
    assert res.cards_drawn == expected_draw
    assert len(player.hand) - before_hand == expected_draw
    assert game.pending_end_turn_hand_discards.get(player.name, 0) == max(0, expected_draw - 1)


@pytest.mark.parametrize("bird_name", GAIN_SUPPLY_BIRDS)
def test_semantic_batch_gain_food_from_supply(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "GainFoodFromSupply"
    food_types = list(getattr(power, "food_types", []))
    count = int(getattr(power, "count", 1))
    all_players = bool(getattr(power, "all_players", False))
    you_also = bool(getattr(power, "you_also", False))
    before = {ft: player.food_supply.get(ft) for ft in food_types}
    before_total = player.food_supply.total()

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    expected_actor_gain = count + (1 if (all_players and you_also) else 0)
    expected_total_gain = expected_actor_gain * len(food_types)
    for ft in food_types:
        if ft == FoodType.WILD:
            continue
        assert player.food_supply.get(ft) - before[ft] == expected_actor_gain
    assert player.food_supply.total() - before_total == expected_total_gain


@pytest.mark.parametrize("bird_name", CHOOSE_OTHER_BOTH_GAIN_FOOD_BIRDS)
def test_semantic_batch_choose_other_player_both_gain_food(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ChooseOtherPlayerBothGainFood"
    ft = getattr(power, "food_type")
    count = int(getattr(power, "count", 1))
    before_a = a.food_supply.get(ft)
    before_b = b.food_supply.get(ft)
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert a.food_supply.get(ft) == before_a + count
    assert b.food_supply.get(ft) == before_b + count


@pytest.mark.parametrize("bird_name", CONDITIONAL_NEIGHBOR_HAS_FOOD_GAIN_BIRDS)
def test_semantic_batch_conditional_neighbor_has_food_gain(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ConditionalNeighborHasFoodGainFood"
    ft = getattr(power, "food_type")
    before = a.food_supply.get(ft)
    # Oceania starts each player with 1 nectar; clear neighbors for a deterministic "no" case.
    for n in (game.player_to_left(a), game.player_to_right(a)):
        if n is not None and n.food_supply.get(ft) > 0:
            n.food_supply.spend(ft, n.food_supply.get(ft))

    # No food on neighbor -> no gain.
    res_no = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert not res_no.executed
    assert a.food_supply.get(ft) == before

    # Add food to neighbor -> gain.
    b.food_supply.add(ft, 1)
    res_yes = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res_yes.executed
    assert a.food_supply.get(ft) == before + 1


@pytest.mark.parametrize("bird_name", DISCARD_FOOD_GAIN_OTHER_BIRDS)
def test_semantic_batch_discard_food_to_gain_other_food(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird
    power = get_power(bird)
    assert type(power).__name__ == "DiscardFoodToGainOtherFood"

    discard_ft = getattr(power, "discard_type")
    gain_ft = getattr(power, "gain_type")
    player.food_supply.add(discard_ft, 2)
    before_discard = player.food_supply.get(discard_ft)
    before_gain = player.food_supply.get(gain_ft)

    requested = 2
    max_discard = getattr(power, "max_discard", None)
    if isinstance(max_discard, int):
        requested = min(requested, max_discard)
    queue_power_choice(game, player.name, bird_name, {"discard_count": requested})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.get(discard_ft) == before_discard - requested
    assert player.food_supply.get(gain_ft) == before_gain + requested


@pytest.mark.parametrize("bird_name", MISTLETOEBIRD_MODE_BIRDS)
def test_semantic_batch_gain_food_or_discard_for_other_food(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird
    power = get_power(bird)
    assert type(power).__name__ == "GainFoodOrDiscardSameGainOther"

    base_ft = getattr(power, "base_food")
    alt_ft = getattr(power, "alternate_gain")
    base_before = player.food_supply.get(base_ft)
    alt_before = player.food_supply.get(alt_ft)
    queue_power_choice(game, player.name, bird_name, {"mode": "gain"})
    res_gain = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_gain.executed
    assert player.food_supply.get(base_ft) == base_before + 1

    queue_power_choice(game, player.name, bird_name, {"mode": "discard_for_other"})
    res_alt = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_alt.executed
    assert player.food_supply.get(alt_ft) == alt_before + 1


@pytest.mark.parametrize("bird_name", WINGSPAN_HABITAT_GAIN_BIRDS)
def test_semantic_batch_gain_food_per_matching_wingspan_in_habitat(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird
    # Place two <49cm birds in forest.
    player.board.forest.slots[1].bird = _make_support_bird_multi(name="Small 1", habitats=frozenset({Habitat.FOREST}), wingspan_cm=20)
    player.board.forest.slots[2].bird = _make_support_bird_multi(name="Small 2", habitats=frozenset({Habitat.FOREST}), wingspan_cm=30)

    power = get_power(bird)
    assert type(power).__name__ == "GainFoodPerBirdWingspanInHabitat"
    ft = getattr(power, "food_type")
    before = player.food_supply.get(ft)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.get(ft) == before + 2


@pytest.mark.parametrize("bird_name", PER_EGGED_BIRD_ROLL_GAIN_BIRDS)
def test_semantic_batch_per_egged_bird_roll_die_gain_food(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PerEggedBirdInHabitatRollDieGainFood"
    target_hab = getattr(power, "habitat")
    row = player.board.get_row(target_hab)
    row.slots[1].bird = _make_support_bird_multi(name="Egged 1", habitats=frozenset({target_hab}))
    row.slots[1].eggs = 1
    row.slots[2].bird = _make_support_bird_multi(name="Egged 2", habitats=frozenset({target_hab}))
    row.slots[2].eggs = 1

    monkeypatch.setattr("random.choice", lambda _seq: FoodType.SEED)
    before_seed = player.food_supply.get(FoodType.SEED)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.get(FoodType.SEED) == before_seed + 2


@pytest.mark.parametrize("bird_name", CONDITIONAL_NEIGHBOR_FOOD_TUCK_BIRDS)
def test_semantic_batch_conditional_neighbor_has_food_tuck_from_deck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a = game.players[0]
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "ConditionalNeighborHasFoodTuckFromDeck"
    ft = getattr(power, "food_type")

    right = game.player_to_right(a)
    assert right is not None
    if right.food_supply.get(ft) > 0:
        right.food_supply.spend(ft, right.food_supply.get(ft))

    before_tucked = slot.tucked_cards
    res_no = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert not res_no.executed
    assert slot.tucked_cards == before_tucked

    right.food_supply.add(ft, 1)
    res_yes = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res_yes.executed
    assert slot.tucked_cards == before_tucked + int(getattr(power, "tuck_count", 1))


@pytest.mark.parametrize("bird_name", CHOOSE_OTHER_RESET_GAIN_THEN_TUCK_BIRDS)
def test_semantic_batch_choose_other_reset_gain_then_tuck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "ChooseOtherPlayerResetFeederGainFoodThenSelfTuck"
    ft = getattr(power, "food_type")
    tuck_count = int(getattr(power, "tuck_count", 2))

    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [ft, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE]
    )
    before_b = b.food_supply.get(ft)
    before_tucked = slot.tucked_cards
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert b.food_supply.get(ft) == before_b + 1
    assert slot.tucked_cards == before_tucked + tuck_count


@pytest.mark.parametrize("bird_name", IF_EGG_LAID_THIS_TURN_TUCK_BIRDS)
def test_semantic_batch_if_egg_laid_this_turn_tuck_from_deck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "IfEggLaidOnThisBirdThisTurnTuckFromDeck"
    before_tucked = slot.tucked_cards

    # Without action marker: no tuck.
    res_no = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_no.executed
    assert slot.tucked_cards == before_tucked

    # With action marker: tuck from deck.
    game._eggs_laid_this_action = {(player.name, habitat.value, 0)}  # type: ignore[attr-defined]
    res_yes = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_yes.executed
    assert slot.tucked_cards == before_tucked + int(getattr(power, "count", 1))


@pytest.mark.parametrize("bird_name", LOOK_KEEP_HABITAT_HAND_OR_TUCK_BIRDS)
def test_semantic_batch_look_keep_habitat_hand_or_tuck_discard_rest(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "LookAtDeckKeepHabitatBirdHandOrTuckDiscardRest"
    look_count = int(getattr(power, "look_count", 3))
    target_hab = getattr(power, "habitat_filter")
    keep_card = _make_support_bird_multi(name=f"{bird_name} Keep", habitats=frozenset({target_hab}))
    other_a = _make_support_bird_multi(name=f"{bird_name} Other A", habitats=frozenset({Habitat.FOREST}))
    other_b = _make_support_bird_multi(name=f"{bird_name} Other B", habitats=frozenset({Habitat.GRASSLAND}))
    game._deck_cards = [other_a, keep_card, other_b]  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    before_hand = len(player.hand)
    before_tucked = slot.tucked_cards
    before_discard = game.discard_pile_count
    queue_power_choice(game, player.name, bird_name, {"keep_name": keep_card.name, "keep_mode": "hand"})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert len(player.hand) == before_hand + 1
    assert slot.tucked_cards == before_tucked
    assert game.discard_pile_count == before_discard + (look_count - 1)


@pytest.mark.parametrize("bird_name", CACHE_OR_TUCK_THEN_TUCK_BIRDS)
def test_semantic_batch_cache_or_tuck_then_tuck_from_deck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    player.food_supply.add(FoodType.FISH, 1)
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "CacheAnyFoodOrTuckFromHandThenTuckDeck"
    before_cached = slot.cached_food.get(FoodType.FISH, 0)
    before_tucked = slot.tucked_cards
    before_supply = player.food_supply.get(FoodType.FISH)
    queue_power_choice(game, player.name, bird_name, {"mode": "cache", "food_type": "fish"})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.FISH, 0) == before_cached + 1
    assert player.food_supply.get(FoodType.FISH) == before_supply - 1
    assert slot.tucked_cards == before_tucked + 1


@pytest.mark.parametrize("bird_name", TUCK_BEHIND_EACH_BIRD_IN_HABITAT_BIRDS)
def test_semantic_batch_tuck_from_deck_behind_each_bird_in_habitat(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird
    row.slots[1].bird = _make_support_bird_multi(name=f"{bird_name} Support 1", habitats=frozenset({habitat}))
    row.slots[2].bird = _make_support_bird_multi(name=f"{bird_name} Support 2", habitats=frozenset({habitat}))
    game._deck_cards = list(birds.all_birds[:20])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "TuckFromDeckBehindEachBirdInHabitat"
    before = [row.slots[i].tucked_cards for i in (0, 1, 2)]
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 3
    assert [row.slots[i].tucked_cards for i in (0, 1, 2)] == [before[0] + 1, before[1] + 1, before[2] + 1]


@pytest.mark.parametrize("bird_name", LAY_EGG_ON_SELF_PER_OTHER_HABITAT_BIRDS)
def test_semantic_batch_lay_egg_on_self_per_other_bird_in_habitat(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "LayEggOnSelfPerOtherBirdInHabitat"
    source_hab = getattr(power, "habitat")
    source_row = player.board.get_row(source_hab)
    source_row.slots[1].bird = _make_support_bird_multi(name=f"{bird_name} Source 1", habitats=frozenset({source_hab}))
    source_row.slots[2].bird = _make_support_bird_multi(name=f"{bird_name} Source 2", habitats=frozenset({source_hab}))

    before = slot.eggs
    expected = sum(1 for s in source_row.slots if s.bird is not None)
    if source_hab == habitat:
        expected -= 1
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.eggs == before + expected


@pytest.mark.parametrize("bird_name", DISCARD_EGG_FOR_FIXED_SUPPLY_BIRDS)
def test_semantic_batch_discard_egg_for_fixed_supply_food(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    slot.eggs = 1

    power = get_power(bird)
    assert type(power).__name__ == "DiscardEggForBenefit"
    ft = getattr(power, "food_type")
    gain = int(getattr(power, "food_gain", 1))
    before_eggs = slot.eggs
    before_food = player.food_supply.get(ft)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.eggs == before_eggs - 1
    assert player.food_supply.get(ft) == before_food + gain


@pytest.mark.parametrize("bird_name", DISCARD_FOOD_LAY_EGG_ON_SELF_BIRDS)
def test_semantic_batch_discard_food_to_lay_eggs_on_self(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "DiscardFoodToLayEggOnSelf"
    food_type = getattr(power, "food_type")
    if food_type == FoodType.WILD:
        player.food_supply.add(FoodType.SEED, 2)
        player.food_supply.add(FoodType.FISH, 1)
    else:
        player.food_supply.add(food_type, 3)

    before_eggs = slot.eggs
    before_total_food = player.food_supply.total()
    queue_power_choice(game, player.name, bird_name, {"discard_count": 2})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.eggs == before_eggs + 2
    assert player.food_supply.total() == before_total_food - 2


@pytest.mark.parametrize("bird_name", DISCARD_SPECIFIC_FOOD_THEN_LAY_UP_TO_BIRDS)
def test_semantic_batch_discard_specific_food_then_lay_up_to(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "DiscardSpecificFoodThenLayEggOnSelfUpTo"
    ft = getattr(power, "food_type")
    player.food_supply.add(ft, 1)
    before_food = player.food_supply.get(ft)
    before_eggs = slot.eggs
    queue_power_choice(game, player.name, bird_name, {"lay_eggs": 1})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.get(ft) == before_food - 1
    assert slot.eggs == before_eggs + 1


@pytest.mark.parametrize("bird_name", DISCARD_CARD_THEN_LAY_ON_SELF_BIRDS)
def test_semantic_batch_discard_card_then_lay_egg_on_self(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    player.hand = list(birds.all_birds[:2])

    power = get_power(bird)
    assert type(power).__name__ == "DiscardCardThenLayEggOnSelf"
    discard_name = player.hand[0].name
    before_hand = len(player.hand)
    before_eggs = slot.eggs
    queue_power_choice(game, player.name, bird_name, {"discard_names": [discard_name]})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert len(player.hand) == before_hand - 1
    assert slot.eggs == before_eggs + 1


@pytest.mark.parametrize("bird_name", CHOOSE_OTHER_BOTH_LAY_EGG_BIRDS)
def test_semantic_batch_choose_other_player_both_lay_egg(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird
    b.board.forest.slots[0].bird = _make_support_bird_multi(name=f"{bird_name} Opp Target", habitats=frozenset({Habitat.FOREST}))

    power = get_power(bird)
    assert type(power).__name__ == "ChooseOtherPlayerBothLayEgg"
    before_a = a.board.total_eggs()
    before_b = b.board.total_eggs()
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert a.board.total_eggs() == before_a + 1
    assert b.board.total_eggs() == before_b + 1


@pytest.mark.parametrize("bird_name", CHOOSE_OTHER_LAY_THEN_DRAW_BIRDS)
def test_semantic_batch_choose_other_lay_egg_then_draw_cards(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird
    b.board.wetland.slots[0].bird = _make_support_bird_multi(name=f"{bird_name} Opp Egg Target", habitats=frozenset({Habitat.WETLAND}))
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "ChooseOtherPlayerLayEggThenDraw"
    draw_count = int(getattr(power, "draw_count", 2))
    before_b = b.board.total_eggs()
    before_hand = len(a.hand)
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert b.board.total_eggs() == before_b + 1
    assert len(a.hand) == before_hand + draw_count


@pytest.mark.parametrize("bird_name", IF_HAS_FOOD_LAY_EGG_ON_SELF_BIRDS)
def test_semantic_batch_if_has_food_lay_egg_on_self(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "IfHasFoodLayEggOnSelf"
    req = getattr(power, "required_food")
    if player.food_supply.get(req) > 0:
        player.food_supply.spend(req, player.food_supply.get(req))
    before_eggs = slot.eggs

    res_no = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_no.executed
    assert slot.eggs == before_eggs

    player.food_supply.add(req, 1)
    res_yes = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_yes.executed
    assert slot.eggs == before_eggs + 1


@pytest.mark.parametrize("bird_name", GIVE_FOOD_THEN_EGGS_OR_DIE_BIRDS)
def test_semantic_batch_give_food_to_other_then_eggs_or_die_reward(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "GiveFoodToOtherThenEggsOrDie"
    ft = getattr(power, "food_type")
    reward = int(getattr(power, "reward_count", 2))
    if a.food_supply.get(ft) < 2:
        a.food_supply.add(ft, 2 - a.food_supply.get(ft))

    before_a = a.food_supply.get(ft)
    before_b = b.food_supply.get(ft)
    before_eggs = slot.eggs
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name, "reward_mode": "eggs"})
    res_eggs = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res_eggs.executed
    assert a.food_supply.get(ft) == before_a - 1
    assert b.food_supply.get(ft) == before_b + 1
    assert slot.eggs == before_eggs + reward

    # Alternate reward branch: gain die-based food instead of eggs.
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name, "reward_mode": "die"})
    before_total_food = a.food_supply.total()
    before_eggs = slot.eggs
    res_die = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res_die.executed
    assert a.food_supply.total() >= before_total_food
    assert slot.eggs == before_eggs


@pytest.mark.parametrize("bird_name", GIVE_CARD_THEN_LAY_EGGS_BIRDS)
def test_semantic_batch_give_card_to_other_then_lay_eggs_on_self(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = list(birds.all_birds[:2])

    power = get_power(bird)
    assert type(power).__name__ == "GiveCardToOtherThenLayEggOnSelf"
    egg_count = int(getattr(power, "egg_count", 2))
    give_name = a.hand[0].name
    before_a_hand = len(a.hand)
    before_b_hand = len(b.hand)
    before_eggs = slot.eggs
    queue_power_choice(game, a.name, bird_name, {"target_player": b.name, "give_name": give_name})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert len(a.hand) == before_a_hand - 1
    assert len(b.hand) == before_b_hand + 1
    assert slot.eggs == before_eggs + egg_count


@pytest.mark.parametrize("bird_name", LAY_EGG_ON_EACH_BIRD_IN_HABITAT_BIRDS)
def test_semantic_batch_lay_egg_on_each_bird_in_habitat(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "EndGameLayEggOnEachBirdInHabitat"
    target_hab = getattr(power, "habitat")
    row = player.board.get_row(target_hab)
    row.slots[1].bird = _make_support_bird_multi(name=f"{bird_name} Habitat A", habitats=frozenset({target_hab}))
    row.slots[2].bird = _make_support_bird_multi(name=f"{bird_name} Habitat B", habitats=frozenset({target_hab}))
    birds_in_target = sum(1 for s in row.slots if s.bird is not None)

    before_total = player.board.total_eggs()
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.eggs_laid == birds_in_target
    assert player.board.total_eggs() == before_total + birds_in_target


@pytest.mark.parametrize("bird_name", ROLL_PER_BIRD_IF_ANY_HIT_GAIN_CACHE_BIRDS)
def test_semantic_batch_roll_per_bird_if_any_hit_gain_may_cache(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "RollOneDiePerBirdInHabitatIfAnyTargetGainMayCache"
    target_hab = getattr(power, "habitat")
    target_food = getattr(power, "target_food")
    gain_food = getattr(power, "gain_food")
    row = player.board.get_row(target_hab)
    row.slots[1].bird = _make_support_bird_multi(name=f"{bird_name} Roll A", habitats=frozenset({target_hab}))
    row.slots[2].bird = _make_support_bird_multi(name=f"{bird_name} Roll B", habitats=frozenset({target_hab}))

    # Hit path: include one matching roll, choose gain-to-supply branch.
    rolls = iter([target_food, FoodType.SEED, FoodType.FRUIT])
    monkeypatch.setattr("random.choice", lambda _seq: next(rolls))
    before_supply = player.food_supply.get(gain_food)
    queue_power_choice(game, player.name, bird_name, {"cache": False})
    res_hit = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_hit.executed
    assert player.food_supply.get(gain_food) == before_supply + 1

    # Miss path: no target food rolled.
    monkeypatch.setattr("random.choice", lambda _seq: FoodType.SEED)
    res_miss = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_miss.executed


@pytest.mark.parametrize("bird_name", RAVEN_EGG_FOR_FOOD_BIRDS)
def test_semantic_batch_raven_discards_egg_from_other_bird_for_two_food(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    raven_slot = player.board.get_row(habitat).slots[0]
    raven_slot.bird = bird
    raven_slot.eggs = 2
    support_hab = Habitat.GRASSLAND if habitat != Habitat.GRASSLAND else Habitat.WETLAND
    support_slot = player.board.get_row(support_hab).slots[0]
    support_slot.bird = _make_support_bird_multi(name="Raven Support", habitats=frozenset({support_hab}))
    support_slot.eggs = 1

    power = get_power(bird)
    before_total_food = player.food_supply.total()
    before_raven_eggs = raven_slot.eggs
    before_support_eggs = support_slot.eggs

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    assert player.food_supply.total() - before_total_food == 2
    assert raven_slot.eggs == before_raven_eggs
    assert support_slot.eggs == before_support_eggs - 1


@pytest.mark.parametrize("bird_name", DISCARD_OTHER_EGG_FOR_ONE_WILD_BIRDS)
def test_semantic_batch_discards_egg_from_other_bird_for_one_wild(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    power_slot = player.board.get_row(habitat).slots[0]
    power_slot.bird = bird
    power_slot.eggs = 2
    support_hab = Habitat.GRASSLAND if habitat != Habitat.GRASSLAND else Habitat.WETLAND
    support_slot = player.board.get_row(support_hab).slots[0]
    support_slot.bird = _make_support_bird_multi(name="Other Egg Source", habitats=frozenset({support_hab}))
    support_slot.eggs = 1

    power = get_power(bird)
    assert type(power).__name__ == "DiscardEggForBenefit"
    before_total_food = player.food_supply.total()
    before_power_eggs = power_slot.eggs
    before_support_eggs = support_slot.eggs

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.total() - before_total_food == 1
    assert power_slot.eggs == before_power_eggs
    assert support_slot.eggs == before_support_eggs - 1


@pytest.mark.parametrize("bird_name", DISCARD_EGG_DRAW_BIRDS)
def test_semantic_batch_discard_egg_to_draw_two_cards(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird
    support_hab = Habitat.GRASSLAND if habitat != Habitat.GRASSLAND else Habitat.WETLAND
    support_slot = player.board.get_row(support_hab).slots[0]
    support_slot.bird = _make_support_bird_multi(name="Egg Support", habitats=frozenset({support_hab}))
    support_slot.eggs = 1
    game._deck_cards = list(birds.all_birds[:20])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "DiscardEggForBenefit"
    before_eggs = player.board.total_eggs()
    before_hand = len(player.hand)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.board.total_eggs() == before_eggs - 1
    assert len(player.hand) == before_hand + 2


@pytest.mark.parametrize("bird_name", TUCK_FROM_HAND_BIRDS)
def test_semantic_batch_tuck_from_hand(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    player.hand = list(birds.all_birds[:8])
    game._deck_cards = list(birds.all_birds[8:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "TuckFromHand"
    tuck_count = int(getattr(power, "tuck_count", 1))
    draw_count = int(getattr(power, "draw_count", 0))
    lay_count = int(getattr(power, "lay_count", 0))
    food_type = getattr(power, "food_type", None)
    food_count = int(getattr(power, "food_count", 0))
    before_hand = len(player.hand)
    before_tucked = slot.tucked_cards
    before_eggs = slot.eggs
    before_food = player.food_supply.get(food_type) if food_type else 0
    deck_before = game.deck_remaining

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    expected_tucked = min(tuck_count, before_hand)
    expected_drawn = min(draw_count, deck_before) if expected_tucked > 0 else 0
    assert res.cards_tucked == expected_tucked
    assert slot.tucked_cards - before_tucked == expected_tucked
    assert len(player.hand) - before_hand == (-expected_tucked + expected_drawn)
    if lay_count > 0 and expected_tucked > 0:
        assert slot.eggs - before_eggs == min(lay_count, bird.egg_limit - before_eggs)
    if food_type and food_count > 0 and expected_tucked > 0:
        assert player.food_supply.get(food_type) - before_food == food_count


@pytest.mark.parametrize("bird_name", CHOOSE_BIRDS_TUCK_FROM_HAND_BIRDS)
def test_semantic_batch_choose_birds_in_habitat_tuck_from_hand_each(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird
    row.slots[1].bird = _make_support_bird_multi(name="Support 1", habitats=frozenset({habitat}))
    row.slots[2].bird = _make_support_bird_multi(name="Support 2", habitats=frozenset({habitat}))
    player.hand = list(birds.all_birds[:8])

    queue_power_choice(game, "A", bird_name, {"target_slots": [0, 2]})
    power = get_power(bird)
    assert type(power).__name__ == "ChooseBirdsInHabitatTuckFromHandEach"
    before_hand = len(player.hand)
    draw_if_tucked = int(getattr(power, "draw_if_tucked", 0))
    deck_before = game.deck_remaining

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 2
    expected_draw = min(draw_if_tucked, deck_before) if res.cards_tucked > 0 else 0
    assert res.cards_drawn == expected_draw
    assert row.slots[0].tucked_cards == 1
    assert row.slots[1].tucked_cards == 0
    assert row.slots[2].tucked_cards == 1
    assert len(player.hand) == before_hand - 2 + expected_draw


@pytest.mark.parametrize("bird_name", TUCK_TO_PAY_COST_BIRDS)
def test_semantic_batch_tuck_to_pay_cost_marker_class(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "TuckToPayCost"
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed


@pytest.mark.parametrize("bird_name", PER_OPPONENT_WETLAND_CUBES_BIRDS)
def test_semantic_batch_tuck_and_draw_per_opponent_wetland_cubes(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B", "C"], board_type=BoardType.OCEANIA)
    a, b, c = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = list(birds.all_birds[:8])
    game._deck_cards = list(birds.all_birds[8:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    b.draw_cards_actions_this_round = 3
    c.draw_cards_actions_this_round = 1
    queue_power_choice(game, "A", bird_name, {"target_player": "C"})

    power = get_power(bird)
    assert type(power).__name__ == "TuckHandThenDrawEqualPerOpponentWetlandCubes"
    before_hand = len(a.hand)
    before_tucked = slot.tucked_cards

    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 1
    assert res.cards_drawn == 1
    assert slot.tucked_cards - before_tucked == 1
    assert len(a.hand) == before_hand


def test_semantic_batch_noisy_miner_tuck_lay_and_others_may_lay(regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get("Noisy Miner")
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird
    a.hand = list(birds.all_birds[:2])
    b.board.forest.slots[0].bird = _make_support_bird_multi(name="Opp Support", habitats=frozenset({Habitat.FOREST}))

    power = get_power(bird)
    assert type(power).__name__ == "NoisyMinerTuckLayOthersLayEgg"
    before_hand = len(a.hand)
    before_tucked = slot.tucked_cards
    before_a_eggs = a.board.total_eggs()
    before_b_eggs = b.board.total_eggs()

    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    assert res.cards_tucked == 1
    assert slot.tucked_cards == before_tucked + 1
    assert len(a.hand) == before_hand - 1
    assert a.board.total_eggs() == before_a_eggs + 2
    assert b.board.total_eggs() == before_b_eggs + 1


@pytest.mark.parametrize("bird_name", LAY_EGGS_BIRDS)
def test_semantic_batch_lay_eggs(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    # Add broad coverage of candidate targets across habitats and nest types.
    player.board.forest.slots[1].bird = _make_support_bird("Support Bowl", NestType.BOWL, Habitat.FOREST)
    player.board.grassland.slots[1].bird = _make_support_bird("Support Cavity", NestType.CAVITY, Habitat.GRASSLAND)
    player.board.wetland.slots[1].bird = _make_support_bird("Support Ground", NestType.GROUND, Habitat.WETLAND)
    player.board.forest.slots[2].bird = _make_support_bird("Support Platform", NestType.PLATFORM, Habitat.FOREST)
    player.board.grassland.slots[2].bird = _make_support_bird("Support Wild", NestType.WILD, Habitat.GRASSLAND)

    power = get_power(bird)
    assert type(power).__name__ == "LayEggs"
    before_total = player.board.total_eggs()
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    assert res.eggs_laid >= 1
    assert player.board.total_eggs() - before_total == res.eggs_laid


@pytest.mark.parametrize("bird_name", ALL_PLAYERS_LAY_ONE_PLUS_SELF_ADDITIONAL_ON_DISTINCT_NEST_BIRDS)
def test_semantic_batch_all_players_lay_then_self_additional_must_be_distinct_bird(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    power = get_power(bird)
    assert type(power).__name__ == "AllPlayersLayEggsSelfBonus"
    assert bool(getattr(power, "self_bonus_must_be_distinct_bird", False))
    nest_filter = getattr(power, "nest_filter")
    assert nest_filter in {NestType.BOWL, NestType.CAVITY, NestType.GROUND}
    habitat = _first_habitat(bird)

    def _eligible_slots(player):
        out = []
        for hab, idx, slot in player.board.all_slots():
            if not slot.bird or not slot.can_hold_more_eggs():
                continue
            if slot.bird.nest_type not in {nest_filter, NestType.WILD}:
                continue
            out.append((hab, idx, slot))
        return out

    def _seed_game(open_eligible_count: int):
        game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
        a, b = game.players
        a.board.get_row(habitat).slots[0].bird = bird
        # Add guaranteed matching-nest supports so we can control eligible count.
        a.board.forest.slots[1].bird = _make_support_bird(f"{bird_name} A", nest_filter, Habitat.FOREST)
        a.board.grassland.slots[1].bird = _make_support_bird(f"{bird_name} B", nest_filter, Habitat.GRASSLAND)
        b.board.forest.slots[0].bird = _make_support_bird(f"{bird_name} Opp", nest_filter, Habitat.FOREST)

        eligible = _eligible_slots(a)
        assert len(eligible) >= open_eligible_count
        # Leave first N eligible slots open; fill the rest to capacity.
        keep = {(h, i) for h, i, _ in eligible[:open_eligible_count]}
        for h, i, slot in eligible:
            if (h, i) not in keep:
                slot.eggs = slot.bird.egg_limit
        return game, a, b, eligible[:open_eligible_count]

    # Case 1: exactly 1 eligible slot open for active player -> self lays only 1 total.
    game1, a1, b1, open1 = _seed_game(open_eligible_count=1)
    before_self_total_1 = a1.board.total_eggs()
    before_opp_total_1 = b1.board.total_eggs()
    res1 = power.execute(PowerContext(game_state=game1, player=a1, bird=bird, slot_index=0, habitat=habitat))
    assert res1.executed
    assert res1.eggs_laid == 1
    assert a1.board.total_eggs() == before_self_total_1 + 1
    assert b1.board.total_eggs() == before_opp_total_1 + 1

    # Case 2: 2 eligible slots open -> active lays 2, one egg on each distinct slot.
    game2, a2, b2, open2 = _seed_game(open_eligible_count=2)
    before_self_total_2 = a2.board.total_eggs()
    before_opp_total_2 = b2.board.total_eggs()
    (h1, i1, s1), (h2, i2, s2) = open2
    before_1 = s1.eggs
    before_2 = s2.eggs
    res2 = power.execute(PowerContext(game_state=game2, player=a2, bird=bird, slot_index=0, habitat=habitat))
    assert res2.executed
    assert res2.eggs_laid == 2
    assert a2.board.total_eggs() == before_self_total_2 + 2
    assert b2.board.total_eggs() == before_opp_total_2 + 1
    assert a2.board.get_row(h1).slots[i1].eggs == before_1 + 1
    assert a2.board.get_row(h2).slots[i2].eggs == before_2 + 1


@pytest.mark.parametrize("bird_name", LAY_EGGS_ON_EACH_OTHER_BIRD_IN_COLUMN_BIRDS)
def test_semantic_batch_lay_eggs_on_each_other_bird_in_column(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot_index = 0
    own_slot = player.board.get_row(habitat).slots[slot_index]
    own_slot.bird = bird

    # Seed both other habitats in same column with room for 2 eggs each.
    other_rows = [row for row in player.board.all_rows() if row.habitat != habitat]
    other_rows[0].slots[slot_index].bird = _make_support_bird("AED Other A", NestType.BOWL, other_rows[0].habitat)
    other_rows[1].slots[slot_index].bird = _make_support_bird("AED Other B", NestType.CAVITY, other_rows[1].habitat)

    power = get_power(bird)
    assert type(power).__name__ == "LayEggsOnEachOtherBirdInColumn"

    before_total = player.board.total_eggs()
    before_other_a = other_rows[0].slots[slot_index].eggs
    before_other_b = other_rows[1].slots[slot_index].eggs
    before_self = own_slot.eggs

    res = power.execute(
        PowerContext(
            game_state=game,
            player=player,
            bird=bird,
            slot_index=slot_index,
            habitat=habitat,
        )
    )

    assert res.executed
    assert res.eggs_laid == 4
    assert other_rows[0].slots[slot_index].eggs == before_other_a + 2
    assert other_rows[1].slots[slot_index].eggs == before_other_b + 2
    assert own_slot.eggs == before_self
    assert player.board.total_eggs() == before_total + 4


@pytest.mark.parametrize("bird_name", TOTAL_EGGS_THRESHOLD_SELF_LAY_BIRDS)
def test_semantic_batch_total_eggs_threshold_self_lay(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    power = get_power(bird)
    assert type(power).__name__ == "LayEggOnSelfIfTotalEggsBelow"
    habitat = _first_habitat(bird)

    # Case 1: total eggs is 5 (<6), so lay 1 on this bird.
    game_ok = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_ok = game_ok.players[0]
    slot_ok = player_ok.board.get_row(habitat).slots[0]
    slot_ok.bird = bird
    player_ok.board.forest.slots[1].bird = _make_support_bird("RJ Support A", NestType.BOWL, Habitat.FOREST)
    player_ok.board.grassland.slots[1].bird = _make_support_bird("RJ Support B", NestType.GROUND, Habitat.GRASSLAND)
    player_ok.board.wetland.slots[1].bird = _make_support_bird("RJ Support C", NestType.CAVITY, Habitat.WETLAND)
    player_ok.board.forest.slots[1].eggs = 2
    player_ok.board.grassland.slots[1].eggs = 2
    player_ok.board.wetland.slots[1].eggs = 1

    before_ok_total = player_ok.board.total_eggs()
    before_ok_self = slot_ok.eggs
    res_ok = power.execute(
        PowerContext(game_state=game_ok, player=player_ok, bird=bird, slot_index=0, habitat=habitat)
    )

    assert before_ok_total == 5
    assert res_ok.executed
    assert res_ok.eggs_laid == 1
    assert slot_ok.eggs == before_ok_self + 1
    assert player_ok.board.total_eggs() == before_ok_total + 1

    # Case 2: total eggs is 6 (not fewer than 6), so do not lay.
    game_blocked = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_blocked = game_blocked.players[0]
    slot_blocked = player_blocked.board.get_row(habitat).slots[0]
    slot_blocked.bird = bird
    player_blocked.board.forest.slots[1].bird = _make_support_bird("RJ Block A", NestType.BOWL, Habitat.FOREST)
    player_blocked.board.grassland.slots[1].bird = _make_support_bird("RJ Block B", NestType.GROUND, Habitat.GRASSLAND)
    player_blocked.board.wetland.slots[1].bird = _make_support_bird("RJ Block C", NestType.CAVITY, Habitat.WETLAND)
    player_blocked.board.forest.slots[1].eggs = 3
    player_blocked.board.grassland.slots[1].eggs = 2
    player_blocked.board.wetland.slots[1].eggs = 1

    before_blocked_total = player_blocked.board.total_eggs()
    before_blocked_self = slot_blocked.eggs
    res_blocked = power.execute(
        PowerContext(
            game_state=game_blocked,
            player=player_blocked,
            bird=bird,
            slot_index=0,
            habitat=habitat,
        )
    )

    assert before_blocked_total == 6
    assert not res_blocked.executed
    assert res_blocked.eggs_laid == 0
    assert slot_blocked.eggs == before_blocked_self
    assert player_blocked.board.total_eggs() == before_blocked_total


@pytest.mark.parametrize("bird_name", LAY_EGG_ON_EACH_MATCHING_NEST_BIRDS)
def test_semantic_batch_lay_egg_on_each_matching_nest_includes_wild_and_played_bird(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird

    # Include one guaranteed wild nest target and one non-matching target.
    player.board.forest.slots[1].bird = _make_support_bird("EachNest Wild", NestType.WILD, Habitat.FOREST)
    player.board.grassland.slots[1].bird = _make_support_bird("EachNest NonMatch", NestType.GROUND, Habitat.GRASSLAND)

    power = get_power(bird)
    assert type(power).__name__ == "EndGameLayEggOnMatchingNest"
    nest_types = set(getattr(power, "nest_types", set()))
    assert nest_types

    targets = []
    before_total = player.board.total_eggs()
    for hab, idx, slot in player.board.all_slots():
        if slot.bird is None or not slot.can_hold_more_eggs():
            continue
        nest = slot.bird.nest_type
        if nest in nest_types or (nest == NestType.WILD and NestType.WILD not in nest_types):
            targets.append((hab, idx))
    assert targets, "expected at least one matching nest target"

    # Newly played bird should be eligible for these birds (matches official text expectation).
    assert (habitat, 0) in targets

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.eggs_laid == len(targets)
    assert player.board.total_eggs() == before_total + len(targets)
    for hab, idx in targets:
        assert player.board.get_row(hab).slots[idx].eggs == 1


@pytest.mark.parametrize("bird_name", DRAW_BONUS_BIRDS)
def test_semantic_batch_draw_bonus_cards(bird_name: str, regs):
    birds, bonus_reg, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    opp = game.players[1]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    if bird_name in {"Spoon-Billed Sandpiper", "Crested Ibis"}:
        assert type(power).__name__ == "SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo"
        # Keep batch expectation focused on active-player semantics.
        queue_power_choice(game, opp.name, bird_name, {"activate": False})
    elif bird_name == "North Island Brown Kiwi":
        assert type(power).__name__ == "NorthIslandBrownKiwiDiscardBonusDrawKeep"
        # Kiwi must discard one existing bonus card first.
        player.bonus_cards = list(bonus_reg.all_cards[:1])
    elif bird_name == "Kea":
        assert type(power).__name__ == "KeaDrawBonusDiscardFoodForMoreKeepOne"
    else:
        assert type(power).__name__ == "DrawBonusCards"
    draw = int(getattr(power, "draw", 2))
    keep = int(getattr(power, "keep", 1))

    # Deterministic finite bonus deck.
    game._bonus_cards = list(bonus_reg.all_cards[:20])  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]

    before_bonus = len(player.bonus_cards)
    before_discard = len(game._bonus_discard_cards)  # type: ignore[attr-defined]
    deck_before = len(game._bonus_cards)  # type: ignore[attr-defined]

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    seen = min(draw, deck_before)
    expected_keep = min(keep, seen)
    expected_bonus_delta = expected_keep - (1 if bird_name == "North Island Brown Kiwi" else 0)
    expected_discard_delta = max(0, seen - expected_keep) + (1 if bird_name == "North Island Brown Kiwi" else 0)
    assert len(player.bonus_cards) - before_bonus == expected_bonus_delta
    assert len(game._bonus_discard_cards) - before_discard == expected_discard_delta  # type: ignore[attr-defined]


@pytest.mark.parametrize("bird_name", COLUMN_EGG_DRAW_KEEP_BIRDS)
def test_semantic_batch_draw_per_column_egg_keep_one(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot_idx = 0
    row = player.board.get_row(habitat)
    row.slots[slot_idx].bird = bird
    row.slots[slot_idx].eggs = 1

    # Seed egged birds in same column across the two other habitats.
    for other_row in player.board.all_rows():
        if other_row.habitat == habitat:
            continue
        other_row.slots[slot_idx].bird = _make_support_bird_multi(
            name=f"{other_row.habitat.value.title()} Col Egg",
            habitats=frozenset({other_row.habitat}),
        )
        other_row.slots[slot_idx].eggs = 1

    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    power = get_power(bird)
    assert type(power).__name__ == "DrawPerColumnBirdWithEggKeepOne"

    before_hand = len(player.hand)
    before_discard = game.discard_pile_count
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=slot_idx, habitat=habitat))
    assert res.executed
    assert res.cards_drawn == 1
    assert len(player.hand) == before_hand + 1
    assert game.discard_pile_count == before_discard + 2


@pytest.mark.parametrize("bird_name", BONUS_THEN_CARD_OR_EGG_BIRDS)
def test_semantic_batch_bonus_then_card_or_egg(bird_name: str, regs):
    birds, bonus_reg, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird
    player.board.get_row(habitat).slots[1].bird = _make_support_bird_multi(
        name="Egg Target",
        habitats=frozenset({habitat}),
    )

    game._bonus_cards = list(bonus_reg.all_cards[:10])  # type: ignore[attr-defined]
    game._bonus_discard_cards = []  # type: ignore[attr-defined]
    game._deck_cards = list(birds.all_birds[:20])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "SnowyOwlBonusThenChoice"
    before_bonus = len(player.bonus_cards)
    before_hand = len(player.hand)
    before_eggs = player.board.total_eggs()

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert len(player.bonus_cards) == before_bonus + 1
    assert (player.board.total_eggs() == before_eggs + 1) or (len(player.hand) == before_hand + 1)


@pytest.mark.parametrize("bird_name", KEEP_GIVE_BIRDS)
def test_semantic_batch_draw_keep_one_give_one(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird
    game._deck_cards = list(birds.all_birds[:20])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "PinkEaredDuckDrawKeepGive"
    before_a = len(a.hand)
    before_b = len(b.hand)

    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert len(a.hand) == before_a + 1
    assert len(b.hand) == before_b + 1


@pytest.mark.parametrize("bird_name", PREDATOR_DICE_BIRDS)
def test_semantic_batch_predator_dice(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PredatorDice"
    target = getattr(power, "target_food")

    # Ensure dice are outside feeder and deterministic "hit".
    game.birdfeeder.set_dice([])

    def _choice(seq):
        return target if target in seq else seq[0]

    monkeypatch.setattr("random.choice", _choice)
    before = slot.cached_food.get(target, 0)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(target, 0) - before == int(getattr(power, "cache_count", 1))


@pytest.mark.parametrize("bird_name", PUSH_YOUR_LUCK_PREDATOR_BIRDS)
def test_semantic_batch_push_your_luck_predator_dice_stop_early(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PushYourLuckPredatorDiceCache"
    cache_food = getattr(power, "cache_food")
    queue_power_choice(game, "A", bird_name, {"roll_attempts": 1})

    monkeypatch.setattr("random.choice", lambda seq: next(iter(getattr(power, "target_foods"))))
    before = slot.cached_food.get(cache_food, 0)
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(cache_food, 0) == before + 1


@pytest.mark.parametrize("bird_name", PUSH_YOUR_LUCK_WINGSPAN_DRAW_BIRDS)
def test_semantic_batch_push_your_luck_wingspan_draw_then_tuck_or_discard(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PushYourLuckDrawByWingspanTotalThenTuck"
    threshold = int(getattr(power, "wingspan_threshold", 110))

    # Success path: stop at 2 with low total wingspan -> tucks drawn cards.
    low_a = _make_support_bird_multi(name=f"{bird_name} Low A", wingspan_cm=20)
    low_b = _make_support_bird_multi(name=f"{bird_name} Low B", wingspan_cm=30)
    game._deck_cards = [low_b, low_a]  # type: ignore[attr-defined]
    game.deck_remaining = 2
    queue_power_choice(game, "A", bird_name, {"draw_attempts": 2})
    tucked_before = slot.tucked_cards
    discard_before = game.discard_pile_count
    res_ok = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_ok.executed
    assert res_ok.cards_tucked == 2
    assert slot.tucked_cards == tucked_before + 2
    assert game.discard_pile_count == discard_before

    # Bust path: 2 draws exceed threshold -> both are discarded.
    hi = max(56, threshold // 2 + 1)
    high_a = _make_support_bird_multi(name=f"{bird_name} High A", wingspan_cm=hi)
    high_b = _make_support_bird_multi(name=f"{bird_name} High B", wingspan_cm=hi)
    game._deck_cards = [high_b, high_a]  # type: ignore[attr-defined]
    game.deck_remaining = 2
    queue_power_choice(game, "A", bird_name, {"draw_attempts": 2})
    tucked_before = slot.tucked_cards
    discard_before = game.discard_pile_count
    res_fail = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_fail.executed
    assert res_fail.cards_tucked == 0
    assert slot.tucked_cards == tucked_before
    assert game.discard_pile_count == discard_before + 2


@pytest.mark.parametrize("bird_name", PLAY_ADDITIONAL_BIRD_BIRDS)
def test_semantic_batch_play_additional_bird_permission(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PlayAdditionalBird"
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert "May play additional bird" in res.description


@pytest.mark.parametrize("bird_name", RESET_FEEDER_BIRDS)
def test_semantic_batch_reset_feeder_gain_food(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ResetFeederGainFood"
    target = getattr(power, "food_type", None)
    cache = bool(getattr(power, "cache", False))
    if target is not None:
        def _choice(seq):
            return target if target in seq else seq[0]
        monkeypatch.setattr("random.choice", _choice)
        before = player.food_supply.get(target)
        before_cached = player.board.get_row(habitat).slots[0].cached_food.get(target, 0)
        res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
        assert res.executed
        if cache:
            assert player.board.get_row(habitat).slots[0].cached_food.get(target, 0) >= before_cached + 1
        else:
            assert player.food_supply.get(target) >= before + 1
    else:
        res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
        assert res.executed


@pytest.mark.parametrize("bird_name", EMU_SPLIT_SEED_BIRDS)
def test_semantic_batch_emu_gain_all_seed_keep_half_distribute_remainder(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)

    # 2-player behavior: actor keeps ceil(n/2), opponent gets remainder.
    game_2p = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a2, b2 = game_2p.players
    habitat = _first_habitat(bird)
    a2.board.get_row(habitat).slots[0].bird = bird
    game_2p.birdfeeder.set_dice([FoodType.SEED, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT])
    before_a2 = a2.food_supply.get(FoodType.SEED)
    before_b2 = b2.food_supply.get(FoodType.SEED)
    power = get_power(bird)
    assert type(power).__name__ == "EmuGainAllSeedKeepHalfDistributeRemainder"
    res_2p = power.execute(PowerContext(game_state=game_2p, player=a2, bird=bird, slot_index=0, habitat=habitat))
    assert res_2p.executed
    assert a2.food_supply.get(FoodType.SEED) == before_a2 + 1
    assert b2.food_supply.get(FoodType.SEED) == before_b2 + 1
    assert not game_2p.birdfeeder.can_take(FoodType.SEED)

    # 3-player behavior with explicit distribution.
    game_3p = create_new_game(["A", "B", "C"], board_type=BoardType.OCEANIA)
    a3, b3, c3 = game_3p.players
    a3.board.get_row(habitat).slots[0].bird = bird
    game_3p.birdfeeder.set_dice([FoodType.SEED, FoodType.SEED, FoodType.SEED, FoodType.FISH, FoodType.FRUIT])
    before_a3 = a3.food_supply.get(FoodType.SEED)
    before_b3 = b3.food_supply.get(FoodType.SEED)
    before_c3 = c3.food_supply.get(FoodType.SEED)
    queue_power_choice(game_3p, a3.name, bird_name, {"distribution": {b3.name: 1, c3.name: 0}})
    res_3p = power.execute(PowerContext(game_state=game_3p, player=a3, bird=bird, slot_index=0, habitat=habitat))
    assert res_3p.executed
    # 3 seeds -> keep 2, distribute 1 (explicitly to B).
    assert a3.food_supply.get(FoodType.SEED) == before_a3 + 2
    assert b3.food_supply.get(FoodType.SEED) == before_b3 + 1
    assert c3.food_supply.get(FoodType.SEED) == before_c3


@pytest.mark.parametrize("bird_name", RESET_FEEDER_GIVE_RODENT_BIRDS)
def test_semantic_batch_reset_feeder_gain_rodent_may_give_for_eggs(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ResetFeederGainRodentMayGiveLayUpToThree"

    # Success path: rodent available, actor gives it, lays up to 3 eggs on this bird.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.RODENT, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE]
    )
    before_a_rodent = a.food_supply.get(FoodType.RODENT)
    before_b_rodent = b.food_supply.get(FoodType.RODENT)
    before_eggs = slot.eggs
    queue_power_choice(game, "A", bird_name, {"give_player": "B", "lay_eggs": 3})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert b.food_supply.get(FoodType.RODENT) == before_b_rodent + 1
    assert a.food_supply.get(FoodType.RODENT) == before_a_rodent
    assert slot.eggs == before_eggs + min(3, slot.bird.egg_limit - before_eggs)
    assert game.birdfeeder.count == 4

    # No-rodent path: no gain, no give, no eggs.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.SEED, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.INVERTEBRATE]
    )
    before_a_rodent = a.food_supply.get(FoodType.RODENT)
    before_b_rodent = b.food_supply.get(FoodType.RODENT)
    before_eggs = slot.eggs
    res_none = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert not res_none.executed
    assert a.food_supply.get(FoodType.RODENT) == before_a_rodent
    assert b.food_supply.get(FoodType.RODENT) == before_b_rodent
    assert slot.eggs == before_eggs
    assert game.birdfeeder.count == 5


@pytest.mark.parametrize("bird_name", RESET_GAIN_OPTIONAL_CACHE_BIRDS)
def test_semantic_batch_reset_gain_optional_cache_split(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ResetFeederGainAllMayCacheAny"
    target = getattr(power, "food_type")
    other = FoodType.SEED if target != FoodType.SEED else FoodType.FRUIT

    # Deterministic feeder result after reset: exactly 3 target dice.
    def _rigged_reroll():
        game.birdfeeder.set_dice([target, target, target, other, FoodType.INVERTEBRATE])

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]
    supply_before = player.food_supply.get(target)
    cached_before = slot.cached_food.get(target, 0)

    queue_power_choice(game, player.name, bird_name, {"cache_count": 2})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(target, 0) == cached_before + 2
    assert player.food_supply.get(target) == supply_before + 1


@pytest.mark.parametrize("bird_name", RESET_FEEDER_CHOOSE_ONE_BIRDS)
def test_semantic_batch_reset_feeder_choose_one_from_options(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    game.players[0].board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "ResetFeederGainOneFromOptions"

    # Explicitly choose fish when available.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.INVERTEBRATE, FoodType.FISH, FoodType.RODENT, FoodType.SEED, FoodType.FRUIT]
    )
    fish_before = player.food_supply.get(FoodType.FISH)
    queue_power_choice(game, player.name, bird_name, {"food_type": "fish"})
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert player.food_supply.get(FoodType.FISH) == fish_before + 1

    # No eligible types rolled => no gain.
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.SEED, FoodType.FRUIT, FoodType.SEED, FoodType.FRUIT, FoodType.NECTAR]
    )
    total_before = player.food_supply.total()
    res_none = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_none.executed
    assert player.food_supply.total() == total_before


@pytest.mark.parametrize("bird_name", GAIN_FEEDER_BIRDS)
def test_semantic_batch_gain_food_from_feeder(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "GainFoodFromFeeder"
    food_types = list(getattr(power, "food_types", []) or [FoodType.SEED])
    count = int(getattr(power, "count", 1))
    first = food_types[0]
    second = FoodType.FISH if first != FoodType.FISH else FoodType.SEED
    # Avoid all-same dice, which would trigger an auto-reroll in power logic.
    game.birdfeeder.set_dice([first, second, first, second, first])
    before_supply = player.food_supply.get(first)
    before_cached = slot.cached_food.get(first, 0)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    # May-cache powers transfer food from supply to cached; track combined gain.
    combined_delta = (player.food_supply.get(first) - before_supply) + (slot.cached_food.get(first, 0) - before_cached)
    assert combined_delta == min(count, 5)


@pytest.mark.parametrize("bird_name", EACH_PLAYER_FEEDER_BIRDS)
def test_semantic_batch_each_player_gains_from_feeder(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    a.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "EachPlayerGainDieFromFeederStartingChoice"
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE])
    before_a = a.food_supply.total()
    before_b = b.food_supply.total()

    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert a.food_supply.total() == before_a + 1
    assert b.food_supply.total() == before_b + 1


@pytest.mark.parametrize("bird_name", FLOCKING_BIRDS or ["__none__"])
def test_semantic_batch_flocking_tucks_from_deck(bird_name: str, regs):
    if bird_name == "__none__":
        pytest.skip("No birds currently use generic FlockingPower mappings")
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "FlockingPower"
    count = int(getattr(power, "count", 1))
    game._deck_cards = list(birds.all_birds[:20])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    before_tucked = slot.tucked_cards

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == count
    assert slot.tucked_cards - before_tucked == count


@pytest.mark.parametrize("bird_name", DISCARD_ALL_EGGS_NEST_TUCK_DOUBLE_BIRDS)
def test_semantic_batch_discard_all_eggs_one_cavity_then_tuck_double(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    source_slot = row.slots[0]
    source_slot.bird = bird

    # Build a cavity target with eggs in another slot/habitat.
    cavity_hab = Habitat.FOREST if habitat != Habitat.FOREST else Habitat.GRASSLAND
    cavity_slot_idx = 1
    cavity_slot = player.board.get_row(cavity_hab).slots[cavity_slot_idx]
    cavity_slot.bird = _make_support_bird("BH Cavity Target", NestType.CAVITY, cavity_hab)
    cavity_slot.eggs = 2

    power = get_power(bird)
    assert type(power).__name__ == "DiscardAllEggsFromOneNestThenTuckDouble"

    game._deck_cards = list(birds.all_birds[:30])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    before_tucked = source_slot.tucked_cards

    queue_power_choice(
        game,
        player.name,
        bird_name,
        {"target_habitat": cavity_hab.value, "target_slot": cavity_slot_idx},
    )
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    assert cavity_slot.eggs == 0
    assert res.cards_tucked == 4
    assert source_slot.tucked_cards == before_tucked + 4


@pytest.mark.parametrize("bird_name", RESET_GAIN_FISH_OPTIONAL_TUCK_BIRDS)
def test_semantic_batch_black_noddy_reset_gain_fish_optional_tuck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "BlackNoddyResetFeederGainFishOptionalDiscardToTuck"

    def _rigged_reroll():
        game.birdfeeder.set_dice(
            [FoodType.FISH, FoodType.FISH, FoodType.FISH, FoodType.SEED, FoodType.FRUIT]
        )

    game.birdfeeder.reroll = _rigged_reroll  # type: ignore[assignment]
    game._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    queue_power_choice(game, player.name, bird_name, {"discard_fish_for_tuck": 2})
    before_tucked = slot.tucked_cards
    fish_before = player.food_supply.get(FoodType.FISH)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.cards_tucked == 2
    assert slot.tucked_cards - before_tucked == 2
    # Gained 3 fish, discarded 2 for tucks -> net +1 fish.
    assert player.food_supply.get(FoodType.FISH) == fish_before + 1


@pytest.mark.parametrize("bird_name", DRAW_KEEP_TUCK_DISCARD_BIRDS)
def test_semantic_batch_draw_keep_tuck_discard(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "DrawKeepTuckDiscard"
    draw_count = int(getattr(power, "draw_count", 2))
    keep_count = int(getattr(power, "keep_count", 1))
    tuck_count = int(getattr(power, "tuck_count", 1))

    game._deck_cards = list(birds.all_birds[:50])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    before_hand = len(player.hand)
    before_tucked = slot.tucked_cards
    before_discard = game.discard_pile_count
    deck_before = game.deck_remaining

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    seen = min(draw_count, deck_before)
    expected_keep = min(keep_count, seen)
    expected_tucked = min(tuck_count, max(0, seen - expected_keep))
    expected_discard = max(0, seen - expected_keep - expected_tucked)
    assert res.cards_drawn == expected_keep
    assert res.cards_tucked == expected_tucked
    assert len(player.hand) - before_hand == expected_keep
    assert slot.tucked_cards - before_tucked == expected_tucked
    assert game.discard_pile_count - before_discard == expected_discard


@pytest.mark.parametrize("bird_name", GAIN_OR_TUCK_BIRDS)
def test_semantic_batch_gain_seed_or_tuck_from_deck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)

    # Gain path
    game_gain = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_gain = game_gain.players[0]
    habitat = _first_habitat(bird)
    slot_gain = player_gain.board.get_row(habitat).slots[0]
    slot_gain.bird = bird
    power = get_power(bird)
    assert type(power).__name__ == "GainSeedFromSupplyOrTuckFromDeck"
    seed_before = player_gain.food_supply.get(FoodType.SEED)
    queue_power_choice(game_gain, player_gain.name, bird_name, {"mode": "gain"})
    res_gain = power.execute(PowerContext(game_state=game_gain, player=player_gain, bird=bird, slot_index=0, habitat=habitat))
    assert res_gain.executed
    assert player_gain.food_supply.get(FoodType.SEED) == seed_before + 1
    assert slot_gain.tucked_cards == 0

    # Tuck path
    game_tuck = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player_tuck = game_tuck.players[0]
    slot_tuck = player_tuck.board.get_row(habitat).slots[0]
    slot_tuck.bird = bird
    game_tuck._deck_cards = list(birds.all_birds[:10])  # type: ignore[attr-defined]
    game_tuck.deck_remaining = len(game_tuck._deck_cards)
    queue_power_choice(game_tuck, player_tuck.name, bird_name, {"mode": "tuck"})
    res_tuck = power.execute(PowerContext(game_state=game_tuck, player=player_tuck, bird=bird, slot_index=0, habitat=habitat))
    assert res_tuck.executed
    assert slot_tuck.tucked_cards == 1


@pytest.mark.parametrize("bird_name", PREDATOR_LOOK_BIRDS)
def test_semantic_batch_predator_look_tucks_matching_card(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "PredatorLookAt"

    if getattr(power, "food_cost_includes", None):
        ft = sorted(list(getattr(power, "food_cost_includes")), key=lambda x: x.value)[0]
        match = _make_support_bird_multi(name="Match Food", food_items=(ft,))
    elif getattr(power, "habitat_filter", None) is not None:
        hab = getattr(power, "habitat_filter")
        match = _make_support_bird_multi(name="Match Habitat", habitats=frozenset({hab}))
    else:
        threshold = int(getattr(power, "wingspan_threshold", 75))
        cmp_mode = str(getattr(power, "wingspan_cmp", "lt"))
        wingspan = threshold + 10 if cmp_mode == "gt" else max(1, threshold - 10)
        match = _make_support_bird_multi(name="Match Wingspan", wingspan_cm=wingspan)

    game._deck_cards = [match]  # type: ignore[attr-defined]
    game.deck_remaining = 1
    before_tucked = slot.tucked_cards
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.tucked_cards == before_tucked + 1


@pytest.mark.parametrize("bird_name", CACHE_FEEDER_BIRDS)
def test_semantic_batch_cache_food_from_feeder(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "CacheFoodFromFeeder"
    food_type = getattr(power, "food_type")
    dice = [food_type, FoodType.SEED, food_type, FoodType.FISH, food_type]
    game.birdfeeder.set_dice(dice)
    before = slot.cached_food.get(food_type, 0)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    expected = sum(
        1
        for d in dice
        if d == food_type or (isinstance(d, tuple) and food_type in d)
    )
    assert slot.cached_food.get(food_type, 0) - before == expected


@pytest.mark.parametrize("bird_name", WILLOW_TIT_BIRDS)
def test_semantic_batch_willow_tit_cache_with_optional_reset_on_all_same(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "WillowTitCacheOneFromFeederOptionalResetOnAllSame"

    # Case 1: all same + explicit reset, then choose fruit from refreshed feeder.
    game.birdfeeder.set_dice([FoodType.RODENT] * 5)
    game.birdfeeder.reroll = lambda: game.birdfeeder.set_dice(  # type: ignore[assignment]
        [FoodType.FRUIT, FoodType.RODENT, FoodType.FISH, FoodType.SEED, FoodType.INVERTEBRATE]
    )
    before_fruit = slot.cached_food.get(FoodType.FRUIT, 0)
    queue_power_choice(game, "A", bird_name, {"reset_feeder": True, "food_type": "fruit"})
    res_reset = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_reset.executed
    assert slot.cached_food.get(FoodType.FRUIT, 0) == before_fruit + 1
    assert game.birdfeeder.count == 4

    # Case 2: reset requested when not all same -> no reset; with no eligible food, gains nothing.
    game.birdfeeder.set_dice([FoodType.RODENT, FoodType.FISH, FoodType.RODENT, FoodType.FISH, FoodType.RODENT])
    before_total = sum(slot.cached_food.values())
    queue_power_choice(game, "A", bird_name, {"reset_feeder": True, "food_type": "seed"})
    res_no = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res_no.executed
    assert sum(slot.cached_food.values()) == before_total


@pytest.mark.parametrize("bird_name", STEAL_CACHE_TARGET_GAIN_DIE_BIRDS)
def test_semantic_batch_steal_cache_then_target_gains_die(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    a, b = game.players
    habitat = _first_habitat(bird)
    slot = a.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "StealSpecificFoodCacheThenTargetGainDieFromFeeder"
    stolen_type = getattr(power, "food_type")

    b.food_supply.add(stolen_type, 1)
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE])
    cache_before = slot.cached_food.get(stolen_type, 0)
    target_total_before = b.food_supply.total()

    queue_power_choice(game, a.name, bird_name, {"target_player": b.name})
    queue_power_choice(game, b.name, bird_name, {"food_type": "fish"})
    res = power.execute(PowerContext(game_state=game, player=a, bird=bird, slot_index=0, habitat=habitat))

    assert res.executed
    assert slot.cached_food.get(stolen_type, 0) == cache_before + 1
    # Target loses one specific stolen food, then gains one die from feeder => total unchanged.
    assert b.food_supply.total() == target_total_before


@pytest.mark.parametrize("bird_name", CACHE_SUPPLY_BIRDS)
def test_semantic_batch_cache_food_from_supply(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "CacheFoodFromSupply"
    food_type = getattr(power, "food_type")
    count = int(getattr(power, "count", 1))
    before = slot.cached_food.get(food_type, 0)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(food_type, 0) - before == count


@pytest.mark.parametrize("bird_name", SPENDABLE_CACHE_SUPPLY_BIRDS)
def test_semantic_batch_spendable_cache_food_can_pay_cost(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    player.food_supply.spend(FoodType.SEED, player.food_supply.get(FoodType.SEED))
    player.food_supply.spend(FoodType.NECTAR, player.food_supply.get(FoodType.NECTAR))

    power = get_power(bird)
    assert type(power).__name__ == "CacheFoodFromSupply"
    assert bool(getattr(power, "spendable", False))
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.SEED, 0) >= 1
    assert slot.spendable_cached_food.get(FoodType.SEED, 0) >= 1

    can_pay, _ = can_pay_food_cost(player, FoodCost(items=(FoodType.SEED,), is_or=False, total=1))
    assert can_pay


def test_semantic_batch_non_spendable_cached_food_does_not_pay_cost(regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get("Carolina Chickadee")
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    player.food_supply.spend(FoodType.SEED, player.food_supply.get(FoodType.SEED))
    player.food_supply.spend(FoodType.NECTAR, player.food_supply.get(FoodType.NECTAR))

    power = get_power(bird)
    assert type(power).__name__ == "CacheFoodFromSupply"
    assert not bool(getattr(power, "spendable", False))
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.cached_food.get(FoodType.SEED, 0) >= 1
    assert slot.spendable_cached_food.get(FoodType.SEED, 0) == 0

    can_pay, _ = can_pay_food_cost(player, FoodCost(items=(FoodType.SEED,), is_or=False, total=1))
    assert not can_pay


@pytest.mark.parametrize("bird_name", MOVE_FISH_OPTIONAL_ROLL_CACHE_BIRDS)
def test_semantic_batch_move_fish_optional_roll_cache(bird_name: str, regs, monkeypatch):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    power = get_power(bird)
    assert type(power).__name__ == "GreatCormorantMoveFishThenOptionalRoll"

    # Case 1: skip move; roll and hit fish => cache +1 fish.
    slot.cache_food(FoodType.FISH, 1)
    fish_cache_before = slot.cached_food.get(FoodType.FISH, 0)
    fish_supply_before = player.food_supply.get(FoodType.FISH)
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE, FoodType.NECTAR])
    rolls = iter([FoodType.SEED, FoodType.FISH])
    monkeypatch.setattr("random.choice", lambda _seq: next(rolls))
    queue_power_choice(game, player.name, bird_name, {"move_fish_to_supply": False, "roll_dice": True})
    res_roll = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_roll.executed
    assert player.food_supply.get(FoodType.FISH) == fish_supply_before
    assert slot.cached_food.get(FoodType.FISH, 0) == fish_cache_before + 1

    # Case 2: move fish; skip roll => supply +1 fish, cache -1 fish.
    fish_cache_before_move = slot.cached_food.get(FoodType.FISH, 0)
    fish_supply_before_move = player.food_supply.get(FoodType.FISH)
    queue_power_choice(game, player.name, bird_name, {"move_fish_to_supply": True, "roll_dice": False})
    res_move = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res_move.executed
    assert player.food_supply.get(FoodType.FISH) == fish_supply_before_move + 1
    assert slot.cached_food.get(FoodType.FISH, 0) == fish_cache_before_move - 1


@pytest.mark.parametrize("bird_name", CHOOSE_BIRDS_CACHE_SUPPLY_BIRDS)
def test_semantic_batch_choose_birds_cache_food_each(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    habitat = _first_habitat(bird)
    row = player.board.get_row(habitat)
    row.slots[0].bird = bird
    row.slots[1].bird = _make_support_bird_multi(name="Cache 1", habitats=frozenset({habitat}))
    row.slots[2].bird = _make_support_bird_multi(name="Cache 2", habitats=frozenset({habitat}))

    power = get_power(bird)
    assert type(power).__name__ == "ChooseBirdsInHabitatCacheFoodEach"
    food_type = getattr(power, "food_type")
    queue_power_choice(game, "A", bird_name, {"target_slots": [0, 2]})
    before_0 = row.slots[0].cached_food.get(food_type, 0)
    before_1 = row.slots[1].cached_food.get(food_type, 0)
    before_2 = row.slots[2].cached_food.get(food_type, 0)

    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert res.food_cached.get(food_type, 0) == 2
    assert row.slots[0].cached_food.get(food_type, 0) == before_0 + 1
    assert row.slots[1].cached_food.get(food_type, 0) == before_1
    assert row.slots[2].cached_food.get(food_type, 0) == before_2 + 1


@pytest.mark.parametrize("bird_name", PINK_LAY_EGG_TRIGGER_BIRDS)
def test_semantic_batch_pink_lay_egg_on_opponent_lay_eggs_trigger(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    actor, owner = game.players[0], game.players[1]
    bird = birds.get(bird_name)
    habitat = _first_habitat(bird)
    owner.board.get_row(habitat).slots[0].bird = bird

    # Provide a broad set of potential targets for nest/wingspan variants.
    owner.board.forest.slots[1].bird = _make_support_bird_multi(name="Pink Bowl", nest_type=NestType.BOWL)
    owner.board.forest.slots[3].bird = _make_support_bird_multi(name="Pink Cavity", nest_type=NestType.CAVITY)
    owner.board.grassland.slots[1].bird = _make_support_bird_multi(name="Pink Ground", nest_type=NestType.GROUND)
    owner.board.wetland.slots[1].bird = _make_support_bird_multi(name="Pink Platform", nest_type=NestType.PLATFORM)
    owner.board.forest.slots[2].bird = _make_support_bird_multi(name="Pink Small", wingspan_cm=20)

    power = get_power(bird)
    assert type(power).__name__ == "PinkLayEggOnTrigger"
    before = owner.board.total_eggs()
    from backend.models.enums import ActionType
    res = power.execute(
        PowerContext(
            game_state=game,
            player=owner,
            bird=bird,
            slot_index=0,
            habitat=habitat,
            trigger_player=actor,
            trigger_action=ActionType.LAY_EGGS,
        )
    )
    assert res.executed
    assert owner.board.total_eggs() == before + 1


@pytest.mark.parametrize("bird_name", MOVE_BIRD_BIRDS)
def test_semantic_batch_move_bird_migrates_when_rightmost(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get(bird_name)
    habitat = _first_habitat(bird)
    src_row = player.board.get_row(habitat)
    src_row.slots[0].bird = bird  # rightmost in its row

    power = get_power(bird)
    assert type(power).__name__ == "MoveBird"
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert src_row.slots[0].bird is None
    assert any(slot.bird and slot.bird.name == bird_name for _, _, slot in player.board.all_slots())


@pytest.mark.parametrize("bird_name", DISCARD_FOOD_TUCK_BIRDS)
def test_semantic_batch_discard_food_to_tuck_from_deck(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get(bird_name)
    habitat = _first_habitat(bird)
    slot = player.board.get_row(habitat).slots[0]
    slot.bird = bird
    for ft in (FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT):
        player.food_supply.add(ft, 5)
    game._deck_cards = list(birds.all_birds[:40])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)

    power = get_power(bird)
    assert type(power).__name__ == "DiscardFoodToTuckFromDeck"
    before_tucked = slot.tucked_cards
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert res.executed
    assert slot.tucked_cards > before_tucked


@pytest.mark.parametrize("bird_name", NO_POWER_BIRDS)
def test_semantic_batch_no_power_birds_execute_as_noop(bird_name: str, regs):
    birds, _, _ = regs
    clear_cache()
    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    player = game.players[0]
    bird = birds.get(bird_name)
    habitat = _first_habitat(bird)
    player.board.get_row(habitat).slots[0].bird = bird

    power = get_power(bird)
    assert type(power).__name__ == "NoPower"
    res = power.execute(PowerContext(game_state=game, player=player, bird=bird, slot_index=0, habitat=habitat))
    assert not res.executed
    assert res.description == "No power"
