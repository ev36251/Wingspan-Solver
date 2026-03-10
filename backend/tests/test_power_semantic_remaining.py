import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard
from backend.models.enums import ActionType, BeakDirection, BoardType, FoodType, GameSet, Habitat, NestType, PowerColor
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext
from backend.powers.registry import clear_cache, get_power


REMAINING_BIRDS = [
    "Australasian Pipit",
    "Australian Owlet-Nightjar",
    "Australian Raven",
    "Bearded Reedling",
    "Black Drongo",
    "Black Redstart",
    "Black Stork",
    "Black-Billed Magpie",
    "Black-Headed Gull",
    "Black-Naped Oriole",
    "Bluethroat",
    "Brambling",
    "Carrion Crow",
    "Common Blackbird",
    "Common Buzzard",
    "Common Goldeneye",
    "Common Myna",
    "Common Tailorbird",
    "Common Teal",
    "Crested Lark",
    "Crested Pigeon",
    "Dunnock",
    "Eastern Imperial Eagle",
    "Eastern Kingbird",
    "Eurasian Golden Oriole",
    "Eurasian Green Woodpecker",
    "Eurasian Hobby",
    "Eurasian Magpie",
    "Eurasian Tree Sparrow",
    "Eurasian Treecreeper",
    "European Roller",
    "Fire-Fronted Serin",
    "Gould's Finch",
    "Gray Catbird",
    "Gray Wagtail",
    "Grey Heron",
    "Grey-Headed Mannikin",
    "Greylag Goose",
    "Griffon Vulture",
    "Himalayan Monal",
    "Hooded Crow",
    "Horned Lark",
    "House Crow",
    "Ibisbill",
    "Indian Vulture",
    "Lazuli Bunting",
    "Lesser Frigatebird",
    "Lesser Whitethroat",
    "Little Pied Cormorant",
    "Loggerhead Shrike",
    "Long-Tailed Tit",
    "Malleefowl",
    "Many-Colored Fruit-Dove",
    "Moltoni's Warbler",
    "Montagu's Harrier",
    "Northern Mockingbird",
    "Orange-Footed Scrubfowl",
    "Oriental Magpie-Robin",
    "Philippine Eagle",
    "Pileated Woodpecker",
    "Red Kite",
    "Red-Backed Fairywren",
    "Red-Bellied Woodpecker",
    "Red-Legged Partridge",
    "Rhinoceros Auklet",
    "Rock Pigeon",
    "Rose-Ringed Parakeet",
    "Ruddy Shelduck",
    "Ruff",
    "Sacred Kingfisher",
    "Snowy Owl",
    "Southern Cassowary",
    "Splendid Fairywren",
    "Spotless Crake",
    "Sri Lanka Blue-Magpie",
    "Sri Lanka Frogmouth",
    "Sulphur-Crested Cockatoo",
    "Superb Lyrebird",
    "T\u016b\u012b",
    "Turkey Vulture",
    "Western Meadowlark",
    "White Stork",
    "White Wagtail",
    "Willie-Wagtail",
    "Wrybill",
]


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


def _first_habitat(bird) -> Habitat:
    for h in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
        if bird.can_live_in(h):
            return h
    return Habitat.FOREST


def _make_custom_bird(
    *,
    name: str,
    color: PowerColor = PowerColor.BROWN,
    power_text: str = "Gain 1 [seed] from the supply.",
    habitats: frozenset[Habitat] | None = None,
    nest_type: NestType = NestType.BOWL,
    food_items: tuple[FoodType, ...] = (FoodType.SEED,),
    is_predator: bool = False,
    wingspan_cm: int = 40,
) -> Bird:
    return Bird(
        name=name,
        scientific_name=name,
        game_set=GameSet.CORE,
        color=color,
        power_text=power_text,
        victory_points=2,
        nest_type=nest_type,
        egg_limit=6,
        wingspan_cm=wingspan_cm,
        habitats=habitats or frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
        food_cost=FoodCost(items=food_items, is_or=False, total=len(food_items)),
        beak_direction=BeakDirection.NONE,
        is_predator=is_predator,
        is_flocking=False,
        is_bonus_card_bird=False,
        bonus_eligibility=frozenset(),
    )


def _seed_players(actor, opp) -> None:
    for p in (actor, opp):
        for ft in (
            FoodType.INVERTEBRATE,
            FoodType.SEED,
            FoodType.FISH,
            FoodType.FRUIT,
            FoodType.RODENT,
            FoodType.NECTAR,
        ):
            p.food_supply.add(ft, 4)


@pytest.mark.parametrize("bird_name", REMAINING_BIRDS)
def test_semantic_remaining_bird_powers_execute_with_class_aware_setup(bird_name: str, regs):
    birds, bonus_reg, _ = regs
    clear_cache()
    bird = birds.get(bird_name)
    assert bird is not None

    game = create_new_game(["A", "B"], board_type=BoardType.OCEANIA)
    actor, opp = game.players
    _seed_players(actor, opp)

    actor.action_types_used_this_round = {
        ActionType.PLAY_BIRD,
        ActionType.GAIN_FOOD,
        ActionType.LAY_EGGS,
        ActionType.DRAW_CARDS,
    }
    opp.lay_eggs_actions_this_round = 2

    actor.hand = list(birds.all_birds[:12])
    opp.hand = list(birds.all_birds[12:24])

    game._deck_cards = list(birds.all_birds[24:220])  # type: ignore[attr-defined]
    game.deck_remaining = len(game._deck_cards)
    game._bonus_cards = list(bonus_reg.all_cards[:25])  # type: ignore[attr-defined]
    game._bonus_discard_cards = list(bonus_reg.all_cards[25:30])  # type: ignore[attr-defined]

    habitat = _first_habitat(bird)
    slot_index = 1
    row = actor.board.get_row(habitat)
    row.slots[slot_index].bird = bird

    # Common board scaffolding for adjacency/column/nest/wingspan tests.
    if row.slots[0].bird is None:
        row.slots[0].bird = _make_custom_bird(name="Adj Left", nest_type=NestType.CAVITY)
    if row.slots[2].bird is None:
        row.slots[2].bird = _make_custom_bird(name="Adj Right", nest_type=NestType.WILD)

    for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
        r = actor.board.get_row(hab)
        if r.slots[slot_index].bird is None:
            r.slots[slot_index].bird = _make_custom_bird(name=f"Col-{hab.value}", nest_type=NestType.BOWL)
        if hab != habitat:
            r.slots[slot_index].eggs = 1

    # Neighbor setup for copy powers.
    opp.board.forest.slots[0].bird = _make_custom_bird(
        name="Neighbor Brown Forest",
        color=PowerColor.BROWN,
        power_text="Gain 1 [seed] from the supply.",
    )
    opp.board.grassland.slots[0].bird = _make_custom_bird(
        name="Neighbor Brown Grass",
        color=PowerColor.BROWN,
        power_text="Gain 1 [fish] from the supply.",
    )
    opp.board.forest.slots[1].bird = _make_custom_bird(
        name="Neighbor White",
        color=PowerColor.WHITE,
        power_text="Draw 1 card.",
    )

    # Tray + feeder defaults.
    game.card_tray.clear()
    game.card_tray.add_card(_make_custom_bird(name="Tray Grass", habitats=frozenset({Habitat.GRASSLAND}), nest_type=NestType.PLATFORM, wingspan_cm=22))
    game.card_tray.add_card(_make_custom_bird(name="Tray Bowl", habitats=frozenset({Habitat.FOREST}), nest_type=NestType.BOWL, wingspan_cm=65))
    game.card_tray.add_card(_make_custom_bird(name="Tray Star", habitats=frozenset({Habitat.WETLAND}), nest_type=NestType.WILD, wingspan_cm=95))
    game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.RODENT, FoodType.FRUIT, FoodType.INVERTEBRATE])

    # Extra setup used by several strict classes.
    for p in (actor, opp):
        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            rs = p.board.get_row(hab).slots[0]
            if rs.bird is None:
                rs.bird = _make_custom_bird(name=f"{p.name}-{hab.value}", habitats=frozenset({hab}))
            rs.eggs = max(rs.eggs, 1)

    opp.bonus_cards = [bonus_reg.all_cards[0]]

    power = get_power(bird)
    cls = type(power).__name__

    trigger_player = None
    trigger_action = None
    trigger_meta = None

    if cls == "ChooseEgglessHabitatLayEggEachBird":
        # Ensure one fully eggless habitat with multiple birds.
        eggless = actor.board.wetland
        eggless.slots[0].bird = _make_custom_bird(name="Eggless A", habitats=frozenset({Habitat.WETLAND}))
        eggless.slots[1].bird = _make_custom_bird(name="Eggless B", habitats=frozenset({Habitat.WETLAND}))
        eggless.slots[0].eggs = 0
        eggless.slots[1].eggs = 0

    if cls in {"PerThreeEggsInHabitatDrawThenTuck", "PerThreeEggsInHabitatGainSeedOrInvertThenCache", "EndGameTuckPerBirdInHabitat", "EndGameLayEggOnEachBirdInHabitat"}:
        target_hab = getattr(power, "habitat", habitat)
        target_row = actor.board.get_row(target_hab)
        target_row.slots[0].bird = target_row.slots[0].bird or _make_custom_bird(name=f"Target-{target_hab.value}", habitats=frozenset({target_hab}))
        target_row.slots[1].bird = target_row.slots[1].bird or _make_custom_bird(name=f"Target2-{target_hab.value}", habitats=frozenset({target_hab}))
        target_row.slots[0].eggs = 3
        target_row.slots[1].eggs = 3

    if cls == "EndGameLayContiguousSameNest":
        actor.board.forest.slots[0].bird = _make_custom_bird(name="NestRun1", nest_type=NestType.BOWL)
        actor.board.forest.slots[1].bird = _make_custom_bird(name="NestRun2", nest_type=NestType.BOWL)
        actor.board.forest.slots[2].bird = _make_custom_bird(name="NestRun3", nest_type=NestType.BOWL)
        actor.board.forest.slots[0].eggs = 0
        actor.board.forest.slots[1].eggs = 0
        actor.board.forest.slots[2].eggs = 0

    if cls == "EndGameLayEggOnMatchingWingspan":
        actor.board.grassland.slots[3].bird = _make_custom_bird(name="SmallWing", wingspan_cm=20)
        actor.board.grassland.slots[4].bird = _make_custom_bird(name="BigWing", wingspan_cm=120)

    if cls in {"ConditionalAllActionsPlayBird", "EndGamePlayBird"}:
        actor.hand.append(
            _make_custom_bird(
                name="Playable Bonus Bird",
                color=PowerColor.NONE,
                habitats=frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
                food_items=(FoodType.SEED,),
            )
        )
        actor.food_supply.add(FoodType.SEED, 3)

    if cls == "RepeatPower":
        actor.board.get_row(habitat).slots[3].bird = _make_custom_bird(
            name="Repeatable Brown",
            color=PowerColor.BROWN,
            power_text="Gain 1 [seed] from the supply.",
        )

    if cls == "CopyNeighborBrownPower":
        target_hab = getattr(power, "target_habitat", Habitat.FOREST)
        opp.board.get_row(target_hab).slots[0].bird = _make_custom_bird(
            name="Copy Source Brown",
            color=PowerColor.BROWN,
            power_text="Gain 1 [seed] from the supply.",
            habitats=frozenset({target_hab}),
        )

    if cls == "AllPlayersMayDiscardEggFromHabitatForWild":
        target_hab = getattr(power, "habitat", Habitat.FOREST)
        for p in (actor, opp):
            s = p.board.get_row(target_hab).slots[0]
            s.bird = s.bird or _make_custom_bird(name=f"EggSource-{p.name}", habitats=frozenset({target_hab}))
            s.eggs = max(s.eggs, 1)

    if cls in {"PinkGainFoodFromFeederOnGainFood", "PinkGainFoodFromSupplyOnPlayInHabitat", "PinkPredatorSuccessGainDie", "PinkCacheRodentOnOpponentRodentGain", "PinkTuckFromHandOnPlayInHabitat"}:
        trigger_player = opp

    if cls == "PinkGainFoodFromFeederOnGainFood":
        trigger_action = ActionType.GAIN_FOOD
        wanted = list(getattr(power, "food_types", []) or [FoodType.SEED])[0]
        game.birdfeeder.set_dice([wanted, FoodType.FISH, FoodType.FRUIT, FoodType.RODENT, FoodType.INVERTEBRATE])

    if cls == "PinkGainFoodFromSupplyOnPlayInHabitat":
        trigger_action = ActionType.PLAY_BIRD
        trigger_meta = {"played_habitat": getattr(power, "trigger_habitat", Habitat.FOREST)}

    if cls == "PinkPredatorSuccessGainDie":
        trigger_action = ActionType.GAIN_FOOD
        trigger_meta = {"predator_successes": 1}

    if cls == "PinkCacheRodentOnOpponentRodentGain":
        trigger_action = ActionType.GAIN_FOOD
        trigger_meta = {"food_gained": {FoodType.RODENT: 1}}

    if cls == "PinkTuckFromHandOnPlayInHabitat":
        trigger_action = ActionType.PLAY_BIRD
        trigger_meta = {"played_habitat": getattr(power, "trigger_habitat", Habitat.FOREST)}
        if not actor.hand:
            actor.hand.append(_make_custom_bird(name="Pink Tuck Hand"))

    if cls == "BlackDrongoDiscardTrayLayEgg":
        game.card_tray.clear()
        game.card_tray.add_card(_make_custom_bird(name="Discard Grass", habitats=frozenset({Habitat.GRASSLAND}), nest_type=NestType.BOWL, wingspan_cm=15))
        game.card_tray.add_card(_make_custom_bird(name="Discard Forest", habitats=frozenset({Habitat.FOREST}), nest_type=NestType.CAVITY, wingspan_cm=45))
        game.card_tray.add_card(_make_custom_bird(name="Discard Wet", habitats=frozenset({Habitat.WETLAND}), nest_type=NestType.PLATFORM, wingspan_cm=70))

    if cls == "RedBelliedWoodpeckerSeedFromFeeder":
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.RODENT, FoodType.FRUIT, FoodType.INVERTEBRATE])

    if cls == "SnowyOwlBonusThenChoice":
        assert isinstance(game._bonus_cards, list) and len(game._bonus_cards) > 0  # type: ignore[attr-defined]

    if cls == "RetrieveDiscardedBonusCard":
        game._bonus_discard_cards = [bonus_reg.all_cards[0]]  # type: ignore[attr-defined]

    if cls == "LayEggOnSelfPerOpponentGrasslandCubes":
        opp.lay_eggs_actions_this_round = 3

    if cls == "TuckHandThenDrawEqualPerOpponentGrasslandCubes":
        opp.lay_eggs_actions_this_round = 3
        if not actor.hand:
            actor.hand.append(_make_custom_bird(name="Tuck Source"))

    if cls == "CacheWildOnAnyBirdPerOpponentGrasslandCubes":
        opp.lay_eggs_actions_this_round = 2

    if cls == "CacheRodentPerPredator":
        opp.board.forest.slots[2].bird = _make_custom_bird(
            name="Pred One",
            is_predator=True,
            habitats=frozenset({Habitat.FOREST}),
        )
        opp.board.forest.slots[3].bird = _make_custom_bird(
            name="Pred Two",
            is_predator=True,
            habitats=frozenset({Habitat.FOREST}),
        )

    ctx = PowerContext(
        game_state=game,
        player=actor,
        bird=bird,
        slot_index=slot_index,
        habitat=habitat,
        trigger_player=trigger_player,
        trigger_action=trigger_action,
        trigger_meta=trigger_meta,
    )

    if cls.startswith("Pink"):
        assert power.can_execute(ctx)

    before_counts_double = row.slots[slot_index].counts_double
    before_sideways = row.slots[slot_index].is_sideways

    res = power.execute(ctx)

    assert isinstance(res.description, str)
    assert res is not None

    if cls == "PlaceBirdSideways":
        assert row.slots[slot_index].is_sideways and not before_sideways

    if cls == "CountsDoubleForGoal":
        assert row.slots[slot_index].counts_double and not before_counts_double

    if cls == "PlayOnTopPower":
        assert "Played on top" in res.description

    if cls == "TuckToPayCost":
        assert "Tuck-to-pay" in res.description

    if cls == "RedBelliedWoodpeckerSeedFromFeeder":
        assert res.executed

    if cls == "BlackHeadedGullStealWild":
        assert res.executed

    if cls == "AllPlayersLayEggsSelfBonus":
        assert actor.board.total_eggs() >= opp.board.total_eggs()

    if cls == "EndGameLayEggOnMatchingNest":
        assert res.eggs_laid >= 1

    if cls == "EndGameLayEggOnEachBirdInHabitat":
        assert res.eggs_laid >= 1

    if cls == "PinkCacheRodentOnOpponentRodentGain":
        assert row.slots[slot_index].cached_food.get(FoodType.RODENT, 0) >= 1

    if cls == "PinkTuckFromHandOnPlayInHabitat":
        assert row.slots[slot_index].tucked_cards >= 1

    if cls == "RetrieveDiscardedBonusCard":
        assert isinstance(actor.bonus_cards[-1], BonusCard)
