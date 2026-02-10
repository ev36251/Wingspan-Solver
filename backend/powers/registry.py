"""Power registry: maps each bird to its PowerEffect implementation.

Uses a regex-based parser to auto-classify ~80% of birds, with
manual overrides for edge cases. Unmatched birds get FallbackPower.
"""

import re
from backend.models.bird import Bird
from backend.models.enums import FoodType, Habitat, NestType, PowerColor
from backend.powers.base import PowerEffect, NoPower, FallbackPower
from backend.powers.templates.gain_food import (
    GainFoodFromSupply, GainFoodFromFeeder, GainFoodFromSupplyOrCache,
    ResetFeederGainFood,
)
from backend.powers.templates.lay_eggs import LayEggs, LayEggsEachBirdInRow
from backend.powers.templates.draw_cards import DrawCards, DrawFromTray, DrawBonusCards
from backend.powers.templates.tuck_cards import TuckFromHand, TuckFromDeck, DiscardToTuck
from backend.powers.templates.predator import PredatorDice, PredatorLookAt
from backend.powers.templates.cache_food import CacheFoodFromSupply, CacheFoodFromFeeder, TradeFood
from backend.powers.templates.play_bird import PlayAdditionalBird
from backend.powers.templates.special import (
    RepeatPower, CopyNeighborBrownPower, FlockingPower,
    EndOfRoundLayEggs, EndOfRoundCacheFood, DiscardEggForBenefit,
)
from backend.powers.templates.unique import (
    CountsDoubleForGoal, ActivateAllPredators, GainFoodMatchingPrevious,
    FewestBirdsGainFood, ScoreBonusCardNow, RetrieveDiscardedBonusCard,
    CopyNeighborBonusCard, TradeFoodForAny, RepeatPredatorPower,
    CopyNeighborWhitePower, ConditionalCacheFromNeighbor,
)
from backend.powers.templates.strict import (
    DrawThenDiscardFromHand,
    AllPlayersLayEggsSelfBonus,
    TuckThenCacheFromSupply,
    EndRoundMandarinDuck,
    SnowyOwlBonusThenChoice,
    CommonNightingaleChooseFoodAllPlayers,
    RedBelliedWoodpeckerSeedFromFeeder,
    WillieWagtailDrawTrayNest,
    WhiteStorkResetTrayThenDraw,
    PinkEaredDuckDrawKeepGive,
    CrestedLarkDiscardSeedLayEgg,
    BlackHeadedGullStealWild,
    BlackDrongoDiscardTrayLayEgg,
)


# Food type text -> FoodType mapping
FOOD_MAP = {
    "invertebrate": FoodType.INVERTEBRATE,
    "seed": FoodType.SEED,
    "fish": FoodType.FISH,
    "fruit": FoodType.FRUIT,
    "rodent": FoodType.RODENT,
    "nectar": FoodType.NECTAR,
    "wild": FoodType.WILD,
}

NEST_MAP = {
    "bowl": NestType.BOWL,
    "cavity": NestType.CAVITY,
    "ground": NestType.GROUND,
    "platform": NestType.PLATFORM,
    "star": NestType.WILD,
}

HABITAT_MAP = {
    "forest": Habitat.FOREST,
    "grassland": Habitat.GRASSLAND,
    "wetland": Habitat.WETLAND,
}


# Manual overrides for the 14 birds with unique powers
_MANUAL_OVERRIDES: dict[str, PowerEffect] = {
    # Teal: counts double toward end-of-round goal
    "Cetti's Warbler": CountsDoubleForGoal(),
    "Eurasian Green Woodpecker": CountsDoubleForGoal(),
    "Greylag Goose": CountsDoubleForGoal(),
    # Teal: activate all predator brown powers
    "Oriental Bay-Owl": ActivateAllPredators(),
    # Brown: gain 1 food of a type you already have
    "European Robin": GainFoodMatchingPrevious(),
    # Brown: fewest birds in forest gain food from feeder
    "Hermit Thrush": FewestBirdsGainFood(habitat=Habitat.FOREST),
    # White: score bonus card now by caching seeds
    "Great Indian Bustard": ScoreBonusCardNow(),
    # White: retrieve discarded bonus card
    "Wrybill": RetrieveDiscardedBonusCard(),
    # Yellow: copy neighbor's bonus card
    "Greater Adjutant": CopyNeighborBonusCard(direction="left"),
    "Indian Vulture": CopyNeighborBonusCard(direction="right"),
    # Brown: trade 1 food for any other type
    "Green Heron": TradeFoodForAny(),
    # Brown: repeat a predator power in this habitat
    "Hooded Merganser": RepeatPredatorPower(),
    # White: copy a white power from neighbor
    "Rose-Ringed Parakeet": CopyNeighborWhitePower(),
    # Brown: cache if neighbor has invertebrate
    "South Island Robin": ConditionalCacheFromNeighbor(
        direction="right", food_type=FoodType.INVERTEBRATE),
    # Brown: discard 1 card from hand to play another bird in forest
    "Goldcrest": PlayAdditionalBird(habitat_filter=Habitat.FOREST),
    # Brown: discard 1 food to play another bird in wetland
    "Common Moorhen": PlayAdditionalBird(habitat_filter=Habitat.WETLAND),
    # Brown: discard 1 egg to play another bird in forest
    "Short-Toed Treecreeper": PlayAdditionalBird(habitat_filter=Habitat.FOREST),
    # Yellow: play a bird at normal cost (triggers when-played powers)
    "Gould's Finch": PlayAdditionalBird(),
    # Yellow: play a bird, ignore 1 egg cost
    "Grey-Headed Mannikin": PlayAdditionalBird(egg_discount=1),
    # Yellow: discard 2 eggs to play 1 bird in grassland
    "Magpie-Lark": PlayAdditionalBird(habitat_filter=Habitat.GRASSLAND),
    # Teal: conditional — if used all 4 action types, play another bird
    "Moltoni's Warbler": PlayAdditionalBird(),
    "White Wagtail": PlayAdditionalBird(),
    "Yellowhammer": PlayAdditionalBird(),
    # Brown: copy a brown power from neighbor's habitat
    "Superb Lyrebird": CopyNeighborBrownPower(
        direction="right", target_habitat=Habitat.FOREST),
    "Tūī": CopyNeighborBrownPower(
        direction="left", target_habitat=Habitat.FOREST),
    "Common Myna": CopyNeighborBrownPower(
        direction="left", target_habitat=Habitat.GRASSLAND),
}


# Explicit per-card powers for high-impact competitive lines.
# These definitions are treated as strict source of truth for listed birds.
_STRICT_CARD_POWERS: dict[str, callable] = {
    "Forster's Tern": lambda: DrawThenDiscardFromHand(draw=1, discard=1),
    "Wood Duck": lambda: DrawThenDiscardFromHand(draw=2, discard=1),
    "Violet-Green Swallow": lambda: TuckFromHand(tuck_count=1, draw_count=1),
    "Golden Pheasant": lambda: AllPlayersLayEggsSelfBonus(all_players_count=2, self_bonus_count=2),
    "Twite": lambda: TuckFromDeck(count=2),
    "Lesser Whitethroat": lambda: EndOfRoundLayEggs(count_per_bird=1, habitat_filter=Habitat.GRASSLAND),
    "Mandarin Duck": lambda: EndRoundMandarinDuck(),
    "Great Hornbill": lambda: TuckThenCacheFromSupply(cache_food_type=FoodType.FRUIT),
    "Magpie-Lark": lambda: PlayAdditionalBird(habitat_filter=Habitat.GRASSLAND),
    "Roseate Spoonbill": lambda: DrawBonusCards(draw=2, keep=1),
    "Baltimore Oriole": lambda: GainFoodFromSupply(food_types=[FoodType.FRUIT], count=1, all_players=True),
    "Snowy Owl": lambda: SnowyOwlBonusThenChoice(),
    "Common Nightingale": lambda: CommonNightingaleChooseFoodAllPlayers(),
    "Red-Bellied Woodpecker": lambda: RedBelliedWoodpeckerSeedFromFeeder(),
    "Willie-Wagtail": lambda: WillieWagtailDrawTrayNest(),
    "White Stork": lambda: WhiteStorkResetTrayThenDraw(),
    "Pink-Eared Duck": lambda: PinkEaredDuckDrawKeepGive(),
    "Crested Lark": lambda: CrestedLarkDiscardSeedLayEgg(),
    "Black-Headed Gull": lambda: BlackHeadedGullStealWild(),
    "Black Drongo": lambda: BlackDrongoDiscardTrayLayEgg(),
}


def _extract_food_type(text: str) -> FoodType | None:
    """Extract first food type mentioned in text."""
    for name, ft in FOOD_MAP.items():
        if f"[{name}]" in text.lower():
            return ft
    return None


def _extract_all_food_types(text: str) -> list[FoodType]:
    """Extract all food types mentioned in text."""
    found = []
    for name, ft in FOOD_MAP.items():
        if f"[{name}]" in text.lower():
            found.append(ft)
    return found


def _extract_count(text: str, keyword: str = "Gain") -> int:
    """Extract a number near a keyword."""
    m = re.search(rf"{keyword}\s+(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else 1


def _extract_nest_type(text: str) -> str | None:
    """Extract nest type from text like '[ground]' or '[bowl]'."""
    for name in NEST_MAP:
        if f"[{name}]" in text.lower():
            return name
    return None


def _extract_habitat(text: str) -> Habitat | None:
    """Extract habitat from text like '[forest]'."""
    for name, hab in HABITAT_MAP.items():
        if f"[{name}]" in text.lower():
            return hab
    return None


def parse_power(bird: Bird) -> PowerEffect:
    """Parse a bird's power text into a PowerEffect.

    Checks manual overrides first, then tries regex patterns in priority
    order. Returns FallbackPower if no pattern matches.
    """
    # Strict per-card mapping for known high-impact birds.
    strict = _STRICT_CARD_POWERS.get(bird.name)
    if strict is not None:
        return strict()

    # Check manual overrides first (14 unique birds)
    if bird.name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[bird.name]

    text = bird.power_text
    if not text or text.strip() == "":
        return NoPower()

    color = bird.color.value if bird.color != PowerColor.NONE else "brown"
    t = text.lower()

    # --- Flocking (very common brown) ---
    if bird.is_flocking and "tuck" in t and "deck" in t:
        return FlockingPower(count=1)

    # --- "All players" variants ---
    if "all players" in t or "each player" in t:
        if "draw" in t and "card" in t:
            count = _extract_count(text, "draw")
            return DrawCards(draw=count, keep=count, all_players=True)
        if "gain" in t:
            foods = _extract_all_food_types(text)
            count = _extract_count(text, "gain")
            if foods:
                return GainFoodFromSupply(food_types=foods, count=count, all_players=True)
        if "lay" in t and "egg" in t:
            count = _extract_count(text, "lay")
            nest = _extract_nest_type(text)
            return LayEggs(count=count, all_players=True,
                           nest_filter=NEST_MAP.get(nest) if nest else None)

    # --- Predator powers ---
    if "roll all dice not in" in t:
        ft = _extract_food_type(text)
        if ft:
            return PredatorDice(target_food=ft)

    if "look at" in t and ("wingspan" in t or "less than" in t):
        m = re.search(r"less than (\d+)", text)
        threshold = int(m.group(1)) if m else 75
        return PredatorLookAt(wingspan_threshold=threshold)

    # --- Reset feeder ---
    if "reset the birdfeeder" in t:
        ft = _extract_food_type(text)
        cache = "cache" in t
        return ResetFeederGainFood(food_type=ft, cache=cache)

    # --- Draw bonus cards (white powers) ---
    if "bonus card" in t and "draw" in t:
        draw_count = _extract_count(text, "draw")
        m = re.search(r"discard (\d+)", text)
        discard = int(m.group(1)) if m else 0
        keep = draw_count - discard
        return DrawBonusCards(draw=draw_count, keep=max(1, keep))

    # --- Play additional bird (white powers) ---
    if "play" in t and ("second bird" in t or "additional bird" in t
                        or "another bird" in t or "a bird" in t):
        if bird.color == PowerColor.WHITE:
            hab = _extract_habitat(text)
            food_discount = 0
            egg_discount = 0
            # Oceania "1 [egg] discount" pattern — egg cost only
            if "egg" in t and "discount" in t:
                egg_discount = 1
            # Asia "ignore 1 [food] or 1 [egg]" pattern — both food and egg
            elif "ignore" in t and "egg" in t:
                egg_discount = 1
                for ft_name in ["invertebrate", "seed", "fruit", "fish", "rodent"]:
                    if ft_name in t:
                        food_discount = 1
                        break
            # Generic food discount ("1 less food" type patterns)
            elif "1 less" in t:
                food_discount = 1
            return PlayAdditionalBird(habitat_filter=hab, food_discount=food_discount,
                                     egg_discount=egg_discount)

    # --- Tuck from hand + draw ---
    if "tuck" in t and "from your hand" in t:
        tuck_count = _extract_count(text, "tuck")
        draw_count = 0
        lay_count = 0
        food_type = None
        food_count = 0

        if "draw" in t:
            draw_count = _extract_count(text, "draw")
        if "lay" in t:
            lay_count = _extract_count(text, "lay")
        if "gain" in t:
            food_type = _extract_food_type(text)
            food_count = _extract_count(text, "gain")

        return TuckFromHand(tuck_count=tuck_count, draw_count=draw_count,
                            lay_count=lay_count, food_type=food_type,
                            food_count=food_count)

    # --- Tuck from deck ---
    if "tuck" in t and "from the deck" in t and "discard" not in t:
        count = _extract_count(text, "tuck")
        return TuckFromDeck(count=count)

    # --- Draw and discard from deck (conditional tuck) ---
    if "draw and discard" in t and "from the deck" in t:
        draw_count = _extract_count(text, "draw")
        ft = _extract_food_type(text)
        return DiscardToTuck(draw_count=draw_count, food_type=ft,
                             food_on_success=1 if ft else 0)

    # --- Cache food from supply ---
    if "cache" in t and "from the supply" in t:
        ft = _extract_food_type(text)
        count = _extract_count(text, "cache")
        if ft:
            return CacheFoodFromSupply(food_type=ft, count=count)

    # --- Cache food from feeder ---
    if "cache" in t and "from the birdfeeder" in t:
        ft = _extract_food_type(text)
        if ft:
            return CacheFoodFromFeeder(food_type=ft)

    # --- Lay eggs (various patterns) ---
    # Use word boundary to avoid matching "lay" inside "play"
    if re.search(r'\blay\b', t) and "egg" in t:
        count = _extract_count(text, "lay")

        # "each bird in this bird's row"
        if "each bird" in t and ("row" in t or "column" in t):
            include_col = "column" in t
            return LayEggsEachBirdInRow(include_column=include_col)

        # "on this bird" or "on another bird"
        if "on this bird" in t:
            return LayEggs(count=count, target="self")

        # Nest filter
        nest = _extract_nest_type(text)
        if nest:
            return LayEggs(count=count, nest_filter=NEST_MAP[nest])

        # Habitat filter
        hab = _extract_habitat(text)
        if hab:
            return LayEggs(count=count, habitat_filter=hab)

        return LayEggs(count=count, target="any")

    # --- Gain food from feeder ---
    if "from the birdfeeder" in t and ("gain" in t or "take" in t):
        foods = _extract_all_food_types(text)
        count = _extract_count(text, "gain") if "gain" in t else _extract_count(text, "take")
        cache = "cache" in t
        return GainFoodFromFeeder(food_types=foods or None, count=count, may_cache=cache)

    # --- Gain food from supply ---
    if "from the supply" in t and "gain" in t:
        foods = _extract_all_food_types(text)
        count = _extract_count(text, "gain")
        if "cache" in t:
            ft = foods[0] if foods else FoodType.WILD
            return GainFoodFromSupplyOrCache(food_types=foods or [ft], count=count)
        if foods:
            return GainFoodFromSupply(food_types=foods, count=count)

    # --- Draw cards from deck ---
    if "draw" in t and "card" in t and ("deck" in t or "from the" not in t):
        draw_count = _extract_count(text, "draw")
        m = re.search(r"discard (\d+)", text)
        discard = int(m.group(1)) if m else 0
        keep = draw_count - discard
        return DrawCards(draw=draw_count, keep=max(1, keep))

    # --- Repeat power ---
    if "repeat" in t and "brown" in t:
        return RepeatPower()

    # --- End of round: lay eggs ---
    if "end of round" in t and "lay" in t:
        count = _extract_count(text, "lay")
        nest = _extract_nest_type(text)
        hab = _extract_habitat(text)
        return EndOfRoundLayEggs(count_per_bird=count, nest_filter=nest,
                                 habitat_filter=hab)

    # --- End of round: cache food ---
    if "end of round" in t and "cache" in t:
        ft = _extract_food_type(text)
        count = _extract_count(text, "cache")
        if ft:
            return EndOfRoundCacheFood(food_type=ft, count=count)

    # --- Discard egg for benefit ---
    if "discard" in t and "egg" in t and ("gain" in t or "draw" in t):
        ft = _extract_food_type(text)
        food_gain = _extract_count(text, "gain") if "gain" in t else 0
        card_gain = _extract_count(text, "draw") if "draw" in t else 0
        return DiscardEggForBenefit(egg_cost=1, food_gain=food_gain,
                                    food_type=ft, card_gain=card_gain)

    # --- "Gain all [food] from birdfeeder" ---
    if "gain all" in t and "birdfeeder" in t:
        ft = _extract_food_type(text)
        cache = "cache" in t
        return ResetFeederGainFood(food_type=ft, cache=cache)

    # --- "Move it to another habitat" (movement birds) ---
    if "move it to another habitat" in t or "move this bird" in t:
        from backend.powers.templates.special import MoveBird
        return MoveBird()

    # --- "Place this bird sideways" (white) ---
    if "sideways" in t:
        from backend.powers.templates.play_bird import PlaceBirdSideways
        return PlaceBirdSideways()

    # --- "Draw 1 face-up card from the tray with a [nest]" ---
    if "face-up" in t and "tray" in t and ("nest" in t or "star" in t):
        nest = _extract_nest_type(text)
        return DrawFromTray(count=1, food_filter=None)

    # --- "Gain 1 face-up card that can live in [habitat]" ---
    if "face-up" in t and "can live in" in t:
        return DrawFromTray(count=1, food_filter=None)

    # --- "Counts double toward end-of-round goal" (teal) ---
    if "counts double" in t and "goal" in t:
        return FallbackPower(color_hint="teal")  # Handled specially in goal scoring

    # --- "Roll any N dice" (alternate predator format) ---
    if "roll" in t and "die" in t and "cache" in t:
        ft = _extract_food_type(text)
        if ft:
            return PredatorDice(target_food=ft)

    # --- "Choose a food type. All players gain" ---
    if "choose a food type" in t and "all players" in t:
        return GainFoodFromSupply(food_types=[FoodType.WILD], count=1, all_players=True)

    # --- "Copy a brown power" / "Copy a white power" ---
    if "copy" in t and ("brown" in t or "when activated" in t):
        return RepeatPower()

    # --- "Tuck" from tray (specific condition) ---
    if "tuck" in t and "tray" in t:
        return TuckFromDeck(count=1)

    # --- "Cache up to N [food] from your supply" (yellow) ---
    if "cache up to" in t and "from your supply" in t:
        ft = _extract_food_type(text)
        m = re.search(r"cache up to (\d+)", t)
        count = int(m.group(1)) if m else 5
        if ft:
            return EndOfRoundCacheFood(food_type=ft, count=count)

    # --- "Look at a card from the deck. If it can live in" (habitat predator) ---
    if "look at" in t and "can live in" in t:
        return PredatorLookAt(wingspan_threshold=75)  # Use as probability proxy

    # --- "Look at a card from the deck. If its food cost includes" ---
    if "look at" in t and "food cost includes" in t:
        return PredatorLookAt(wingspan_threshold=75)

    # --- Neighbor supply powers ("if player to your left/right has") ---
    if ("player to your left" in t or "player to your right" in t) and "gain" in t:
        ft = _extract_food_type(text)
        if ft:
            return GainFoodFromSupply(food_types=[ft], count=1)

    # --- "Each player may roll any 1 die" ---
    if "each player" in t and "roll" in t and "die" in t:
        return GainFoodFromSupply(food_types=[FoodType.WILD], count=1, all_players=True)

    # --- "Copy one bonus card" (yellow) ---
    if "copy" in t and "bonus card" in t:
        return FallbackPower(color_hint="yellow")

    # --- "For each bird in your [habitat] with an egg" ---
    if "for each bird" in t and "egg" in t and "roll" in t:
        return GainFoodFromSupply(food_types=[FoodType.WILD], count=1)

    # --- "All players may cache" / "All players may tuck" ---
    if "all players may" in t:
        if "cache" in t:
            return GainFoodFromSupplyOrCache(food_types=[FoodType.WILD], count=1)
        if "tuck" in t:
            return TuckFromHand(tuck_count=1, draw_count=0)

    # --- "Discard 1 [food] to" (trade/conditional) ---
    if "discard" in t and ("to choose" in t or "to gain" in t or "to tuck" in t):
        ft = _extract_food_type(text)
        if "tuck" in t:
            return TuckFromDeck(count=1)
        return TradeFood(discard_type=ft, gain_type=FoodType.WILD, gain_count=1)

    # --- "Cache 1 [food] from your supply" / "cache from supply" (on other birds) ---
    if "cache" in t and ("from your supply" in t or "from their supply" in t):
        ft = _extract_food_type(text)
        if ft:
            return CacheFoodFromSupply(food_type=ft, count=1)

    # --- Fallback ---
    return FallbackPower(color_hint=color)


# --- Registry ---

_power_cache: dict[str, PowerEffect] = {}


def get_power(bird: Bird) -> PowerEffect:
    """Get the PowerEffect for a bird, using cache."""
    if bird.name not in _power_cache:
        _power_cache[bird.name] = parse_power(bird)
    return _power_cache[bird.name]


def get_power_source(bird: Bird) -> str:
    """Return mapping source for a bird power.

    One of: strict, manual, no_power, fallback, parsed
    """
    if bird.name in _STRICT_CARD_POWERS:
        return "strict"
    if bird.name in _MANUAL_OVERRIDES:
        return "manual"
    power = get_power(bird)
    if isinstance(power, NoPower):
        return "no_power"
    if isinstance(power, FallbackPower):
        return "fallback"
    return "parsed"


def is_strict_power_source_allowed(source: str) -> bool:
    return source in {"strict", "manual", "no_power"}


def assert_power_allowed_for_strict_mode(game_state, bird: Bird) -> None:
    """Raise if strict rules mode is enabled and this bird uses non-strict mapping."""
    if not getattr(game_state, "strict_rules_mode", False):
        return
    source = get_power_source(bird)
    if not is_strict_power_source_allowed(source):
        raise RuntimeError(
            f"Strict rules mode rejected non-strict power mapping for '{bird.name}' (source={source})"
        )


def clear_cache() -> None:
    """Clear the power cache (for testing)."""
    _power_cache.clear()


def get_registry_stats(birds: list[Bird]) -> dict[str, int]:
    """Analyze how many birds are mapped vs fallback.

    Returns counts by power class name.
    """
    stats: dict[str, int] = {}
    for bird in birds:
        power = get_power(bird)
        cls_name = type(power).__name__
        stats[cls_name] = stats.get(cls_name, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1]))
