"""Tests for the 14 unique bird powers implemented in Phase 9."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat, PowerColor
from backend.models.game_state import create_new_game
from backend.models.bird import Bird, FoodCost
from backend.powers.base import PowerContext, FallbackPower
from backend.powers.registry import get_power, clear_cache


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture(scope="module")
def bonus_reg(regs):
    return regs[1]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


def _make_ctx(game, bird, habitat=Habitat.FOREST, slot_index=0):
    """Helper to create a PowerContext for testing."""
    return PowerContext(
        game_state=game,
        player=game.current_player,
        bird=bird,
        slot_index=slot_index,
        habitat=habitat,
    )


# --- Zero fallback birds ---

class TestZeroFallback:
    def test_no_fallback_birds(self, bird_reg):
        """All 446 birds should have dedicated power implementations."""
        clear_cache()
        fallbacks = []
        for bird in bird_reg.all_birds:
            power = get_power(bird)
            if isinstance(power, FallbackPower):
                fallbacks.append(bird.name)
        assert fallbacks == [], f"Unexpected fallback birds: {fallbacks}"

    def test_coverage_100_percent(self, bird_reg):
        """Coverage should be 100% (446/446)."""
        clear_cache()
        from backend.powers.registry import get_registry_stats
        stats = get_registry_stats(bird_reg.all_birds)
        assert "FallbackPower" not in stats


# --- CountsDoubleForGoal ---

class TestCountsDoubleForGoal:
    def test_cettis_warbler_mapped(self, bird_reg):
        bird = bird_reg.get("Cetti's Warbler")
        assert bird is not None
        from backend.powers.templates.unique import CountsDoubleForGoal
        power = get_power(bird)
        assert isinstance(power, CountsDoubleForGoal)

    def test_sets_flag_on_slot(self, game, bird_reg):
        bird = bird_reg.get("Cetti's Warbler")
        player = game.current_player
        player.board.forest.slots[0].bird = bird

        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)

        assert result.executed
        assert player.board.forest.slots[0].counts_double is True

    def test_all_three_birds(self, bird_reg):
        for name in ["Cetti's Warbler", "Eurasian Green Woodpecker", "Greylag Goose"]:
            bird = bird_reg.get(name)
            assert bird is not None, f"{name} not found"
            from backend.powers.templates.unique import CountsDoubleForGoal
            assert isinstance(get_power(bird), CountsDoubleForGoal)


# --- ActivateAllPredators ---

class TestActivateAllPredators:
    def test_oriental_bay_owl_mapped(self, bird_reg):
        bird = bird_reg.get("Oriental Bay-Owl")
        assert bird is not None
        from backend.powers.templates.unique import ActivateAllPredators
        assert isinstance(get_power(bird), ActivateAllPredators)

    def test_activates_predator_powers(self, game, bird_reg):
        """Should activate predator brown powers on other birds."""
        player = game.current_player
        owl = bird_reg.get("Oriental Bay-Owl")
        player.board.forest.slots[0].bird = owl

        # Add a predator bird to the board
        # Find a known predator bird (PredatorDice)
        kestrel = bird_reg.get("American Kestrel")
        if kestrel:
            player.board.forest.slots[1].bird = kestrel
            # Set up feeder so predator can work (take dice out)
            game.birdfeeder.set_dice([FoodType.SEED])  # 1 die in, 4 out

            ctx = _make_ctx(game, owl)
            power = get_power(owl)
            result = power.execute(ctx)
            assert result.executed
            assert "predator" in result.description.lower()


# --- GainFoodMatchingPrevious ---

class TestGainFoodMatchingPrevious:
    def test_european_robin_mapped(self, bird_reg):
        bird = bird_reg.get("European Robin")
        assert bird is not None
        from backend.powers.templates.unique import GainFoodMatchingPrevious
        assert isinstance(get_power(bird), GainFoodMatchingPrevious)

    def test_gains_food_from_supply(self, game, bird_reg):
        player = game.current_player
        bird = bird_reg.get("European Robin")
        player.board.forest.slots[0].bird = bird
        player.food_supply.add(FoodType.SEED, 3)

        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)

        assert result.executed
        assert FoodType.SEED in result.food_gained
        assert player.food_supply.get(FoodType.SEED) == 4

    def test_no_food_in_supply(self, game, bird_reg):
        bird = bird_reg.get("European Robin")
        player = game.current_player
        player.board.forest.slots[0].bird = bird

        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)
        assert not result.executed


# --- FewestBirdsGainFood ---

class TestFewestBirdsGainFood:
    def test_hermit_thrush_mapped(self, bird_reg):
        bird = bird_reg.get("Hermit Thrush")
        assert bird is not None
        from backend.powers.templates.unique import FewestBirdsGainFood
        assert isinstance(get_power(bird), FewestBirdsGainFood)

    def test_gains_food_when_fewest(self, game, bird_reg):
        bird = bird_reg.get("Hermit Thrush")
        player = game.current_player
        player.board.forest.slots[0].bird = bird
        game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])

        # Alice has 1 bird (Hermit Thrush), Bob has 0 — but we check forest
        # Alice has 1 in forest, Bob has 0 — Alice NOT fewest, Bob is
        # Actually both could qualify... let me think
        # Alice: 1 bird in forest. Bob: 0 birds in forest.
        # min_count = 0 (Bob). Alice count = 1 > 0, so she doesn't qualify.
        # Give Bob a bird in forest so they tie at 1 each
        other = game.players[1]
        other.board.forest.slots[0].bird = bird_reg.get("Acorn Woodpecker")

        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)
        # Both have 1 bird in forest = tied for fewest, Alice should qualify
        assert result.executed


# --- ScoreBonusCardNow ---

class TestScoreBonusCardNow:
    def test_great_indian_bustard_mapped(self, bird_reg):
        bird = bird_reg.get("Great Indian Bustard")
        assert bird is not None
        from backend.powers.templates.unique import ScoreBonusCardNow
        assert isinstance(get_power(bird), ScoreBonusCardNow)

    def test_no_bonus_cards(self, game, bird_reg):
        bird = bird_reg.get("Great Indian Bustard")
        player = game.current_player
        player.board.grassland.slots[0].bird = bird

        ctx = _make_ctx(game, bird, habitat=Habitat.GRASSLAND)
        power = get_power(bird)
        result = power.execute(ctx)
        assert not result.executed


# --- TradeFoodForAny ---

class TestTradeFoodForAny:
    def test_green_heron_mapped(self, bird_reg):
        bird = bird_reg.get("Green Heron")
        assert bird is not None
        from backend.powers.templates.unique import TradeFoodForAny
        assert isinstance(get_power(bird), TradeFoodForAny)

    def test_trades_food(self, game, bird_reg):
        bird = bird_reg.get("Green Heron")
        player = game.current_player
        player.board.wetland.slots[0].bird = bird
        player.food_supply.add(FoodType.SEED, 5)

        ctx = _make_ctx(game, bird, habitat=Habitat.WETLAND)
        power = get_power(bird)
        result = power.execute(ctx)

        assert result.executed
        # Should have traded 1 seed for something else
        assert player.food_supply.get(FoodType.SEED) == 4  # Lost 1
        total_food = sum(player.food_supply.get(ft) for ft in
                        [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                         FoodType.FRUIT, FoodType.RODENT])
        assert total_food == 5  # Net zero: lost 1, gained 1

    def test_no_food_to_trade(self, game, bird_reg):
        bird = bird_reg.get("Green Heron")
        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)
        assert not result.executed


# --- RepeatPredatorPower ---

class TestRepeatPredatorPower:
    def test_hooded_merganser_mapped(self, bird_reg):
        bird = bird_reg.get("Hooded Merganser")
        assert bird is not None
        from backend.powers.templates.unique import RepeatPredatorPower
        assert isinstance(get_power(bird), RepeatPredatorPower)

    def test_no_predator_in_habitat(self, game, bird_reg):
        bird = bird_reg.get("Hooded Merganser")
        player = game.current_player
        player.board.wetland.slots[0].bird = bird

        ctx = _make_ctx(game, bird, habitat=Habitat.WETLAND)
        power = get_power(bird)
        result = power.execute(ctx)
        assert not result.executed


# --- CopyNeighborBonusCard ---

class TestCopyNeighborBonusCard:
    def test_greater_adjutant_mapped(self, bird_reg):
        bird = bird_reg.get("Greater Adjutant")
        assert bird is not None
        from backend.powers.templates.unique import CopyNeighborBonusCard
        power = get_power(bird)
        assert isinstance(power, CopyNeighborBonusCard)
        assert power.direction == "left"

    def test_indian_vulture_mapped(self, bird_reg):
        bird = bird_reg.get("Indian Vulture")
        assert bird is not None
        from backend.powers.templates.unique import CopyNeighborBonusCard
        power = get_power(bird)
        assert isinstance(power, CopyNeighborBonusCard)
        assert power.direction == "right"

    def test_no_neighbor_bonus_cards(self, game, bird_reg):
        bird = bird_reg.get("Greater Adjutant")
        player = game.current_player
        player.board.grassland.slots[0].bird = bird

        ctx = _make_ctx(game, bird, habitat=Habitat.GRASSLAND)
        power = get_power(bird)
        result = power.execute(ctx)
        assert not result.executed


# --- Remaining birds mapped correctly ---

class TestRemainingMappings:
    def test_wrybill(self, bird_reg):
        bird = bird_reg.get("Wrybill")
        assert bird is not None
        from backend.powers.templates.unique import RetrieveDiscardedBonusCard
        assert isinstance(get_power(bird), RetrieveDiscardedBonusCard)

    def test_rose_ringed_parakeet(self, bird_reg):
        bird = bird_reg.get("Rose-Ringed Parakeet")
        assert bird is not None
        from backend.powers.templates.unique import CopyNeighborWhitePower
        assert isinstance(get_power(bird), CopyNeighborWhitePower)

    def test_south_island_robin(self, bird_reg):
        bird = bird_reg.get("South Island Robin")
        assert bird is not None
        from backend.powers.templates.unique import ConditionalCacheFromNeighbor
        power = get_power(bird)
        assert isinstance(power, ConditionalCacheFromNeighbor)
        assert power.food_type == FoodType.INVERTEBRATE

    def test_conditional_cache_works(self, game, bird_reg):
        """South Island Robin should cache if right neighbor has invertebrate."""
        bird = bird_reg.get("South Island Robin")
        player = game.current_player
        player.board.forest.slots[0].bird = bird

        # Give Bob (right neighbor) some invertebrate
        bob = game.players[1]
        bob.food_supply.add(FoodType.INVERTEBRATE, 2)

        ctx = _make_ctx(game, bird)
        power = get_power(bird)
        result = power.execute(ctx)

        assert result.executed
        assert result.food_cached.get(FoodType.INVERTEBRATE) == 1


# --- Neighbor helpers ---

class TestNeighborHelpers:
    def test_left_neighbor(self, game):
        alice = game.players[0]
        bob = game.players[1]
        assert game.player_to_left(alice).name == "Bob"
        assert game.player_to_left(bob).name == "Alice"

    def test_right_neighbor(self, game):
        alice = game.players[0]
        bob = game.players[1]
        assert game.player_to_right(alice).name == "Bob"
        assert game.player_to_right(bob).name == "Alice"

    def test_single_player_no_neighbor(self):
        solo_game = create_new_game(["Solo"])
        player = solo_game.players[0]
        assert solo_game.player_to_left(player) is None
        assert solo_game.player_to_right(player) is None

    def test_three_players(self):
        game3 = create_new_game(["A", "B", "C"])
        a, b, c = game3.players
        assert game3.player_to_left(a).name == "B"
        assert game3.player_to_right(a).name == "C"
        assert game3.player_to_left(b).name == "C"
        assert game3.player_to_right(b).name == "A"


# --- Estimate values ---

class TestEstimateValues:
    def test_all_unique_powers_have_positive_estimates(self, game, bird_reg):
        """All unique power birds should have non-negative estimate values."""
        unique_names = [
            "Cetti's Warbler", "Eurasian Green Woodpecker", "Greylag Goose",
            "Oriental Bay-Owl", "European Robin", "Hermit Thrush",
            "Great Indian Bustard", "Wrybill", "Greater Adjutant",
            "Indian Vulture", "Green Heron", "Hooded Merganser",
            "Rose-Ringed Parakeet", "South Island Robin",
        ]
        player = game.current_player
        for name in unique_names:
            bird = bird_reg.get(name)
            assert bird is not None, f"{name} not found"
            power = get_power(bird)
            ctx = _make_ctx(game, bird)
            value = power.estimate_value(ctx)
            assert value >= 0.0, f"{name} has negative estimate: {value}"
