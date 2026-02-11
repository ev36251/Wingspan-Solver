"""Tests for the bird power system: templates, parsing, and registry."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat, PowerColor
from backend.models.board import BirdSlot, PlayerBoard
from backend.models.birdfeeder import Birdfeeder
from backend.models.player import Player
from backend.models.game_state import create_new_game
from backend.powers.base import PowerContext, NoPower, FallbackPower
from backend.powers.registry import get_power, get_registry_stats, clear_cache
from backend.powers.templates.gain_food import GainFoodFromSupply, GainFoodFromFeeder
from backend.powers.templates.lay_eggs import LayEggs
from backend.powers.templates.draw_cards import DrawCards
from backend.powers.templates.tuck_cards import TuckFromHand
from backend.powers.templates.special import FlockingPower
from backend.powers.templates.predator import PredatorDice, PredatorLookAt
from backend.powers.templates.cache_food import CacheFoodFromSupply


@pytest.fixture(scope="module")
def regs():
    clear_cache()
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


@pytest.fixture
def ctx(game, bird_reg):
    """Create a PowerContext with a bird on the board."""
    player = game.players[0]
    bird = bird_reg.get("Acorn Woodpecker")
    player.board.forest.slots[0].bird = bird
    return PowerContext(
        game_state=game,
        player=player,
        bird=bird,
        slot_index=0,
        habitat=Habitat.FOREST,
    )


# --- Registry coverage ---

class TestRegistryCoverage:
    def test_total_coverage(self, bird_reg):
        stats = get_registry_stats(bird_reg.all_birds)
        total = sum(stats.values())
        fallback = stats.get("FallbackPower", 0)
        coverage = (total - fallback) / total
        assert coverage >= 0.95, f"Coverage {coverage:.1%} is below 95%"

    def test_no_power_birds(self, bird_reg):
        """6 birds with no power should get NoPower."""
        no_power_count = 0
        for bird in bird_reg.all_birds:
            if bird.color == PowerColor.NONE:
                power = get_power(bird)
                assert isinstance(power, NoPower), f"{bird.name} should be NoPower"
                no_power_count += 1
        assert no_power_count == 6

    def test_all_birds_get_a_power(self, bird_reg):
        """Every bird should get some power (even if FallbackPower)."""
        for bird in bird_reg.all_birds:
            power = get_power(bird)
            assert power is not None, f"{bird.name} has no power assigned"

    def test_brown_birds_mapped(self, bird_reg):
        """Brown birds should have high coverage (most common)."""
        brown_birds = bird_reg.by_color(PowerColor.BROWN)
        fallback_count = sum(
            1 for b in brown_birds
            if isinstance(get_power(b), FallbackPower)
        )
        coverage = (len(brown_birds) - fallback_count) / len(brown_birds)
        assert coverage >= 0.90, f"Brown coverage {coverage:.1%}"


# --- Template execution ---

class TestGainFoodFromSupply:
    def test_gain_seed(self, game):
        player = game.players[0]
        bird = player.board.forest  # doesn't matter for this test
        power = GainFoodFromSupply(food_types=[FoodType.SEED], count=2)
        ctx = PowerContext(game, player, bird=None, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        assert result.food_gained[FoodType.SEED] == 2
        assert player.food_supply.get(FoodType.SEED) == 2

    def test_gain_all_players(self, game):
        power = GainFoodFromSupply(
            food_types=[FoodType.FISH], count=1, all_players=True
        )
        ctx = PowerContext(game, game.players[0], bird=None,
                           slot_index=0, habitat=Habitat.FOREST)
        power.execute(ctx)
        for p in game.players:
            assert p.food_supply.get(FoodType.FISH) >= 1


class TestGainFoodFromFeeder:
    def test_gain_from_feeder(self, game):
        player = game.players[0]
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
        ])

        power = GainFoodFromFeeder(food_types=[FoodType.SEED], count=1)
        ctx = PowerContext(game, player, bird=None, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        assert result.food_gained.get(FoodType.SEED, 0) == 1
        assert game.birdfeeder.count == 2

    def test_feeder_empty_rerolls(self, game):
        player = game.players[0]
        game.birdfeeder.set_dice([])

        power = GainFoodFromFeeder(food_types=None, count=1)
        ctx = PowerContext(game, player, bird=None, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        # Should have rerolled and taken something
        assert sum(result.food_gained.values()) >= 1


class TestLayEggs:
    def test_lay_on_self(self, game, bird_reg):
        player = game.players[0]
        bird = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = bird

        power = LayEggs(count=2, target="self")
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.WETLAND)
        result = power.execute(ctx)
        assert result.eggs_laid == 2
        assert player.board.wetland.slots[0].eggs == 2

    def test_lay_respects_limit(self, game, bird_reg):
        player = game.players[0]
        bird = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = bird
        player.board.wetland.slots[0].eggs = bird.egg_limit - 1

        power = LayEggs(count=5, target="self")
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.WETLAND)
        result = power.execute(ctx)
        assert result.eggs_laid == 1  # Only space for 1 more

    def test_lay_on_any(self, game, bird_reg):
        player = game.players[0]
        bird1 = bird_reg.get("Trumpeter Swan")
        bird2 = bird_reg.get("Acorn Woodpecker")
        player.board.wetland.slots[0].bird = bird1
        player.board.forest.slots[0].bird = bird2

        power = LayEggs(count=3, target="any")
        ctx = PowerContext(game, player, bird=bird1, slot_index=0,
                           habitat=Habitat.WETLAND)
        result = power.execute(ctx)
        assert result.eggs_laid == 3


class TestDrawCards:
    def test_draw_from_deck(self, game):
        game.deck_remaining = 50
        player = game.players[0]

        power = DrawCards(draw=2, keep=2)
        ctx = PowerContext(game, player, bird=None, slot_index=0,
                           habitat=Habitat.WETLAND)
        result = power.execute(ctx)
        assert result.cards_drawn == 2
        assert game.deck_remaining == 48


class TestTuckFromHand:
    def test_tuck_and_draw(self, game, bird_reg):
        player = game.players[0]
        game.deck_remaining = 50
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        # Give player some cards to tuck
        player.hand.append(bird_reg.get("Trumpeter Swan"))

        power = TuckFromHand(tuck_count=1, draw_count=1)
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        assert result.cards_tucked == 1
        assert result.cards_drawn == 1
        assert player.board.forest.slots[0].tucked_cards == 1
        assert player.hand_size == 1  # Tucked one card, then drew one card

    def test_no_cards_to_tuck(self, game, bird_reg):
        player = game.players[0]
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        player.hand.clear()

        power = TuckFromHand(tuck_count=1, draw_count=1)
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        assert not result.executed  # Can't tuck with empty hand


class TestPredatorDice:
    def test_predator_with_dice_out(self, game, bird_reg):
        player = game.players[0]
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        # Remove some dice from feeder to have dice "out"
        game.birdfeeder.set_dice([FoodType.SEED])  # 1 in, 4 out

        power = PredatorDice(target_food=FoodType.RODENT)
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.FOREST)
        # Run multiple times to test both hit and miss (randomized)
        results = [power.execute(ctx) for _ in range(20)]
        # At least some should hit with 4 dice out
        hits = sum(1 for r in results if r.food_cached)
        assert hits >= 1  # Very unlikely to miss 20 times with 4 dice


class TestCacheFoodFromSupply:
    def test_cache_seed(self, game, bird_reg):
        player = game.players[0]
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird

        power = CacheFoodFromSupply(food_type=FoodType.SEED, count=2)
        ctx = PowerContext(game, player, bird=bird, slot_index=0,
                           habitat=Habitat.FOREST)
        result = power.execute(ctx)
        assert result.food_cached[FoodType.SEED] == 2
        assert player.board.forest.slots[0].total_cached_food == 2


class TestEstimateValue:
    def test_no_power_zero(self):
        power = NoPower()
        assert power.estimate_value(None) == 0.0

    def test_fallback_positive(self):
        power = FallbackPower(color_hint="brown")
        assert power.estimate_value(None) > 0

    def test_gain_food_value(self):
        power = GainFoodFromSupply(food_types=[FoodType.SEED], count=1)
        assert 0 < power.estimate_value(None) < 3

    def test_cache_value(self):
        power = CacheFoodFromSupply(food_type=FoodType.FISH, count=1)
        assert power.estimate_value(None) > 0.5


# --- Specific bird mapping tests ---

class TestSpecificBirds:
    def test_acorn_woodpecker_is_food_power(self, bird_reg):
        bird = bird_reg.get("Acorn Woodpecker")
        power = get_power(bird)
        # "Gain 1 seed from birdfeeder, cache on this bird" — could be GainFoodFromFeeder or CacheFoodFromFeeder
        assert not isinstance(power, (NoPower, FallbackPower)), f"Got {type(power).__name__}"

    def test_trumpeter_swan_is_no_power(self, bird_reg):
        bird = bird_reg.get("Trumpeter Swan")
        power = get_power(bird)
        assert isinstance(power, NoPower)

    def test_american_robin_is_gain_food(self, bird_reg):
        """American Robin: 'All players gain 1 [invertebrate] from supply'."""
        bird = bird_reg.get("American Robin")
        power = get_power(bird)
        # Should be some form of food gain or tuck
        assert not isinstance(power, (NoPower, FallbackPower))

    def test_flocking_bird_mapped(self, bird_reg):
        """Flocking birds should all get a real power (not FallbackPower)."""
        flocking_birds = [b for b in bird_reg.all_birds if b.is_flocking]
        assert len(flocking_birds) > 20
        for bird in flocking_birds:
            power = get_power(bird)
            assert not isinstance(power, FallbackPower), f"{bird.name} is FallbackPower"

    def test_predator_bird_gets_predator(self, bird_reg):
        """Most predator birds should get PredatorDice or PredatorLookAt."""
        predators = [b for b in bird_reg.all_birds if b.is_predator]
        mapped = 0
        for bird in predators:
            power = get_power(bird)
            if isinstance(power, (PredatorDice, PredatorLookAt)):
                mapped += 1
        # Many predators have complex/unique powers; 50% mapping to PredatorDice/LookAt is good
        assert mapped >= len(predators) * 0.5, f"Only {mapped}/{len(predators)} predators mapped"
