"""Tests for game state models: board, player, birdfeeder, game state."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat, NestType, PowerColor
from backend.models.board import BirdSlot, HabitatRow, PlayerBoard
from backend.models.birdfeeder import Birdfeeder
from backend.models.card_tray import CardTray
from backend.models.player import Player, FoodSupply
from backend.models.game_state import GameState, create_new_game


@pytest.fixture(scope="module")
def bird_reg():
    regs = load_all(EXCEL_FILE)
    return regs[0]


@pytest.fixture
def sample_bird(bird_reg):
    """A known bird for testing."""
    return bird_reg.get("Acorn Woodpecker")


@pytest.fixture
def high_vp_bird(bird_reg):
    """A no-power high-VP bird."""
    return bird_reg.get("Trumpeter Swan")


# --- BirdSlot ---

class TestBirdSlot:
    def test_empty_slot(self):
        slot = BirdSlot()
        assert slot.is_empty
        assert slot.eggs == 0
        assert slot.total_cached_food == 0
        assert slot.tucked_cards == 0

    def test_slot_with_bird(self, sample_bird):
        slot = BirdSlot(bird=sample_bird)
        assert not slot.is_empty
        assert slot.can_hold_more_eggs()

    def test_egg_limit(self, sample_bird):
        slot = BirdSlot(bird=sample_bird, eggs=sample_bird.egg_limit)
        assert not slot.can_hold_more_eggs()
        assert slot.eggs_space() == 0

    def test_cache_food(self):
        slot = BirdSlot()
        slot.cache_food(FoodType.SEED, 2)
        slot.cache_food(FoodType.FISH, 1)
        assert slot.total_cached_food == 3
        assert slot.cached_food[FoodType.SEED] == 2

    def test_eggs_space(self, sample_bird):
        slot = BirdSlot(bird=sample_bird, eggs=1)
        assert slot.eggs_space() == sample_bird.egg_limit - 1


# --- HabitatRow ---

class TestHabitatRow:
    def test_empty_row(self):
        row = HabitatRow(Habitat.FOREST)
        assert row.bird_count == 0
        assert not row.is_full
        assert row.next_empty_slot() == 0
        assert row.total_eggs() == 0

    def test_add_bird(self, sample_bird):
        row = HabitatRow(Habitat.FOREST)
        idx = row.next_empty_slot()
        row.slots[idx].bird = sample_bird
        assert row.bird_count == 1
        assert row.next_empty_slot() == 1

    def test_full_row(self, sample_bird):
        row = HabitatRow(Habitat.FOREST)
        for i in range(5):
            row.slots[i].bird = sample_bird
        assert row.is_full
        assert row.next_empty_slot() is None

    def test_occupied_slots(self, sample_bird):
        row = HabitatRow(Habitat.WETLAND)
        row.slots[0].bird = sample_bird
        row.slots[2].bird = sample_bird
        occupied = row.occupied_slots()
        assert len(occupied) == 2
        assert occupied[0][0] == 0
        assert occupied[1][0] == 2

    def test_birds_list(self, sample_bird, high_vp_bird):
        row = HabitatRow(Habitat.FOREST)
        row.slots[0].bird = sample_bird
        row.slots[1].bird = high_vp_bird
        birds = row.birds()
        assert len(birds) == 2
        assert birds[0].name == sample_bird.name
        assert birds[1].name == high_vp_bird.name


# --- PlayerBoard ---

class TestPlayerBoard:
    def test_empty_board(self):
        board = PlayerBoard()
        assert board.total_birds() == 0
        assert board.total_eggs() == 0
        assert len(board.all_birds()) == 0

    def test_get_row(self):
        board = PlayerBoard()
        assert board.get_row(Habitat.FOREST) is board.forest
        assert board.get_row(Habitat.GRASSLAND) is board.grassland
        assert board.get_row(Habitat.WETLAND) is board.wetland

    def test_add_birds_across_habitats(self, sample_bird, high_vp_bird):
        board = PlayerBoard()
        board.forest.slots[0].bird = sample_bird
        board.wetland.slots[0].bird = high_vp_bird
        assert board.total_birds() == 2
        assert len(board.all_birds()) == 2

    def test_find_bird(self, sample_bird):
        board = PlayerBoard()
        board.grassland.slots[2].bird = sample_bird
        result = board.find_bird(sample_bird.name)
        assert result is not None
        assert result[0] == Habitat.GRASSLAND
        assert result[1] == 2

    def test_find_bird_not_found(self):
        board = PlayerBoard()
        assert board.find_bird("Nonexistent Bird") is None

    def test_all_slots(self):
        board = PlayerBoard()
        slots = board.all_slots()
        assert len(slots) == 15  # 3 habitats x 5 slots


# --- Birdfeeder ---

class TestBirdfeeder:
    def test_initial_roll(self):
        feeder = Birdfeeder()
        assert feeder.count == 5
        assert not feeder.is_empty

    def test_take_food(self):
        feeder = Birdfeeder()
        feeder.set_dice([FoodType.SEED, FoodType.FISH, FoodType.SEED,
                         FoodType.FRUIT, FoodType.RODENT])
        assert feeder.take_food(FoodType.SEED)
        assert feeder.count == 4

    def test_take_choice_die(self):
        feeder = Birdfeeder()
        feeder.set_dice([(FoodType.INVERTEBRATE, FoodType.SEED)])
        assert feeder.can_take(FoodType.INVERTEBRATE)
        assert feeder.can_take(FoodType.SEED)
        assert feeder.take_food(FoodType.INVERTEBRATE)
        assert feeder.count == 0

    def test_cant_take_unavailable(self):
        feeder = Birdfeeder()
        feeder.set_dice([FoodType.SEED, FoodType.SEED])
        assert not feeder.can_take(FoodType.FISH)
        assert not feeder.take_food(FoodType.FISH)

    def test_all_same_face(self):
        feeder = Birdfeeder()
        feeder.set_dice([FoodType.SEED, FoodType.SEED, FoodType.SEED])
        assert feeder.all_same_face()
        assert feeder.should_reroll()

    def test_not_all_same(self):
        feeder = Birdfeeder()
        feeder.set_dice([FoodType.SEED, FoodType.FISH])
        assert not feeder.all_same_face()
        assert not feeder.should_reroll()

    def test_empty_should_reroll(self):
        feeder = Birdfeeder()
        feeder.set_dice([])
        assert feeder.is_empty
        assert feeder.should_reroll()

    def test_reroll_gives_5_dice(self):
        feeder = Birdfeeder()
        feeder.set_dice([])
        feeder.reroll()
        assert feeder.count == 5

    def test_available_food_types(self):
        feeder = Birdfeeder()
        feeder.set_dice([FoodType.SEED, FoodType.FISH,
                         (FoodType.INVERTEBRATE, FoodType.SEED)])
        types = feeder.available_food_types()
        assert FoodType.SEED in types
        assert FoodType.FISH in types
        assert FoodType.INVERTEBRATE in types


# --- CardTray ---

class TestCardTray:
    def test_empty_tray(self):
        tray = CardTray()
        assert tray.count == 0
        assert tray.needs_refill() == 3

    def test_add_and_take(self, sample_bird):
        tray = CardTray()
        tray.add_card(sample_bird)
        assert tray.count == 1
        taken = tray.take_card(0)
        assert taken is sample_bird
        assert tray.count == 0

    def test_take_by_name(self, sample_bird):
        tray = CardTray()
        tray.add_card(sample_bird)
        taken = tray.take_by_name(sample_bird.name)
        assert taken is sample_bird

    def test_max_capacity(self, sample_bird):
        tray = CardTray()
        for _ in range(5):
            tray.add_card(sample_bird)
        assert tray.count == 3  # Max is 3


# --- FoodSupply ---

class TestFoodSupply:
    def test_initial_empty(self):
        supply = FoodSupply()
        assert supply.total() == 0

    def test_add_and_get(self):
        supply = FoodSupply()
        supply.add(FoodType.SEED, 3)
        assert supply.get(FoodType.SEED) == 3
        assert supply.has(FoodType.SEED, 3)
        assert not supply.has(FoodType.SEED, 4)

    def test_spend(self):
        supply = FoodSupply()
        supply.add(FoodType.FISH, 2)
        assert supply.spend(FoodType.FISH, 1)
        assert supply.get(FoodType.FISH) == 1
        assert not supply.spend(FoodType.FISH, 5)  # Can't overspend

    def test_as_dict(self):
        supply = FoodSupply()
        supply.add(FoodType.SEED, 2)
        supply.add(FoodType.FISH, 1)
        d = supply.as_dict()
        assert d == {FoodType.SEED: 2, FoodType.FISH: 1}


# --- Player ---

class TestPlayer:
    def test_new_player(self):
        player = Player(name="Alice")
        assert player.name == "Alice"
        assert player.total_birds == 0
        assert player.hand_size == 0

    def test_hand_management(self, sample_bird, high_vp_bird):
        player = Player(name="Alice")
        player.hand.append(sample_bird)
        player.hand.append(high_vp_bird)
        assert player.hand_size == 2
        assert player.has_bird_in_hand(sample_bird.name)

        removed = player.remove_from_hand(sample_bird.name)
        assert removed is sample_bird
        assert player.hand_size == 1
        assert not player.has_bird_in_hand(sample_bird.name)


# --- GameState ---

class TestGameState:
    def test_create_new_game(self):
        game = create_new_game(["Alice", "Bob"])
        assert game.num_players == 2
        assert game.current_round == 1
        assert not game.is_game_over
        assert game.current_player.name == "Alice"
        # Round 1 = 8 actions per player
        assert game.players[0].action_cubes_remaining == 8
        assert game.players[1].action_cubes_remaining == 8

    def test_advance_turn(self):
        game = create_new_game(["Alice", "Bob"])
        assert game.current_player.name == "Alice"
        game.advance_turn()
        assert game.current_player.name == "Bob"
        game.advance_turn()
        assert game.current_player.name == "Alice"

    def test_advance_round(self):
        game = create_new_game(["Alice", "Bob"])
        # Exhaust all actions in round 1
        for p in game.players:
            p.action_cubes_remaining = 0
        game.advance_round()
        assert game.current_round == 2
        # Round 2 = 7 actions
        assert game.players[0].action_cubes_remaining == 7

    def test_game_over(self):
        game = create_new_game(["Alice"])
        game.current_round = 5  # Past round 4
        assert game.is_game_over

    def test_get_player(self):
        game = create_new_game(["Alice", "Bob", "Charlie"])
        assert game.get_player("Bob").name == "Bob"
        assert game.get_player("Nobody") is None

    def test_other_players(self):
        game = create_new_game(["Alice", "Bob", "Charlie"])
        others = game.other_players(game.players[0])
        assert len(others) == 2
        assert all(p.name != "Alice" for p in others)

    def test_actions_this_round(self):
        game = create_new_game(["Alice"])
        assert game.actions_this_round == 8  # Round 1
        game.current_round = 2
        assert game.actions_this_round == 7
        game.current_round = 3
        assert game.actions_this_round == 6
        game.current_round = 4
        assert game.actions_this_round == 5
