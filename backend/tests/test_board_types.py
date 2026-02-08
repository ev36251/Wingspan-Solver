"""Tests for board type support: Base vs Oceania boards with bonus trades."""

import pytest
from backend.config import (
    EXCEL_FILE, get_action_column,
    BASE_BOARD, OCEANIA_BOARD, BOARDS,
)
from backend.data.registries import load_all
from backend.models.enums import BoardType, FoodType, Habitat
from backend.models.board import PlayerBoard
from backend.models.birdfeeder import (
    Birdfeeder, BASE_DICE_FACES, OCEANIA_DICE_FACES,
)
from backend.models.card_tray import CardTray
from backend.models.player import Player, FoodSupply
from backend.models.game_state import GameState, create_new_game
from backend.engine.actions import (
    execute_gain_food, execute_lay_eggs, execute_draw_cards, _pay_bonus_cost,
)
from backend.solver.move_generator import (
    generate_gain_food_moves, generate_lay_eggs_moves,
    generate_draw_cards_moves, generate_all_moves,
)


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture
def base_game():
    return create_new_game(["Alice", "Bob"], board_type=BoardType.BASE)


@pytest.fixture
def oceania_game():
    return create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)


# --- Board Config Tests ---

class TestBoardConfig:
    def test_base_board_exists(self):
        assert BoardType.BASE in BOARDS

    def test_oceania_board_exists(self):
        assert BoardType.OCEANIA in BOARDS

    def test_all_habitats_defined(self):
        for board in BOARDS.values():
            assert Habitat.FOREST in board
            assert Habitat.GRASSLAND in board
            assert Habitat.WETLAND in board

    def test_columns_per_habitat(self):
        """Both boards have 6 columns per habitat (0-4 birds + all-filled)."""
        for habitat, columns in BOARDS[BoardType.BASE].items():
            assert len(columns) == 6
        for habitat, columns in BOARDS[BoardType.OCEANIA].items():
            assert len(columns) == 6

    def test_get_action_column_clamped(self):
        """bird_count beyond max returns the last column."""
        col = get_action_column(BoardType.BASE, Habitat.FOREST, 10)
        assert col.base_gain == 3  # Last base forest column
        assert col.bonus is not None  # All-filled column has bonus
        col = get_action_column(BoardType.OCEANIA, Habitat.FOREST, 10)
        assert col.base_gain == 4  # Last Oceania forest column

    def test_base_forest_gains(self):
        gains = [get_action_column(BoardType.BASE, Habitat.FOREST, i).base_gain for i in range(6)]
        assert gains == [1, 1, 2, 2, 3, 3]

    def test_base_grassland_gains(self):
        gains = [get_action_column(BoardType.BASE, Habitat.GRASSLAND, i).base_gain for i in range(6)]
        assert gains == [2, 2, 3, 3, 4, 4]

    def test_base_wetland_gains(self):
        gains = [get_action_column(BoardType.BASE, Habitat.WETLAND, i).base_gain for i in range(6)]
        assert gains == [1, 1, 2, 2, 3, 3]

    def test_oceania_forest_gains(self):
        gains = [get_action_column(BoardType.OCEANIA, Habitat.FOREST, i).base_gain for i in range(6)]
        assert gains == [1, 2, 2, 2, 3, 4]

    def test_oceania_grassland_gains(self):
        gains = [get_action_column(BoardType.OCEANIA, Habitat.GRASSLAND, i).base_gain for i in range(6)]
        assert gains == [1, 2, 2, 2, 3, 4]

    def test_oceania_wetland_gains(self):
        gains = [get_action_column(BoardType.OCEANIA, Habitat.WETLAND, i).base_gain for i in range(6)]
        assert gains == [1, 2, 2, 2, 3, 4]

    def test_base_forest_bonuses(self):
        """Base forest: cols 1,3,5 have 'extra' bonus (discard card for +1 food)."""
        for i in range(6):
            col = get_action_column(BoardType.BASE, Habitat.FOREST, i)
            if i in (1, 3, 5):
                assert col.bonus is not None
                assert col.bonus.bonus_type == "extra"
                assert "card" in col.bonus.cost_options
            else:
                assert col.bonus is None

    def test_oceania_forest_extra_bonuses(self):
        """Oceania forest: cols 0,2,3,4 have 'extra' bonus (card for +1 food)."""
        for i in (0, 2, 3, 4):
            col = get_action_column(BoardType.OCEANIA, Habitat.FOREST, i)
            assert col.bonus is not None, f"Col {i} should have extra bonus"
            assert col.bonus.bonus_type == "extra"
            assert "card" in col.bonus.cost_options

    def test_oceania_forest_reset_bonuses(self):
        """Oceania forest: cols 1,3 have 'reset_feeder' as reset_bonus."""
        for i in (1, 3):
            col = get_action_column(BoardType.OCEANIA, Habitat.FOREST, i)
            assert col.reset_bonus is not None, f"Col {i} should have reset_bonus"
            assert col.reset_bonus.bonus_type == "reset_feeder"
            assert "food" in col.reset_bonus.cost_options

    def test_oceania_forest_col3_dual_bonus(self):
        """Oceania forest col 3 has both extra and reset_feeder bonuses."""
        col3 = get_action_column(BoardType.OCEANIA, Habitat.FOREST, 3)
        assert col3.bonus is not None
        assert col3.bonus.bonus_type == "extra"
        assert col3.reset_bonus is not None
        assert col3.reset_bonus.bonus_type == "reset_feeder"

    def test_oceania_forest_col5_no_bonus(self):
        """Oceania forest col 5 (all filled) has no bonus."""
        col5 = get_action_column(BoardType.OCEANIA, Habitat.FOREST, 5)
        assert col5.bonus is None
        assert col5.reset_bonus is None
        assert col5.base_gain == 4

    def test_oceania_grassland_double_bonus(self):
        """Oceania grassland col 3 allows 2 bonus trades."""
        col3 = get_action_column(BoardType.OCEANIA, Habitat.GRASSLAND, 3)
        assert col3.bonus is not None
        assert col3.bonus.max_uses == 2

    def test_oceania_wetland_extra_bonuses(self):
        """Oceania wetland: cols 0,2,3,4 have 'extra' bonus (egg/nectar for +1 card)."""
        for i in (0, 2, 3, 4):
            col = get_action_column(BoardType.OCEANIA, Habitat.WETLAND, i)
            assert col.bonus is not None, f"Col {i} should have extra bonus"
            assert col.bonus.bonus_type == "extra"

    def test_oceania_wetland_reset_tray(self):
        """Oceania wetland cols 1,3 have 'reset_tray' as reset_bonus."""
        for i in (1, 3):
            col = get_action_column(BoardType.OCEANIA, Habitat.WETLAND, i)
            assert col.reset_bonus is not None, f"Col {i} should have reset_bonus"
            assert col.reset_bonus.bonus_type == "reset_tray"

    def test_oceania_wetland_col3_dual_bonus(self):
        """Oceania wetland col 3 has both extra and reset_tray bonuses."""
        col3 = get_action_column(BoardType.OCEANIA, Habitat.WETLAND, 3)
        assert col3.bonus is not None
        assert col3.bonus.bonus_type == "extra"
        assert col3.reset_bonus is not None
        assert col3.reset_bonus.bonus_type == "reset_tray"


# --- Game Creation Tests ---

class TestGameCreation:
    def test_base_game_board_type(self, base_game):
        assert base_game.board_type == BoardType.BASE

    def test_oceania_game_board_type(self, oceania_game):
        assert oceania_game.board_type == BoardType.OCEANIA

    def test_default_board_is_base(self):
        game = create_new_game(["Alice"])
        assert game.board_type == BoardType.BASE


# --- Bonus Cost Payment Tests ---

class TestBonusCostPayment:
    def test_pay_card_cost(self, bird_reg):
        player = Player(name="Test")
        filler = bird_reg.get("American Crow")
        player.hand = [filler]
        assert _pay_bonus_cost(player, ("card",)) is True
        assert len(player.hand) == 0

    def test_pay_card_fails_empty_hand(self):
        player = Player(name="Test")
        player.hand = []
        assert _pay_bonus_cost(player, ("card",)) is False

    def test_pay_food_cost(self):
        player = Player(name="Test")
        player.food_supply.add(FoodType.SEED, 2)
        assert _pay_bonus_cost(player, ("food",)) is True
        assert player.food_supply.get(FoodType.SEED) == 1

    def test_pay_food_fails_no_food(self):
        player = Player(name="Test")
        assert _pay_bonus_cost(player, ("food",)) is False

    def test_pay_egg_cost(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("Trumpeter Swan")
        player.board.forest.slots[0].bird = bird
        player.board.forest.slots[0].eggs = 2
        assert _pay_bonus_cost(player, ("egg",)) is True
        assert player.board.forest.slots[0].eggs == 1

    def test_pay_nectar_cost(self):
        player = Player(name="Test")
        player.food_supply.add(FoodType.NECTAR, 1)
        assert _pay_bonus_cost(player, ("nectar",)) is True
        assert player.food_supply.get(FoodType.NECTAR) == 0

    def test_pay_either_prefers_first(self):
        """When multiple options, pays the first available."""
        player = Player(name="Test")
        player.food_supply.add(FoodType.SEED, 1)
        # cost_options=("card", "food") — no cards, so pays food
        assert _pay_bonus_cost(player, ("card", "food")) is True
        assert player.food_supply.get(FoodType.SEED) == 0


# --- Base Board Action Tests ---

class TestBaseActions:
    def test_gain_food_no_birds(self, base_game):
        """0 birds in forest = col 0: base_gain=1, no bonus."""
        player = base_game.players[0]
        base_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
            FoodType.INVERTEBRATE, FoodType.RODENT,
        ])
        result = execute_gain_food(base_game, player, [FoodType.SEED, FoodType.FISH])
        assert result.success
        # Should gain only 1 food (base_gain=1 at 0 birds)
        assert sum(result.food_gained.values()) == 1

    def test_gain_food_with_bonus_discard_card(self, bird_reg, base_game):
        """1 bird in forest = col 1: base_gain=1, bonus=extra (card)."""
        player = base_game.players[0]
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler
        # Give player a card to discard
        player.hand = [filler]
        base_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.SEED, FoodType.FISH,
            FoodType.FRUIT, FoodType.RODENT,
        ])
        result = execute_gain_food(
            base_game, player,
            [FoodType.SEED, FoodType.SEED],
            bonus_count=1,
        )
        assert result.success
        # base_gain=1 + 1 bonus = 2 food
        assert sum(result.food_gained.values()) == 2
        assert result.bonus_activated == 1
        assert len(player.hand) == 0  # Card was discarded

    def test_gain_food_bonus_not_activated_without_request(self, bird_reg, base_game):
        """Bonus only activates when bonus_count > 0."""
        player = base_game.players[0]
        # 1 bird in forest (from previous test)
        player.hand = [bird_reg.get("American Crow")]
        base_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
            FoodType.INVERTEBRATE, FoodType.RODENT,
        ])
        result = execute_gain_food(base_game, player, [FoodType.SEED, FoodType.FISH])
        assert result.success
        assert result.bonus_activated == 0

    def test_lay_eggs_with_bonus(self, bird_reg, base_game):
        """Base grassland col 1: base_gain=2, bonus=extra (food for +1 egg)."""
        player = base_game.players[0]
        # Use House Sparrow (egg_limit=5) — enough room for 3 eggs
        bird = bird_reg.get("House Sparrow")
        player.board.grassland.slots[0].bird = bird
        player.board.grassland.slots[0].eggs = 0
        # 1 bird in grassland = col 1 (has bonus)
        player.food_supply.add(FoodType.SEED, 2)
        result = execute_lay_eggs(
            base_game, player,
            egg_distribution={(Habitat.GRASSLAND, 0): 5},
            bonus_count=1,
        )
        assert result.success
        # base=2 + 1 bonus = 3 eggs
        assert result.eggs_laid == 3
        assert result.bonus_activated == 1

    def test_draw_cards_with_bonus(self, bird_reg, base_game):
        """Base wetland col 1: base_gain=1, bonus=extra (egg for +1 card)."""
        player = base_game.players[0]
        filler = bird_reg.get("American Crow")
        player.board.wetland.slots[0].bird = filler
        # Add an egg to pay for bonus
        player.board.wetland.slots[0].eggs = 1
        base_game.deck_remaining = 10
        result = execute_draw_cards(
            base_game, player,
            from_deck_count=5,
            bonus_count=1,
        )
        assert result.success
        # base=1 + 1 bonus = 2 cards
        assert result.cards_drawn == 2
        assert result.bonus_activated == 1
        assert player.board.wetland.slots[0].eggs == 0


# --- Oceania Board Action Tests ---

class TestOceaniaActions:
    def test_gain_food_col0_bonus(self, bird_reg, oceania_game):
        """Oceania forest col 0: base=1, bonus=extra (card for +1 food)."""
        player = oceania_game.players[0]
        filler = bird_reg.get("American Crow")
        player.hand = [filler]  # Card to discard for bonus
        oceania_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.SEED, FoodType.FISH,
            FoodType.FRUIT, FoodType.RODENT,
        ])
        result = execute_gain_food(
            oceania_game, player,
            [FoodType.SEED, FoodType.SEED],
            bonus_count=1,
        )
        assert result.success
        assert sum(result.food_gained.values()) == 2  # 1 base + 1 bonus
        assert result.bonus_activated == 1

    def test_gain_food_reset_feeder(self, bird_reg, oceania_game):
        """Oceania forest col 1: base=2, reset_bonus=reset_feeder (discard food)."""
        player = oceania_game.players[0]
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler  # 1 bird = col 1
        player.food_supply.add(FoodType.RODENT, 1)  # Food to pay for reset
        oceania_game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.SEED, FoodType.SEED,
            FoodType.SEED, FoodType.SEED,
        ])
        result = execute_gain_food(
            oceania_game, player,
            [FoodType.SEED, FoodType.SEED],
            reset_bonus=True,
        )
        assert result.success
        assert result.bonus_activated == 1
        # Feeder was reset (rerolled), food was paid

    def test_lay_eggs_double_bonus(self, bird_reg, oceania_game):
        """Oceania grassland col 3: base=2, bonus=extra x2."""
        player = oceania_game.players[0]
        bird = bird_reg.get("House Sparrow")  # egg_limit=5
        filler = bird_reg.get("American Crow")
        player.board.grassland.slots[0].bird = bird
        player.board.grassland.slots[1].bird = filler
        player.board.grassland.slots[2].bird = filler
        player.board.grassland.slots[0].eggs = 0
        # 3 birds = col 3, max_uses=2
        player.food_supply.add(FoodType.SEED, 5)
        result = execute_lay_eggs(
            oceania_game, player,
            egg_distribution={(Habitat.GRASSLAND, 0): 10},
            bonus_count=2,
        )
        assert result.success
        # base=2 + 2 bonus = 4 eggs
        assert result.eggs_laid == 4
        assert result.bonus_activated == 2

    def test_draw_cards_reset_tray(self, bird_reg, oceania_game):
        """Oceania wetland col 1: base=2, reset_bonus=reset_tray (discard food)."""
        player = oceania_game.players[0]
        filler = bird_reg.get("American Crow")
        player.board.wetland.slots[0].bird = filler  # 1 bird = col 1
        player.food_supply.add(FoodType.SEED, 1)  # Food to pay for tray reset
        # Put some cards in the tray
        tray_bird = bird_reg.get("Acorn Woodpecker")
        oceania_game.card_tray.face_up = [tray_bird, tray_bird]
        oceania_game.deck_remaining = 10
        result = execute_draw_cards(
            oceania_game, player,
            from_deck_count=5,
            reset_bonus=True,
        )
        assert result.success
        assert result.bonus_activated == 1
        # Tray was cleared, then drew from deck


# --- Move Generator Tests ---

class TestMoveGeneratorBonus:
    def test_base_no_bonus_at_col0(self, base_game):
        """Base forest with 0 birds: no bonus moves generated."""
        player = base_game.players[0]
        player.board = PlayerBoard()  # Fresh board
        base_game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        moves = generate_gain_food_moves(base_game, player)
        assert all(m.bonus_count == 0 for m in moves)

    def test_base_bonus_at_col1(self, bird_reg, base_game):
        """Base forest with 1 bird: bonus moves generated if player has card."""
        player = base_game.players[0]
        player.board = PlayerBoard()
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler
        player.hand = [filler]  # Has a card to discard
        base_game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        moves = generate_gain_food_moves(base_game, player)
        has_bonus = any(m.bonus_count > 0 for m in moves)
        has_no_bonus = any(m.bonus_count == 0 for m in moves)
        assert has_bonus
        assert has_no_bonus  # Both options offered

    def test_oceania_double_bonus_moves(self, bird_reg, oceania_game):
        """Oceania grassland col 3 generates moves with bonus_count=1 and 2."""
        player = oceania_game.players[0]
        player.board = PlayerBoard()
        filler = bird_reg.get("American Crow")
        bird = bird_reg.get("Trumpeter Swan")
        player.board.grassland.slots[0].bird = filler
        player.board.grassland.slots[1].bird = filler
        player.board.grassland.slots[2].bird = filler
        # Need a bird that can receive eggs
        player.board.forest.slots[0].bird = bird
        player.food_supply = FoodSupply()
        player.food_supply.add(FoodType.SEED, 5)
        moves = generate_lay_eggs_moves(oceania_game, player)
        bonus_counts = {m.bonus_count for m in moves}
        assert 0 in bonus_counts
        assert 1 in bonus_counts
        assert 2 in bonus_counts

    def test_all_moves_include_bonus_fields(self, bird_reg, oceania_game):
        """All generated moves have bonus_count and reset_bonus fields."""
        player = oceania_game.players[0]
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler
        player.hand = [filler]
        player.food_supply.add(FoodType.SEED, 5)
        oceania_game.birdfeeder.set_dice([FoodType.SEED] * 5)
        oceania_game.deck_remaining = 10
        moves = generate_all_moves(oceania_game, player)
        for m in moves:
            assert hasattr(m, 'bonus_count')
            assert hasattr(m, 'reset_bonus')

    def test_oceania_forest_col1_generates_reset_moves(self, bird_reg, oceania_game):
        """Oceania forest col 1 (reset_feeder only) generates reset_bonus moves."""
        player = oceania_game.players[0]
        player.board = PlayerBoard()
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler  # 1 bird = col 1
        player.food_supply = FoodSupply()
        player.food_supply.add(FoodType.SEED, 3)  # Food to pay for reset
        oceania_game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        moves = generate_gain_food_moves(oceania_game, player)
        has_reset = any(m.reset_bonus for m in moves)
        has_no_reset = any(not m.reset_bonus for m in moves)
        assert has_reset, "Should offer reset_bonus moves"
        assert has_no_reset, "Should also offer non-reset moves"
        # Col 1 has no extra bonus, only reset
        assert all(m.bonus_count == 0 for m in moves)

    def test_oceania_forest_col3_generates_dual_moves(self, bird_reg, oceania_game):
        """Oceania forest col 3 (both bonuses) generates all four variants."""
        player = oceania_game.players[0]
        player.board = PlayerBoard()
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler
        player.board.forest.slots[1].bird = filler
        player.board.forest.slots[2].bird = filler  # 3 birds = col 3
        player.hand = [filler, filler]  # Cards for extra bonus
        player.food_supply = FoodSupply()
        player.food_supply.add(FoodType.SEED, 3)  # Food for reset
        oceania_game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH])
        moves = generate_gain_food_moves(oceania_game, player)
        combos = {(m.bonus_count, m.reset_bonus) for m in moves}
        assert (0, False) in combos, "Should have base moves"
        assert (1, False) in combos, "Should have extra-only moves"
        assert (0, True) in combos, "Should have reset-only moves"
        assert (1, True) in combos, "Should have dual-bonus moves"

    def test_oceania_all_filled_col5_no_bonus(self, bird_reg, oceania_game):
        """When all 5 slots filled (col 5), gain max with no bonus options."""
        player = oceania_game.players[0]
        player.board = PlayerBoard()
        filler = bird_reg.get("American Crow")
        for i in range(5):
            player.board.forest.slots[i].bird = filler  # 5 birds = col 5
        player.hand = [filler]
        player.food_supply = FoodSupply()
        player.food_supply.add(FoodType.SEED, 3)
        oceania_game.birdfeeder.set_dice([FoodType.SEED, FoodType.FISH,
                                           FoodType.FRUIT, FoodType.RODENT])
        moves = generate_gain_food_moves(oceania_game, player)
        # All moves should be base only (no bonus at col 5)
        assert all(m.bonus_count == 0 for m in moves)
        assert all(not m.reset_bonus for m in moves)
        # Should request 4 food (base_gain=4)
        assert any(len(m.food_choices) == 4 for m in moves)


# --- Nectar Dice Tests ---

class TestNectarDice:
    def test_base_dice_have_no_nectar(self):
        """Base game dice faces should not include nectar."""
        for face in BASE_DICE_FACES:
            if isinstance(face, tuple):
                assert FoodType.NECTAR not in face
            else:
                assert face != FoodType.NECTAR

    def test_oceania_dice_have_nectar(self):
        """Oceania dice should have nectar as a choice option."""
        nectar_faces = [f for f in OCEANIA_DICE_FACES
                        if isinstance(f, tuple) and FoodType.NECTAR in f]
        assert len(nectar_faces) == 2

    def test_oceania_nectar_fruit_choice(self):
        """Oceania dice have a nectar/fruit choice face."""
        assert (FoodType.NECTAR, FoodType.FRUIT) in OCEANIA_DICE_FACES

    def test_oceania_nectar_seed_choice(self):
        """Oceania dice have a nectar/seed choice face."""
        assert (FoodType.NECTAR, FoodType.SEED) in OCEANIA_DICE_FACES

    def test_base_feeder_uses_base_dice(self):
        feeder = Birdfeeder(board_type=BoardType.BASE)
        assert feeder.dice_faces is BASE_DICE_FACES

    def test_oceania_feeder_uses_oceania_dice(self):
        feeder = Birdfeeder(board_type=BoardType.OCEANIA)
        assert feeder.dice_faces is OCEANIA_DICE_FACES

    def test_oceania_feeder_can_offer_nectar(self):
        """Oceania feeder with nectar choice dice should offer nectar."""
        feeder = Birdfeeder(board_type=BoardType.OCEANIA)
        feeder.set_dice([(FoodType.NECTAR, FoodType.FRUIT)])
        assert FoodType.NECTAR in feeder.available_food_types()
        assert FoodType.FRUIT in feeder.available_food_types()

    def test_take_nectar_from_choice_die(self):
        """Player can take nectar from a nectar/fruit choice die."""
        feeder = Birdfeeder(board_type=BoardType.OCEANIA)
        feeder.set_dice([
            (FoodType.NECTAR, FoodType.FRUIT),
            (FoodType.NECTAR, FoodType.SEED),
        ])
        assert feeder.take_food(FoodType.NECTAR)
        assert feeder.count == 1

    def test_take_fruit_from_nectar_choice_die(self):
        """Player can take fruit from a nectar/fruit choice die."""
        feeder = Birdfeeder(board_type=BoardType.OCEANIA)
        feeder.set_dice([(FoodType.NECTAR, FoodType.FRUIT)])
        assert feeder.take_food(FoodType.FRUIT)
        assert feeder.count == 0

    def test_create_game_oceania_feeder(self):
        """New Oceania game should have a feeder with Oceania dice."""
        game = create_new_game(["Alice"], board_type=BoardType.OCEANIA)
        assert game.birdfeeder.board_type == BoardType.OCEANIA
        assert game.birdfeeder.dice_faces is OCEANIA_DICE_FACES

    def test_create_game_base_feeder(self):
        """New Base game should have a feeder with base dice."""
        game = create_new_game(["Alice"], board_type=BoardType.BASE)
        assert game.birdfeeder.board_type == BoardType.BASE

    def test_gain_nectar_from_oceania_feeder(self):
        """Player can gain nectar food from Oceania feeder."""
        game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
        player = game.players[0]
        # Player starts with 1 nectar (Oceania starting bonus)
        assert player.food_supply.nectar == 1
        game.birdfeeder.set_dice([
            (FoodType.NECTAR, FoodType.FRUIT),
            FoodType.SEED, FoodType.FISH,
        ])
        result = execute_gain_food(game, player, [FoodType.NECTAR])
        assert result.success
        assert player.food_supply.nectar == 2  # 1 starting + 1 gained

    def test_oceania_dice_have_8_faces(self):
        """Oceania dice have 8 faces: all 6 base + 2 nectar choices."""
        assert len(OCEANIA_DICE_FACES) == 8
        assert len(BASE_DICE_FACES) == 6

    def test_oceania_dice_include_all_base_faces(self):
        """Oceania dice include all single-food faces from base dice."""
        base_singles = {f for f in BASE_DICE_FACES if not isinstance(f, tuple)}
        oceania_singles = {f for f in OCEANIA_DICE_FACES if not isinstance(f, tuple)}
        assert base_singles == oceania_singles

    def test_oceania_dice_have_base_choice_face(self):
        """Oceania dice still include the invertebrate/seed choice from base."""
        assert (FoodType.INVERTEBRATE, FoodType.SEED) in OCEANIA_DICE_FACES

    def test_oceania_starting_nectar(self):
        """Each player starts with 1 free nectar in Oceania games."""
        game = create_new_game(["Alice", "Bob", "Charlie"], board_type=BoardType.OCEANIA)
        for p in game.players:
            assert p.food_supply.nectar == 1

    def test_base_no_starting_nectar(self):
        """Base game players do NOT start with nectar."""
        game = create_new_game(["Alice", "Bob"], board_type=BoardType.BASE)
        for p in game.players:
            assert p.food_supply.nectar == 0


# --- Nectar End-of-Round Clearing Tests ---

class TestNectarClearing:
    def test_nectar_cleared_on_round_advance_oceania(self):
        """On Oceania board, nectar is discarded at end of each round."""
        game = create_new_game(["Alice", "Bob"], board_type=BoardType.OCEANIA)
        alice = game.players[0]
        bob = game.players[1]
        alice.food_supply.add(FoodType.NECTAR, 3)
        bob.food_supply.add(FoodType.NECTAR, 2)
        # Also give some regular food that should NOT be cleared
        alice.food_supply.add(FoodType.SEED, 5)

        game.advance_round()

        assert alice.food_supply.nectar == 0
        assert bob.food_supply.nectar == 0
        assert alice.food_supply.seed == 5  # Regular food preserved

    def test_nectar_not_cleared_on_base_board(self):
        """On Base board, nectar is NOT discarded (shouldn't have any, but test the logic)."""
        game = create_new_game(["Alice", "Bob"], board_type=BoardType.BASE)
        alice = game.players[0]
        alice.food_supply.add(FoodType.NECTAR, 2)

        game.advance_round()

        # Base board doesn't clear nectar
        assert alice.food_supply.nectar == 2

    def test_nectar_cleared_each_round(self):
        """Nectar is cleared at the end of every round, not just the first."""
        game = create_new_game(["Alice"], board_type=BoardType.OCEANIA)
        alice = game.players[0]

        # Round 1 -> 2
        alice.food_supply.add(FoodType.NECTAR, 3)
        game.advance_round()
        assert alice.food_supply.nectar == 0

        # Round 2 -> 3
        alice.food_supply.add(FoodType.NECTAR, 1)
        game.advance_round()
        assert alice.food_supply.nectar == 0

    def test_nectar_spent_on_birds_preserved(self):
        """Nectar spent on habitats (for scoring) is NOT cleared."""
        game = create_new_game(["Alice"], board_type=BoardType.OCEANIA)
        alice = game.players[0]
        alice.board.forest.nectar_spent = 2
        alice.food_supply.add(FoodType.NECTAR, 3)

        game.advance_round()

        assert alice.food_supply.nectar == 0
        assert alice.board.forest.nectar_spent == 2  # Preserved for scoring
