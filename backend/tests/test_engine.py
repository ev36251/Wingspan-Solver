"""Tests for the core game engine: rules, actions, and scoring."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, Habitat, PowerColor, GameSet
from backend.models.board import PlayerBoard, HabitatRow, BirdSlot
from backend.models.birdfeeder import Birdfeeder
from backend.models.card_tray import CardTray
from backend.models.player import Player, FoodSupply
from backend.models.game_state import GameState, create_new_game
from backend.engine.rules import (
    can_play_bird, can_pay_food_cost, find_food_payment_options,
    egg_cost_for_slot,
)
from backend.engine.actions import (
    execute_play_bird, execute_gain_food, execute_lay_eggs, execute_draw_cards,
)
from backend.engine.scoring import (
    score_birds, score_eggs, score_cached_food, score_tucked_cards,
    score_bonus_cards, calculate_score, ScoreBreakdown,
)


@pytest.fixture(scope="module")
def regs():
    return load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def bird_reg(regs):
    return regs[0]


@pytest.fixture(scope="module")
def bonus_reg(regs):
    return regs[1]


@pytest.fixture(scope="module")
def goal_reg(regs):
    return regs[2]


@pytest.fixture
def game():
    return create_new_game(["Alice", "Bob"])


@pytest.fixture
def alice(game) -> Player:
    return game.players[0]


# --- Rules ---

class TestRules:
    def test_egg_cost_by_column(self):
        assert egg_cost_for_slot(0) == 0
        assert egg_cost_for_slot(1) == 1
        assert egg_cost_for_slot(2) == 1
        assert egg_cost_for_slot(3) == 2
        assert egg_cost_for_slot(4) == 2

    def test_can_play_bird_no_food(self, bird_reg, game, alice):
        bird = bird_reg.get("Acorn Woodpecker")  # Costs 1 seed, 1 fruit
        alice.hand.append(bird)
        # No food in supply
        legal, reason = can_play_bird(alice, bird, Habitat.FOREST, game)
        assert not legal
        assert "Not enough" in reason or "Cannot pay" in reason

    def test_can_play_bird_success(self, bird_reg, game, alice):
        bird = bird_reg.get("American Crow")  # Costs 1 wild
        alice.hand.append(bird)
        alice.food_supply.add(FoodType.SEED, 1)
        legal, reason = can_play_bird(alice, bird, Habitat.FOREST, game)
        assert legal, reason

    def test_can_play_bird_wrong_habitat(self, bird_reg, game, alice):
        bird = bird_reg.get("Acorn Woodpecker")
        alice.hand.append(bird)
        alice.food_supply.add(FoodType.SEED, 5)
        alice.food_supply.add(FoodType.FRUIT, 5)
        # Check which habitats it CAN live in
        valid_habs = [h for h in Habitat if bird.can_live_in(h)]
        invalid_habs = [h for h in Habitat if not bird.can_live_in(h)]
        if invalid_habs:
            legal, reason = can_play_bird(alice, bird, invalid_habs[0], game)
            assert not legal
            assert "cannot live" in reason

    def test_can_play_bird_not_in_hand(self, bird_reg, game, alice):
        bird = bird_reg.get("Trumpeter Swan")
        legal, reason = can_play_bird(alice, bird, Habitat.WETLAND, game)
        assert not legal
        assert "not in hand" in reason

    def test_can_play_bird_no_cubes(self, bird_reg, game, alice):
        bird = bird_reg.get("American Crow")
        alice.hand.append(bird)
        alice.food_supply.add(FoodType.SEED, 5)
        alice.action_cubes_remaining = 0
        legal, reason = can_play_bird(alice, bird, Habitat.FOREST, game)
        assert not legal
        assert "cubes" in reason.lower()

    def test_can_pay_or_cost(self, bird_reg, alice):
        bird = bird_reg.get("American Robin")
        # OR cost: invertebrate or fruit
        alice.food_supply.add(FoodType.FRUIT, 1)
        can_pay, _ = can_pay_food_cost(alice, bird.food_cost)
        assert can_pay

    def test_cant_pay_or_cost(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Robin")
        # No invertebrate or fruit
        player.food_supply.add(FoodType.FISH, 5)
        can_pay, _ = can_pay_food_cost(player, bird.food_cost)
        assert not can_pay

    def test_find_food_payment_options_or(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Robin")
        player.food_supply.add(FoodType.INVERTEBRATE, 1)
        player.food_supply.add(FoodType.FRUIT, 1)
        options = find_food_payment_options(player, bird.food_cost)
        assert len(options) == 2  # Can pay with either

    def test_wild_food_payment(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Crow")  # 1 wild
        player.food_supply.add(FoodType.FISH, 1)
        options = find_food_payment_options(player, bird.food_cost)
        assert len(options) >= 1
        assert any(FoodType.FISH in opt for opt in options)


# --- Actions ---

class TestActions:
    def test_play_bird_success(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("American Crow")
        player.hand.append(bird)
        player.food_supply.add(FoodType.SEED, 3)

        result = execute_play_bird(
            game, player, bird, Habitat.FOREST,
            food_payment={FoodType.SEED: 1},
        )
        assert result.success, result.message
        assert player.board.forest.bird_count == 1
        assert player.board.forest.slots[0].bird.name == "American Crow"
        assert not player.has_bird_in_hand("American Crow")
        assert player.food_supply.get(FoodType.SEED) == 2

    def test_play_bird_with_egg_cost(self, bird_reg, game):
        player = game.players[0]
        # Fill first 2 slots so the next bird goes in column 2 (egg cost = 1)
        filler = bird_reg.get("American Crow")
        for i in range(2):
            player.board.forest.slots[i].bird = filler

        # Put an egg on a bird to pay with
        player.board.forest.slots[0].eggs = 1

        # Play a bird that can go in forest
        bird = bird_reg.get("Acorn Woodpecker")
        if Habitat.FOREST not in bird.habitats:
            pytest.skip("Acorn Woodpecker can't go in forest")

        player.hand.append(bird)
        player.food_supply.add(FoodType.SEED, 5)
        player.food_supply.add(FoodType.FRUIT, 5)

        result = execute_play_bird(
            game, player, bird, Habitat.FOREST,
            food_payment={FoodType.SEED: 1, FoodType.FRUIT: 1},
        )
        assert result.success, result.message
        assert player.board.forest.bird_count == 3
        # Egg should have been removed
        assert player.board.forest.slots[0].eggs == 0

    def test_gain_food(self, game):
        player = game.players[0]
        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
            FoodType.INVERTEBRATE, FoodType.RODENT,
        ])

        result = execute_gain_food(
            game, player,
            food_choices=[FoodType.SEED],
        )
        assert result.success
        assert player.food_supply.get(FoodType.SEED) == 1
        assert game.birdfeeder.count == 4

    def test_gain_food_multiple(self, bird_reg, game):
        player = game.players[0]
        # Put birds in forest so we gain more food
        filler = bird_reg.get("American Crow")
        player.board.forest.slots[0].bird = filler
        player.board.forest.slots[1].bird = filler

        game.birdfeeder.set_dice([
            FoodType.SEED, FoodType.FISH, FoodType.FRUIT,
            FoodType.INVERTEBRATE, FoodType.RODENT,
        ])

        # 2 birds = FOREST_FOOD_GAIN[2] = 1 food
        result = execute_gain_food(
            game, player,
            food_choices=[FoodType.SEED, FoodType.FISH],
        )
        assert result.success

    def test_lay_eggs(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("Trumpeter Swan")  # High egg limit
        player.board.grassland.slots[0].bird = bird

        result = execute_lay_eggs(
            game, player,
            egg_distribution={(Habitat.GRASSLAND, 0): 2},
        )
        assert result.success
        assert player.board.grassland.slots[0].eggs == 2

    def test_lay_eggs_respects_limit(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = bird
        limit = bird.egg_limit

        result = execute_lay_eggs(
            game, player,
            egg_distribution={(Habitat.WETLAND, 0): 99},
        )
        assert result.success
        # Should be capped at egg limit (or available eggs to lay)
        assert player.board.wetland.slots[0].eggs <= limit

    def test_draw_cards_from_tray(self, bird_reg, game):
        player = game.players[0]
        tray_bird = bird_reg.get("Acorn Woodpecker")
        game.card_tray.face_up = [tray_bird]

        result = execute_draw_cards(
            game, player,
            from_tray_indices=[0],
        )
        assert result.success
        assert player.has_bird_in_hand("Acorn Woodpecker")
        assert game.card_tray.count == 0


# --- Scoring ---

class TestScoring:
    def test_round_goal_scored_on_advance_round(self, bird_reg, goal_reg):
        # Pick a simple bird-count goal if available.
        goal = next((g for g in goal_reg.all_goals if "[bird] in [forest]" in g.description.lower()), None)
        if goal is None:
            pytest.skip("No simple forest bird count goal found")

        game = create_new_game(["Alice", "Bob"], round_goals=[goal])
        alice = game.players[0]
        bob = game.players[1]
        forest_birds = [b for b in bird_reg.all_birds if Habitat.FOREST in b.habitats]
        if len(forest_birds) < 3:
            pytest.skip("Not enough forest birds to validate round-goal scoring")
        alice.board.forest.slots[0].bird = forest_birds[0]
        bob.board.forest.slots[0].bird = forest_birds[1]
        bob.board.forest.slots[1].bird = forest_birds[2]

        # End round 1
        game.advance_round()
        assert 1 in game.round_goal_scores
        assert game.round_goal_scores[1]["Bob"] >= game.round_goal_scores[1]["Alice"]

    def test_empty_board_score(self, game):
        player = game.players[0]
        breakdown = calculate_score(game, player)
        assert breakdown.total == 0

    def test_bird_vp_scoring(self, bird_reg, game):
        player = game.players[0]
        swan = bird_reg.get("Trumpeter Swan")  # 9 VP
        player.board.wetland.slots[0].bird = swan
        assert score_birds(player) == 9

    def test_egg_scoring(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("Trumpeter Swan")
        player.board.wetland.slots[0].bird = bird
        player.board.wetland.slots[0].eggs = 3
        assert score_eggs(player) == 3

    def test_cached_food_scoring(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        player.board.forest.slots[0].cache_food(FoodType.SEED, 4)
        assert score_cached_food(player) == 4

    def test_tucked_card_scoring(self, bird_reg, game):
        player = game.players[0]
        bird = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = bird
        player.board.forest.slots[0].tucked_cards = 5
        assert score_tucked_cards(player) == 5

    def test_full_score_breakdown(self, bird_reg, bonus_reg, game):
        player = game.players[0]

        # Place a bird with VP
        swan = bird_reg.get("Trumpeter Swan")  # 9 VP
        player.board.wetland.slots[0].bird = swan
        player.board.wetland.slots[0].eggs = 2

        woodpecker = bird_reg.get("Acorn Woodpecker")
        player.board.forest.slots[0].bird = woodpecker
        player.board.forest.slots[0].cache_food(FoodType.SEED, 3)
        player.board.forest.slots[0].tucked_cards = 1

        # Add goal scores
        game.round_goal_scores = {1: {"Alice": 3, "Bob": 1}}

        breakdown = calculate_score(game, player)
        assert breakdown.bird_vp == 9 + woodpecker.victory_points
        assert breakdown.eggs == 2
        assert breakdown.cached_food == 3
        assert breakdown.tucked_cards == 1
        assert breakdown.round_goals == 3
        assert breakdown.total == (9 + woodpecker.victory_points + 2 + 3 + 1 + 3)

    def test_bonus_card_scoring(self, bird_reg, bonus_reg, game):
        player = game.players[0]

        # Place several birds eligible for a bonus card
        bonus = bonus_reg.get("Bird Counter")  # 2 per flocking bird
        if bonus is None:
            pytest.skip("Bird Counter not found")
        player.bonus_cards.append(bonus)

        # Find some flocking birds
        flocking_birds = [b for b in bird_reg.all_birds if b.is_flocking]
        placed = 0
        for bird in flocking_birds[:3]:
            for hab in bird.habitats:
                row = player.board.get_row(hab)
                idx = row.next_empty_slot()
                if idx is not None:
                    row.slots[idx].bird = bird
                    placed += 1
                    break
            if placed >= 3:
                break

        # Score should reflect flocking birds
        bonus_score = score_bonus_cards(player)
        # Bird Counter: 2 per flocking bird. We placed up to 3.
        # But the bird also needs "Bird Counter" in its bonus_eligibility
        eligible = sum(
            1 for b in player.board.all_birds()
            if "Bird Counter" in b.bonus_eligibility
        )
        assert bonus_score == 2 * eligible

    def test_nectar_scoring_two_players(self, game):
        alice = game.players[0]
        bob = game.players[1]

        # Alice spent 3 nectar in forest, Bob spent 1
        alice.board.forest.nectar_spent = 3
        bob.board.forest.nectar_spent = 1

        breakdown_a = calculate_score(game, alice)
        breakdown_b = calculate_score(game, bob)
        assert breakdown_a.nectar == 5  # 1st place
        assert breakdown_b.nectar == 2  # 2nd place

    def test_nectar_scoring_tie(self, game):
        alice = game.players[0]
        bob = game.players[1]

        # Tied in forest
        alice.board.forest.nectar_spent = 2
        bob.board.forest.nectar_spent = 2

        breakdown_a = calculate_score(game, alice)
        breakdown_b = calculate_score(game, bob)
        # Tied for 1st: (5+2) // 2 = 3 each
        assert breakdown_a.nectar == 3
        assert breakdown_b.nectar == 3

    def test_nectar_zero_disqualifies(self, game):
        """Players with 0 nectar spent don't qualify for any placement."""
        alice = game.players[0]
        bob = game.players[1]

        # Alice spent 1 nectar, Bob spent 0
        alice.board.forest.nectar_spent = 1
        bob.board.forest.nectar_spent = 0

        breakdown_a = calculate_score(game, alice)
        breakdown_b = calculate_score(game, bob)
        assert breakdown_a.nectar == 5  # 1st place (only qualifier)
        assert breakdown_b.nectar == 0  # 0 nectar = no points
