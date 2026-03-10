"""Tests for the core game engine: rules, actions, and scoring."""

import pytest
from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import (
    FoodType,
    Habitat,
    PowerColor,
    GameSet,
    BeakDirection,
    NestType,
)
from backend.models.board import PlayerBoard, HabitatRow, BirdSlot
from backend.models.bird import Bird, FoodCost
from backend.models.birdfeeder import Birdfeeder
from backend.models.card_tray import CardTray
from backend.models.goal import Goal
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
        # No invertebrate, no fruit, and only 1 fish (not enough for 2-for-1)
        player.food_supply.add(FoodType.FISH, 1)
        can_pay, _ = can_pay_food_cost(player, bird.food_cost)
        assert not can_pay

    def test_can_pay_or_cost_via_two_for_one(self, bird_reg):
        player = Player(name="Test")
        bird = bird_reg.get("American Robin")
        # No invertebrate or fruit, but 2+ fish allows 2-for-1 substitution
        player.food_supply.add(FoodType.FISH, 2)
        can_pay, _ = can_pay_food_cost(player, bird.food_cost)
        assert can_pay

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

    def test_play_on_top_discards_eggs_food_and_keeps_prior_tucked_plus_one(self, bird_reg, game):
        player = game.players[0]
        red_kite = bird_reg.get("Red Kite")
        covered = bird_reg.get("American Crow")
        assert red_kite is not None and covered is not None
        habitat = next(iter(red_kite.habitats))

        slot = player.board.get_row(habitat).slots[0]
        slot.bird = covered
        slot.eggs = 2
        slot.cached_food = {FoodType.FISH: 2}
        slot.tucked_cards = 3

        player.hand.append(red_kite)
        fish_before = player.food_supply.get(FoodType.FISH)

        result = execute_play_bird(
            game,
            player,
            red_kite,
            habitat,
            food_payment={},
            target_slot=0,
            play_on_top=True,
        )
        assert result.success, result.message
        assert slot.bird.name == "Red Kite"
        assert slot.eggs == 0
        assert slot.cached_food == {}
        assert slot.tucked_cards == 4  # prior tucked cards remain; covered bird becomes tucked
        assert player.food_supply.get(FoodType.FISH) == fish_before
        assert any(
            act.bird_name == "Red Kite" and act.result.cards_tucked >= 1
            for act in result.power_activations
        )

    def test_play_on_top_predator_counts_as_tuck_and_predator_success_for_pink(self, bird_reg):
        game = create_new_game(["Alice", "Bob"])
        actor = game.players[0]
        opponent = game.players[1]

        red_kite = bird_reg.get("Red Kite")
        covered = bird_reg.get("American Crow")
        pink_on_tuck = bird_reg.get("European Goldfinch")
        pink_on_predator = bird_reg.get("Black Vulture")
        deck_card = bird_reg.get("Spotted Dove")
        assert all(x is not None for x in (red_kite, covered, pink_on_tuck, pink_on_predator, deck_card))

        # Actor setup for play-on-top.
        habitat = next(iter(red_kite.habitats))
        actor_slot = actor.board.get_row(habitat).slots[0]
        actor_slot.bird = covered
        actor.hand.append(red_kite)

        # Opponent setup: one pink bird that reacts to "another player tucks",
        # and one that reacts to "another player's predator succeeds".
        def _place_any(p: Player, b: Bird) -> tuple[Habitat, int]:
            for h in b.habitats:
                row = p.board.get_row(h)
                for idx, s in enumerate(row.slots):
                    if s.bird is None:
                        s.bird = b
                        return h, idx
            raise AssertionError(f"No open slot for {b.name}")

        tuck_hab, tuck_idx = _place_any(opponent, pink_on_tuck)
        _place_any(opponent, pink_on_predator)

        # Ensure tuck-trigger pink can draw from deck and predator-success pink can take a die.
        game._deck_cards = [deck_card]  # type: ignore[attr-defined]
        game.deck_remaining = 1
        game.birdfeeder.set_dice([FoodType.FISH, FoodType.SEED, FoodType.FRUIT])

        opponent_tucked_before = opponent.board.get_row(tuck_hab).slots[tuck_idx].tucked_cards
        opponent_food_before = opponent.food_supply.total()

        result = execute_play_bird(
            game,
            actor,
            red_kite,
            habitat,
            food_payment={},
            target_slot=0,
            play_on_top=True,
        )

        assert result.success, result.message
        # European Goldfinch (pink on opponent tuck) should tuck 1 from deck.
        assert opponent.board.get_row(tuck_hab).slots[tuck_idx].tucked_cards == opponent_tucked_before + 1
        # Black Vulture (pink on predator success) should gain one feeder die.
        assert opponent.food_supply.total() == opponent_food_before + 1
        assert game.birdfeeder.count == 2

    def test_eastern_imperial_eagle_tuck_payment_triggers_pink_tuck_and_predator_success(self, bird_reg):
        game = create_new_game(["Alice", "Bob"])
        actor = game.players[0]
        opponent = game.players[1]

        eagle = bird_reg.get("Eastern Imperial Eagle")
        tuck_a = bird_reg.get("Spotted Dove")
        tuck_b = bird_reg.get("American Crow")
        pink_on_tuck = bird_reg.get("European Goldfinch")
        pink_on_predator = bird_reg.get("Black Vulture")
        deck_card = bird_reg.get("Trumpeter Swan")
        assert all(x is not None for x in (eagle, tuck_a, tuck_b, pink_on_tuck, pink_on_predator, deck_card))

        # Actor setup: pay part of rodent cost with rodent, rest with hand tucks.
        habitat = next(iter(eagle.habitats))
        actor.hand = [eagle, tuck_a, tuck_b]
        actor.food_supply.add(FoodType.RODENT, 3)

        def _place_any(p: Player, b: Bird) -> tuple[Habitat, int]:
            for h in b.habitats:
                row = p.board.get_row(h)
                for idx, s in enumerate(row.slots):
                    if s.bird is None:
                        s.bird = b
                        return h, idx
            raise AssertionError(f"No open slot for {b.name}")

        tuck_hab, tuck_idx = _place_any(opponent, pink_on_tuck)
        _place_any(opponent, pink_on_predator)

        # Ensure pink powers have resources to resolve.
        game._deck_cards = [deck_card]  # type: ignore[attr-defined]
        game.deck_remaining = 1
        game.birdfeeder.set_dice([FoodType.FISH, FoodType.SEED, FoodType.FRUIT])

        opponent_tucked_before = opponent.board.get_row(tuck_hab).slots[tuck_idx].tucked_cards
        opponent_food_before = opponent.food_supply.total()

        result = execute_play_bird(
            game,
            actor,
            eagle,
            habitat,
            food_payment={FoodType.RODENT: 1},
            hand_tuck_payment=2,
        )

        assert result.success, result.message
        assert actor.board.get_row(habitat).slots[0].bird is not None
        assert actor.board.get_row(habitat).slots[0].bird.name == "Eastern Imperial Eagle"
        assert actor.board.get_row(habitat).slots[0].tucked_cards == 2
        assert any(
            act.bird_name == "Eastern Imperial Eagle" and act.result.cards_tucked == 2
            for act in result.power_activations
        )
        # European Goldfinch (pink on opponent tuck) should tuck 1 from deck.
        assert opponent.board.get_row(tuck_hab).slots[tuck_idx].tucked_cards == opponent_tucked_before + 1
        # Black Vulture (pink on predator success) should gain one feeder die.
        assert opponent.food_supply.total() == opponent_food_before + 1
        assert game.birdfeeder.count == 2

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

    def test_round_end_powers_resolve_before_round_goal_scoring(self):
        goal = Goal(
            description="[Egg] in [Ground]",
            game_set=GameSet.CORE,
            scoring=(0, 1, 3, 5),
            reverse_description="[Egg] on [Platform] [Bird]",
        )
        game = create_new_game(["Alice", "Bob"], round_goals=[goal])
        alice = game.players[0]
        bob = game.players[1]

        teal = Bird(
            name="Teal Round Layer",
            scientific_name="Teal Round Layer",
            game_set=GameSet.EUROPEAN,
            color=PowerColor.TEAL,
            power_text="At end of round, lay 1 [egg] on each of your [ground] nesting birds.",
            victory_points=2,
            nest_type=NestType.BOWL,
            egg_limit=4,
            wingspan_cm=20,
            habitats=frozenset({Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND}),
            food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
            beak_direction=BeakDirection.LEFT,
            is_predator=False,
            is_flocking=False,
            is_bonus_card_bird=False,
            bonus_eligibility=frozenset(),
        )
        ground = Bird(
            name="Ground Target",
            scientific_name="Ground Target",
            game_set=GameSet.CORE,
            color=PowerColor.NONE,
            power_text="",
            victory_points=1,
            nest_type=NestType.GROUND,
            egg_limit=4,
            wingspan_cm=25,
            habitats=frozenset({Habitat.GRASSLAND}),
            food_cost=FoodCost(items=(FoodType.SEED,), is_or=False, total=1),
            beak_direction=BeakDirection.LEFT,
            is_predator=False,
            is_flocking=False,
            is_bonus_card_bird=False,
            bonus_eligibility=frozenset(),
        )

        alice.board.forest.slots[0].bird = teal
        alice.board.grassland.slots[0].bird = ground
        alice.board.grassland.slots[0].eggs = 0
        bob.board.grassland.slots[0].bird = ground
        bob.board.grassland.slots[0].eggs = 0

        game.advance_round()

        assert alice.board.grassland.slots[0].eggs == 1
        assert bob.board.grassland.slots[0].eggs == 0
        assert 1 in game.round_goal_scores
        assert game.round_goal_scores[1]["Alice"] > game.round_goal_scores[1]["Bob"]

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
