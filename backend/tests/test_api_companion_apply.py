"""Tests for engine-backed move application via the action API (companion mode).

Companion mode is how the frontend "apply move" works: the engine executes the
full action (powers, cube spend, turn advancement) but never invents hidden
card identities — deck draws become face-down counts and the tray is left
short for the user to enter the real revealed cards.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _fresh_game(client, board_type="base", players=("Alice", "Bob")):
    resp = client.post(
        "/api/games",
        json={"player_names": list(players), "board_type": board_type},
    )
    assert resp.status_code == 201
    data = resp.json()
    return data["game_id"], data["state"]


def _put_state(client, game_id, state):
    resp = client.put(f"/api/games/{game_id}/state", json=state)
    assert resp.status_code == 200
    return resp.json()


class TestPlayBirdApply:
    def test_play_bird_engine_applies_and_advances_turn(self, client):
        game_id, state = _fresh_game(client)
        state["players"][0]["hand"] = ["Hooded Warbler"]
        state["players"][0]["food_supply"]["invertebrate"] = 3
        state["current_player_idx"] = 0
        state = _put_state(client, game_id, state)
        cubes_before = state["players"][0]["action_cubes_remaining"]

        resp = client.post(
            f"/api/games/{game_id}/play-bird",
            json={
                "bird_name": "Hooded Warbler",
                "habitat": "forest",
                "food_payment": {"invertebrate": 2},
                "companion": True,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"], result["message"]
        assert result["state"] is not None

        new_state = result["state"]
        alice = new_state["players"][0]
        assert alice["board"][0]["slots"][0]["bird_name"] == "Hooded Warbler"
        assert alice["hand"] == []
        assert alice["food_supply"]["invertebrate"] == 1
        # Engine spent the cube and rotated to Bob.
        assert alice["action_cubes_remaining"] == cubes_before - 1
        assert new_state["current_player_idx"] == 1

    def test_play_bird_nectar_payment_records_nectar_spent(self, client):
        game_id, state = _fresh_game(client, board_type="oceania")
        state["players"][0]["hand"] = ["Hooded Warbler"]
        state["players"][0]["food_supply"]["nectar"] = 2
        state["current_player_idx"] = 0
        state = _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/play-bird",
            json={
                "bird_name": "Hooded Warbler",
                "habitat": "forest",
                "food_payment": {"nectar": 2},
                "companion": True,
            },
        )
        result = resp.json()
        assert result["success"], result["message"]
        row = result["state"]["players"][0]["board"][0]
        assert row["nectar_spent"] == 2
        assert result["state"]["players"][0]["food_supply"]["nectar"] == 0

    def test_failed_play_reports_error_without_advancing(self, client):
        game_id, state = _fresh_game(client)
        state["players"][0]["hand"] = ["Hooded Warbler"]
        # No food to pay with.
        state["current_player_idx"] = 0
        _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/play-bird",
            json={
                "bird_name": "Hooded Warbler",
                "habitat": "forest",
                "food_payment": {"invertebrate": 2},
                "companion": True,
            },
        )
        result = resp.json()
        assert not result["success"]
        assert result["state"] is None
        # Turn did not advance.
        check = client.get(f"/api/games/{game_id}").json()
        assert check["current_player_idx"] == 0
        assert check["players"][0]["hand"] == ["Hooded Warbler"]


class TestGainFoodApply:
    def test_brown_power_fires_and_is_reported(self, client):
        game_id, state = _fresh_game(client)
        p0 = state["players"][0]
        p0["board"][0]["slots"][0]["bird_name"] = "Blue-Gray Gnatcatcher"
        state["birdfeeder"]["dice"] = ["seed", "seed", "fish", "invertebrate", "rodent"]
        state["current_player_idx"] = 0
        state = _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/gain-food",
            json={"food_choices": ["seed"], "companion": True},
        )
        result = resp.json()
        assert result["success"], result["message"]
        alice = result["state"]["players"][0]
        # Feeder seed taken...
        assert alice["food_supply"]["seed"] == 1
        # ...and the Gnatcatcher's brown power gained 1 invertebrate from supply.
        assert alice["food_supply"]["invertebrate"] == 1
        assert any("Blue-Gray Gnatcatcher" in ev for ev in result["power_events"])
        # The taken die left the feeder.
        assert result["state"]["birdfeeder"]["dice"].count("seed") == 1


class TestLayEggsApply:
    def test_multi_slot_same_habitat_distribution(self, client):
        game_id, state = _fresh_game(client)
        p0 = state["players"][0]
        p0["board"][0]["slots"][0]["bird_name"] = "Hooded Warbler"
        p0["board"][0]["slots"][1]["bird_name"] = "Blue-Winged Warbler"
        state["current_player_idx"] = 0
        state = _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/lay-eggs",
            json={
                "egg_distribution": {"forest:0": 1, "forest:1": 1},
                "companion": True,
            },
        )
        result = resp.json()
        assert result["success"], result["message"]
        assert result["eggs_laid"] == 2
        slots = result["state"]["players"][0]["board"][0]["slots"]
        assert slots[0]["eggs"] == 1
        assert slots[1]["eggs"] == 1


class TestDrawCardsApply:
    def test_companion_tray_take_is_known_and_tray_not_refilled(self, client):
        game_id, state = _fresh_game(client)
        tray_before = list(state["card_tray"]["face_up"])
        assert len(tray_before) == 3
        state["current_player_idx"] = 0
        state = _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/draw-cards",
            json={"from_tray_indices": [0], "companion": True},
        )
        result = resp.json()
        assert result["success"], result["message"]
        alice = result["state"]["players"][0]
        # Took the real face-up card into hand.
        assert alice["hand"] == [tray_before[0]]
        assert alice["unknown_hand_count"] == 0
        # Companion mode: tray is NOT refilled with an invented card; the user
        # enters the actually revealed card.
        assert result["state"]["card_tray"]["face_up"] == tray_before[1:]

    def test_companion_deck_draw_becomes_unknown_count(self, client):
        game_id, state = _fresh_game(client)
        state["current_player_idx"] = 0
        state["deck_remaining"] = 40
        state = _put_state(client, game_id, state)
        deck_before = state["deck_remaining"]

        resp = client.post(
            f"/api/games/{game_id}/draw-cards",
            json={"from_deck_count": 1, "companion": True},
        )
        result = resp.json()
        assert result["success"], result["message"]
        alice = result["state"]["players"][0]
        # No invented identity: the draw is tracked face-down.
        assert alice["hand"] == []
        assert alice["unknown_hand_count"] == 1
        assert result["state"]["deck_remaining"] == deck_before - 1

    def test_simulated_mode_still_draws_identities_and_refills(self, client):
        game_id, state = _fresh_game(client)
        tray_before = list(state["card_tray"]["face_up"])
        state["current_player_idx"] = 0
        _put_state(client, game_id, state)

        resp = client.post(
            f"/api/games/{game_id}/draw-cards",
            json={"from_tray_indices": [0], "companion": False},
        )
        result = resp.json()
        assert result["success"], result["message"]
        alice = result["state"]["players"][0]
        assert alice["hand"] == [tray_before[0]]
        assert alice["unknown_hand_count"] == 0
        # Non-companion (simulated) games refill the tray from the deck model.
        assert len(result["state"]["card_tray"]["face_up"]) == 3


class TestSolverDetailSerialization:
    def test_egg_distribution_merges_slots_in_same_habitat(self):
        from backend.api.routes_solver import _egg_distribution_to_details
        from backend.models.enums import Habitat

        dist = _egg_distribution_to_details({
            (Habitat.FOREST, 0): 1,
            (Habitat.FOREST, 1): 2,
            (Habitat.WETLAND, 4): 1,
        })
        assert dist == {"forest": {"0": 1, "1": 2}, "wetland": {"4": 1}}
