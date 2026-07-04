"""Tests for the screenshot import pipeline (matching/merge + API route).

The Claude vision call itself is never made here: `build_proposed_state` is a
pure function, and the route tests monkeypatch `extract_from_images`.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.main import app
from backend.vision.screenshot_import import (
    ExtractedDraft,
    ExtractedGameState,
    ScreenshotImportError,
    XBoardBird,
    XFoodSupply,
    XHabitatRow,
    XPlayer,
    build_draft_proposal,
    build_proposed_state,
    decode_and_check_image,
)


def setup_module():
    load_all(EXCEL_FILE)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestBuildProposedStateFresh:
    def test_board_hand_food_from_scratch(self):
        extracted = ExtractedGameState(
            players=[
                XPlayer(
                    name="Evan",
                    board=[
                        XHabitatRow(
                            habitat="forest",
                            birds=[
                                XBoardBird(
                                    bird_name="Hooded Warbler",
                                    eggs=2,
                                    cached_food=["worm", "invertebrate"],
                                    tucked_cards=1,
                                ),
                            ],
                        ),
                    ],
                    food_supply=XFoodSupply(seed=3, fish=1),
                    hand=["Blue-gray Gnatcatcher"],  # case differs from registry
                    action_cubes_remaining=6,
                ),
            ],
            birdfeeder_dice=["seed", "seed/invertebrate", "fish"],
            current_round=2,
            turn_in_round=3,
            deck_remaining=100,
        )

        proposed, warnings = build_proposed_state(extracted, current=None)
        player = proposed.players[0]
        assert player.name == "Evan"

        forest = player.board[0]
        assert forest.habitat == "forest"
        slot = forest.slots[0]
        assert slot.bird_name == "Hooded Warbler"
        assert slot.eggs == 2
        assert slot.tucked_cards == 1
        # Both "worm" and "invertebrate" count as invertebrate tokens.
        assert slot.cached_food == {"invertebrate": 2}
        # Slot metadata was filled from the registry for the UI.
        assert slot.egg_limit > 0
        # Remaining forest slots stay empty.
        assert all(s.bird_name is None for s in forest.slots[1:])

        assert player.food_supply.seed == 3
        assert player.food_supply.fish == 1
        assert player.hand == ["Blue-Gray Gnatcatcher"]
        assert player.action_cubes_remaining == 6

        assert proposed.birdfeeder.dice == ["seed", ["seed", "invertebrate"], "fish"]
        assert proposed.current_round == 2
        assert proposed.turn_in_round == 3
        assert proposed.deck_remaining == 100
        # Exact-case fuzz on Blue-Gray produces no warning noise.
        assert warnings == []

    def test_misread_names_are_fuzzy_matched_with_warning(self):
        extracted = ExtractedGameState(
            players=[XPlayer(hand=["Hoodd Warbler", "Definitely Not A Bird"])],
        )
        proposed, warnings = build_proposed_state(extracted, current=None)
        assert proposed.players[0].hand == ["Hooded Warbler"]
        assert any("Hoodd Warbler" in w and "Hooded Warbler" in w for w in warnings)
        assert any("Definitely Not A Bird" in w for w in warnings)

    def test_eggs_clamped_to_limit(self):
        extracted = ExtractedGameState(
            players=[
                XPlayer(
                    board=[
                        XHabitatRow(
                            habitat="grassland",
                            birds=[XBoardBird(bird_name="Hooded Warbler", eggs=99)],
                        ),
                    ],
                ),
            ],
        )
        proposed, warnings = build_proposed_state(extracted, current=None)
        slot = proposed.players[0].board[1].slots[0]
        assert slot.eggs == slot.egg_limit
        assert any("clamped" in w for w in warnings)


class TestBuildProposedStateMerge:
    def _current_state(self, client):
        resp = client.post(
            "/api/games",
            json={"player_names": ["Alice", "Bob"], "board_type": "oceania"},
        )
        assert resp.status_code == 201
        from backend.api.schemas import GameStateSchema

        return GameStateSchema(**resp.json()["state"])

    def test_null_fields_keep_current_values(self, client):
        current = self._current_state(client)
        current.players[1].food_supply.rodent = 4
        tray_before = list(current.card_tray.face_up)

        extracted = ExtractedGameState(
            players=[XPlayer(name="alice", food_supply=XFoodSupply(seed=2))],
        )
        proposed, _ = build_proposed_state(extracted, current=current)
        # Name-matched onto Alice (player 0).
        assert proposed.players[0].food_supply.seed == 2
        # Everything not extracted is untouched.
        assert proposed.players[1].food_supply.rodent == 4
        assert proposed.card_tray.face_up == tray_before
        assert proposed.board_type == "oceania"

    def test_target_player_idx_routes_single_player_extraction(self, client):
        current = self._current_state(client)
        extracted = ExtractedGameState(
            players=[XPlayer(food_supply=XFoodSupply(fruit=5))],
        )
        proposed, _ = build_proposed_state(
            extracted, current=current, target_player_idx=1
        )
        assert proposed.players[1].food_supply.fruit == 5
        assert proposed.players[0].food_supply.fruit != 5

    def test_extra_players_ignored_with_warning(self, client):
        current = self._current_state(client)
        extracted = ExtractedGameState(
            players=[XPlayer(), XPlayer(), XPlayer(name="Ghost")],
        )
        _, warnings = build_proposed_state(extracted, current=current)
        assert any("more players" in w.lower() for w in warnings)


class TestBuildDraftProposal:
    def test_matches_dedupes_and_caps(self):
        extracted = ExtractedDraft(
            dealt_birds=[
                "Garden Warbler",
                "Pheasant Coucal",
                "Rose-breasted Grosbeak",  # case differs from registry
                "Eleonora's Falcon",
                "Pesquet's Parrot",
                "Garden Warbler",  # duplicate reading
            ],
            bonus_cards=["Rodentologist", "Not A Real Bonus Card At All"],
        )
        draft, warnings = build_draft_proposal(extracted)
        assert draft.dealt_birds == [
            "Garden Warbler",
            "Pheasant Coucal",
            "Rose-Breasted Grosbeak",
            "Eleonora's Falcon",
            "Pesquet's Parrot",
        ]
        assert draft.bonus_cards == ["Rodentologist"]
        assert any("Not A Real Bonus Card" in w for w in warnings)
        # Goals default to No Goal when the tiles weren't read.
        assert draft.round_goals == ["No Goal"] * 4

    def test_goal_substring_fallback(self):
        from backend.data.registries import get_goal_registry

        # Take a real goal and feed a truncated icon-tile style reading.
        goal = get_goal_registry().all_goals[0]
        partial = goal.description[: max(6, len(goal.description) // 2)]
        extracted = ExtractedDraft(round_goals=[partial])
        draft, _ = build_draft_proposal(extracted)
        # Either matched to the full description or left as No Goal —
        # never passed through as unvalidated text.
        assert draft.round_goals[0] == goal.description or draft.round_goals[0] == "No Goal"


class TestImageValidation:
    def test_rejects_bad_media_type(self):
        with pytest.raises(ScreenshotImportError, match="Unsupported image type"):
            decode_and_check_image("image/bmp", base64.b64encode(b"x").decode())

    def test_rejects_invalid_base64(self):
        with pytest.raises(ScreenshotImportError, match="not valid base64"):
            decode_and_check_image("image/png", "!!not-base64!!")


class TestImportRoute:
    def _payload(self, **extra):
        img = base64.b64encode(b"fake-png-bytes").decode()
        return {"images": [{"media_type": "image/png", "data": img}], **extra}

    def test_route_returns_proposed_state_and_warnings(self, client, monkeypatch):
        def fake_extract(images, notes=None, current=None):
            assert len(images) == 1
            return ExtractedGameState(
                players=[XPlayer(hand=["Hooded Warbler"])],
                card_tray=["Hoodd Warbler"],
                current_round=3,
                uncertainties=["Feeder dice partially cut off"],
            )

        monkeypatch.setattr(
            "backend.api.routes_import.extract_from_images", fake_extract
        )
        resp = client.post("/api/import/screenshot", json=self._payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["proposed"]["players"][0]["hand"] == ["Hooded Warbler"]
        assert data["proposed"]["card_tray"]["face_up"] == ["Hooded Warbler"]
        assert data["proposed"]["current_round"] == 3
        assert data["uncertainties"] == ["Feeder dice partially cut off"]
        assert any("Hoodd Warbler" in w for w in data["warnings"])

    def test_route_proposed_state_round_trips_through_put(self, client, monkeypatch):
        """The proposed state must be accepted verbatim by PUT /state."""

        def fake_extract(images, notes=None, current=None):
            return ExtractedGameState(
                players=[
                    XPlayer(
                        board=[
                            XHabitatRow(
                                habitat="wetland",
                                birds=[XBoardBird(bird_name="Great Crested Grebe", eggs=1)],
                            ),
                        ],
                        food_supply=XFoodSupply(fish=2),
                    ),
                ],
                birdfeeder_dice=["fish", "seed/invertebrate"],
            )

        monkeypatch.setattr(
            "backend.api.routes_import.extract_from_images", fake_extract
        )
        game = client.post(
            "/api/games",
            json={"player_names": ["Alice", "Bob"], "board_type": "oceania"},
        ).json()

        resp = client.post(
            "/api/import/screenshot",
            json=self._payload(current_state=game["state"]),
        )
        assert resp.status_code == 200
        proposed = resp.json()["proposed"]

        put = client.put(f"/api/games/{game['game_id']}/state", json=proposed)
        assert put.status_code == 200
        saved = put.json()
        wetland = saved["players"][0]["board"][2]
        assert wetland["slots"][0]["bird_name"] == "Great Crested Grebe"
        assert saved["players"][0]["food_supply"]["fish"] == 2

    def test_draft_mode_returns_draft_proposal(self, client, monkeypatch):
        def fake_extract_draft(images, notes=None):
            return ExtractedDraft(
                dealt_birds=["Garden Warbler", "Pesquet's Parrot"],
                bonus_cards=["Rodentologist"],
                uncertainties=["Bonus card thumbnails too small to read fully"],
            )

        monkeypatch.setattr(
            "backend.api.routes_import.extract_draft_from_images", fake_extract_draft
        )
        resp = client.post(
            "/api/import/screenshot", json=self._payload(mode="draft")
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["proposed"] is None
        assert data["draft"]["dealt_birds"] == ["Garden Warbler", "Pesquet's Parrot"]
        assert data["draft"]["bonus_cards"] == ["Rodentologist"]
        assert data["uncertainties"] == ["Bonus card thumbnails too small to read fully"]

    def test_route_maps_import_error_to_400(self, client, monkeypatch):
        def fake_extract(images, notes=None, current=None):
            raise ScreenshotImportError("ANTHROPIC_API_KEY is not set.")

        monkeypatch.setattr(
            "backend.api.routes_import.extract_from_images", fake_extract
        )
        resp = client.post("/api/import/screenshot", json=self._payload())
        assert resp.status_code == 400
        assert "ANTHROPIC_API_KEY" in resp.json()["detail"]
