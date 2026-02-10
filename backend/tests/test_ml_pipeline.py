from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import ActionType, BoardType, FoodType, Habitat
from backend.solver.move_generator import Move
from backend.ml.action_codec import ActionCodec, action_signature
from backend.ml.state_encoder import StateEncoder
from backend.ml.self_play_dataset import generate_dataset
from backend.solver.self_play import create_training_game


def setup_module() -> None:
    load_all(EXCEL_FILE)


def test_action_signature_is_stable() -> None:
    m1 = Move(
        action_type=ActionType.PLAY_BIRD,
        description="x",
        bird_name="Tui",
        habitat=Habitat.FOREST,
        food_payment={FoodType.NECTAR: 2, FoodType.INVERTEBRATE: 1},
    )
    m2 = Move(
        action_type=ActionType.PLAY_BIRD,
        description="y",
        bird_name="Tui",
        habitat=Habitat.FOREST,
        food_payment={FoodType.INVERTEBRATE: 1, FoodType.NECTAR: 2},
    )
    assert action_signature(m1) == action_signature(m2)


def test_state_encoder_dim_matches_feature_names() -> None:
    game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
    enc = StateEncoder()
    vec = enc.encode(game, player_index=0)
    assert len(vec) == len(enc.feature_names())


def test_generate_dataset_smoke(tmp_path: Path) -> None:
    out = tmp_path / "dataset.jsonl"
    meta = tmp_path / "dataset.meta.json"
    info = generate_dataset(
        output_jsonl=str(out),
        metadata_path=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=7,
    )
    assert out.exists()
    assert meta.exists()
    assert info["samples"] > 0
    assert info["feature_dim"] > 0
    assert info["action_space"]["num_actions"] > 0
    assert info["strict_rules_only"] is True
    assert info["reject_non_strict_powers"] is True

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == info["samples"]

    assert len(lines) > 0
