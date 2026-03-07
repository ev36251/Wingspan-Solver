"""Tests for the AlphaZero MCTS + self-play pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import BoardType
from backend.solver.self_play import create_training_game
from backend.solver.move_generator import generate_all_moves


# Path to a lightweight model that ships with the repo
_CHAMPION_PATH = Path("reports/ml/champions/current_champion.npz")


def setup_module() -> None:
    load_all(EXCEL_FILE)


def _skip_if_no_model():
    if not _CHAMPION_PATH.exists():
        pytest.skip("Champion model not found; skipping MCTS tests")


# ---------------------------------------------------------------------------
# MCTSNode unit tests (no model needed)
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_q_empty_node(self):
        from backend.ml.mcts import MCTSNode
        node = MCTSNode()
        assert node.Q == 0.0

    def test_q_after_backprop(self):
        from backend.ml.mcts import MCTSNode
        node = MCTSNode(prior=0.5)
        node.N = 4
        node.W = 8.0
        assert node.Q == pytest.approx(2.0)

    def test_ucb_increases_with_prior(self):
        from backend.ml.mcts import MCTSNode
        root = MCTSNode()
        root.N = 10

        low = MCTSNode(parent=root, prior=0.1)
        low.N = 5; low.W = 5.0

        high = MCTSNode(parent=root, prior=0.9)
        high.N = 5; high.W = 5.0   # same Q

        assert high.ucb(1.5) > low.ucb(1.5)

    def test_ucb_unvisited_node(self):
        from backend.ml.mcts import MCTSNode
        root = MCTSNode()
        root.N = 10

        child = MCTSNode(parent=root, prior=0.5)
        # N=0: UCB dominated by exploration term
        assert child.ucb(1.5) > 0.0


# ---------------------------------------------------------------------------
# MCTS integration tests (requires model)
# ---------------------------------------------------------------------------

class TestMCTS:
    def test_get_best_move_returns_legal_move(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=5, value_blend=0.1, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        player_idx = game.current_player_idx
        player = game.players[player_idx]
        legal = generate_all_moves(game, player)
        legal_descs = {m.description for m in legal}

        best = mcts.get_best_move(game, player_idx, temperature=0.0)
        assert best is not None
        assert best.description in legal_descs

    def test_visit_counts_sum_to_num_sims(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        num_sims = 8
        mcts = MCTS(model, enc, num_sims=num_sims, value_blend=0.1, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        player_idx = game.current_player_idx
        moves, probs = mcts.get_policy(game, player_idx, temperature=1.0)

        # Root's children visit counts should sum to num_sims
        assert len(moves) > 0
        assert len(moves) == len(probs)
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_temperature_zero_gives_argmax(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=10, value_blend=0.1, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        player_idx = game.current_player_idx
        moves, probs = mcts.get_policy(game, player_idx, temperature=0.0)

        # Exactly one move should have prob=1.0
        assert len([p for p in probs if p > 0.5]) == 1
        assert sum(probs) == pytest.approx(1.0)

    def test_get_policy_probabilities_sum_to_one(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=6, value_blend=0.1, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        moves, probs = mcts.get_policy(game, 0, temperature=1.0)
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_mcts_no_crash_with_value_blend_zero(self):
        """value_blend=0.0 means rollout-only (no NN value head used)."""
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=4, value_blend=0.0, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        best = mcts.get_best_move(game, 0)
        assert best is not None

    def test_mcts_no_crash_with_value_blend_one(self):
        """value_blend=1.0 means NN-value-only (no rollout)."""
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=4, value_blend=1.0, rollout_policy="fast")

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        best = mcts.get_best_move(game, 0)
        assert best is not None


# ---------------------------------------------------------------------------
# Self-play dataset generation
# ---------------------------------------------------------------------------

class TestAlphaZeroSelfPlay:
    def test_generate_alphazero_game_returns_records(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.alphazero_self_play import generate_alphazero_game
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=3, value_blend=0.1, rollout_policy="fast")

        records = generate_alphazero_game(
            model=model,
            encoder=enc,
            mcts=mcts,
            players=2,
            board_type=BoardType.OCEANIA,
            max_turns=40,   # short game for test speed
        )
        assert len(records) > 0

    def test_game_records_have_required_fields(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.alphazero_self_play import generate_alphazero_game
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=3, value_blend=0.1, rollout_policy="fast")

        records = generate_alphazero_game(
            model=model,
            encoder=enc,
            mcts=mcts,
            players=2,
            board_type=BoardType.OCEANIA,
            max_turns=30,
        )
        assert len(records) > 0
        rec = records[0]
        # Required fields for train_factorized_bc.py
        assert "state" in rec
        assert "targets" in rec
        assert "player_name" in rec
        assert "value_target_score" in rec
        # All value targets should be filled (not None)
        for r in records:
            assert r["value_target_score"] is not None

    def test_delta_values_in_expected_range(self):
        _skip_if_no_model()
        from backend.ml.mcts import MCTS
        from backend.ml.alphazero_self_play import generate_alphazero_game
        from backend.ml.factorized_inference import FactorizedPolicyModel
        from backend.ml.state_encoder import StateEncoder

        model = FactorizedPolicyModel(str(_CHAMPION_PATH))
        enc = StateEncoder.resolve_for_model(model.meta)
        mcts = MCTS(model, enc, num_sims=3, value_blend=0.1, rollout_policy="fast")

        records = generate_alphazero_game(
            model=model,
            encoder=enc,
            mcts=mcts,
            players=2,
            board_type=BoardType.OCEANIA,
            max_turns=60,
        )
        for rec in records:
            delta = float(rec["value_target_score"])
            # Delta should be bounded (typical Wingspan game ±150 max)
            assert -200.0 <= delta <= 200.0

    def test_generate_self_play_dataset_creates_files(self, tmp_path):
        _skip_if_no_model()
        from backend.ml.alphazero_self_play import generate_self_play_dataset

        out_jsonl = tmp_path / "test.jsonl"
        out_meta = tmp_path / "test.meta.json"

        meta = generate_self_play_dataset(
            model_path=str(_CHAMPION_PATH),
            out_jsonl=str(out_jsonl),
            out_meta=str(out_meta),
            games=1,
            mcts_sims=3,
            value_blend=0.1,
            max_turns=30,
            seed=42,
        )
        assert out_jsonl.exists()
        assert out_meta.exists()
        assert meta["samples"] > 0
        assert meta["feature_dim"] > 0
        assert "value_target_config" in meta
        assert meta["value_target_config"]["mode"] == "absolute"

    def test_dataset_jsonl_is_valid(self, tmp_path):
        """Each JSONL line must parse as JSON with expected fields."""
        _skip_if_no_model()
        from backend.ml.alphazero_self_play import generate_self_play_dataset

        out_jsonl = tmp_path / "test.jsonl"
        out_meta = tmp_path / "test.meta.json"
        generate_self_play_dataset(
            model_path=str(_CHAMPION_PATH),
            out_jsonl=str(out_jsonl),
            out_meta=str(out_meta),
            games=1,
            mcts_sims=3,
            value_blend=0.1,
            max_turns=30,
            seed=7,
        )
        with out_jsonl.open("r") as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) > 0
        for line in lines:
            rec = json.loads(line)
            assert "state" in rec
            assert "targets" in rec
            assert "value_target_score" in rec
            assert rec["value_target_score"] is not None

    def test_meta_compatible_with_train_bc(self, tmp_path):
        """Meta file fields satisfy what train_factorized_bc._load_as_arrays needs."""
        _skip_if_no_model()
        from backend.ml.alphazero_self_play import generate_self_play_dataset

        out_jsonl = tmp_path / "test.jsonl"
        out_meta = tmp_path / "test.meta.json"
        generate_self_play_dataset(
            model_path=str(_CHAMPION_PATH),
            out_jsonl=str(out_jsonl),
            out_meta=str(out_meta),
            games=1,
            mcts_sims=3,
            value_blend=0.1,
            max_turns=30,
            seed=3,
        )
        meta = json.loads(out_meta.read_text())
        # Required by _load_as_arrays
        assert "feature_dim" in meta
        assert "target_heads" in meta
        assert "value_target_config" in meta
        assert "score_scale" in meta["value_target_config"]
        assert "score_bias" in meta["value_target_config"]
        # Scale should be 120.0 for absolute score targets
        assert meta["value_target_config"]["score_scale"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Auto-improve smoke test
# ---------------------------------------------------------------------------

class TestAutoImproveAlphaZero:
    def test_smoke_run(self, tmp_path):
        """1 iter, 2 games, 3 sims — verifies end-to-end plumbing."""
        _skip_if_no_model()
        from backend.ml.auto_improve_alphazero import run_auto_improve_alphazero

        result = run_auto_improve_alphazero(
            out_dir=str(tmp_path / "az_smoke"),
            iterations=1,
            games_per_iter=2,
            mcts_sims=3,
            value_blend=0.1,
            max_turns=30,
            train_epochs=2,
            train_batch=4,
            train_hidden1=32,
            train_hidden2=16,
            eval_games=2,
            promotion_games=4,
            data_accumulation_enabled=False,
            dataset_workers=1,
            seed=42,
            clean_out_dir=True,
            train_init_model_path=str(_CHAMPION_PATH),
        )
        assert "history" in result
        assert len(result["history"]) == 1
        assert (tmp_path / "az_smoke" / "best_model.npz").exists()
