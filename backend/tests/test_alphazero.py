"""Tests for the AlphaZero MCTS + self-play pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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

    def test_evaluate_terminal_uses_absolute_score_units(self, monkeypatch):
        from backend.ml.mcts import MCTS

        class DummyModel:
            has_value_head = False

        class DummyEncoder:
            def encode(self, game, player_idx):
                return [0.0]

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        game.current_round = 5  # game-over sentinel
        p0 = game.players[0].name
        p1 = game.players[1].name

        def _fake_calc(_game, player):
            return SimpleNamespace(total=72 if player.name == p0 else 61)

        monkeypatch.setattr("backend.ml.mcts.calculate_score", _fake_calc)

        mcts = MCTS(DummyModel(), DummyEncoder(), num_sims=1, value_blend=0.0, rollout_policy="fast")
        val = mcts._evaluate(game, 0)
        assert val == pytest.approx(72.0)

    def test_evaluate_rollout_uses_absolute_score_units(self, monkeypatch):
        from backend.ml.mcts import MCTS

        class DummyModel:
            has_value_head = False

        class DummyEncoder:
            def encode(self, game, player_idx):
                return [0.0]

        game = create_training_game(num_players=2, board_type=BoardType.OCEANIA)
        p0 = game.players[0].name
        p1 = game.players[1].name

        monkeypatch.setattr(
            "backend.ml.mcts.simulate_playout",
            lambda _sim, rollout_policy="fast": {p0: 64.0, p1: 58.0},
        )

        mcts = MCTS(DummyModel(), DummyEncoder(), num_sims=1, value_blend=0.0, rollout_policy="fast")
        val = mcts._evaluate(game, 0)
        assert val == pytest.approx(64.0)


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

class TestAutoImproveShardMerge:
    def test_merge_az_shards_uses_weighted_stats_without_expansion(self, tmp_path):
        from backend.ml.auto_improve_alphazero import _merge_az_shards

        shard1 = tmp_path / "s1.jsonl"
        shard2 = tmp_path / "s2.jsonl"
        shard1.write_text('{"x":1}\n\n{"x":2}\n', encoding="utf-8")
        shard2.write_text('{"x":3}\n', encoding="utf-8")

        shard_results = [
            {
                "jsonl": str(shard1),
                "meta": {
                    "games": 1,
                    "samples": 2,
                    "elapsed_sec": 1.5,
                    "mean_score": 10.0,
                    "std_score": 2.0,
                    "mean_delta": 10.0,
                    "std_delta": 2.0,
                },
            },
            {
                "jsonl": str(shard2),
                "meta": {
                    "games": 2,
                    "samples": 6,
                    "elapsed_sec": 2.5,
                    "mean_score": 20.0,
                    "std_score": 4.0,
                    "mean_delta": 20.0,
                    "std_delta": 4.0,
                },
            },
        ]

        out_jsonl = tmp_path / "merged.jsonl"
        out_meta = tmp_path / "merged.meta.json"
        merged = _merge_az_shards(shard_results, out_jsonl=out_jsonl, out_meta=out_meta)

        assert merged["games"] == 3
        assert merged["samples"] == 8
        assert merged["elapsed_sec"] == pytest.approx(4.0)
        assert merged["mean_score"] == pytest.approx(17.5, abs=1e-3)
        assert merged["std_score"] == pytest.approx(5.635, abs=1e-3)
        assert merged["mean_delta"] == pytest.approx(17.5, abs=1e-3)
        assert merged["std_delta"] == pytest.approx(5.635, abs=1e-3)

        lines = [ln for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 3
        meta_json = json.loads(out_meta.read_text(encoding="utf-8"))
        assert meta_json["mean_score"] == pytest.approx(17.5, abs=1e-3)


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
        rec = result["history"][0]
        assert "timing_sec" in rec
        for k in ("data_gen", "train", "eval", "gate", "iter_total"):
            assert k in rec["timing_sec"]
            assert isinstance(rec["timing_sec"][k], (int, float))
            assert rec["timing_sec"][k] >= 0.0
        assert (tmp_path / "az_smoke" / "best_model.npz").exists()


class TestAutoImprovePreflightGates:
    def test_bird_audit_gate_raises_when_semantic_untested_nonzero(self, monkeypatch):
        from backend.ml import auto_improve_alphazero as az

        monkeypatch.setattr(
            az,
            "_run_bird_audit_gate",
            lambda expected_semantic_untested=0: (_ for _ in ()).throw(
                RuntimeError("Bird semantic audit gate failed")
            ),
        )

        with pytest.raises(RuntimeError, match="Bird semantic audit gate failed"):
            az.run_auto_improve_alphazero(
                out_dir="reports/ml/az_gate_fail_test",
                iterations=1,
                games_per_iter=1,
                mcts_sims=1,
                eval_games=1,
                promotion_games=1,
                train_epochs=1,
                train_batch=2,
                train_hidden1=16,
                train_hidden2=8,
                dataset_workers=1,
                require_bird_audit_gate=True,
                clean_out_dir=True,
            )

    def test_stale_lineage_resume_is_blocked(self, tmp_path):
        from backend.ml.auto_improve_alphazero import _assert_fresh_lineage_or_raise

        out_dir = tmp_path / "az_stale"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "auto_improve_alphazero_manifest.json").write_text(
            json.dumps(
                {
                    "iterations_completed": 2,
                    "rules_baseline": {"id": "old_rules_baseline"},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="stale"):
            _assert_fresh_lineage_or_raise(
                out_dir=out_dir,
                clean_out_dir=False,
                start_iter=2,
                rules_baseline_id="rules_2026_03_07_semantic446",
                allow_stale_lineage=False,
            )
