from pathlib import Path

import json

from backend.models.enums import BoardType
from backend.ml.factorized_policy import encode_factorized_targets
from backend.ml.generate_bc_dataset import generate_bc_dataset
from backend.ml.move_features import encode_move_features
from backend.ml.train_factorized_bc import train_bc
from backend.ml.evaluate_factorized_bc import evaluate_factorized_vs_heuristic
from backend.ml.auto_improve_factorized import run_auto_improve_factorized


def test_factorized_eval_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "bc.jsonl"
    meta = tmp_path / "bc.meta.json"
    model = tmp_path / "bc_model.npz"

    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=8,
        lookahead_depth=1,
        n_step=1,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden1=64,
        hidden2=32,
        dropout=0.1,
        lr_init=1e-4,
        lr_peak=1e-3,
        lr_warmup_epochs=1,
        lr_decay_every=3,
        lr_decay_factor=0.5,
        val_split=0.2,
        seed=8,
    )

    ev = evaluate_factorized_vs_heuristic(
        model_path=str(model),
        games=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=8,
    )
    assert ev.games == 2
    assert ev.nn_wins + ev.heuristic_wins + ev.ties == 2


def test_auto_improve_factorized_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        games_per_iter=2,
        proposal_top_k=3,
        lookahead_depth=1,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=1,
        train_epochs=1,
        train_batch=32,
        train_hidden1=64,
        train_hidden2=32,
        train_dropout=0.1,
        train_lr_init=1e-4,
        train_lr_peak=1e-3,
        train_lr_warmup_epochs=1,
        train_lr_decay_every=3,
        train_lr_decay_factor=0.5,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=2,
        promotion_games=4,
        pool_games_per_opponent=2,
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=0.0,
        min_gate_mean_score=0.0,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        engine_teacher_prob=0.0,
        engine_time_budget_ms=25,
        engine_num_determinizations=0,
        engine_max_rollout_depth=24,
        strict_kpi_gate_enabled=False,
        seed=9,
    )

    assert (out_dir / "auto_improve_factorized_manifest.json").exists()
    assert len(manifest["history"]) == 1
    # strict gating field exists
    assert "promotion_gate" in manifest["history"][0]
    assert "promotion_primary_eval" in manifest["history"][0]
    assert "generation_mode" in manifest["history"][0]
    assert "pool_eval" in manifest["history"][0]
    assert "kpi_gate" in manifest["history"][0]
    assert "strict_kpi_gate" in manifest["history"][0]
    assert manifest["history"][0]["strict_kpi_gate"]["skipped"] is True
    assert manifest["config"]["strict_rules_only"] is True
    assert manifest["config"]["reject_non_strict_powers"] is True
    assert "promotion_primary_opponent" in manifest["config"]


def test_auto_improve_factorized_data_accumulation_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac_accum"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=2,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=100,
        games_per_iter=1,
        proposal_top_k=2,
        lookahead_depth=0,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=1,
        train_epochs=1,
        train_batch=32,
        train_hidden1=64,
        train_hidden2=32,
        train_dropout=0.1,
        train_lr_init=1e-4,
        train_lr_peak=1e-3,
        train_lr_warmup_epochs=1,
        train_lr_decay_every=3,
        train_lr_decay_factor=0.5,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=1,
        promotion_games=2,
        pool_games_per_opponent=1,
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=0.0,
        min_gate_mean_score=0.0,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        data_accumulation_enabled=True,
        data_accumulation_decay=0.5,
        max_accumulated_samples=100,
        strict_kpi_gate_enabled=False,
        seed=901,
    )
    row2 = manifest["history"][1]
    assert "combined_dataset_summary" in row2
    summary = row2["combined_dataset_summary"]
    assert summary["enabled"] is True
    assert summary["combined_samples"] <= 100
    assert summary["combined_samples"] >= row2["dataset_summary"]["samples"]
    assert (out_dir / "iter_002" / "bc_dataset_combined.jsonl").exists()
    assert (out_dir / "iter_002" / "bc_dataset_combined.meta.json").exists()
    assert manifest["config"]["data_accumulation_enabled"] is True
    assert manifest["config"]["data_accumulation_decay"] == 0.5
    assert manifest["config"]["max_accumulated_samples"] == 100


def test_generate_bc_dataset_engine_teacher_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "bc_engine.jsonl"
    meta = tmp_path / "bc_engine.meta.json"
    out = generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=10,
        lookahead_depth=0,
        n_step=1,
        engine_teacher_prob=1.0,
        engine_time_budget_ms=5,
        engine_num_determinizations=2,
        engine_max_rollout_depth=16,
    )
    pi = out["policy_improvement"]
    assert pi["engine_teacher_calls"] >= 1
    assert pi["engine_teacher_applied"] >= 1
    assert "value_target_config" in out
    assert out["move_feature_dim"] > 0
    assert out["move_value_enabled"] is True
    assert out["move_value_num_negatives"] >= 1
    assert out["strict_rules_only"] is True
    assert out["reject_non_strict_powers"] is True


def test_generate_bc_dataset_records_fallback_targets(monkeypatch, tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod

    original_exec = mod.execute_move_on_sim
    original_create_game = mod.create_training_game
    root_game_id = {"id": None}
    seen_keys: set[tuple[int, int, int, int, str]] = set()
    pending_pre: dict[tuple[int, int, int], object] = {}
    captured: dict[tuple[int, int, int], tuple[dict, list[float]]] = {}

    def tracked_create_game(*args, **kwargs):
        game = original_create_game(*args, **kwargs)
        root_game_id["id"] = id(game)
        return game

    def flaky_exec(game, player, move):
        if root_game_id["id"] is not None and id(game) != root_game_id["id"]:
            return original_exec(game, player, move)
        key = (id(game), game.current_round, game.turn_in_round, game.current_player_idx, player.name)
        sample_key = (game.current_round, game.turn_in_round, game.current_player_idx)
        if key not in seen_keys:
            seen_keys.add(key)
            pre = mod.deep_copy_game(game)
            pending_pre[sample_key] = pre.players[game.current_player_idx]
            return False
        ok = original_exec(game, player, move)
        if ok and sample_key in pending_pre and sample_key not in captured:
            pre_player = pending_pre[sample_key]
            captured[sample_key] = (
                encode_factorized_targets(move, pre_player),
                encode_move_features(move, pre_player),
            )
        return ok

    monkeypatch.setattr(mod, "create_training_game", tracked_create_game)
    monkeypatch.setattr(mod, "execute_move_on_sim", flaky_exec)

    ds = tmp_path / "bc_fallback.jsonl"
    meta = tmp_path / "bc_fallback.meta.json"
    out = generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=21,
        proposal_top_k=1,
        lookahead_depth=0,
        engine_teacher_prob=0.0,
    )
    pi = out["policy_improvement"]
    assert pi["move_execute_attempts"] >= 1
    assert pi["move_execute_successes"] >= 1
    assert pi["move_execute_fallback_used"] >= 1
    assert pi["move_execute_dropped"] >= 0
    assert captured

    # Ensure output remained writable/valid JSONL rows.
    lines = [ln for ln in ds.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == out["samples"]
    rows = [json.loads(ln) for ln in lines]
    # Match a row that corresponds to one of the fallback events we captured.
    row = None
    expected_targets = None
    expected_pos = None
    for r in rows:
        k = (int(r["round_num"]), int(r["turn_in_round"]), int(r["player_index"]))
        if k in captured:
            row = r
            expected_targets, expected_pos = captured[k]
            break
    assert row is not None
    assert expected_targets is not None
    assert expected_pos is not None
    assert "move_pos" in row
    assert "move_negs" in row
    assert isinstance(row["move_negs"], list)
    assert len(row["move_pos"]) == out["move_feature_dim"]
    assert row["targets"] == expected_targets
    assert row["move_pos"] == expected_pos
    if row["move_negs"]:
        assert len(row["move_negs"][0]) == out["move_feature_dim"]


def test_generate_bc_dataset_play_bird_move_pos_uses_pre_action_hand(monkeypatch, tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod

    original_select = mod._select_policy_move

    def force_play_if_available(*, moves, **kwargs):
        for m in moves:
            if m.action_type.value == "play_bird":
                return m
        return original_select(moves=moves, **kwargs)

    monkeypatch.setattr(mod, "_select_policy_move", force_play_if_available)
    ds = tmp_path / "bc_play_pos.jsonl"
    meta = tmp_path / "bc_play_pos.meta.json"
    out = generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=120,
        seed=71,
        lookahead_depth=0,
        engine_teacher_prob=0.0,
    )
    rows = [json.loads(ln) for ln in ds.read_text(encoding="utf-8").splitlines() if ln.strip()]
    play_rows = [r for r in rows if int(r["targets"]["action_type"]) == 0]
    assert play_rows, "expected at least one play_bird row"
    # play_vp/play_cost/play_egg_cap should be populated from pre-action hand.
    assert any((r["move_pos"][4] > 0.0) or (r["move_pos"][5] > 0.0) or (r["move_pos"][6] > 0.0) for r in play_rows)
    assert out["move_feature_dim"] == len(play_rows[0]["move_pos"])


def test_generate_bc_dataset_champion_nn_vs_nn_engine_only_smoke(tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod
    from backend.solver.move_generator import generate_all_moves

    def fast_teacher(game, player_idx, **kwargs):
        moves = generate_all_moves(game, game.players[player_idx])
        return moves[0] if moves else None

    original_engine = mod._select_engine_teacher_move
    mod._select_engine_teacher_move = fast_teacher
    ds = tmp_path / "bc_src.jsonl"
    meta = tmp_path / "bc_src.meta.json"
    model = tmp_path / "champion_model.npz"
    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=31,
        lookahead_depth=0,
        n_step=1,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden1=64,
        hidden2=32,
        dropout=0.1,
        lr_init=1e-4,
        lr_peak=1e-3,
        lr_warmup_epochs=1,
        lr_decay_every=3,
        lr_decay_factor=0.5,
        val_split=0.2,
        seed=31,
    )

    try:
        out = generate_bc_dataset(
            out_jsonl=str(tmp_path / "bc_champion.jsonl"),
            out_meta=str(tmp_path / "bc_champion.meta.json"),
            games=1,
            players=2,
            board_type=BoardType.OCEANIA,
            max_turns=40,
            seed=32,
            proposal_model_path=str(model),
            opponent_model_path=str(model),
            self_play_policy="champion_nn_vs_nn",
            lookahead_depth=0,
            teacher_source="engine_only",
            engine_time_budget_ms=1,
            engine_num_determinizations=0,
        )
    finally:
        mod._select_engine_teacher_move = original_engine
    pi = out["policy_improvement"]
    assert pi["self_play_policy"] == "champion_nn_vs_nn"
    assert pi["teacher_source"] == "engine_only"
    assert pi["engine_teacher_calls"] >= 1


def test_generate_bc_dataset_engine_only_teacher_fallback_counter(monkeypatch, tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod
    from backend.solver.move_generator import generate_all_moves

    def flaky_engine(*args, **kwargs):
        game = kwargs["game"]
        player_idx = kwargs["player_idx"]
        seed = int(kwargs.get("seed", 1))
        if seed % 2 == 0:  # Force some misses to verify fallback accounting.
            return None
        moves = generate_all_moves(game, game.players[player_idx])
        return moves[0] if moves else None

    monkeypatch.setattr(mod, "_select_engine_teacher_move", flaky_engine)

    ds = tmp_path / "bc_teacher_fb.jsonl"
    meta = tmp_path / "bc_teacher_fb.meta.json"
    out = generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=40,
        seed=40,
        teacher_source="engine_only",
        engine_time_budget_ms=1,
        engine_num_determinizations=0,
        lookahead_depth=0,
    )
    pi = out["policy_improvement"]
    assert pi["engine_teacher_calls"] >= 1
    assert pi["engine_teacher_miss_fallback_used"] >= 1


def test_generate_bc_dataset_champion_mode_uses_model_for_both_sides(monkeypatch, tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod

    ds = tmp_path / "bc_model_src.jsonl"
    meta = tmp_path / "bc_model_src.meta.json"
    model = tmp_path / "both_sides_model.npz"
    generate_bc_dataset(
        out_jsonl=str(ds),
        out_meta=str(meta),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        seed=50,
        lookahead_depth=0,
        n_step=1,
    )
    train_bc(
        dataset_jsonl=str(ds),
        meta_json=str(meta),
        out_model=str(model),
        epochs=1,
        batch_size=32,
        hidden1=64,
        hidden2=32,
        dropout=0.1,
        lr_init=1e-4,
        lr_peak=1e-3,
        lr_warmup_epochs=1,
        lr_decay_every=3,
        lr_decay_factor=0.5,
        val_split=0.2,
        seed=50,
    )

    seen_player_indices: set[int] = set()
    original_model_for_player = mod._model_for_player

    def tracked_model_for_player(player_idx, **kwargs):
        seen_player_indices.add(int(player_idx))
        return original_model_for_player(player_idx, **kwargs)

    monkeypatch.setattr(mod, "_model_for_player", tracked_model_for_player)
    monkeypatch.setattr(mod, "_select_engine_teacher_move", lambda **kwargs: None)

    _ = generate_bc_dataset(
        out_jsonl=str(tmp_path / "bc_both_sides.jsonl"),
        out_meta=str(tmp_path / "bc_both_sides.meta.json"),
        games=1,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=40,
        seed=51,
        proposal_model_path=str(model),
        opponent_model_path=str(model),
        self_play_policy="champion_nn_vs_nn",
        lookahead_depth=0,
        teacher_source="engine_only",
        engine_time_budget_ms=1,
        engine_num_determinizations=0,
    )
    assert 0 in seen_player_indices
    assert 1 in seen_player_indices


def test_auto_improve_factorized_switches_to_champion_mode(tmp_path: Path) -> None:
    import backend.ml.generate_bc_dataset as mod

    # Keep the loop fast by bypassing heavy engine search in tests.
    original_engine = mod._select_engine_teacher_move
    mod._select_engine_teacher_move = lambda **kwargs: None
    out_dir = tmp_path / "auto_fac_switch"
    try:
        manifest = run_auto_improve_factorized(
            out_dir=str(out_dir),
            iterations=2,
            players=2,
            board_type=BoardType.OCEANIA,
            max_turns=100,
            games_per_iter=1,
            proposal_top_k=3,
            lookahead_depth=0,
            n_step=1,
            gamma=0.97,
            bootstrap_mix=0.35,
            value_target_score_scale=160.0,
            value_target_score_bias=0.0,
            late_round_oversample_factor=1,
            train_epochs=1,
            train_batch=32,
            train_hidden1=64,
            train_hidden2=32,
            train_dropout=0.1,
            train_lr_init=1e-4,
            train_lr_peak=1e-3,
            train_lr_warmup_epochs=1,
            train_lr_decay_every=3,
            train_lr_decay_factor=0.5,
            train_value_weight=0.5,
            val_split=0.2,
            eval_games=1,
            promotion_games=1,
            pool_games_per_opponent=1,
            min_pool_win_rate=0.0,
            min_pool_mean_score=0.0,
            min_pool_rate_ge_100=0.0,
            min_pool_rate_ge_120=0.0,
            require_pool_non_regression=False,
            min_gate_win_rate=0.0,
            min_gate_mean_score=0.0,
            min_gate_rate_ge_100=0.0,
            min_gate_rate_ge_120=0.0,
            champion_self_play_enabled=True,
            champion_switch_after_first_promotion=True,
            promotion_primary_opponent="champion",
            champion_engine_time_budget_ms=1,
            champion_engine_num_determinizations=0,
            champion_engine_max_rollout_depth=8,
            strict_kpi_gate_enabled=False,
            seed=61,
        )
    finally:
        mod._select_engine_teacher_move = original_engine
    assert manifest["config"]["promotion_primary_opponent"] == "champion"
    assert manifest["history"][0]["generation_mode"] == "bootstrap_mixed"
    assert manifest["history"][1]["generation_mode"] == "champion_nn_vs_nn"
    assert "promotion_primary_eval" in manifest["history"][1]
    assert manifest["history"][1]["promotion_primary_eval"]["primary_opponent"] == "champion"


def test_auto_improve_factorized_bootstrap_carries_latest_candidate(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac_bootstrap_carry"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=2,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        games_per_iter=1,
        proposal_top_k=3,
        lookahead_depth=0,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=1,
        train_epochs=1,
        train_batch=32,
        train_hidden1=64,
        train_hidden2=32,
        train_dropout=0.1,
        train_lr_init=1e-4,
        train_lr_peak=1e-3,
        train_lr_warmup_epochs=1,
        train_lr_decay_every=3,
        train_lr_decay_factor=0.5,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=1,
        promotion_games=1,
        pool_games_per_opponent=1,
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=1.0,
        min_gate_mean_score=999.0,
        min_gate_rate_ge_100=1.0,
        min_gate_rate_ge_120=1.0,
        champion_self_play_enabled=True,
        champion_switch_after_first_promotion=True,
        promotion_primary_opponent="champion",
        strict_kpi_gate_enabled=False,
        seed=75,
    )
    assert manifest["history"][0]["promotion_gate"]["promoted"] is False
    assert manifest["history"][1]["generation_mode"] == "bootstrap_mixed"
    meta2 = json.loads((out_dir / "iter_002" / "bc_dataset.meta.json").read_text(encoding="utf-8"))
    proposal_path = meta2["policy_improvement"]["proposal_model_path"]
    assert proposal_path is not None
    assert proposal_path.endswith("iter_001/factorized_model.npz")


def test_auto_improve_factorized_strict_curriculum_tapers(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto_fac_strict_curriculum"
    manifest = run_auto_improve_factorized(
        out_dir=str(out_dir),
        iterations=2,
        players=2,
        board_type=BoardType.OCEANIA,
        max_turns=80,
        games_per_iter=2,
        proposal_top_k=3,
        lookahead_depth=0,
        n_step=1,
        gamma=0.97,
        bootstrap_mix=0.35,
        value_target_score_scale=160.0,
        value_target_score_bias=0.0,
        late_round_oversample_factor=1,
        train_epochs=1,
        train_batch=32,
        train_hidden1=64,
        train_hidden2=32,
        train_dropout=0.1,
        train_lr_init=1e-4,
        train_lr_peak=1e-3,
        train_lr_warmup_epochs=1,
        train_lr_decay_every=3,
        train_lr_decay_factor=0.5,
        train_value_weight=0.5,
        val_split=0.2,
        eval_games=1,
        promotion_games=1,
        pool_games_per_opponent=1,
        min_pool_win_rate=0.0,
        min_pool_mean_score=0.0,
        min_pool_rate_ge_100=0.0,
        min_pool_rate_ge_120=0.0,
        require_pool_non_regression=False,
        min_gate_win_rate=0.0,
        min_gate_mean_score=0.0,
        min_gate_rate_ge_100=0.0,
        min_gate_rate_ge_120=0.0,
        strict_rules_only=True,
        reject_non_strict_powers=True,
        strict_curriculum_enabled=True,
        strict_fraction_start=1.0,
        strict_fraction_end=0.0,
        strict_fraction_warmup_iters=1,
        strict_kpi_gate_enabled=False,
        seed=89,
    )
    assert manifest["history"][0]["strict_game_fraction"] == 1.0
    assert manifest["history"][1]["strict_game_fraction"] == 0.0
    meta1 = json.loads((out_dir / "iter_001" / "bc_dataset.meta.json").read_text(encoding="utf-8"))
    meta2 = json.loads((out_dir / "iter_002" / "bc_dataset.meta.json").read_text(encoding="utf-8"))
    assert int(meta1["strict_games"]) == 2
    assert int(meta1["relaxed_games"]) == 0
    assert int(meta2["strict_games"]) == 0
    assert int(meta2["relaxed_games"]) == 2
