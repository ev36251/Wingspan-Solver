# v17.1 Rules-Stable Restart Runbook

This runbook enforces pre-training rules stability before launching a new
AlphaZero lineage.

## 1) Pre-training hard gates (must pass)

```bash
python -m backend.scripts.audit_bird_power_tests
python -m backend.ml.strict_kpi_runner --out reports/ml/strict_kpi_pretrain.json
pytest backend/tests/test_power_strict_mapping.py backend/tests/test_power_semantic_batches.py backend/tests/test_power_semantic_remaining.py
```

Expected:
- `semantic_untested_count == 0`
- `strict_kpi_runner` returns `"passed": true`
- All three pytest suites pass

## 2) Strict smoke self-play gate (10 games)

```bash
python -m backend.scripts.v17_1_rules_stable_restart --only-gates
```

Expected:
- strict smoke self-play runs with `strict_rules_mode=True`
- no self-play game exceptions

## 3) Launch v17.1 pilot (3 iterations, Modal, full features)

```bash
python -m backend.scripts.v17_1_rules_stable_restart
```

Defaults enforced by the launcher:
- 2-player scope
- strict rules mode
- self-play fail-fast
- full feature pack:
  - `use_per_slot_encoding`
  - `use_hand_habitat_features`
  - `use_tray_per_slot_encoding`
  - `use_opponent_board_encoding`
  - `use_power_features`
- champion gate config for meaningful iteration-1 comparisons
- eval-vs-heuristic uses MCTS move selection (aligned eval mode)

## 4) Pilot acceptance criteria

At end of iter 3:
- no self-play/eval/gate exceptions, and
- at least one:
  - eval win-rate vs heuristic `>= 0.15`, or
  - champion gate stage1 win-rate `>= 0.45` with non-worsening eval margin.

If acceptance fails, stop scaling and triage the top failure mode first.
