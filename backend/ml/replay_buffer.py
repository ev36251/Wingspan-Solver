"""Replay buffer writer for self-play training data."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingSample:
    state: list[float]
    legal_action_ids: list[int]
    action_id: int
    reward: float
    final_score: int
    won: int
    player_index: int
    num_players: int
    round_num: int
    turn_in_round: int


class JsonlReplayWriter:
    """Append-only JSONL writer for model-training samples."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_many(self, samples: list[TrainingSample]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), separators=(",", ":")) + "\n")
