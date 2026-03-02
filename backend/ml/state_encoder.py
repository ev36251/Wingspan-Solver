"""State vector encoding for Wingspan neural-network training."""

from __future__ import annotations

import hashlib
import re as _re
from dataclasses import dataclass

import numpy as np

from backend.config import ACTIONS_PER_ROUND, EGG_COST_BY_COLUMN, get_action_column
from backend.engine.rules import can_pay_food_cost
from backend.engine.scoring import bonus_card_details, calculate_score, goal_progress_for_round
from backend.models.bird import Bird
from backend.models.enums import BoardType, FoodType, GameSet, Habitat, NestType, PowerColor
from backend.models.game_state import GameState


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class BirdFeatureEncoder:
    """Encode a single Bird instance as a 33-dimensional float vector.

    Returns all-zeros for None (empty slot or absent hand card).
    """

    DIM = 33

    @staticmethod
    def encode(bird: Bird | None) -> np.ndarray:
        """Return a 33-float array representing the bird's attributes."""
        vec = np.zeros(BirdFeatureEncoder.DIM, dtype=np.float32)
        if bird is None:
            return vec

        # f[0] VP
        vec[0] = _clamp01(bird.victory_points / 9.0)
        # f[1] egg limit
        vec[1] = _clamp01(bird.egg_limit / 6.0)
        # f[2] wingspan: 0.5 if flightless (wildcard), else normalized
        if bird.wingspan_cm is None:
            vec[2] = 0.5
        else:
            vec[2] = _clamp01(bird.wingspan_cm / 150.0)
        # f[3] is_flightless
        vec[3] = 1.0 if bird.wingspan_cm is None else 0.0
        # f[4] food cost total
        vec[4] = _clamp01(bird.food_cost.total / 6.0)
        # f[5] is_or_cost
        vec[5] = 1.0 if bird.food_cost.is_or else 0.0
        # f[6-12] food types in cost (binary flags)
        food_order = [
            FoodType.INVERTEBRATE,
            FoodType.SEED,
            FoodType.FISH,
            FoodType.FRUIT,
            FoodType.RODENT,
            FoodType.NECTAR,
            FoodType.WILD,
        ]
        cost_types = bird.food_cost.distinct_types
        for i, ft in enumerate(food_order):
            vec[6 + i] = 1.0 if ft in cost_types else 0.0
        # f[13-15] habitats
        vec[13] = 1.0 if Habitat.FOREST in bird.habitats else 0.0
        vec[14] = 1.0 if Habitat.GRASSLAND in bird.habitats else 0.0
        vec[15] = 1.0 if Habitat.WETLAND in bird.habitats else 0.0
        # f[16] multi-habitat
        vec[16] = 1.0 if len(bird.habitats) > 1 else 0.0
        # f[17-21] nest type
        nest_order = [NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM, NestType.WILD]
        for i, nt in enumerate(nest_order):
            vec[17 + i] = 1.0 if bird.nest_type == nt else 0.0
        # f[22-27] power color
        color_order = [
            PowerColor.BROWN,
            PowerColor.WHITE,
            PowerColor.PINK,
            PowerColor.TEAL,
            PowerColor.YELLOW,
            PowerColor.NONE,
        ]
        for i, pc in enumerate(color_order):
            vec[22 + i] = 1.0 if bird.color == pc else 0.0
        # f[28] is_predator
        vec[28] = 1.0 if bird.is_predator else 0.0
        # f[29] is_flocking
        vec[29] = 1.0 if bird.is_flocking else 0.0
        # f[30-32] game set: core/european/oceania (asia = all-zero)
        vec[30] = 1.0 if bird.game_set == GameSet.CORE else 0.0
        vec[31] = 1.0 if bird.game_set == GameSet.EUROPEAN else 0.0
        vec[32] = 1.0 if bird.game_set == GameSet.OCEANIA else 0.0

        return vec


_FOOD_WORDS = (
    "seed", "fish", "fruit", "invertebrate", "rodent", "nectar", "wild",
    "any food", "food token", "food from", "food of",
)


class PowerFeatureEncoder:
    """Encode a bird's power EFFECT as a 14-dimensional float vector.

    Captures WHAT the power does and WHO benefits — things BirdFeatureEncoder's
    power-color flags cannot express:
      - A brown bird might give ALL players eggs (Noisy Miner: "All other players
        may lay 1 egg") vs. give only you food.
      - A tuck power is engine fuel; a cache power scores at game end.
      - A compound power (tuck + lay eggs) is stronger than a single-effect one.

    Features:
      [0]  gain_food        power gives food to the food supply
      [1]  lay_eggs         power places eggs
      [2]  tuck_cards       power tucks cards behind birds (engine fuel)
      [3]  cache_food       power caches food ON a bird (end-game pts)
      [4]  draw_cards       power draws cards to hand
      [5]  predator         predator hunt mechanic
      [6]  play_bird        free bird placement from hand
      [7]  no_power         bird has no power at all
      [8]  quantity_norm    primary resource amount / 5  (magnitude proxy)
      [9]  all_players      ALL players receive a benefit (key: brown powers
                            can also give opponents resources)
      [10] per_bird         "for each X" — scales with engine size
      [11] conditional      has an "if" / "when" constraint
      [12] compound         two or more distinct effect types combined
      [13] repeat_or_copy   power repeats another bird's power or copies it
    """

    DIM = 14
    _POWER_FEATURE_NAMES = [
        "gain_food", "lay_eggs", "tuck_cards", "cache_food", "draw_cards",
        "predator", "play_bird", "no_power", "quantity_norm", "all_players",
        "per_bird", "conditional", "compound", "repeat_or_copy",
    ]

    @staticmethod
    def encode(bird: "Bird | None") -> list[float]:
        if bird is None:
            return [0.0] * PowerFeatureEncoder.DIM

        text = (bird.power_text or "").lower()

        if not text:
            # No power text (high-VP birds with power_color=NONE)
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def _has(*kws: str) -> bool:
            return any(k in text for k in kws)

        # Effect types
        gain_food = 1.0 if (
            _has("gain", "take", "receive") and _has(*_FOOD_WORDS)
        ) else 0.0

        lay_eggs = 1.0 if ("egg" in text and _has("lay", "place")) else 0.0

        tuck_cards = 1.0 if "tuck" in text else 0.0

        cache_food = 1.0 if "cache" in text else 0.0

        draw_cards = 1.0 if (
            _has("draw", "add to your hand", "keep") and _has("card", "bird")
        ) else 0.0

        predator = 1.0 if bird.is_predator else 0.0

        # Use word boundary for "play" to avoid false match on "players"
        play_bird = 1.0 if (
            bool(_re.search(r'\bplay\b|\bplace\b', text)) and "bird" in text and
            _has("free", "without paying")
        ) else 0.0

        no_power = 0.0  # we have power text, so there is a power

        # Magnitude: max small integer in the text (1–5), normalised by 5.
        # Taking max avoids "discard 1 egg to gain 3 food" picking up the cost (1)
        # rather than the gain (3).
        nums = [int(m) for m in _re.findall(r'\b([1-5])\b', text)]
        quantity_norm = float(max(nums)) / 5.0 if nums else 0.2

        # ALL players benefit — user's key concern: some brown powers share
        all_players = 1.0 if _has(
            "all players", "all other players", "each other player",
            "each player", "players may", "other players may",
        ) else 0.0

        # Scales with board state ("for each X")
        per_bird = 1.0 if "for each" in text else 0.0

        # Conditional / situational
        conditional = 1.0 if _has("if ", "when ", "for each") else 0.0

        # Compound: multiple distinct effect types
        n_effects = sum([
            gain_food > 0, lay_eggs > 0, tuck_cards > 0,
            cache_food > 0, draw_cards > 0, predator > 0, play_bird > 0,
        ])
        compound = 1.0 if n_effects >= 2 else 0.0

        # Repeat/copy: mirrors another bird's power (very powerful combinators)
        repeat_or_copy = 1.0 if _has("repeat", "copy", "mimic") else 0.0

        return [
            gain_food, lay_eggs, tuck_cards, cache_food, draw_cards,
            predator, play_bird, no_power, quantity_norm, all_players,
            per_bird, conditional, compound, repeat_or_copy,
        ]


@dataclass
class StateEncoder:
    """Encode a game state into a fixed-size float feature vector.

    Features are player-centric: the acting player is encoded first,
    then opponent aggregate features.

    When use_per_slot_encoding=True, appends:
      - 15 board slots × 44 features = 660
      - max_hand_slots hand cards × 33 features = 264 (default)
    Total with per-slot: 145 + 660 + 264 = 1069

    Optional appended blocks (all backward-compatible via zero-init warm-start):
      use_hand_habitat_features  → +15  hand-board synergy (5 per habitat)
      use_tray_per_slot_encoding → +156 tray per-bird features (3 slots × 52)
                                         (38 structural + 14 power effect)
      use_opponent_board_encoding→ +750 leading opponent board (15 slots × 50)
                                         (36 structural + 14 power effect)
      use_power_features         → +322 power effect vectors for self board+hand
                                         (15 board + 8 hand) × 14
    """

    max_hand: float = 25.0
    max_food: float = 12.0
    max_score: float = 180.0
    max_deck: float = 200.0
    enable_identity_features: bool = False
    identity_hash_dim: int = 128
    use_per_slot_encoding: bool = False
    max_hand_slots: int = 8
    use_hand_habitat_features: bool = False
    use_tray_per_slot_encoding: bool = False
    use_opponent_board_encoding: bool = False
    use_power_features: bool = False  # power-effect block for self board + hand

    @classmethod
    def from_metadata(cls, meta: dict | None) -> "StateEncoder":
        cfg = (meta or {}).get("state_encoder", {}) if isinstance(meta, dict) else {}
        return cls(
            enable_identity_features=bool(cfg.get("enable_identity_features", False)),
            identity_hash_dim=max(1, int(cfg.get("identity_hash_dim", 128))),
            use_per_slot_encoding=bool(cfg.get("use_per_slot_encoding", False)),
            max_hand_slots=max(1, int(cfg.get("max_hand_slots", 8))),
            use_hand_habitat_features=bool(cfg.get("use_hand_habitat_features", False)),
            use_tray_per_slot_encoding=bool(cfg.get("use_tray_per_slot_encoding", False)),
            use_opponent_board_encoding=bool(cfg.get("use_opponent_board_encoding", False)),
            use_power_features=bool(cfg.get("use_power_features", False)),
        )

    @classmethod
    def resolve_for_model(
        cls,
        model_meta: dict | None,
        *,
        fallback_enable_identity_features: bool = False,
        fallback_identity_hash_dim: int = 128,
        fallback_use_per_slot: bool = False,
        fallback_max_hand_slots: int = 8,
        fallback_use_hand_habitat_features: bool = False,
        fallback_use_tray_per_slot: bool = False,
        fallback_use_opponent_board: bool = False,
        fallback_use_power_features: bool = False,
    ) -> "StateEncoder":
        if isinstance(model_meta, dict) and isinstance(model_meta.get("state_encoder"), dict):
            return cls.from_metadata(model_meta)
        return cls(
            enable_identity_features=bool(fallback_enable_identity_features),
            identity_hash_dim=max(1, int(fallback_identity_hash_dim)),
            use_per_slot_encoding=bool(fallback_use_per_slot),
            max_hand_slots=max(1, int(fallback_max_hand_slots)),
            use_hand_habitat_features=bool(fallback_use_hand_habitat_features),
            use_tray_per_slot_encoding=bool(fallback_use_tray_per_slot),
            use_opponent_board_encoding=bool(fallback_use_opponent_board),
            use_power_features=bool(fallback_use_power_features),
        )

    def feature_names(self) -> list[str]:
        base = [
            "global.round", "global.turn_in_round", "global.current_player_idx",
            "global.total_actions_remaining", "global.deck_remaining", "global.tray_count",
            "global.feeder.invertebrate", "global.feeder.seed", "global.feeder.fish",
            "global.feeder.fruit", "global.feeder.rodent", "global.feeder.nectar",
            "self.cubes", "self.hand_size", "self.food_total", "self.food_invertebrate",
            "self.food_seed", "self.food_fish", "self.food_fruit", "self.food_rodent",
            "self.food_nectar", "self.birds_total", "self.birds_forest", "self.birds_grassland",
            "self.birds_wetland", "self.eggs_total", "self.cached_food_total", "self.tucked_total",
            "self.brown_birds", "self.white_birds", "self.pink_birds", "self.teal_birds",
            "self.high_vp_birds", "self.nectar_spent_forest", "self.nectar_spent_grassland",
            "self.nectar_spent_wetland", "opp.mean_cubes", "opp.mean_hand_size", "opp.mean_food_total",
            "opp.mean_birds_total", "opp.mean_eggs_total", "opp.mean_birds_forest", "opp.mean_birds_grassland",
            "opp.mean_birds_wetland", "opp.max_birds_total", "opp.max_eggs_total", "opp.mean_est_score",
            "opp.lead_over_self_est_score",
            "self.score.bird_vp", "self.score.eggs", "self.score.cached_food", "self.score.tucked_cards",
            "self.score.bonus_cards", "self.score.round_goals", "self.score.nectar",
            "goal.r1.active", "goal.r1.self_progress", "goal.r1.opp_max_progress", "goal.r1.earned_points",
            "goal.r2.active", "goal.r2.self_progress", "goal.r2.opp_max_progress", "goal.r2.earned_points",
            "goal.r3.active", "goal.r3.self_progress", "goal.r3.opp_max_progress", "goal.r3.earned_points",
            "goal.r4.active", "goal.r4.self_progress", "goal.r4.opp_max_progress", "goal.r4.earned_points",
            "self.bonus.count", "self.bonus.potential_vp", "self.bonus.best_single_vp",
            "self.bonus.min_single_vp", "self.bonus.improvable_count", "self.bonus.maxed_count",
            "self.hand.total_vp", "self.hand.mean_vp", "self.hand.best_vp", "self.hand.mean_cost",
            "self.hand.affordable_count", "self.hand.affordable_frac", "self.hand.mean_egg_limit",
            "self.hand.multi_habitat_frac",
            "self.board.slots_remaining_forest", "self.board.slots_remaining_grassland",
            "self.board.slots_remaining_wetland", "self.board.egg_capacity_remaining",
            "self.board.egg_fill_rate", "self.board.action_gain_forest", "self.board.action_gain_grassland",
            "self.board.action_gain_wetland", "self.board.has_bonus_forest", "self.board.has_bonus_grassland",
            "self.board.has_bonus_wetland", "self.board.bird_vp_total",
            "self.nest.bowl_count", "self.nest.cavity_count", "self.nest.ground_count",
            "self.nest.platform_count", "self.nest.wild_count",
            "phase.game_progress", "phase.cubes_spent_this_round", "phase.num_players",
            "phase.board_is_oceania", "phase.rounds_remaining",
            "opp.score.mean_bird_vp", "opp.score.mean_eggs", "opp.score.mean_cached_food",
            "opp.score.mean_tucked_cards", "opp.score.mean_bonus_cards", "opp.score.mean_round_goals",
            "opp.score.mean_nectar",
            "opp.res.mean_cached_food", "opp.res.max_cached_food", "opp.res.mean_tucked_cards",
            "opp.res.max_tucked_cards", "opp.res.mean_brown_birds", "opp.res.mean_pink_birds",
            "nectar.lead_forest", "nectar.lead_grassland", "nectar.lead_wetland",
            "nectar.habitats_winning", "nectar.habitats_tied", "nectar.total_spent",
            "playable.count", "playable.frac", "playable.best_vp", "playable.food_gap_best_bird",
            "playable.egg_gap", "playable.can_play_any", "playable.food_type_diversity",
            "playable.total_resources",
            "position.score_total", "position.score_rank", "position.gap_to_first", "position.gap_to_last",
            "position.efficiency", "position.tray_mean_vp", "position.tray_affordable_count",
            "position.no_power_birds",
            "goal.current_self_progress", "goal.current_self_rank", "goal.current_gap_to_first",
            "goal.past_total_earned", "goal.future_goals_remaining", "goal.has_no_goal_round",
        ]
        if self.enable_identity_features:
            base.extend([f"identity.hash_{i:03d}" for i in range(max(1, int(self.identity_hash_dim)))])
        if self.use_per_slot_encoding:
            base.extend(self._per_slot_feature_names())
        if self.use_hand_habitat_features:
            base.extend(self._hand_habitat_feature_names())
        if self.use_tray_per_slot_encoding:
            base.extend(self._tray_per_slot_feature_names())
        if self.use_opponent_board_encoding:
            base.extend(self._opponent_board_feature_names())
        if self.use_power_features:
            base.extend(self._self_power_feature_names())
        return base

    def _per_slot_feature_names(self) -> list[str]:
        """Names for the per-slot board + hand encoding block."""
        bird_attr_names = [
            "vp", "egg_limit", "wingspan_norm", "is_flightless",
            "cost_total", "is_or_cost",
            "cost_invertebrate", "cost_seed", "cost_fish", "cost_fruit", "cost_rodent", "cost_nectar", "cost_wild",
            "hab_forest", "hab_grassland", "hab_wetland", "multi_habitat",
            "nest_bowl", "nest_cavity", "nest_ground", "nest_platform", "nest_wild",
            "color_brown", "color_white", "color_pink", "color_teal", "color_yellow", "color_none",
            "is_predator", "is_flocking",
            "set_core", "set_european", "set_oceania",
        ]
        slot_ctx_names = [
            "eggs_norm", "cached_food_norm", "tucked_norm",
            "col_0", "col_1", "col_2", "col_3", "col_4",
            "hab_is_forest", "hab_is_grassland", "hab_is_wetland",
        ]
        names: list[str] = []
        habitat_order = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND]
        for hab in habitat_order:
            for col in range(5):
                prefix = f"board.{hab.value}.{col}"
                for attr in bird_attr_names:
                    names.append(f"{prefix}.bird.{attr}")
                for ctx in slot_ctx_names:
                    names.append(f"{prefix}.{ctx}")
        n = max(1, int(self.max_hand_slots))
        for i in range(n):
            for attr in bird_attr_names:
                names.append(f"hand.{i}.{attr}")
        return names

    def _identity_bucket(self, token: str, dim: int) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, byteorder="little", signed=False) % dim

    def _encode_identity_features(self, game: GameState, player_index: int) -> list[float]:
        dim = max(1, int(self.identity_hash_dim))
        vec = [0.0 for _ in range(dim)]
        p = game.players[player_index]
        opponents = [op for i, op in enumerate(game.players) if i != player_index]

        def add(token: str, weight: float) -> None:
            if not token:
                return
            idx = self._identity_bucket(token, dim)
            vec[idx] += float(weight)

        for b in p.hand:
            add(f"self_hand:{b.name}", 1.0)
        for b in p.board.all_birds():
            add(f"self_board:{b.name}", 1.5)
        for b in game.card_tray.face_up:
            add(f"tray:{b.name}", 0.5)
        for bc in getattr(p, "bonus_cards", []):
            add(f"self_bonus:{getattr(bc, 'name', '')}", 1.0)

        for op in opponents:
            for b in op.board.all_birds():
                add(f"opp_board:{b.name}", 0.4)
            for bc in getattr(op, "bonus_cards", []):
                add(f"opp_bonus:{getattr(bc, 'name', '')}", 0.25)

        max_v = max(1.0, max(vec))
        return [_clamp01(v / max_v) for v in vec]

    def _encode_per_slot(self, game: GameState, player_index: int) -> list[float]:
        """Encode per-slot board features + hand cards as a flat float list.

        Board: 15 slots × 44 features = 660
        Hand:  max_hand_slots × 33 features = 264 (default)
        """
        p = game.players[player_index]
        result: list[float] = []

        habitat_order = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND]
        hab_one_hot = {
            Habitat.FOREST:    [1.0, 0.0, 0.0],
            Habitat.GRASSLAND: [0.0, 1.0, 0.0],
            Habitat.WETLAND:   [0.0, 0.0, 1.0],
        }

        for hab in habitat_order:
            row = p.board.get_row(hab)
            for col, slot in enumerate(row.slots):
                # 33 bird features
                bird_vec = BirdFeatureEncoder.encode(slot.bird)
                result.extend(bird_vec.tolist())
                # 11 slot context features
                bird_egg_limit = slot.bird.egg_limit if slot.bird is not None else 1
                eggs_norm = _clamp01(slot.eggs / max(1, bird_egg_limit))
                cached_norm = _clamp01(slot.total_cached_food / 10.0)
                tucked_norm = _clamp01(slot.tucked_cards / 20.0)
                col_one_hot = [1.0 if col == c else 0.0 for c in range(5)]
                result.append(eggs_norm)
                result.append(cached_norm)
                result.append(tucked_norm)
                result.extend(col_one_hot)
                result.extend(hab_one_hot[hab])

        # Hand encoding: top-N cards by VP, padded with zeros
        n = max(1, int(self.max_hand_slots))
        hand_sorted = sorted(p.hand, key=lambda b: b.victory_points, reverse=True)
        for i in range(n):
            bird = hand_sorted[i] if i < len(hand_sorted) else None
            result.extend(BirdFeatureEncoder.encode(bird).tolist())

        return result

    def _hand_habitat_feature_names(self) -> list[str]:
        """Names for the 15 hand-board synergy features (5 per habitat)."""
        names: list[str] = []
        for hab in ("forest", "grassland", "wetland"):
            names.extend([
                f"hand.hab.{hab}.count",
                f"hand.hab.{hab}.affordable",
                f"hand.hab.{hab}.brown_count",
                f"hand.hab.{hab}.best_vp",
                f"hand.hab.{hab}.engine_synergy",
            ])
        return names

    def _encode_hand_habitat_features(self, game: GameState, player_index: int) -> list[float]:
        """Encode 15 hand-board synergy features (5 per habitat).

        For each habitat:
          [0] hand_count      — hand birds placeable in this habitat / 8
          [1] hand_affordable — affordable hand birds for this habitat / 8
          [2] hand_brown      — brown-power hand birds for this habitat / 5
          [3] hand_best_vp    — best VP of hand birds for this habitat / 9
          [4] engine_synergy  — board engine strength × hand depth interaction:
                                (board_brown_count × hand_count) / 25

        Appended AFTER per-slot so existing model weights are untouched
        during warm-start (new weights zero-initialised).
        """
        p = game.players[player_index]
        result: list[float] = []

        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            row = p.board.get_row(hab)

            # Existing engine strength in this habitat row
            board_brown_count = sum(
                1 for slot in row.slots
                if slot.bird and slot.bird.color == PowerColor.BROWN
            )

            # Hand birds compatible with this habitat
            hab_hand = [b for b in p.hand if hab in b.habitats]
            hand_count = len(hab_hand)
            hand_affordable = sum(
                1 for b in hab_hand if can_pay_food_cost(p, b.food_cost)[0]
            )
            hand_brown = sum(1 for b in hab_hand if b.color == PowerColor.BROWN)
            hand_best_vp = float(max((b.victory_points for b in hab_hand), default=0))

            # Interaction term: existing engine × future engine potential
            # High when both the board already has engine birds AND there are
            # more engine candidates in hand (Raven + Killdeer + NM scenario).
            engine_synergy = float(board_brown_count * hand_count)

            result.append(_clamp01(hand_count / 8.0))
            result.append(_clamp01(hand_affordable / 8.0))
            result.append(_clamp01(hand_brown / 5.0))
            result.append(_clamp01(hand_best_vp / 9.0))
            result.append(_clamp01(engine_synergy / 25.0))

        return result

    # ------------------------------------------------------------------
    # Tray per-slot encoding  (3 slots × 38 features = 114)
    # ------------------------------------------------------------------

    _BIRD_ATTR_NAMES = [
        "vp", "egg_limit", "wingspan_norm", "is_flightless",
        "cost_total", "is_or_cost",
        "cost_invertebrate", "cost_seed", "cost_fish", "cost_fruit",
        "cost_rodent", "cost_nectar", "cost_wild",
        "hab_forest", "hab_grassland", "hab_wetland", "multi_habitat",
        "nest_bowl", "nest_cavity", "nest_ground", "nest_platform", "nest_wild",
        "color_brown", "color_white", "color_pink", "color_teal",
        "color_yellow", "color_none",
        "is_predator", "is_flocking",
        "set_core", "set_european", "set_oceania",
    ]

    def _tray_per_slot_feature_names(self) -> list[str]:
        """3 tray slots × 52 features = 156 names.

        Per slot: 33 structural (BirdFeatureEncoder) + 5 play-context
                + 14 power-effect (PowerFeatureEncoder) = 52.
        """
        names: list[str] = []
        for i in range(3):
            prefix = f"tray.{i}"
            for attr in self._BIRD_ATTR_NAMES:
                names.append(f"{prefix}.bird.{attr}")
            names.extend([
                f"{prefix}.is_affordable",
                f"{prefix}.egg_gap_norm",
                f"{prefix}.hab_forest_open",
                f"{prefix}.hab_grass_open",
                f"{prefix}.hab_wet_open",
            ])
            for pf in PowerFeatureEncoder._POWER_FEATURE_NAMES:
                names.append(f"{prefix}.power.{pf}")
        return names

    def _encode_tray_per_slot(self, game: GameState, player_index: int) -> list[float]:
        """Encode up to 3 tray cards: 52 features each = 156 total.

        Per tray slot:
          [0-32]  BirdFeatureEncoder  — structural attributes
          [33-37] Play-context        — affordable, egg_gap, open hab slots
          [38-51] PowerFeatureEncoder — what the power does, who benefits
        """
        p = game.players[player_index]
        tray_birds = sorted(
            game.card_tray.face_up,
            key=lambda b: b.victory_points,
            reverse=True,
        )
        eggs_total = p.board.total_eggs()
        open_forest = p.board.forest.next_empty_slot() is not None
        open_grass = p.board.grassland.next_empty_slot() is not None
        open_wet = p.board.wetland.next_empty_slot() is not None

        result: list[float] = []
        for i in range(3):
            bird = tray_birds[i] if i < len(tray_birds) else None
            result.extend(BirdFeatureEncoder.encode(bird).tolist())
            if bird is None:
                result.extend([0.0] * 5)
                result.extend([0.0] * PowerFeatureEncoder.DIM)
            else:
                affordable, _ = can_pay_food_cost(p, bird.food_cost)
                egg_gaps = []
                for hab in bird.habitats:
                    row = p.board.get_row(hab)
                    slot_idx = row.next_empty_slot()
                    if slot_idx is not None:
                        egg_cost = EGG_COST_BY_COLUMN[slot_idx]
                        egg_gaps.append(max(0, egg_cost - eggs_total))
                egg_gap = min(egg_gaps) if egg_gaps else 10
                result.append(1.0 if affordable else 0.0)
                result.append(_clamp01(egg_gap / 10.0))
                result.append(1.0 if (Habitat.FOREST in bird.habitats and open_forest) else 0.0)
                result.append(1.0 if (Habitat.GRASSLAND in bird.habitats and open_grass) else 0.0)
                result.append(1.0 if (Habitat.WETLAND in bird.habitats and open_wet) else 0.0)
                result.extend(PowerFeatureEncoder.encode(bird))
        return result

    # ------------------------------------------------------------------
    # Opponent board per-slot encoding  (15 slots × 50 features = 750)
    # ------------------------------------------------------------------

    def _opponent_board_feature_names(self) -> list[str]:
        """15 opponent board slots × 50 features = 750 names."""
        names: list[str] = []
        for hab in ("forest", "grassland", "wetland"):
            for col in range(5):
                prefix = f"opp.board.{hab}.{col}"
                for attr in self._BIRD_ATTR_NAMES:
                    names.append(f"{prefix}.bird.{attr}")
                names.extend([
                    f"{prefix}.eggs_norm",
                    f"{prefix}.cached_norm",
                    f"{prefix}.tucked_norm",
                ])
                for pf in PowerFeatureEncoder._POWER_FEATURE_NAMES:
                    names.append(f"{prefix}.power.{pf}")
        return names

    def _encode_opponent_board(self, game: GameState, player_index: int) -> list[float]:
        """Encode the leading opponent's board: 15 slots × 50 features = 750.

        Per slot (50 features):
          [0-32]  BirdFeatureEncoder  — structural attributes (power color tells
                  timing: brown=on-action, white=on-your-action, pink=between-turns)
          [33-35] Slot state          — eggs_norm, cached_norm, tucked_norm
          [36-49] PowerFeatureEncoder — what the power actually does and who
                  benefits. A pink bird with all_players=1 means free resources
                  for you on the opponent's turns; a white bird with all_players=0
                  means nothing for you.

        Leading opponent = highest current estimated score.
        In 2-player this is always the only opponent.
        Empty slots are zero-padded.
        """
        opponents = [op for i, op in enumerate(game.players) if i != player_index]
        if not opponents:
            return [0.0] * (15 * 50)

        leading_opp = max(opponents, key=lambda op: calculate_score(game, op).total)

        result: list[float] = []
        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            row = leading_opp.board.get_row(hab)
            for slot in row.slots:
                result.extend(BirdFeatureEncoder.encode(slot.bird).tolist())
                if slot.bird is not None:
                    eggs_norm = _clamp01(slot.eggs / max(1, slot.bird.egg_limit))
                else:
                    eggs_norm = 0.0
                result.append(eggs_norm)
                result.append(_clamp01(slot.total_cached_food / 10.0))
                result.append(_clamp01(slot.tucked_cards / 20.0))
                result.extend(PowerFeatureEncoder.encode(slot.bird))
        return result

    # ------------------------------------------------------------------
    # Self board + hand power features  (322 features when enabled)
    # 15 board slots × 14 + max_hand_slots × 14
    # Appended AFTER tray/opp-board blocks → backward compatible.
    # Gives the NN power-effect info for its own birds (the per-slot
    # encoding already has structural BirdFeatureEncoder features but
    # no PowerFeatureEncoder — this fills that gap without changing
    # per-slot slot sizes, preserving warm-start compatibility).
    # ------------------------------------------------------------------

    def _self_power_feature_names(self) -> list[str]:
        """(15 board + max_hand_slots hand) × 14 power features."""
        names: list[str] = []
        n = max(1, int(self.max_hand_slots))
        for hab in ("forest", "grassland", "wetland"):
            for col in range(5):
                prefix = f"self.board.{hab}.{col}.power"
                for pf in PowerFeatureEncoder._POWER_FEATURE_NAMES:
                    names.append(f"{prefix}.{pf}")
        for i in range(n):
            prefix = f"self.hand.{i}.power"
            for pf in PowerFeatureEncoder._POWER_FEATURE_NAMES:
                names.append(f"{prefix}.{pf}")
        return names

    def _encode_self_power_features(self, game: GameState, player_index: int) -> list[float]:
        """Power-effect vectors for own board (15 slots) + hand (max_hand_slots).

        The per-slot encoding already carries structural BirdFeatureEncoder
        features. This block adds the 14-feature PowerFeatureEncoder on top
        of those, appended at the end of the vector so v8 warm-start weights
        remain valid for the first 1599 dimensions.
        """
        p = game.players[player_index]
        result: list[float] = []

        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            row = p.board.get_row(hab)
            for slot in row.slots:
                result.extend(PowerFeatureEncoder.encode(slot.bird))

        n = max(1, int(self.max_hand_slots))
        hand_sorted = sorted(p.hand, key=lambda b: b.victory_points, reverse=True)
        for i in range(n):
            bird = hand_sorted[i] if i < len(hand_sorted) else None
            result.extend(PowerFeatureEncoder.encode(bird))

        return result

    def encode(self, game: GameState, player_index: int) -> list[float]:
        p = game.players[player_index]
        opponents = [op for i, op in enumerate(game.players) if i != player_index]

        feeder_counts = {ft: 0 for ft in (
            FoodType.INVERTEBRATE,
            FoodType.SEED,
            FoodType.FISH,
            FoodType.FRUIT,
            FoodType.RODENT,
            FoodType.NECTAR,
        )}
        for die in game.birdfeeder.dice:
            if isinstance(die, tuple):
                for ft in die:
                    feeder_counts[ft] = feeder_counts.get(ft, 0) + 1
            else:
                feeder_counts[die] = feeder_counts.get(die, 0) + 1

        def _effective_actions(round_num: int) -> int:
            if round_num < 1 or round_num > len(ACTIONS_PER_ROUND):
                return 0
            base = ACTIONS_PER_ROUND[round_num - 1]
            no_goal_bonus = 0
            for i in range(round_num - 1):
                if i < len(game.round_goals) and game.round_goals[i].description.lower() == "no goal":
                    no_goal_bonus += 1
            return base + no_goal_bonus

        def _player_est_score(player):
            return calculate_score(game, player)

        score_breakdowns = {pl.name: _player_est_score(pl) for pl in game.players}
        self_breakdown = score_breakdowns[p.name]
        self_score = float(self_breakdown.total)

        def _board_stats(player):
            board = player.board
            brown = 0
            white = 0
            pink = 0
            teal = 0
            no_power = 0
            high_vp = 0
            cached = 0
            tucked = 0
            nest_bowl = 0
            nest_cavity = 0
            nest_ground = 0
            nest_platform = 0
            nest_wild = 0
            for _, _, slot in board.all_slots():
                if not slot.bird:
                    continue
                cached += slot.total_cached_food
                tucked += slot.tucked_cards
                if slot.bird.victory_points >= 5:
                    high_vp += 1
                if slot.bird.color == PowerColor.BROWN:
                    brown += 1
                elif slot.bird.color == PowerColor.WHITE:
                    white += 1
                elif slot.bird.color == PowerColor.PINK:
                    pink += 1
                elif slot.bird.color == PowerColor.TEAL:
                    teal += 1
                elif slot.bird.color == PowerColor.NONE:
                    no_power += 1
                if slot.bird.nest_type == NestType.BOWL:
                    nest_bowl += 1
                elif slot.bird.nest_type == NestType.CAVITY:
                    nest_cavity += 1
                elif slot.bird.nest_type == NestType.GROUND:
                    nest_ground += 1
                elif slot.bird.nest_type == NestType.PLATFORM:
                    nest_platform += 1
                elif slot.bird.nest_type == NestType.WILD:
                    nest_wild += 1
            return {
                "birds_total": board.total_birds(),
                "birds_forest": board.forest.bird_count,
                "birds_grassland": board.grassland.bird_count,
                "birds_wetland": board.wetland.bird_count,
                "eggs_total": board.total_eggs(),
                "cached_total": cached,
                "tucked_total": tucked,
                "brown": brown,
                "white": white,
                "pink": pink,
                "teal": teal,
                "no_power": no_power,
                "high_vp": high_vp,
                "nest_bowl": nest_bowl,
                "nest_cavity": nest_cavity,
                "nest_ground": nest_ground,
                "nest_platform": nest_platform,
                "nest_wild": nest_wild,
            }

        self_stats = _board_stats(p)

        if opponents:
            opp_stats = [_board_stats(op) for op in opponents]
            opp_breakdowns = [score_breakdowns[op.name] for op in opponents]
            opp_scores = [float(b.total) for b in opp_breakdowns]
            opp_mean_cubes = sum(op.action_cubes_remaining for op in opponents) / len(opponents)
            opp_mean_hand = sum(op.hand_size for op in opponents) / len(opponents)
            opp_mean_food = sum(op.food_supply.total() for op in opponents) / len(opponents)
            opp_mean_birds = sum(s["birds_total"] for s in opp_stats) / len(opponents)
            opp_mean_eggs = sum(s["eggs_total"] for s in opp_stats) / len(opponents)
            opp_mean_forest = sum(s["birds_forest"] for s in opp_stats) / len(opponents)
            opp_mean_grass = sum(s["birds_grassland"] for s in opp_stats) / len(opponents)
            opp_mean_wet = sum(s["birds_wetland"] for s in opp_stats) / len(opponents)
            opp_max_birds = max(s["birds_total"] for s in opp_stats)
            opp_max_eggs = max(s["eggs_total"] for s in opp_stats)
            opp_mean_score = sum(opp_scores) / len(opp_scores)
            opp_lead = max(opp_scores) - self_score
            opp_mean_bird_vp = sum(b.bird_vp for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_eggs = sum(b.eggs for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_cached = sum(b.cached_food for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_tucked = sum(b.tucked_cards for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_bonus = sum(b.bonus_cards for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_goals = sum(b.round_goals for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_sb_nectar = sum(b.nectar for b in opp_breakdowns) / len(opp_breakdowns)
            opp_mean_cached = sum(s["cached_total"] for s in opp_stats) / len(opp_stats)
            opp_max_cached = max(s["cached_total"] for s in opp_stats)
            opp_mean_tucked = sum(s["tucked_total"] for s in opp_stats) / len(opp_stats)
            opp_max_tucked = max(s["tucked_total"] for s in opp_stats)
            opp_mean_brown = sum(s["brown"] for s in opp_stats) / len(opp_stats)
            opp_mean_pink = sum(s["pink"] for s in opp_stats) / len(opp_stats)
        else:
            opp_mean_cubes = opp_mean_hand = opp_mean_food = 0.0
            opp_mean_birds = opp_mean_eggs = opp_mean_forest = 0.0
            opp_mean_grass = opp_mean_wet = opp_max_birds = opp_max_eggs = 0.0
            opp_mean_score = 0.0
            opp_lead = 0.0
            opp_mean_bird_vp = opp_mean_sb_eggs = 0.0
            opp_mean_sb_cached = opp_mean_sb_tucked = 0.0
            opp_mean_sb_bonus = opp_mean_sb_goals = opp_mean_sb_nectar = 0.0
            opp_mean_cached = opp_max_cached = 0.0
            opp_mean_tucked = opp_max_tucked = 0.0
            opp_mean_brown = opp_mean_pink = 0.0

        bonus_details = bonus_card_details(p)
        bonus_scores = [score for _, _, score in bonus_details]

        def _bonus_improvable(card, qualifying: int, score: int) -> tuple[bool, bool]:
            # Custom full scorers do not map from qualifying count.
            if card.score(qualifying) != score:
                return (False, False)
            improvable = card.score(qualifying + 1) > score
            if card.is_per_bird:
                return (improvable, False)
            bounded_tiers = [t.max_count for t in card.scoring_tiers if t.max_count is not None]
            maxed = bool(bounded_tiers) and qualifying >= max(bounded_tiers)
            return (improvable, maxed)

        bonus_improvable = 0
        bonus_maxed = 0
        for card, qualifying, score in bonus_details:
            improvable, maxed = _bonus_improvable(card, qualifying, score)
            if improvable:
                bonus_improvable += 1
            if maxed:
                bonus_maxed += 1

        hand = p.hand
        hand_total_vp = float(sum(b.victory_points for b in hand))
        hand_mean_vp = hand_total_vp / len(hand) if hand else 0.0
        hand_best_vp = float(max((b.victory_points for b in hand), default=0))
        hand_mean_cost = (
            sum(b.food_cost.total for b in hand) / len(hand) if hand else 0.0
        )
        hand_affordable_count = sum(1 for b in hand if can_pay_food_cost(p, b.food_cost)[0])
        hand_affordable_frac = hand_affordable_count / len(hand) if hand else 0.0
        hand_mean_egg_limit = (
            sum(b.egg_limit for b in hand) / len(hand) if hand else 0.0
        )
        hand_multi_habitat_frac = (
            sum(1 for b in hand if len(b.habitats) > 1) / len(hand) if hand else 0.0
        )

        board = p.board
        slots_remaining_forest = 5 - board.forest.bird_count
        slots_remaining_grassland = 5 - board.grassland.bird_count
        slots_remaining_wetland = 5 - board.wetland.bird_count
        egg_capacity_remaining = sum(slot.eggs_space() for _, _, slot in board.all_slots())
        egg_total = board.total_eggs()
        egg_fill_rate = egg_total / (egg_total + egg_capacity_remaining) if (egg_total + egg_capacity_remaining) > 0 else 0.0

        forest_col = get_action_column(game.board_type, Habitat.FOREST, board.forest.bird_count)
        grass_col = get_action_column(game.board_type, Habitat.GRASSLAND, board.grassland.bird_count)
        wet_col = get_action_column(game.board_type, Habitat.WETLAND, board.wetland.bird_count)

        action_gain_forest = forest_col.base_gain
        action_gain_grassland = grass_col.base_gain
        action_gain_wetland = wet_col.base_gain
        has_bonus_forest = 1.0 if (forest_col.bonus or forest_col.reset_bonus) else 0.0
        has_bonus_grassland = 1.0 if (grass_col.bonus or grass_col.reset_bonus) else 0.0
        has_bonus_wetland = 1.0 if (wet_col.bonus or wet_col.reset_bonus) else 0.0

        def _round_goal_features(round_num: int) -> tuple[float, float, float, float]:
            if round_num - 1 >= len(game.round_goals):
                return (0.0, 0.0, 0.0, 0.0)
            goal = game.round_goals[round_num - 1]
            is_no_goal = goal.description.lower() == "no goal"
            active = 1.0 if (round_num >= game.current_round and not is_no_goal) else 0.0
            self_progress = 0.0 if is_no_goal else float(goal_progress_for_round(p, goal))
            opp_max_progress = 0.0
            if opponents and not is_no_goal:
                opp_max_progress = max(float(goal_progress_for_round(op, goal)) for op in opponents)
            earned = float(game.round_goal_scores.get(round_num, {}).get(p.name, 0))
            return (active, self_progress, opp_max_progress, earned)

        goal_features: list[float] = []
        for rn in (1, 2, 3, 4):
            active, self_prog, opp_prog, earned = _round_goal_features(rn)
            goal_features.extend([
                _clamp01(active),
                _clamp01(self_prog / 10.0),
                _clamp01(opp_prog / 10.0),
                _clamp01(earned / 5.0),
            ])

        cubes_spent_this_round = max(0, game.actions_this_round - p.action_cubes_remaining)
        mean_round_spent = 0.0
        if game.actions_this_round > 0:
            mean_round_spent = sum(max(0, game.actions_this_round - pl.action_cubes_remaining) for pl in game.players) / (
                game.num_players * game.actions_this_round
            )
        game_progress = ((game.current_round - 1) + mean_round_spent) / 4.0
        rounds_remaining = max(0, 4 - game.current_round)

        all_scores = [float(score_breakdowns[pl.name].total) for pl in game.players]
        best_score = max(all_scores) if all_scores else 0.0
        worst_score = min(all_scores) if all_scores else 0.0
        score_rank = sum(1 for sc in all_scores if sc > self_score)
        rank_den = max(1, game.num_players - 1)
        score_gap_to_first = max(0.0, best_score - self_score)
        score_gap_to_last = max(0.0, self_score - worst_score)

        prev_actions = sum(_effective_actions(rn) for rn in range(1, game.current_round))
        cubes_spent_total = max(0, prev_actions + cubes_spent_this_round)
        efficiency = self_score / max(1, cubes_spent_total)

        tray_birds = game.card_tray.face_up
        tray_mean_vp = sum(b.victory_points for b in tray_birds) / len(tray_birds) if tray_birds else 0.0
        tray_affordable_count = sum(1 for b in tray_birds if can_pay_food_cost(p, b.food_cost)[0])

        nectar_leads: list[float] = []
        nectar_winning = 0
        nectar_tied = 0
        for hab in (Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND):
            self_nectar = p.board.get_row(hab).nectar_spent
            opp_max = max((op.board.get_row(hab).nectar_spent for op in opponents), default=0)
            lead = float(self_nectar - opp_max)
            nectar_leads.append(lead)
            max_all = max(self_nectar, opp_max)
            if self_nectar == max_all and self_nectar > opp_max:
                nectar_winning += 1
            elif self_nectar == max_all and any(op.board.get_row(hab).nectar_spent == self_nectar for op in opponents):
                nectar_tied += 1
        nectar_total_spent = (
            p.board.forest.nectar_spent + p.board.grassland.nectar_spent + p.board.wetland.nectar_spent
        )

        def _food_gap_for_bird(bird) -> int:
            cost = bird.food_cost
            if cost.total == 0:
                return 0
            if cost.is_or:
                if FoodType.WILD in cost.distinct_types and p.food_supply.total() > 0:
                    return 0
                if any(p.food_supply.has(ft) for ft in cost.distinct_types if ft != FoodType.WILD):
                    return 0
                if p.food_supply.has(FoodType.NECTAR):
                    return 0
                return 1

            required: dict[FoodType, int] = {}
            for ft in cost.items:
                required[ft] = required.get(ft, 0) + 1
            wild_needed = required.pop(FoodType.WILD, 0)

            specific_shortfall = 0
            non_nectar_committed = 0
            for ft, count in required.items():
                have = p.food_supply.get(ft)
                covered = min(have, count)
                non_nectar_committed += covered
                specific_shortfall += count - covered

            non_nectar_surplus = max(0, p.food_supply.total_non_nectar() - non_nectar_committed)
            flexible = non_nectar_surplus + p.food_supply.get(FoodType.NECTAR)
            return max(0, specific_shortfall + wild_needed - flexible)

        eggs_total = p.board.total_eggs()
        playable_birds: list = []
        egg_gaps_all: list[int] = []
        for bird in hand:
            affordable, _ = can_pay_food_cost(p, bird.food_cost)
            row_egg_gaps: list[int] = []
            for hab in bird.habitats:
                row = p.board.get_row(hab)
                slot_idx = row.next_empty_slot()
                if slot_idx is None:
                    continue
                egg_cost = EGG_COST_BY_COLUMN[slot_idx]
                row_egg_gaps.append(max(0, egg_cost - eggs_total))
            if row_egg_gaps:
                egg_gaps_all.append(min(row_egg_gaps))
            if affordable and row_egg_gaps and min(row_egg_gaps) == 0:
                playable_birds.append(bird)

        playable_count = len(playable_birds)
        playable_frac = playable_count / len(hand) if hand else 0.0
        best_playable_vp = float(max((b.victory_points for b in playable_birds), default=0))

        if hand:
            best_hand_bird = max(hand, key=lambda b: b.victory_points)
            food_gap_best_bird = float(_food_gap_for_bird(best_hand_bird))
        else:
            food_gap_best_bird = 0.0

        egg_gap = float(min(egg_gaps_all)) if egg_gaps_all else 10.0

        food_type_diversity = sum(
            1 for ft in (
                FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                FoodType.FRUIT, FoodType.RODENT, FoodType.NECTAR,
            )
            if p.food_supply.get(ft) > 0
        )
        total_resources = p.food_supply.total() + p.board.total_eggs()

        current_goal = game.current_round_goal()
        if current_goal and current_goal.description.lower() != "no goal":
            cur_progress_by_player = {
                pl.name: float(goal_progress_for_round(pl, current_goal))
                for pl in game.players
            }
            cur_self_progress = cur_progress_by_player[p.name]
            cur_best = max(cur_progress_by_player.values()) if cur_progress_by_player else 0.0
            cur_rank = sum(1 for val in cur_progress_by_player.values() if val > cur_self_progress)
            cur_gap_to_first = max(0.0, cur_best - cur_self_progress)
        else:
            cur_self_progress = 0.0
            cur_rank = 0
            cur_gap_to_first = 0.0

        past_total_earned = sum(
            game.round_goal_scores.get(rn, {}).get(p.name, 0)
            for rn in range(1, game.current_round)
        )
        future_goals_remaining = sum(
            1
            for rn in range(game.current_round + 1, 5)
            if rn - 1 < len(game.round_goals) and game.round_goals[rn - 1].description.lower() != "no goal"
        )
        has_no_goal_round = 1.0 if any(g.description.lower() == "no goal" for g in game.round_goals) else 0.0

        vec = [
            _clamp01(game.current_round / 4.0),
            _clamp01(game.turn_in_round / 8.0),
            _clamp01(game.current_player_idx / max(1, game.num_players - 1)),
            _clamp01(game.total_actions_remaining / 120.0),
            _clamp01(game.deck_remaining / self.max_deck),
            _clamp01(len(game.card_tray.face_up) / 3.0),
            _clamp01(feeder_counts[FoodType.INVERTEBRATE] / 5.0),
            _clamp01(feeder_counts[FoodType.SEED] / 5.0),
            _clamp01(feeder_counts[FoodType.FISH] / 5.0),
            _clamp01(feeder_counts[FoodType.FRUIT] / 5.0),
            _clamp01(feeder_counts[FoodType.RODENT] / 5.0),
            _clamp01(feeder_counts[FoodType.NECTAR] / 5.0),
            _clamp01(p.action_cubes_remaining / 8.0),
            _clamp01(p.hand_size / self.max_hand),
            _clamp01(p.food_supply.total() / self.max_food),
            _clamp01(p.food_supply.invertebrate / self.max_food),
            _clamp01(p.food_supply.seed / self.max_food),
            _clamp01(p.food_supply.fish / self.max_food),
            _clamp01(p.food_supply.fruit / self.max_food),
            _clamp01(p.food_supply.rodent / self.max_food),
            _clamp01(p.food_supply.nectar / self.max_food),
            _clamp01(self_stats["birds_total"] / 15.0),
            _clamp01(self_stats["birds_forest"] / 5.0),
            _clamp01(self_stats["birds_grassland"] / 5.0),
            _clamp01(self_stats["birds_wetland"] / 5.0),
            _clamp01(self_stats["eggs_total"] / 30.0),
            _clamp01(self_stats["cached_total"] / 20.0),
            _clamp01(self_stats["tucked_total"] / 30.0),
            _clamp01(self_stats["brown"] / 15.0),
            _clamp01(self_stats["white"] / 15.0),
            _clamp01(self_stats["pink"] / 15.0),
            _clamp01(self_stats["teal"] / 15.0),
            _clamp01(self_stats["high_vp"] / 15.0),
            _clamp01(p.board.forest.nectar_spent / 10.0),
            _clamp01(p.board.grassland.nectar_spent / 10.0),
            _clamp01(p.board.wetland.nectar_spent / 10.0),
            _clamp01(opp_mean_cubes / 8.0),
            _clamp01(opp_mean_hand / self.max_hand),
            _clamp01(opp_mean_food / self.max_food),
            _clamp01(opp_mean_birds / 15.0),
            _clamp01(opp_mean_eggs / 30.0),
            _clamp01(opp_mean_forest / 5.0),
            _clamp01(opp_mean_grass / 5.0),
            _clamp01(opp_mean_wet / 5.0),
            _clamp01(opp_max_birds / 15.0),
            _clamp01(opp_max_eggs / 30.0),
            _clamp01(opp_mean_score / self.max_score),
            _clamp01((opp_lead + self.max_score) / (2.0 * self.max_score)),
            _clamp01(self_breakdown.bird_vp / 80.0),
            _clamp01(self_breakdown.eggs / 30.0),
            _clamp01(self_breakdown.cached_food / 20.0),
            _clamp01(self_breakdown.tucked_cards / 30.0),
            _clamp01(self_breakdown.bonus_cards / 30.0),
            _clamp01(self_breakdown.round_goals / 20.0),
            _clamp01(self_breakdown.nectar / 15.0),
        ]

        vec.extend(goal_features)

        vec.extend([
            _clamp01(len(bonus_details) / 4.0),
            _clamp01(sum(bonus_scores) / 30.0),
            _clamp01(max(bonus_scores, default=0) / 10.0),
            _clamp01(min(bonus_scores, default=0) / 10.0),
            _clamp01(bonus_improvable / 4.0),
            _clamp01(bonus_maxed / 4.0),
            _clamp01(hand_total_vp / 60.0),
            _clamp01(hand_mean_vp / 9.0),
            _clamp01(hand_best_vp / 9.0),
            _clamp01(hand_mean_cost / 5.0),
            _clamp01(hand_affordable_count / 10.0),
            _clamp01(hand_affordable_frac),
            _clamp01(hand_mean_egg_limit / 6.0),
            _clamp01(hand_multi_habitat_frac),
            _clamp01(slots_remaining_forest / 5.0),
            _clamp01(slots_remaining_grassland / 5.0),
            _clamp01(slots_remaining_wetland / 5.0),
            _clamp01(egg_capacity_remaining / 30.0),
            _clamp01(egg_fill_rate),
            _clamp01(action_gain_forest / 4.0),
            _clamp01(action_gain_grassland / 4.0),
            _clamp01(action_gain_wetland / 4.0),
            _clamp01(has_bonus_forest),
            _clamp01(has_bonus_grassland),
            _clamp01(has_bonus_wetland),
            _clamp01(self_breakdown.bird_vp / 80.0),
            _clamp01(self_stats["nest_bowl"] / 5.0),
            _clamp01(self_stats["nest_cavity"] / 5.0),
            _clamp01(self_stats["nest_ground"] / 5.0),
            _clamp01(self_stats["nest_platform"] / 5.0),
            _clamp01(self_stats["nest_wild"] / 5.0),
            _clamp01(game_progress),
            _clamp01(cubes_spent_this_round / 8.0),
            _clamp01(game.num_players / 4.0),
            _clamp01(1.0 if game.board_type == BoardType.OCEANIA else 0.0),
            _clamp01(rounds_remaining / 4.0),
            _clamp01(opp_mean_bird_vp / 80.0),
            _clamp01(opp_mean_sb_eggs / 30.0),
            _clamp01(opp_mean_sb_cached / 20.0),
            _clamp01(opp_mean_sb_tucked / 30.0),
            _clamp01(opp_mean_sb_bonus / 30.0),
            _clamp01(opp_mean_sb_goals / 20.0),
            _clamp01(opp_mean_sb_nectar / 15.0),
            _clamp01(opp_mean_cached / 20.0),
            _clamp01(opp_max_cached / 30.0),
            _clamp01(opp_mean_tucked / 30.0),
            _clamp01(opp_max_tucked / 40.0),
            _clamp01(opp_mean_brown / 15.0),
            _clamp01(opp_mean_pink / 15.0),
            _clamp01((nectar_leads[0] + 10.0) / 20.0),
            _clamp01((nectar_leads[1] + 10.0) / 20.0),
            _clamp01((nectar_leads[2] + 10.0) / 20.0),
            _clamp01(nectar_winning / 3.0),
            _clamp01(nectar_tied / 3.0),
            _clamp01(nectar_total_spent / 15.0),
            _clamp01(playable_count / 10.0),
            _clamp01(playable_frac),
            _clamp01(best_playable_vp / 9.0),
            _clamp01(food_gap_best_bird / 5.0),
            _clamp01(egg_gap / 10.0),
            _clamp01(1.0 if playable_count > 0 else 0.0),
            _clamp01(food_type_diversity / 6.0),
            _clamp01(total_resources / 20.0),
            _clamp01(self_score / self.max_score),
            _clamp01(score_rank / rank_den),
            _clamp01(score_gap_to_first / 50.0),
            _clamp01(score_gap_to_last / 50.0),
            _clamp01(efficiency / 5.0),
            _clamp01(tray_mean_vp / 9.0),
            _clamp01(tray_affordable_count / 3.0),
            _clamp01(self_stats["no_power"] / 5.0),
            _clamp01(cur_self_progress / 10.0),
            _clamp01(cur_rank / rank_den),
            _clamp01(cur_gap_to_first / 5.0),
            _clamp01(past_total_earned / 20.0),
            _clamp01(future_goals_remaining / 4.0),
            _clamp01(has_no_goal_round),
        ])

        if self.enable_identity_features:
            vec.extend(self._encode_identity_features(game, player_index))
        if self.use_per_slot_encoding:
            vec.extend(self._encode_per_slot(game, player_index))
        if self.use_hand_habitat_features:
            vec.extend(self._encode_hand_habitat_features(game, player_index))
        if self.use_tray_per_slot_encoding:
            vec.extend(self._encode_tray_per_slot(game, player_index))
        if self.use_opponent_board_encoding:
            vec.extend(self._encode_opponent_board(game, player_index))
        if self.use_power_features:
            vec.extend(self._encode_self_power_features(game, player_index))
        return vec
