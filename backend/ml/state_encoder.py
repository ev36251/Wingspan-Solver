"""State vector encoding for Wingspan neural-network training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from backend.config import ACTIONS_PER_ROUND, EGG_COST_BY_COLUMN, get_action_column
from backend.engine.rules import can_pay_food_cost
from backend.engine.scoring import bonus_card_details, calculate_score, goal_progress_for_round
from backend.models.enums import BoardType, FoodType, Habitat, NestType, PowerColor
from backend.models.game_state import GameState


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class StateEncoder:
    """Encode a game state into a fixed-size float feature vector.

    Features are player-centric: the acting player is encoded first,
    then opponent aggregate features.
    """

    max_hand: float = 25.0
    max_food: float = 12.0
    max_score: float = 180.0
    max_deck: float = 200.0
    enable_identity_features: bool = False
    identity_hash_dim: int = 128

    @classmethod
    def from_metadata(cls, meta: dict | None) -> "StateEncoder":
        cfg = (meta or {}).get("state_encoder", {}) if isinstance(meta, dict) else {}
        return cls(
            enable_identity_features=bool(cfg.get("enable_identity_features", False)),
            identity_hash_dim=max(1, int(cfg.get("identity_hash_dim", 128))),
        )

    @classmethod
    def resolve_for_model(
        cls,
        model_meta: dict | None,
        *,
        fallback_enable_identity_features: bool = False,
        fallback_identity_hash_dim: int = 128,
    ) -> "StateEncoder":
        if isinstance(model_meta, dict) and isinstance(model_meta.get("state_encoder"), dict):
            return cls.from_metadata(model_meta)
        return cls(
            enable_identity_features=bool(fallback_enable_identity_features),
            identity_hash_dim=max(1, int(fallback_identity_hash_dim)),
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
        return base

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
        return vec
