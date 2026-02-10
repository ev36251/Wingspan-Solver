"""State vector encoding for Wingspan neural-network training."""

from __future__ import annotations

from dataclasses import dataclass

from backend.models.enums import FoodType, PowerColor
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

    def feature_names(self) -> list[str]:
        return [
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
        ]

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

        def _player_est_score(player) -> float:
            from backend.engine.scoring import calculate_score
            return float(calculate_score(game, player).total)

        self_score = _player_est_score(p)

        def _board_stats(player):
            board = player.board
            brown = 0
            white = 0
            pink = 0
            teal = 0
            high_vp = 0
            cached = 0
            tucked = 0
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
                "high_vp": high_vp,
            }

        self_stats = _board_stats(p)

        if opponents:
            opp_stats = [_board_stats(op) for op in opponents]
            opp_scores = [_player_est_score(op) for op in opponents]
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
        else:
            opp_mean_cubes = opp_mean_hand = opp_mean_food = 0.0
            opp_mean_birds = opp_mean_eggs = opp_mean_forest = 0.0
            opp_mean_grass = opp_mean_wet = opp_max_birds = opp_max_eggs = 0.0
            opp_mean_score = 0.0
            opp_lead = 0.0

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
        ]

        return vec
