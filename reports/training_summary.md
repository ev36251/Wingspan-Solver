# Training Summary

Runs: self-play evolutionary training using current heuristics baseline.

## 2 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 51.27
Best win rate: 0.592
Best fitness: 53.65

Top weight deltas vs defaults:
- bird_vp: 1.000 (default 1.000, +0%)
- egg_points: 1.000 (default 1.000, +0%)
- cached_food_points: 1.000 (default 1.000, +0%)
- tucked_card_points: 1.000 (default 1.000, +0%)
- bonus_card_points: 1.000 (default 1.000, +0%)
- round_goal_points: 1.000 (default 1.000, +0%)
- nectar_points: 1.000 (default 1.000, +0%)
- engine_value: 0.700 (default 0.700, +0%)

## 3 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 50.08
Best win rate: 0.535
Best fitness: 51.1

Top weight deltas vs defaults:
- early_game_engine_bonus: 0.562 (default 0.300, +87%)
- nectar_early_bonus: 0.677 (default 0.400, +69%)
- goal_alignment: 1.995 (default 1.200, +66%)
- cards_in_hand: 0.547 (default 0.350, +56%)
- egg_points: 1.480 (default 1.000, +48%)
- bird_vp: 0.555 (default 1.000, -45%)
- bonus_card_points: 0.573 (default 1.000, -43%)
- tucked_card_points: 0.576 (default 1.000, -42%)

## 4 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 50.18
Best win rate: 0.27
Best fitness: 43.21

Top weight deltas vs defaults:
- cards_in_hand: 0.658 (default 0.350, +88%)
- egg_points: 1.806 (default 1.000, +81%)
- tucked_card_points: 1.630 (default 1.000, +63%)
- nectar_points: 0.513 (default 1.000, -49%)
- play_another_bird_bonus: 3.532 (default 2.500, +41%)
- predator_penalty: 0.706 (default 0.500, +41%)
- habitat_diversity_bonus: 0.701 (default 0.500, +40%)
- bonus_card_points: 0.614 (default 1.000, -39%)

## Notes

- Training used 3 generations with population size 4 for each player count (1000 games per matchup per generation).
- Results saved as JSON in `reports/` for inspection or future training continuation.