# Training Summary

Runs: self-play evolutionary training using current heuristics with round-goal scoring and tray refills enabled.

## 2 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 34.37
Best win rate: 0.592
Best fitness: 41.82

Top weight deltas vs defaults:
- egg_points: 1.893 (default 1.000, +89%)
- habitat_diversity_bonus: 0.798 (default 0.500, +60%)
- cached_food_points: 1.489 (default 1.000, +49%)
- nectar_early_bonus: 0.579 (default 0.400, +45%)
- engine_value: 0.391 (default 0.700, -44%)
- grassland_egg_synergy: 0.842 (default 0.600, +40%)
- tucked_card_points: 0.618 (default 1.000, -38%)
- food_for_birds_bonus: 0.460 (default 0.400, +15%)

## 3 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 31.91
Best win rate: 0.635
Best fitness: 41.39

Top weight deltas vs defaults:
- egg_points: 1.813 (default 1.000, +81%)
- habitat_diversity_bonus: 0.886 (default 0.500, +77%)
- action_cubes: 0.316 (default 0.200, +58%)
- food_in_supply: 0.457 (default 0.300, +52%)
- early_game_engine_bonus: 0.150 (default 0.300, -50%)
- play_another_bird_bonus: 1.270 (default 2.500, -49%)
- nectar_early_bonus: 0.236 (default 0.400, -41%)
- bonus_card_points: 0.612 (default 1.000, -39%)

## 4 Players

Config: games_per_matchup=1000 · generations=3 · pop_size=4 · board_type=oceania

Best avg score: 29.32
Best win rate: 0.345
Best fitness: 30.87

Top weight deltas vs defaults:
- food_for_birds_bonus: 0.754 (default 0.400, +89%)
- egg_points: 1.728 (default 1.000, +73%)
- nectar_points: 1.547 (default 1.000, +55%)
- predator_penalty: 0.765 (default 0.500, +53%)
- bird_vp: 1.513 (default 1.000, +51%)
- cards_in_hand: 0.178 (default 0.350, -49%)
- tucked_card_points: 1.410 (default 1.000, +41%)
- action_cubes: 0.123 (default 0.200, -39%)

## Notes

- Training used 3 generations with population size 4 for each player count (1000 games per matchup per generation).
- Round-goal scores are now computed at the end of each round during simulation.
- Card tray refills are simulated from the deck after each turn.
- Results saved as JSON in `reports/` for inspection or future training continuation.