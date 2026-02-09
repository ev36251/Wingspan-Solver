"""Self-play training to discover improved heuristic weights.

Runs an evolutionary strategy:
1. Maintain a population of HeuristicWeights configurations
2. For each generation, play tournament matches between configurations
3. Select winners, apply crossover and mutation
4. Repeat until convergence

Usage:
    python -m backend.solver.self_play --games 10 --players 2
    python -m backend.solver.self_play --generations 50 --pop-size 20 --players 3
"""

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass, fields, asdict
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry, get_bonus_registry, get_goal_registry
from backend.models.enums import BoardType, ActionType, FoodType
from backend.models.game_state import create_new_game, GameState
from backend.models.player import Player
from backend.engine.scoring import calculate_score
from backend.solver.heuristics import HeuristicWeights, DEFAULT_WEIGHTS, dynamic_weights
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import deep_copy_game, execute_move_on_sim, _refill_tray, _score_round_goal


@dataclass
class TrainingConfig:
    """Configuration for the evolutionary training run."""
    population_size: int = 20
    generations: int = 50
    games_per_matchup: int = 10
    num_players: int = 2
    board_type: str = "oceania"
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.15
    crossover_rate: float = 0.5
    elitism: int = 2
    tournament_k: int = 3
    convergence_patience: int = 10
    convergence_threshold: float = 0.01
    output_path: str = "training_results.json"
    seed: int | None = None


@dataclass
class Individual:
    """One member of the population."""
    weights: HeuristicWeights
    fitness: float = 0.0
    avg_score: float = 0.0
    win_rate: float = 0.0
    games_played: int = 0


WEIGHT_FIELDS = [f.name for f in fields(HeuristicWeights)]
WEIGHT_BOUNDS = (0.05, 5.0)


def _clamp(val: float) -> float:
    return max(WEIGHT_BOUNDS[0], min(WEIGHT_BOUNDS[1], val))


def random_individual() -> Individual:
    """Create a random individual near the defaults."""
    w = HeuristicWeights()
    for f in WEIGHT_FIELDS:
        default_val = getattr(w, f)
        multiplier = 2.0 ** random.uniform(-1, 1)  # 0.5x to 2.0x
        setattr(w, f, _clamp(default_val * multiplier))
    return Individual(weights=w)


def mutate(individual: Individual, config: TrainingConfig) -> Individual:
    """Apply Gaussian mutation to a fraction of genes."""
    w = copy.deepcopy(individual.weights)
    for f in WEIGHT_FIELDS:
        if random.random() < config.mutation_rate:
            val = getattr(w, f)
            factor = 2.0 ** random.gauss(0, config.mutation_sigma)
            setattr(w, f, _clamp(val * factor))
    return Individual(weights=w)


def crossover(parent_a: Individual, parent_b: Individual,
              config: TrainingConfig) -> Individual:
    """Uniform crossover: each gene from A or B."""
    w = HeuristicWeights()
    for f in WEIGHT_FIELDS:
        if random.random() < config.crossover_rate:
            setattr(w, f, getattr(parent_b.weights, f))
        else:
            setattr(w, f, getattr(parent_a.weights, f))
    return Individual(weights=w)


def tournament_select(population: list[Individual], k: int) -> Individual:
    """Select the best individual from a random sample of k."""
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def create_training_game(num_players: int, board_type: BoardType) -> GameState:
    """Create a game state suitable for self-play training."""
    bird_reg = get_bird_registry()
    goal_reg = get_goal_registry()
    bonus_reg = get_bonus_registry()

    player_names = [f"Player_{i+1}" for i in range(num_players)]

    # Pick 4 random round goals
    all_goals = goal_reg.all_goals
    round_goals = random.sample(all_goals, min(4, len(all_goals)))

    game = create_new_game(player_names, round_goals, board_type)

    # Deal random hands (5 birds each) and bonus cards (1 each)
    all_birds = list(bird_reg.all_birds)
    random.shuffle(all_birds)
    all_bonus = list(bonus_reg.all_cards)

    idx = 0
    for player in game.players:
        hand_size = min(5, len(all_birds) - idx)
        player.hand = all_birds[idx:idx + hand_size]
        idx += hand_size
        player.bonus_cards = [random.choice(all_bonus)]

        # Starting food: 1 of each non-nectar type
        for ft in [FoodType.INVERTEBRATE, FoodType.SEED, FoodType.FISH,
                   FoodType.FRUIT, FoodType.RODENT]:
            player.food_supply.add(ft, 1)

    # Set up card tray
    tray_count = min(3, len(all_birds) - idx)
    for _ in range(tray_count):
        if idx < len(all_birds):
            game.card_tray.add_card(all_birds[idx])
            idx += 1

    # Set up birdfeeder with initial roll
    game.birdfeeder.reroll()
    game.deck_remaining = max(0, len(all_birds) - idx)

    return game


def _pick_move_with_weights(moves: list[Move], game: GameState,
                            player: Player,
                            base_weights: HeuristicWeights) -> Move:
    """Epsilon-greedy move selection using specific weights."""
    from backend.solver.heuristics import _estimate_move_value

    weights = dynamic_weights(game, base_weights)
    scored = [(m, _estimate_move_value(game, player, m, weights)) for m in moves]
    scored.sort(key=lambda x: -x[1])

    if random.random() < 0.9:
        return scored[0][0]

    top_n = scored[:min(3, len(scored))]
    min_score = min(s for _, s in top_n)
    sel_weights = [max(s - min_score + 0.5, 0.1) for _, s in top_n]
    return random.choices([m for m, _ in top_n], weights=sel_weights, k=1)[0]


def play_head_to_head(game: GameState,
                      player_weights: dict[str, HeuristicWeights],
                      max_turns: int = 200) -> dict[str, int]:
    """Play a full game where each player uses their own weight configuration."""
    sim = deep_copy_game(game)
    turns = 0

    while not sim.is_game_over and turns < max_turns:
        player = sim.current_player
        if player.action_cubes_remaining <= 0:
            if all(p.action_cubes_remaining <= 0 for p in sim.players):
                _score_round_goal(sim, sim.current_round)
                sim.advance_round()
                continue
            sim.current_player_idx = (sim.current_player_idx + 1) % sim.num_players
            continue

        moves = generate_all_moves(sim, player)
        if not moves:
            player.action_cubes_remaining = 0
            sim.current_player_idx = (sim.current_player_idx + 1) % sim.num_players
            continue

        base = player_weights.get(player.name, DEFAULT_WEIGHTS)
        move = _pick_move_with_weights(moves, sim, player, base)
        success = execute_move_on_sim(sim, player, move)

        if success:
            sim.advance_turn()
            _refill_tray(sim)
        else:
            fallback_ok = False
            for m in moves:
                if m.action_type in (ActionType.GAIN_FOOD, ActionType.LAY_EGGS):
                    if execute_move_on_sim(sim, player, m):
                        sim.advance_turn()
                        _refill_tray(sim)
                        fallback_ok = True
                        break
            if not fallback_ok:
                player.action_cubes_remaining -= 1
                sim.advance_turn()
                _refill_tray(sim)

        turns += 1

    return {p.name: calculate_score(sim, p).total for p in sim.players}


def evaluate_population(population: list[Individual],
                        config: TrainingConfig) -> None:
    """Evaluate all individuals via Swiss-style pairing."""
    board_type = BoardType(config.board_type)

    for ind in population:
        ind.fitness = 0.0
        ind.avg_score = 0.0
        ind.win_rate = 0.0
        ind.games_played = 0

    # Sort by current fitness for Swiss pairing
    population.sort(key=lambda ind: -ind.fitness)

    scores_by_idx: dict[int, list[int]] = {i: [] for i in range(len(population))}
    wins_by_idx: dict[int, int] = {i: 0 for i in range(len(population))}

    # Pair adjacent individuals
    for i in range(0, len(population) - 1, 2):
        for _ in range(config.games_per_matchup):
            game = create_training_game(config.num_players, board_type)

            # Assign weights: Player_1 uses population[i], Player_2 uses population[i+1]
            pw: dict[str, HeuristicWeights] = {}
            for pi, p in enumerate(game.players):
                if pi % 2 == 0:
                    pw[p.name] = population[i].weights
                else:
                    pw[p.name] = population[i + 1].weights

            results = play_head_to_head(game, pw)

            # Tally: even-indexed players for population[i], odd for population[i+1]
            a_scores = [s for name, s in results.items()
                        if int(name.split('_')[1]) % 2 == 1]  # Player_1 is odd (1-based)
            b_scores = [s for name, s in results.items()
                        if int(name.split('_')[1]) % 2 == 0]  # Player_2 is even

            if a_scores:
                scores_by_idx[i].extend(a_scores)
                population[i].games_played += len(a_scores)
            if b_scores:
                scores_by_idx[i + 1].extend(b_scores)
                population[i + 1].games_played += len(b_scores)

            # Win tracking (use Player_1 vs Player_2 for 2-player)
            p1_score = results.get("Player_1", 0)
            p2_score = results.get("Player_2", 0)
            if p1_score > p2_score:
                wins_by_idx[i] += 1
            elif p2_score > p1_score:
                wins_by_idx[i + 1] += 1

    # Compute fitness
    for idx, ind in enumerate(population):
        if scores_by_idx[idx]:
            ind.avg_score = sum(scores_by_idx[idx]) / len(scores_by_idx[idx])
        if ind.games_played > 0:
            ind.win_rate = wins_by_idx[idx] / ind.games_played
        ind.fitness = ind.avg_score * 0.7 + ind.win_rate * 30.0


def evolve(config: TrainingConfig) -> dict:
    """Run the full evolutionary training loop."""
    if config.seed is not None:
        random.seed(config.seed)

    # Initialize population: default weights + random variations
    population = [Individual(weights=copy.deepcopy(DEFAULT_WEIGHTS))]
    for _ in range(config.population_size - 1):
        population.append(random_individual())

    best_ever: Individual | None = None
    best_history: list[dict] = []
    no_improvement_count = 0

    for gen in range(config.generations):
        gen_start = time.time()

        evaluate_population(population, config)

        population.sort(key=lambda ind: -ind.fitness)
        gen_best = population[0]
        gen_elapsed = time.time() - gen_start

        print(f"Gen {gen + 1}/{config.generations}: "
              f"best_fitness={gen_best.fitness:.1f} "
              f"avg_score={gen_best.avg_score:.1f} "
              f"win_rate={gen_best.win_rate:.1%} "
              f"({gen_elapsed:.1f}s)")

        best_history.append({
            "generation": gen + 1,
            "best_fitness": round(gen_best.fitness, 2),
            "best_avg_score": round(gen_best.avg_score, 2),
            "best_win_rate": round(gen_best.win_rate, 3),
            "weights": asdict(gen_best.weights),
        })

        # Track best ever
        if best_ever is None or gen_best.fitness > best_ever.fitness:
            if best_ever is not None:
                improvement = ((gen_best.fitness - best_ever.fitness)
                               / max(1, abs(best_ever.fitness)))
                if improvement > config.convergence_threshold:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            best_ever = copy.deepcopy(gen_best)
        else:
            no_improvement_count += 1

        # Check convergence
        if no_improvement_count >= config.convergence_patience:
            print(f"Converged after {gen + 1} generations "
                  f"(no improvement for {config.convergence_patience} generations)")
            break

        # Create next generation
        next_gen: list[Individual] = []

        # Elitism: keep top performers
        for i in range(min(config.elitism, len(population))):
            next_gen.append(copy.deepcopy(population[i]))

        # Fill rest with crossover + mutation
        while len(next_gen) < config.population_size:
            parent_a = tournament_select(population, config.tournament_k)
            parent_b = tournament_select(population, config.tournament_k)
            child = crossover(parent_a, parent_b, config)
            child = mutate(child, config)
            next_gen.append(child)

        population = next_gen

    return {
        "best_weights": asdict(best_ever.weights) if best_ever else {},
        "best_fitness": round(best_ever.fitness, 2) if best_ever else 0,
        "best_avg_score": round(best_ever.avg_score, 2) if best_ever else 0,
        "best_win_rate": round(best_ever.win_rate, 3) if best_ever else 0,
        "default_weights": asdict(DEFAULT_WEIGHTS),
        "config": asdict(config),
        "history": best_history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Self-play training for Wingspan heuristic weights")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per matchup evaluation")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--board-type", default="oceania",
                        choices=["base", "oceania"])
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="training_results.json")
    args = parser.parse_args()

    print("Loading data registries...")
    load_all(EXCEL_FILE)

    config = TrainingConfig(
        population_size=args.pop_size,
        generations=args.generations,
        games_per_matchup=args.games,
        num_players=args.players,
        board_type=args.board_type,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        seed=args.seed,
        output_path=args.output,
    )

    print(f"Starting self-play training: {config.population_size} individuals, "
          f"{config.generations} generations, {config.num_players} players")

    results = evolve(config)

    output_path = Path(config.output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_path}")
    print(f"Best avg score: {results['best_avg_score']}")
    print(f"Best win rate: {results['best_win_rate']:.1%}")
    print(f"\nBest weights (compare to defaults):")
    for key in results['best_weights']:
        best_val = results['best_weights'][key]
        default_val = results['default_weights'][key]
        diff = ((best_val / default_val) - 1) * 100 if default_val else 0
        marker = " *" if abs(diff) > 20 else ""
        print(f"  {key}: {best_val:.3f} (default: {default_val:.3f}, "
              f"{diff:+.0f}%){marker}")


if __name__ == "__main__":
    main()
