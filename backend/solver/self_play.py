"""Self-play training to discover improved heuristic weights.

Primary optimizer: CMA-ES against a fixed baseline opponent.
Fallback optimizer: legacy evolutionary tournament.

Usage:
    python -m backend.solver.self_play --games 10 --players 2
    python -m backend.solver.self_play --generations 50 --pop-size 20 --players 3
"""

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, fields, asdict
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all, get_bird_registry, get_bonus_registry, get_goal_registry
from backend.models.enums import BoardType, ActionType, FoodType, PowerColor
from backend.models.game_state import create_new_game, GameState
from backend.models.player import Player
from backend.engine.scoring import calculate_score
from backend.solver.setup_advisor import analyze_setup
from backend.solver.heuristics import HeuristicWeights, DEFAULT_WEIGHTS, dynamic_weights
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import deep_copy_game, execute_move_on_sim, _refill_tray, _score_round_goal
from backend.solver.deck_tracker import DeckTracker

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None


@dataclass
class TrainingConfig:
    """Configuration for self-play optimization run."""
    population_size: int = 20
    generations: int = 50
    games_per_matchup: int = 10
    num_players: int = 2
    board_type: str = "oceania"
    optimizer: str = "cmaes"
    cma_sigma: float = 0.15
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.15
    crossover_rate: float = 0.5
    elitism: int = 2
    tournament_k: int = 3
    convergence_patience: int = 10
    convergence_threshold: float = 0.01
    output_path: str = "training_results.json"
    seed: int | None = None
    strict_rules_mode: bool = False
    training_mode: str = "global"


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
WeightProfile = HeuristicWeights | dict[int, HeuristicWeights]


def _clamp(val: float) -> float:
    return max(WEIGHT_BOUNDS[0], min(WEIGHT_BOUNDS[1], val))


def _weights_to_vector(w: HeuristicWeights) -> list[float]:
    return [float(getattr(w, f)) for f in WEIGHT_FIELDS]


def _vector_to_weights(vec: list[float]) -> HeuristicWeights:
    w = HeuristicWeights()
    for i, f in enumerate(WEIGHT_FIELDS):
        setattr(w, f, _clamp(float(vec[i])))
    return w


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


def create_training_game(
    num_players: int,
    board_type: BoardType,
    strict_rules_mode: bool = False,
    setup_mode: str = "real5_softmax",
    draft_temperature: float = 0.85,
    draft_sample_top_k: int = 12,
    coverage_mode: str = "off",
    coverage_seed_birds: bool = False,
) -> GameState:
    """Create a game state suitable for self-play training."""
    bird_reg = get_bird_registry()
    goal_reg = get_goal_registry()
    bonus_reg = get_bonus_registry()

    player_names = [f"Player_{i+1}" for i in range(num_players)]

    # Pick 4 random round goals
    all_goals = goal_reg.all_goals
    round_goals = random.sample(all_goals, min(4, len(all_goals)))

    game = create_new_game(
        player_names,
        round_goals,
        board_type,
        strict_rules_mode=strict_rules_mode,
    )

    # Deal random hands (5 birds each) and bonus cards (1 each).
    # In strict mode, only use strict-certified bird powers so playouts are
    # rules-certified end-to-end.
    if strict_rules_mode:
        from backend.powers.registry import get_power_source, is_strict_power_source_allowed
        all_birds = [
            b for b in bird_reg.all_birds
            if is_strict_power_source_allowed(get_power_source(b))
        ]
    else:
        all_birds = list(bird_reg.all_birds)
    if len(all_birds) < (num_players * 5 + 3):
        raise ValueError(
            "Not enough strict-certified birds to initialize training game "
            f"(have {len(all_birds)}, need at least {num_players * 5 + 3})"
        )
    random.shuffle(all_birds)
    all_bonus = list(bonus_reg.all_cards)
    random.shuffle(all_bonus)

    coverage_mode_norm = coverage_mode.strip().lower()
    if coverage_mode_norm not in {"off", "all_5_colors_exec"}:
        coverage_mode_norm = "off"

    # Validation-only hook: ensure at least one strict-certified bird of each
    # power color is reachable in the opening pool (hands/tray/early deck).
    if coverage_mode_norm == "all_5_colors_exec" and coverage_seed_birds:
        needed_colors = [
            PowerColor.WHITE,
            PowerColor.BROWN,
            PowerColor.PINK,
            PowerColor.TEAL,
            PowerColor.YELLOW,
        ]
        from backend.powers.registry import get_power_source, is_strict_power_source_allowed

        preferred_names = {
            PowerColor.WHITE: "Roseate Spoonbill",
            PowerColor.BROWN: "Forster's Tern",
            PowerColor.PINK: "Sacred Kingfisher",
            PowerColor.TEAL: "Cetti's Warbler",
            PowerColor.YELLOW: "Crested Pigeon",
        }
        selected_by_color: dict[PowerColor, object] = {}
        for color in needed_colors:
            preferred = preferred_names.get(color)
            bird = next(
                (
                    b for b in all_birds
                    if b.color == color
                    and b.name == preferred
                    and (not strict_rules_mode or is_strict_power_source_allowed(get_power_source(b)))
                ),
                None,
            )
            if bird is None:
                bird = next(
                    (
                        b for b in all_birds
                        if b.color == color
                        and (not strict_rules_mode or is_strict_power_source_allowed(get_power_source(b)))
                    ),
                    None,
                )
            if bird is None:
                raise ValueError(f"No eligible bird found for required coverage color '{color.value}'")
            selected_by_color[color] = bird

        # Keep seeded colors in immediate opening reach (candidate hands + tray).
        reachable_window = min(len(all_birds), num_players * 5 + 3)
        if reachable_window < len(needed_colors):
            raise ValueError("Insufficient deck window to seed all coverage colors")

        if len(needed_colors) == 1:
            target_positions = [0]
        else:
            step = (reachable_window - 1) / float(len(needed_colors) - 1)
            target_positions = [int(round(step * i)) for i in range(len(needed_colors))]
            target_positions = [max(0, min(reachable_window - 1, p)) for p in target_positions]
            # Ensure uniqueness in-order.
            seen = set()
            uniq_positions = []
            for p in target_positions:
                while p in seen and p + 1 < reachable_window:
                    p += 1
                if p in seen:
                    p = 0
                    while p in seen and p + 1 < reachable_window:
                        p += 1
                seen.add(p)
                uniq_positions.append(p)
            target_positions = uniq_positions

        for color, pos in zip(needed_colors, target_positions):
            seeded_bird = selected_by_color[color]
            current_idx = all_birds.index(seeded_bird)
            all_birds[pos], all_birds[current_idx] = all_birds[current_idx], all_birds[pos]

    idx = 0
    non_nectar_foods = [
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
    ]
    mode = setup_mode.strip().lower()

    if mode == "legacy_fixed5":
        for player in game.players:
            hand_size = min(5, len(all_birds) - idx)
            player.hand = all_birds[idx:idx + hand_size]
            idx += hand_size
            player.bonus_cards = [random.choice(all_bonus)]

            # Starting food: 1 of each non-nectar type
            for ft in non_nectar_foods:
                player.food_supply.add(ft, 1)
        draft_discard_total = 0
        draft_fallbacks = 0
    else:
        if mode != "real5_softmax":
            mode = "real5_softmax"
        top_k = max(1, int(draft_sample_top_k))
        temp = float(draft_temperature)
        if temp <= 0.0:
            temp = 0.85

        candidate_hands: list[list] = []
        for _ in game.players:
            hand_size = min(5, len(all_birds) - idx)
            candidate_hands.append(all_birds[idx:idx + hand_size])
            idx += hand_size

        # Draw tray before setup choice so setup scoring can account for tray access.
        tray_candidates: list = []
        tray_count = min(3, len(all_birds) - idx)
        for _ in range(tray_count):
            tray_candidates.append(all_birds[idx])
            idx += 1

        if len(all_bonus) < num_players * 2:
            raise ValueError(
                "Not enough bonus cards to initialize training game "
                f"(have {len(all_bonus)}, need at least {num_players * 2})"
            )

        draft_discard_total = 0
        draft_fallbacks = 0
        bonus_idx = 0
        for pi, player in enumerate(game.players):
            candidate = candidate_hands[pi]
            bonus_options = all_bonus[bonus_idx:bonus_idx + 2]
            bonus_idx += 2
            recommendations = analyze_setup(
                birds=candidate,
                bonus_cards=bonus_options,
                round_goals=round_goals,
                top_n=top_k,
                tray_birds=tray_candidates,
                turn_order=pi + 1,
                num_players=game.num_players,
            )

            selected = None
            if recommendations:
                max_score = max(r.score for r in recommendations)
                weights: list[float] = []
                for rec in recommendations:
                    shifted = (float(rec.score) - float(max_score)) / max(1e-6, temp)
                    weights.append(math.exp(shifted))
                selected = random.choices(recommendations, weights=weights, k=1)[0]

            if selected is None:
                # Should not occur; fallback preserves playability.
                draft_fallbacks += 1
                player.hand = list(candidate)
                player.bonus_cards = [bonus_options[0]]
                for ft in non_nectar_foods:
                    player.food_supply.add(ft, 1)
                continue

            by_name: dict[str, list] = {}
            for b in candidate:
                by_name.setdefault(b.name, []).append(b)

            kept_birds = []
            for name in selected.birds_to_keep:
                pool = by_name.get(name)
                if pool:
                    kept_birds.append(pool.pop())

            player.hand = kept_birds
            draft_discard_total += max(0, len(candidate) - len(kept_birds))

            # Keep base-game/Oceania nectar initialization; only overwrite non-nectar.
            player.food_supply.invertebrate = 0
            player.food_supply.seed = 0
            player.food_supply.fish = 0
            player.food_supply.fruit = 0
            player.food_supply.rodent = 0
            for ft_name, count in selected.food_to_keep.items():
                try:
                    ft = FoodType(ft_name)
                except ValueError:
                    continue
                if ft != FoodType.NECTAR and count > 0:
                    player.food_supply.add(ft, int(count))

            chosen_bonus = next((b for b in bonus_options if b.name == selected.bonus_card), None)
            player.bonus_cards = [chosen_bonus if chosen_bonus is not None else bonus_options[0]]

        for b in tray_candidates:
            game.card_tray.add_card(b)

    if mode == "legacy_fixed5":
        # Set up card tray from deck top.
        tray_count = min(3, len(all_birds) - idx)
        for _ in range(tray_count):
            if idx < len(all_birds):
                game.card_tray.add_card(all_birds[idx])
                idx += 1

    game.discard_pile_count = draft_discard_total
    game._setup_stats = {  # type: ignore[attr-defined]
        "setup_mode": mode,
        "draft_temperature": draft_temperature,
        "draft_sample_top_k": draft_sample_top_k,
        "draft_discard_total": draft_discard_total,
        "draft_fallbacks": draft_fallbacks,
        "coverage_mode": coverage_mode_norm,
        "coverage_seed_birds": bool(coverage_seed_birds),
    }

    # Keep finite deck identities for higher-fidelity simulation.
    game._deck_cards = list(all_birds[idx:])  # type: ignore[attr-defined]
    idx = len(all_birds)

    # Set up birdfeeder with initial roll
    game.birdfeeder.reroll()
    game.deck_remaining = max(0, len(getattr(game, "_deck_cards", [])))
    if game.deck_tracker is None:
        game.deck_tracker = DeckTracker()
    game.deck_tracker.reset_from_cards(getattr(game, "_deck_cards", []))

    return game


def _pick_move_with_weights(moves: list[Move], game: GameState,
                            player: Player,
                            base_weights: WeightProfile) -> Move:
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
                      player_weights: dict[str, WeightProfile],
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
            game = create_training_game(
                config.num_players,
                board_type,
                strict_rules_mode=config.strict_rules_mode,
            )

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


def evolve_evolutionary(config: TrainingConfig) -> dict:
    """Run the legacy evolutionary training loop."""
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
        "optimizer_used": "evolutionary",
        "history": best_history,
    }


def _evaluate_weights_vs_baseline(
    candidate: HeuristicWeights,
    config: TrainingConfig,
    target_round: int | None = None,
    locked_round_weights: dict[int, HeuristicWeights] | None = None,
) -> tuple[float, float, float]:
    """Evaluate one candidate against fixed default heuristic weights."""
    board_type = BoardType(config.board_type)
    games = max(1, int(config.games_per_matchup))
    cand_scores: list[float] = []
    cand_wins = 0.0

    for gi in range(games):
        game = create_training_game(
            config.num_players,
            board_type,
            strict_rules_mode=config.strict_rules_mode,
        )
        candidate_on_p1 = (gi % 2 == 0)
        candidate_name = "Player_1" if candidate_on_p1 else "Player_2"

        candidate_profile: WeightProfile
        if target_round is not None:
            profile: dict[int, HeuristicWeights] = {0: DEFAULT_WEIGHTS}
            profile.update(dict(locked_round_weights or {}))
            profile[int(target_round)] = candidate
            candidate_profile = profile
        else:
            candidate_profile = candidate

        pw: dict[str, WeightProfile] = {}
        for p in game.players:
            if p.name == candidate_name:
                pw[p.name] = candidate_profile
            else:
                pw[p.name] = DEFAULT_WEIGHTS

        results = play_head_to_head(game, pw)
        cand_score = float(results.get(candidate_name, 0))
        opp_scores = [float(v) for k, v in results.items() if k != candidate_name]
        opp_best = max(opp_scores) if opp_scores else 0.0
        cand_scores.append(cand_score)

        if cand_score > opp_best:
            cand_wins += 1.0
        elif cand_score == opp_best:
            cand_wins += 0.5

    avg_score = sum(cand_scores) / len(cand_scores) if cand_scores else 0.0
    win_rate = cand_wins / games if games > 0 else 0.0
    fitness = avg_score * 0.7 + win_rate * 30.0
    return avg_score, win_rate, fitness


def _run_cmaes_optimization(
    config: TrainingConfig,
    evaluator,
    *,
    seed_offset: int = 0,
    label: str = "",
) -> tuple[Individual | None, list[dict]]:
    """Shared CMA-ES loop that optimizes weights using an evaluator callback."""
    if config.seed is not None:
        random.seed(int(config.seed) + int(seed_offset))

    x0 = _weights_to_vector(DEFAULT_WEIGHTS)
    sigma = max(0.01, float(config.cma_sigma))
    options = {
        "popsize": max(4, int(config.population_size)),
        "verbose": -9,
    }
    if config.seed is not None:
        options["seed"] = int(config.seed) + int(seed_offset)

    es = cma.CMAEvolutionStrategy(x0, sigma, options)

    best_ever: Individual | None = None
    best_history: list[dict] = []
    no_improvement_count = 0
    tag = f"[{label}] " if label else ""

    for gen in range(config.generations):
        gen_start = time.time()

        solutions = es.ask()
        losses: list[float] = []
        generation_candidates: list[Individual] = []

        for vec in solutions:
            w = _vector_to_weights([float(v) for v in vec])
            avg_score, win_rate, fitness = evaluator(w)
            ind = Individual(
                weights=w,
                fitness=fitness,
                avg_score=avg_score,
                win_rate=win_rate,
                games_played=max(1, int(config.games_per_matchup)),
            )
            generation_candidates.append(ind)
            losses.append(-fitness)  # CMA-ES minimizes

        es.tell(solutions, losses)
        generation_candidates.sort(key=lambda ind: -ind.fitness)
        gen_best = generation_candidates[0]
        gen_elapsed = time.time() - gen_start

        print(
            f"{tag}Gen {gen + 1}/{config.generations}: "
            f"best_fitness={gen_best.fitness:.1f} "
            f"avg_score={gen_best.avg_score:.1f} "
            f"win_rate={gen_best.win_rate:.1%} "
            f"({gen_elapsed:.1f}s)"
        )

        best_history.append({
            "generation": gen + 1,
            "best_fitness": round(gen_best.fitness, 2),
            "best_avg_score": round(gen_best.avg_score, 2),
            "best_win_rate": round(gen_best.win_rate, 3),
            "weights": asdict(gen_best.weights),
        })

        if best_ever is None or gen_best.fitness > best_ever.fitness:
            if best_ever is not None:
                improvement = ((gen_best.fitness - best_ever.fitness) / max(1, abs(best_ever.fitness)))
                if improvement > config.convergence_threshold:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            best_ever = copy.deepcopy(gen_best)
        else:
            no_improvement_count += 1

        if no_improvement_count >= config.convergence_patience:
            print(
                f"{tag}Converged after {gen + 1} generations "
                f"(no improvement for {config.convergence_patience} generations)"
            )
            break

        if es.stop():
            print(f"{tag}CMA-ES stop condition met after generation {gen + 1}.")
            break

    return best_ever, best_history


def evolve_cmaes(config: TrainingConfig) -> dict:
    """Run CMA-ES optimization against fixed DEFAULT_WEIGHTS baseline."""
    if cma is None:
        print("cma package not installed; falling back to legacy evolutionary optimizer.")
        return evolve_evolutionary(config)

    best_ever, best_history = _run_cmaes_optimization(
        config,
        lambda w: _evaluate_weights_vs_baseline(w, config),
    )
    return {
        "best_weights": asdict(best_ever.weights) if best_ever else {},
        "best_fitness": round(best_ever.fitness, 2) if best_ever else 0,
        "best_avg_score": round(best_ever.avg_score, 2) if best_ever else 0,
        "best_win_rate": round(best_ever.win_rate, 3) if best_ever else 0,
        "default_weights": asdict(DEFAULT_WEIGHTS),
        "config": asdict(config),
        "optimizer_used": "cmaes" if cma is not None else "evolutionary",
        "history": best_history,
    }


def evolve_cmaes_per_round(config: TrainingConfig) -> dict:
    """Run CMA-ES independently per round and return round-indexed weights."""
    if cma is None:
        print("cma package not installed; falling back to global evolutionary optimizer.")
        return evolve_evolutionary(config)

    locked_round_weights: dict[int, HeuristicWeights] = {}
    per_round_results: list[dict] = []

    for round_num in range(1, 5):
        best_ever, history = _run_cmaes_optimization(
            config,
            lambda w, rn=round_num, locked=locked_round_weights: _evaluate_weights_vs_baseline(
                w,
                config,
                target_round=rn,
                locked_round_weights=locked,
            ),
            seed_offset=round_num * 10_000,
            label=f"R{round_num}",
        )
        if best_ever is None:
            continue
        locked_round_weights[round_num] = copy.deepcopy(best_ever.weights)
        per_round_results.append(
            {
                "round": round_num,
                "best_fitness": round(best_ever.fitness, 2),
                "best_avg_score": round(best_ever.avg_score, 2),
                "best_win_rate": round(best_ever.win_rate, 3),
                "weights": asdict(best_ever.weights),
                "history": history,
            }
        )

    combined_profile: WeightProfile = ({0: DEFAULT_WEIGHTS} | locked_round_weights) if locked_round_weights else DEFAULT_WEIGHTS
    agg_avg, agg_wr, agg_fit = _evaluate_weights_vs_baseline(
        DEFAULT_WEIGHTS,
        config,
        target_round=None,
        locked_round_weights=None,
    )
    if locked_round_weights:
        # Evaluate a representative combined profile by plugging all learned rounds into candidate side.
        board_type = BoardType(config.board_type)
        games = max(1, int(config.games_per_matchup))
        scores: list[float] = []
        wins = 0.0
        for gi in range(games):
            game = create_training_game(
                config.num_players,
                board_type,
                strict_rules_mode=config.strict_rules_mode,
            )
            candidate_on_p1 = (gi % 2 == 0)
            candidate_name = "Player_1" if candidate_on_p1 else "Player_2"
            pw: dict[str, WeightProfile] = {}
            for p in game.players:
                pw[p.name] = combined_profile if p.name == candidate_name else DEFAULT_WEIGHTS
            results = play_head_to_head(game, pw)
            cand_score = float(results.get(candidate_name, 0.0))
            opp_scores = [float(v) for k, v in results.items() if k != candidate_name]
            opp_best = max(opp_scores) if opp_scores else 0.0
            scores.append(cand_score)
            if cand_score > opp_best:
                wins += 1.0
            elif cand_score == opp_best:
                wins += 0.5
        agg_avg = sum(scores) / len(scores) if scores else 0.0
        agg_wr = wins / games if games > 0 else 0.0
        agg_fit = agg_avg * 0.7 + agg_wr * 30.0

    return {
        "best_weights": asdict(DEFAULT_WEIGHTS),
        "best_weights_by_round": {str(r): asdict(w) for r, w in locked_round_weights.items()},
        "best_fitness": round(agg_fit, 2),
        "best_avg_score": round(agg_avg, 2),
        "best_win_rate": round(agg_wr, 3),
        "default_weights": asdict(DEFAULT_WEIGHTS),
        "config": asdict(config),
        "optimizer_used": "cmaes_per_round",
        "history": per_round_results,
    }


def evolve(config: TrainingConfig) -> dict:
    """Run the configured optimizer (CMA-ES by default)."""
    mode = (config.training_mode or "global").strip().lower()
    if mode in {"per-round", "per_round", "round"}:
        return evolve_cmaes_per_round(config)

    optimizer = (config.optimizer or "cmaes").strip().lower()
    if optimizer == "evolutionary":
        return evolve_evolutionary(config)
    return evolve_cmaes(config)


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
    parser.add_argument("--optimizer", default="cmaes", choices=["cmaes", "evolutionary"])
    parser.add_argument("--training-mode", default="global", choices=["global", "per-round"])
    parser.add_argument("--cma-sigma", type=float, default=0.15)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strict-rules-only", action="store_true")
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
        optimizer=args.optimizer,
        cma_sigma=args.cma_sigma,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        seed=args.seed,
        output_path=args.output,
        strict_rules_mode=args.strict_rules_only,
        training_mode=args.training_mode,
    )

    print(
        f"Starting self-play training ({config.optimizer}, mode={config.training_mode}): {config.population_size} individuals, "
        f"{config.generations} generations, {config.num_players} players"
    )

    results = evolve(config)

    output_path = Path(config.output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_path}")
    print(f"Best avg score: {results['best_avg_score']}")
    print(f"Best win rate: {results['best_win_rate']:.1%}")
    by_round = results.get("best_weights_by_round")
    if isinstance(by_round, dict) and by_round:
        print("\nBest weights by round:")
        for round_key in sorted(by_round.keys(), key=lambda x: int(x)):
            print(f" Round {round_key}:")
            round_weights = by_round[round_key]
            for key in round_weights:
                best_val = round_weights[key]
                default_val = results["default_weights"][key]
                diff = ((best_val / default_val) - 1) * 100 if default_val else 0
                marker = " *" if abs(diff) > 20 else ""
                print(
                    f"  {key}: {best_val:.3f} (default: {default_val:.3f}, "
                    f"{diff:+.0f}%){marker}"
                )
    else:
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
