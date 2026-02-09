"""Evaluate starting draft options and recommend the best combination.

In Wingspan, each player is dealt 5 bird cards and 5 food tokens (1 of each
basic type). They choose N birds to keep and discard the rest, keeping (5 - N)
food tokens. They also pick 1 of 2 bonus cards. This module evaluates all
252 possible bird/food combinations x 2 bonus cards = 504 options.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from backend.models.bird import Bird, FoodCost
from backend.models.bonus_card import BonusCard
from backend.models.goal import Goal
from backend.models.enums import (
    FoodType, Habitat, PowerColor, NestType, BeakDirection,
)

STARTING_FOOD = [
    FoodType.INVERTEBRATE,
    FoodType.SEED,
    FoodType.FISH,
    FoodType.FRUIT,
    FoodType.RODENT,
]


@dataclass
class SetupRecommendation:
    rank: int
    score: float
    birds_to_keep: list[str]
    food_to_keep: dict[str, int]
    bonus_card: str
    reasoning: str


def _food_list_to_dict(food: tuple[FoodType, ...] | list[FoodType]) -> dict[str, int]:
    d: dict[str, int] = {}
    for f in food:
        d[f.value] = d.get(f.value, 0) + 1
    return d


# --- Custom Draft Synergy Functions ---
# For the 21 bonus cards without spreadsheet eligibility columns,
# estimate synergy from bird attributes known at draft time.


def _draft_synergy_behaviorist(birds: list[Bird]) -> int:
    """Behaviorist: columns with 3 different power colors.

    If draft birds have 3+ distinct colors, all could contribute to
    mixed columns. Otherwise, no meaningful synergy.
    """
    colors = {PowerColor.WHITE if b.color == PowerColor.NONE else b.color
              for b in birds}
    return len(birds) if len(colors) >= 3 else 0


def _draft_synergy_ethologist(birds: list[Bird]) -> int:
    """Ethologist: max distinct power colors in any one habitat.

    For each habitat, count distinct colors among draft birds that can live there.
    Return the best habitat's eligible bird count.
    """
    best = 0
    for habitat in Habitat:
        colors = set()
        eligible = 0
        for b in birds:
            if habitat in b.habitats:
                c = PowerColor.WHITE if b.color == PowerColor.NONE else b.color
                colors.add(c)
                eligible += 1
        if len(colors) >= 2:
            best = max(best, eligible)
    return best


def _draft_synergy_population_monitor(birds: list[Bird], habitat: Habitat) -> int:
    """Population Monitor: distinct nest types in a specific habitat.

    Count distinct nest types among birds that can live in the target habitat.
    Wild nests add diversity.
    """
    nests = set()
    for b in birds:
        if habitat in b.habitats:
            nests.add(b.nest_type)
    return min(5, len(nests))


def _draft_synergy_mechanical_engineer(birds: list[Bird]) -> int:
    """Mechanical Engineer: sets of 4 nest types (bowl, cavity, ground, platform).

    Count how many of the 4 required types are covered. Wild fills gaps.
    """
    non_wild = set()
    wild_count = 0
    for b in birds:
        if b.nest_type == NestType.WILD:
            wild_count += 1
        else:
            non_wild.add(b.nest_type)
    return min(4, len(non_wild) + wild_count)


def _draft_synergy_site_selection(birds: list[Bird]) -> int:
    """Site Selection Expert: columns with matching nest pairs/trios.

    Count birds that share a nest type with another draft bird (could
    form a column pair). Wild nests always contribute.
    """
    type_counts: Counter[NestType] = Counter()
    has_wild = 0
    for b in birds:
        if b.nest_type == NestType.WILD:
            has_wild += 1
        else:
            type_counts[b.nest_type] += 1

    matching = has_wild  # wild always contributes
    for b in birds:
        if b.nest_type != NestType.WILD:
            if type_counts[b.nest_type] >= 2 or has_wild > 0:
                matching += 1
    return matching


def _draft_synergy_data_analyst(birds: list[Bird], habitat: Habitat) -> int:
    """Data Analyst: consecutive ascending/descending wingspans.

    Count birds eligible for the habitat with non-None wingspan.
    """
    return sum(
        1 for b in birds
        if habitat in b.habitats and b.wingspan_cm is not None
    )


def _draft_synergy_ranger(birds: list[Bird], habitat: Habitat) -> int:
    """Ranger: consecutive ascending/descending VP.

    Count birds eligible for the habitat.
    """
    return sum(1 for b in birds if habitat in b.habitats)


def _draft_synergy_ecologist(birds: list[Bird]) -> int:
    """Ecologist: birds in habitat with fewest birds.

    Multi-habitat birds give flexibility to balance habitats.
    """
    return sum(1 for b in birds if len(b.habitats) >= 2)


def _draft_synergy_avian_therio(birds: list[Bird]) -> int:
    """Avian Theriogenologist: birds with completely full nests.

    Birds with small egg limits (1-3) are easier to fill.
    """
    return sum(1 for b in birds if 0 < b.egg_limit <= 3)


def _draft_synergy_breeding_manager(birds: list[Bird]) -> int:
    """Breeding Manager: birds with 4+ eggs.

    Birds with egg_limit >= 4 have the potential.
    """
    return sum(1 for b in birds if b.egg_limit >= 4)


def _draft_synergy_oologist(birds: list[Bird]) -> int:
    """Oologist: birds with 1+ egg.

    Nearly universal — most birds have egg capacity.
    """
    return sum(1 for b in birds if b.egg_limit > 0)


def _draft_synergy_citizen_scientist(birds: list[Bird]) -> int:
    """Citizen Scientist: birds with tucked cards.

    Flocking birds or those with tuck-related powers contribute.
    """
    count = 0
    for b in birds:
        if b.is_flocking:
            count += 1
        elif "tuck" in b.power_text.lower():
            count += 1
    return count


def _draft_synergy_pellet_dissector(birds: list[Bird]) -> int:
    """Pellet Dissector: fish + rodent cached on birds.

    Predator birds cache food from kills.
    """
    return sum(1 for b in birds if b.is_predator)


def _draft_synergy_no_signal(birds: list[Bird]) -> int:
    """Cards with no bird-specific draft signal."""
    return 0


CUSTOM_DRAFT_SYNERGY: dict[str, Callable[[list[Bird]], int]] = {
    "Behaviorist": _draft_synergy_behaviorist,
    "Ethologist": _draft_synergy_ethologist,
    "Forest Population Monitor": lambda birds: _draft_synergy_population_monitor(birds, Habitat.FOREST),
    "Grassland Population Monitor": lambda birds: _draft_synergy_population_monitor(birds, Habitat.GRASSLAND),
    "Wetland Population Monitor": lambda birds: _draft_synergy_population_monitor(birds, Habitat.WETLAND),
    "Mechanical Engineer": _draft_synergy_mechanical_engineer,
    "Site Selection Expert": _draft_synergy_site_selection,
    "Forest Data Analyst": lambda birds: _draft_synergy_data_analyst(birds, Habitat.FOREST),
    "Grassland Data Analyst": lambda birds: _draft_synergy_data_analyst(birds, Habitat.GRASSLAND),
    "Wetland Data Analyst": lambda birds: _draft_synergy_data_analyst(birds, Habitat.WETLAND),
    "Forest Ranger": lambda birds: _draft_synergy_ranger(birds, Habitat.FOREST),
    "Grassland Ranger": lambda birds: _draft_synergy_ranger(birds, Habitat.GRASSLAND),
    "Wetland Ranger": lambda birds: _draft_synergy_ranger(birds, Habitat.WETLAND),
    "Ecologist": _draft_synergy_ecologist,
    "Avian Theriogenologist": _draft_synergy_avian_therio,
    "Breeding Manager": _draft_synergy_breeding_manager,
    "Oologist": _draft_synergy_oologist,
    "Citizen Scientist": _draft_synergy_citizen_scientist,
    "Pellet Dissector": _draft_synergy_pellet_dissector,
    "Winter Feeder": _draft_synergy_no_signal,
    "Visionary Leader": _draft_synergy_no_signal,
}


def _can_afford_bird(bird: Bird, available: dict[FoodType, int]) -> bool:
    """Check if a bird can be played with available food (simplified check)."""
    cost = bird.food_cost
    if cost.total == 0:
        return True

    if cost.is_or:
        # Pay any one of the distinct types
        for ft in cost.distinct_types:
            if ft == FoodType.WILD:
                if any(available.get(f, 0) > 0 for f in STARTING_FOOD):
                    return True
            elif available.get(ft, 0) > 0:
                return True
            # Nectar can substitute
            if available.get(FoodType.NECTAR, 0) > 0:
                return True
        return False

    # AND cost: need all specified items
    required: dict[FoodType, int] = {}
    for ft in cost.items:
        required[ft] = required.get(ft, 0) + 1

    wild_needed = required.pop(FoodType.WILD, 0)

    # Check specific food requirements
    remaining = dict(available)
    for ft, count in required.items():
        if remaining.get(ft, 0) >= count:
            remaining[ft] -= count
        elif ft != FoodType.NECTAR and remaining.get(FoodType.NECTAR, 0) > 0:
            # Nectar can substitute for any food
            shortfall = count - remaining.get(ft, 0)
            remaining[ft] = 0
            if remaining.get(FoodType.NECTAR, 0) >= shortfall:
                remaining[FoodType.NECTAR] -= shortfall
            else:
                return False
        else:
            return False

    # Fill wild slots with whatever remains
    total_remaining = sum(v for k, v in remaining.items() if k != FoodType.WILD)
    return total_remaining >= wild_needed


def _play_birds_greedily(
    birds: list[Bird], food: dict[FoodType, int]
) -> tuple[list[Bird], dict[FoodType, int]]:
    """Try to play birds in order of cheapness, returning (playable, remaining_food).

    Simulates sequential bird plays, deducting food as birds are played.
    """
    remaining = dict(food)
    playable = []

    # Sort: cheapest first, then by VP (higher VP breaks ties)
    sorted_birds = sorted(birds, key=lambda b: (b.food_cost.total, -b.victory_points))

    for bird in sorted_birds:
        if _can_afford_bird(bird, remaining):
            # Deduct food cost
            cost = bird.food_cost
            if cost.total == 0:
                playable.append(bird)
                continue

            if cost.is_or:
                # Pay cheapest available option
                paid = False
                for ft in cost.distinct_types:
                    if ft == FoodType.WILD:
                        continue
                    if remaining.get(ft, 0) > 0:
                        remaining[ft] -= 1
                        paid = True
                        break
                if not paid and remaining.get(FoodType.NECTAR, 0) > 0:
                    remaining[FoodType.NECTAR] -= 1
                    paid = True
                if paid:
                    playable.append(bird)
            else:
                # Pay AND cost
                can_pay = True
                temp = dict(remaining)
                for ft in cost.items:
                    if ft == FoodType.WILD:
                        # Find any available food
                        found = False
                        for wft in STARTING_FOOD + [FoodType.NECTAR]:
                            if temp.get(wft, 0) > 0:
                                temp[wft] -= 1
                                found = True
                                break
                        if not found:
                            can_pay = False
                            break
                    elif temp.get(ft, 0) > 0:
                        temp[ft] -= 1
                    elif temp.get(FoodType.NECTAR, 0) > 0:
                        temp[FoodType.NECTAR] -= 1
                    else:
                        can_pay = False
                        break
                if can_pay:
                    remaining = temp
                    playable.append(bird)

    return playable, remaining


def _estimate_goal_contribution(bird: Bird, goal: Goal) -> float:
    """Estimate how much a bird might help with a goal (habitat-agnostic)."""
    desc = goal.description.lower()

    # Bird-count goals
    if "total [bird]" in desc:
        return 1.0
    for hab in ("forest", "grassland", "wetland"):
        if f"[bird] in [{hab}]" in desc:
            if Habitat(hab) in bird.habitats:
                return 1.0

    # Egg-related goals
    if "[egg]" in desc and bird.egg_limit > 0:
        for hab in ("forest", "grassland", "wetland"):
            if f"[egg] in [{hab}]" in desc and Habitat(hab) in bird.habitats:
                return bird.egg_limit * 0.2
        nest_val = bird.nest_type.value if bird.nest_type != NestType.WILD else None
        if nest_val:
            if f"[egg] in [{nest_val}]" in desc:
                return bird.egg_limit * 0.2
        if bird.nest_type == NestType.WILD:
            for nt in ("bowl", "cavity", "ground", "platform"):
                if f"[egg] in [{nt}]" in desc:
                    return bird.egg_limit * 0.2

    # Nest + bird + egg goals
    for nt in ("bowl", "cavity", "ground", "platform"):
        if f"[{nt}] [bird] with [egg]" in desc:
            nest_val = bird.nest_type.value if bird.nest_type != NestType.WILD else None
            if nest_val == nt or bird.nest_type == NestType.WILD:
                return 1.0 if bird.egg_limit > 0 else 0.0

    # VP threshold goals
    if "[bird] worth >4 [feather]" in desc:
        return 1.0 if bird.victory_points > 4 else 0.0
    if "\u22643" in desc:
        return 1.0 if bird.victory_points <= 3 else 0.0

    # Power color goals
    if "brown powers" in desc:
        return 1.0 if bird.color == PowerColor.BROWN else 0.0
    if "white & no powers" in desc:
        return 1.0 if bird.color in (PowerColor.WHITE, PowerColor.NONE) else 0.0

    # Beak direction goals
    if "[beak_pointing_left]" in desc:
        return 1.0 if bird.beak_direction == BeakDirection.LEFT else 0.0
    if "[beak_pointing_right]" in desc:
        return 1.0 if bird.beak_direction == BeakDirection.RIGHT else 0.0

    # Food cost goals
    if "[invertebrate] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft == FoodType.INVERTEBRATE)
    if "[fruit] + [seed] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft in (FoodType.FRUIT, FoodType.SEED))
    if "[rodent] + [fish] in food cost" in desc:
        return sum(1 for ft in bird.food_cost.items if ft in (FoodType.RODENT, FoodType.FISH))
    if "food cost of played [bird]" in desc:
        return bird.food_cost.total * 0.3

    # Tucked card goals
    if "[bird_with_tucked_card]" in desc and bird.is_flocking:
        return 0.5

    return 0.0


def _evaluate_option(
    birds: list[Bird],
    food_kept: tuple[FoodType, ...],
    bonus_card: BonusCard,
    round_goals: list[Goal],
    tray_birds: list[Bird],
    turn_order: int,
    num_players: int,
) -> float:
    """Score a single draft option."""
    score = 0.0

    # Build food supply (kept food + 1 free nectar for Oceania)
    food_dict: dict[FoodType, int] = {}
    for f in food_kept:
        food_dict[f] = food_dict.get(f, 0) + 1
    food_dict[FoodType.NECTAR] = food_dict.get(FoodType.NECTAR, 0) + 1

    # Check which birds can be played with available food
    playable, remaining_food = _play_birds_greedily(birds, food_dict)
    playable_names = {b.name for b in playable}

    # 1. Bird VP value
    for bird in birds:
        vp = bird.victory_points
        if bird.name in playable_names:
            score += vp * 1.5  # Playable birds worth more
        else:
            score += vp * 0.6  # Stuck birds still have some value

    # 2. Playability bonus
    score += len(playable) * 3.0

    # 3. Engine value (brown powers activated many times)
    for bird in birds:
        if bird.color == PowerColor.BROWN:
            if bird.name in playable_names:
                score += 4.0  # Huge value for playable engine birds
            else:
                score += 1.0

    # 4. White power bonus (play-another-bird especially)
    for bird in birds:
        if bird.color == PowerColor.WHITE:
            power = bird.power_text.lower()
            if "play" in power and "bird" in power:
                score += 3.0 if bird.name in playable_names else 0.5

    # 5. Cheap birds that can be played early
    for bird in birds:
        if bird.food_cost.total <= 1 and bird.name in playable_names:
            score += 1.5

    # 6. Egg capacity
    for bird in birds:
        score += bird.egg_limit * 0.3

    # 7. Bonus card synergy
    matching = sum(1 for b in birds if bonus_card.name in b.bonus_eligibility)
    if matching > 0:
        score += matching * 2.5
        if matching >= 3:
            score += 2.0
        elif matching >= 2:
            score += 1.0
    elif bonus_card.name in CUSTOM_DRAFT_SYNERGY:
        custom_matching = CUSTOM_DRAFT_SYNERGY[bonus_card.name](birds)
        score += custom_matching * 1.5  # Lower multiplier (less precise signal)
        if custom_matching >= 3:
            score += 1.5
        elif custom_matching >= 2:
            score += 0.7

    # 8. Goal alignment
    for goal in round_goals:
        for bird in birds:
            contribution = _estimate_goal_contribution(bird, goal)
            score += contribution * 0.8

    # 9. Habitat diversity (can use all three habitats)
    habitats_covered = set()
    for bird in birds:
        habitats_covered.update(bird.habitats)
    score += len(habitats_covered) * 0.5

    # 10. Food diversity of remaining supply (flexibility)
    remaining_types = sum(1 for v in remaining_food.values() if v > 0)
    score += remaining_types * 0.3

    # 11. Tray access value (early pick advantage)
    if tray_birds:
        def _tray_value(b: Bird) -> float:
            val = b.victory_points * 1.1
            val += min(b.egg_limit, 3) * 0.4
            if b.color == PowerColor.BROWN:
                val += 1.2
            if b.color == PowerColor.WHITE and "play" in b.power_text.lower() and "bird" in b.power_text.lower():
                val += 1.0
            if bonus_card.name in b.bonus_eligibility:
                val += 2.0
            for goal in round_goals:
                val += _estimate_goal_contribution(b, goal) * 0.9
            # Habitat diversity with kept birds
            existing_habs = set()
            for kept in birds:
                existing_habs.update(kept.habitats)
            if any(h not in existing_habs for h in b.habitats):
                val += 0.6
            # Affordability using kept food
            if _can_afford_bird(b, food_dict):
                val += 1.0
            if b.is_predator:
                val += 0.3
            return val

        tray_values = [(b, _tray_value(b)) for b in tray_birds]
        tray_values.sort(key=lambda x: -x[1])
        values_only = [v for _, v in tray_values]

        weights_by_turn: dict[int, list[float]] = {
            1: [1.0],
            2: [0.35, 0.65],
            3: [0.2, 0.3, 0.5],
            4: [0.15, 0.25, 0.3, 0.3],
            5: [0.1, 0.2, 0.25, 0.25, 0.2],
        }
        weights = weights_by_turn.get(max(1, min(5, turn_order)), [1.0])
        weights = weights[:len(values_only)]
        if weights:
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
        expected_tray = 0.0
        for i, val in enumerate(values_only):
            w = weights[i] if i < len(weights) else 0.0
            expected_tray += val * w

        # Slightly boost tray impact to be more aggressive about face-up picks
        score += expected_tray * 1.2

    return score


def _generate_reasoning(
    birds: list[Bird],
    food_kept: tuple[FoodType, ...],
    bonus_card: BonusCard,
    round_goals: list[Goal],
    tray_birds: list[Bird],
    turn_order: int,
) -> str:
    """Generate human-readable reasoning for a recommendation."""
    parts = []

    # Food supply for analysis
    food_dict: dict[FoodType, int] = {}
    for f in food_kept:
        food_dict[f] = food_dict.get(f, 0) + 1
    food_dict[FoodType.NECTAR] = food_dict.get(FoodType.NECTAR, 0) + 1

    playable, _ = _play_birds_greedily(birds, food_dict)
    playable_names = {b.name for b in playable}

    if not birds:
        parts.append("All food, maximum flexibility")
    else:
        total_vp = sum(b.victory_points for b in birds)
        parts.append(f"{len(birds)} birds ({total_vp} VP)")

        if playable:
            can_play = [b.name for b in birds if b.name in playable_names]
            if len(can_play) == len(birds):
                parts.append("all playable with kept food")
            else:
                parts.append(f"can play {len(can_play)}/{len(birds)} immediately")

        engine_birds = [b for b in birds if b.color == PowerColor.BROWN]
        if engine_birds:
            parts.append(f"{len(engine_birds)} engine bird{'s' if len(engine_birds) > 1 else ''}")

    matching = sum(1 for b in birds if bonus_card.name in b.bonus_eligibility)
    if matching == 0 and bonus_card.name in CUSTOM_DRAFT_SYNERGY:
        matching = CUSTOM_DRAFT_SYNERGY[bonus_card.name](birds)
    if matching:
        parts.append(f"{matching} match bonus card")

    goal_hits = 0
    for goal in round_goals:
        for bird in birds:
            if _estimate_goal_contribution(bird, goal) > 0:
                goal_hits += 1
                break
    if goal_hits:
        parts.append(f"helps {goal_hits}/{len(round_goals)} goals")

    if tray_birds:
        # Prefer the highest VP as a simple, readable tray cue
        best = max(tray_birds, key=lambda b: b.victory_points)
        if len(tray_birds) >= 2 and turn_order > 1:
            # Also mention a likely fallback if you're not first
            sorted_vp = sorted(tray_birds, key=lambda b: -b.victory_points)
            fallback = sorted_vp[min(turn_order - 1, len(sorted_vp) - 1)]
            if fallback.name != best.name:
                parts.append(f"tray access: likely {fallback.name} (turn {turn_order}), backup {best.name}")
            else:
                parts.append(f"tray access: likely {best.name} (turn {turn_order})")
        else:
            parts.append(f"tray access: likely {best.name} (turn {turn_order})")

    return "; ".join(parts)


def analyze_setup(
    birds: list[Bird],
    bonus_cards: list[BonusCard],
    round_goals: list[Goal],
    top_n: int = 10,
    tray_birds: list[Bird] | None = None,
    turn_order: int = 1,
    num_players: int = 2,
) -> list[SetupRecommendation]:
    """Evaluate all starting draft combinations and return the best options.

    Args:
        birds: The 5 dealt bird cards
        bonus_cards: The 2 dealt bonus cards
        round_goals: The 4 round goals for this game
        top_n: How many recommendations to return

    Returns:
        Top N recommendations sorted by score (best first)
    """
    options: list[tuple[float, list[Bird], tuple[FoodType, ...], BonusCard]] = []
    tray_birds = tray_birds or []

    for num_birds in range(len(birds) + 1):
        num_food = 5 - num_birds
        if num_food < 0 or num_food > len(STARTING_FOOD):
            continue

        for bird_combo in combinations(birds, num_birds):
            for food_combo in combinations(STARTING_FOOD, num_food):
                for bonus in bonus_cards:
                    score = _evaluate_option(
                        list(bird_combo), food_combo, bonus, round_goals,
                        tray_birds, turn_order, num_players,
                    )
                    options.append((score, list(bird_combo), food_combo, bonus))

    # Sort by score descending
    options.sort(key=lambda o: -o[0])

    # Build recommendations
    results = []
    for i, (score, kept_birds, kept_food, bonus) in enumerate(options[:top_n]):
        reasoning = _generate_reasoning(
            kept_birds, kept_food, bonus, round_goals,
            tray_birds, turn_order,
        )
        results.append(SetupRecommendation(
            rank=i + 1,
            score=round(score, 1),
            birds_to_keep=[b.name for b in kept_birds],
            food_to_keep=_food_list_to_dict(kept_food),
            bonus_card=bonus.name,
            reasoning=reasoning,
        ))

    return results
