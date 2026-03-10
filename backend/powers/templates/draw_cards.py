"""Card drawing power templates — covers ~70 birds."""

import random
from itertools import combinations
from backend.models.enums import FoodType
from backend.powers.base import PowerEffect, PowerContext, PowerResult
from backend.powers.choices import consume_power_choice


def _draw_context_factor(ctx: PowerContext) -> float:
    deck_remaining = ctx.deck_remaining or ctx.game_state.deck_remaining
    deck_factor = max(0.15, min(1.0, deck_remaining / 200.0))
    hand_size = ctx.hand_size or len(ctx.player.hand)
    if hand_size >= 12:
        hand_factor = 0.55
    elif hand_size >= 8:
        hand_factor = 0.75
    else:
        hand_factor = 1.0
    return deck_factor * hand_factor


def _draw_one_bird(game_state):
    """Draw one concrete bird card from deck identity model if available."""
    if game_state.deck_remaining <= 0:
        return None

    deck_cards = getattr(game_state, "_deck_cards", None)
    if isinstance(deck_cards, list) and deck_cards:
        card = deck_cards.pop()
        game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
        if game_state.deck_tracker is not None:
            game_state.deck_tracker.mark_drawn(card.name)
        return card

    from backend.data.registries import get_bird_registry
    pool = list(get_bird_registry().all_birds)
    if not pool:
        return None
    game_state.deck_remaining = max(0, game_state.deck_remaining - 1)
    card = random.choice(pool)
    if game_state.deck_tracker is not None:
        game_state.deck_tracker.mark_drawn(card.name)
    return card


def _init_bonus_deck_if_needed(game_state) -> None:
    """Ensure a concrete bonus deck exists on the game state."""
    deck = getattr(game_state, "_bonus_cards", None)
    if isinstance(deck, list):
        if not hasattr(game_state, "_bonus_discard_cards"):
            setattr(game_state, "_bonus_discard_cards", [])
        return

    from backend.data.registries import get_bonus_registry

    used_names = set()
    for p in game_state.players:
        used_names.update(bc.name for bc in p.bonus_cards)
    discarded = getattr(game_state, "_bonus_discard_cards", None)
    if isinstance(discarded, list):
        used_names.update(bc.name for bc in discarded)

    deck_cards = [bc for bc in get_bonus_registry().all_cards if bc.name not in used_names]
    random.shuffle(deck_cards)
    setattr(game_state, "_bonus_cards", deck_cards)
    if not isinstance(discarded, list):
        setattr(game_state, "_bonus_discard_cards", [])


def _draw_one_bonus(game_state):
    _init_bonus_deck_if_needed(game_state)
    deck = getattr(game_state, "_bonus_cards", None)
    if not isinstance(deck, list):
        return None
    if not deck:
        # Wingspan rule: when bonus deck is exhausted, reshuffle bonus discard.
        discard = getattr(game_state, "_bonus_discard_cards", None)
        if isinstance(discard, list) and discard:
            random.shuffle(discard)
            deck.extend(discard)
            discard.clear()
    if not deck:
        return None
    return deck.pop()


def _best_bonus_subset_by_score(ctx: PowerContext, drawn: list, keep_n: int) -> list:
    if keep_n <= 0 or not drawn:
        return []
    keep_n = min(keep_n, len(drawn))

    choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) if ctx.bird else None
    if isinstance(choice, dict):
        names = []
        keep_name = choice.get("keep_bonus_name")
        keep_names = choice.get("keep_bonus_names")
        if isinstance(keep_name, str) and keep_name:
            names.append(keep_name)
        if isinstance(keep_names, list):
            names.extend(str(n) for n in keep_names if isinstance(n, str))
        if names:
            selected = []
            remaining = list(drawn)
            for name in names:
                idx = next((i for i, bc in enumerate(remaining) if bc.name == name), None)
                if idx is None:
                    continue
                selected.append(remaining.pop(idx))
                if len(selected) >= keep_n:
                    break
            if len(selected) >= keep_n:
                return selected[:keep_n]

    from backend.engine.scoring import score_bonus_cards

    original_cards = list(ctx.player.bonus_cards)
    base_score = score_bonus_cards(ctx.player)
    best_combo = tuple(range(keep_n))
    best_key = None

    for combo in combinations(range(len(drawn)), keep_n):
        chosen = [drawn[i] for i in combo]
        ctx.player.bonus_cards = original_cards + chosen
        score = score_bonus_cards(ctx.player)
        delta = score - base_score
        draft = sum(float(getattr(bc, "draft_value_pct", 0.0) or 0.0) for bc in chosen)
        names = tuple(sorted(bc.name for bc in chosen))
        key = (delta, draft, names)
        if best_key is None or key > best_key:
            best_key = key
            best_combo = combo

    ctx.player.bonus_cards = original_cards
    return [drawn[i] for i in best_combo]


def _choose_bonus_discards_by_score(ctx: PowerContext, candidates: list, discard_n: int) -> list:
    if discard_n <= 0 or not candidates:
        return []
    discard_n = min(discard_n, len(candidates))

    choice = consume_power_choice(ctx.game_state, ctx.player.name, ctx.bird.name) if ctx.bird else None
    selected = []
    remaining = list(candidates)
    if isinstance(choice, dict):
        names = []
        discard_name = choice.get("discard_bonus_name")
        discard_names = choice.get("discard_bonus_names")
        if isinstance(discard_name, str) and discard_name:
            names.append(discard_name)
        if isinstance(discard_names, list):
            names.extend(str(n) for n in discard_names if isinstance(n, str))
        if names:
            for name in names:
                idx = next((i for i, bc in enumerate(remaining) if bc.name == name), None)
                if idx is None:
                    continue
                selected.append(remaining.pop(idx))
                if len(selected) >= discard_n:
                    return selected[:discard_n]

    extra_needed = discard_n - len(selected)
    if extra_needed <= 0:
        return selected[:discard_n]

    from backend.engine.scoring import score_bonus_cards

    original_cards = list(ctx.player.bonus_cards)
    best_combo = tuple(range(extra_needed))
    best_key = None

    for combo in combinations(range(len(remaining)), extra_needed):
        kept = [bc for i, bc in enumerate(remaining) if i not in combo]
        ctx.player.bonus_cards = kept
        score = score_bonus_cards(ctx.player)
        draft = sum(float(getattr(bc, "draft_value_pct", 0.0) or 0.0) for bc in kept)
        names = tuple(sorted(bc.name for bc in kept))
        key = (score, draft, names)
        if best_key is None or key > best_key:
            best_key = key
            best_combo = combo

    ctx.player.bonus_cards = original_cards
    selected.extend(remaining[i] for i in best_combo)
    return selected[:discard_n]


class DrawCards(PowerEffect):
    """Draw cards from the deck, optionally keeping some.

    Examples:
    - "Draw 2 [card] from the deck."
    - "Draw 3 [card], then discard 1."
    - "All players draw 1 [card] from the deck."
    """

    def __init__(self, draw: int = 1, keep: int | None = None,
                 all_players: bool = False):
        self.draw = draw
        self.keep = keep if keep is not None else draw
        self.all_players = all_players

    def execute(self, ctx: PowerContext) -> PowerResult:
        kept_for_actor = 0

        def _resolve_for_player(p) -> int:
            seen = []
            for _ in range(self.draw):
                card = _draw_one_bird(ctx.game_state)
                if card is None:
                    break
                seen.append(card)
            keep_n = min(self.keep, len(seen))
            if keep_n > 0:
                p.hand.extend(seen[:keep_n])
            discarded = max(0, len(seen) - keep_n)
            if discarded > 0:
                ctx.game_state.discard_pile_count += discarded
            return keep_n

        if self.all_players:
            for p in ctx.game_state.players:
                kept = _resolve_for_player(p)
                if p.name == ctx.player.name:
                    kept_for_actor = kept
        else:
            kept_for_actor = _resolve_for_player(ctx.player)

        return PowerResult(
            cards_drawn=kept_for_actor,
            description=f"Drew {kept_for_actor} cards",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        prefix = "ALL PLAYERS: " if self.all_players else ""
        if self.keep < self.draw:
            return f"{prefix}draw {self.draw}, keep {self.keep} card{'s' if self.keep > 1 else ''}"
        return f"{prefix}draw {self.draw} card{'s' if self.draw > 1 else ''}"

    def estimate_value(self, ctx: PowerContext) -> float:
        base = self.keep * 0.4  # Cards are potential, not points
        if self.all_players:
            base *= 0.7
        return base * _draw_context_factor(ctx)


class DrawFromTray(PowerEffect):
    """Draw face-up cards with optional filtering.

    Examples:
    - "Draw 1 [card] from the tray whose food cost includes [invertebrate]."
    """

    def __init__(self, count: int = 1, food_filter: FoodType | None = None):
        self.count = count
        self.food_filter = food_filter

    def execute(self, ctx: PowerContext) -> PowerResult:
        drawn = 0
        tray = ctx.game_state.card_tray

        for _ in range(self.count):
            for i, bird in enumerate(tray.face_up):
                if self.food_filter:
                    if self.food_filter not in bird.food_cost.distinct_types:
                        continue
                card = tray.take_card(i)
                if card:
                    ctx.player.hand.append(card)
                    drawn += 1
                    break

        return PowerResult(cards_drawn=drawn,
                           description=f"Drew {drawn} from tray")

    def describe_activation(self, ctx: PowerContext) -> str:
        if self.food_filter:
            return f"draw {self.count} from tray (matching {self.food_filter.value} cost)"
        return f"draw {self.count} from tray"

    def estimate_value(self, ctx: PowerContext) -> float:
        hand_size = ctx.hand_size or len(ctx.player.hand)
        hand_factor = 0.65 if hand_size >= 10 else 1.0
        return self.count * 0.35 * hand_factor


class DrawBonusCards(PowerEffect):
    """Draw bonus cards from the bonus deck.

    Example: "Draw 2 bonus cards, then discard 1."
    """

    def __init__(self, draw: int = 2, keep: int = 1, discard_from_all_bonus: bool = False):
        self.draw = draw
        self.keep = keep
        self.discard_from_all_bonus = discard_from_all_bonus

    def execute(self, ctx: PowerContext) -> PowerResult:
        drawn = []
        for _ in range(max(0, self.draw)):
            card = _draw_one_bonus(ctx.game_state)
            if card is None:
                break
            drawn.append(card)

        if not drawn:
            return PowerResult(executed=False, cards_drawn=0, description="No bonus cards available")

        keep_n = min(max(0, self.keep), len(drawn))

        if self.discard_from_all_bonus:
            ctx.player.bonus_cards.extend(drawn)
            discard_n = max(0, len(drawn) - keep_n)
            to_discard = _choose_bonus_discards_by_score(
                ctx,
                list(ctx.player.bonus_cards),
                discard_n,
            )
            discard_pile = getattr(ctx.game_state, "_bonus_discard_cards", None)
            if not isinstance(discard_pile, list):
                discard_pile = []
                setattr(ctx.game_state, "_bonus_discard_cards", discard_pile)

            removed = []
            for target in to_discard:
                idx = next((i for i, bc in enumerate(ctx.player.bonus_cards) if bc is target), None)
                if idx is None:
                    continue
                removed.append(ctx.player.bonus_cards.pop(idx))
            discard_pile.extend(removed)

            removed_ids = {id(bc) for bc in removed}
            kept_from_drawn = sum(1 for bc in drawn if id(bc) not in removed_ids)
            return PowerResult(
                cards_drawn=kept_from_drawn,
                description=f"Drew {len(drawn)} bonus cards, discarded {len(removed)}",
            )

        kept_cards = _best_bonus_subset_by_score(ctx, drawn, keep_n)
        for bc in kept_cards:
            ctx.player.bonus_cards.append(bc)

        kept_ids = {id(bc) for bc in kept_cards}
        discarded = [bc for bc in drawn if id(bc) not in kept_ids]
        discard_pile = getattr(ctx.game_state, "_bonus_discard_cards", None)
        if not isinstance(discard_pile, list):
            discard_pile = []
            setattr(ctx.game_state, "_bonus_discard_cards", discard_pile)
        discard_pile.extend(discarded)

        return PowerResult(
            cards_drawn=len(kept_cards),
            description=f"Drew {len(drawn)} bonus cards, kept {len(kept_cards)}",
        )

    def describe_activation(self, ctx: PowerContext) -> str:
        if self.keep < self.draw:
            return f"draw {self.draw} bonus cards, keep {self.keep}"
        return f"draw {self.draw} bonus card{'s' if self.draw > 1 else ''}"

    def estimate_value(self, ctx: PowerContext) -> float:
        return self.keep * 1.5  # Bonus cards are high value
