"""Full game state for a Wingspan game in progress."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from .player import Player
from .birdfeeder import Birdfeeder
from .card_tray import CardTray
from .goal import Goal
from backend.config import ACTIONS_PER_ROUND, ROUNDS
from backend.models.enums import BoardType, PowerColor

if TYPE_CHECKING:
    from backend.solver.deck_tracker import DeckTracker


@dataclass
class MoveRecord:
    """Record of a single move made during the game."""
    round: int
    turn: int
    player_name: str
    action_type: str
    description: str
    solver_rank: int | None = None
    total_moves: int = 0
    best_move_description: str = ""


@dataclass
class GameState:
    """Complete state of a Wingspan game at any point in time."""
    players: list[Player]
    board_type: BoardType = BoardType.BASE
    current_player_idx: int = 0
    current_round: int = 1  # 1-4
    turn_in_round: int = 1  # 1-based within the round
    birdfeeder: Birdfeeder = field(default_factory=Birdfeeder)
    card_tray: CardTray = field(default_factory=CardTray)
    round_goals: list[Goal] = field(default_factory=list)  # 4 goals, one per round
    # round -> {player_name: score}
    round_goal_scores: dict[int, dict[str, int]] = field(default_factory=dict)
    strict_rules_mode: bool = False
    deck_remaining: int = 0
    discard_pile_count: int = 0
    deck_tracker: "DeckTracker | None" = None
    move_history: list[MoveRecord] = field(default_factory=list)
    # Optional deterministic choices for power resolution.
    # Key: "<player_name>::<bird_name>" -> FIFO list of choice payload dicts.
    power_choice_queues: dict[str, list[dict]] = field(default_factory=dict)
    # Debug/event stream for bird-power execution visibility in tests.
    power_events: list[dict] = field(default_factory=list)
    _end_game_powers_resolved: bool = False

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def is_game_over(self) -> bool:
        return self.current_round > ROUNDS

    @property
    def actions_this_round(self) -> int:
        """Total actions available per player this round.

        Each 'No Goal' round adds +1 action to all subsequent rounds.
        """
        if self.current_round < 1 or self.current_round > ROUNDS:
            return 0
        return self._effective_actions(self.current_round)

    def _effective_actions(self, round_num: int) -> int:
        """Compute actions for a round, adding +1 per prior 'No Goal' round."""
        base = ACTIONS_PER_ROUND[round_num - 1]
        no_goal_bonus = 0
        for i in range(round_num - 1):
            if i < len(self.round_goals) and self._is_no_goal(self.round_goals[i]):
                no_goal_bonus += 1
        return base + no_goal_bonus

    @staticmethod
    def _is_no_goal(goal: Goal) -> bool:
        return goal.description.lower() == "no goal"

    @property
    def total_actions_remaining(self) -> int:
        """Total actions remaining for all players in the game."""
        total = 0
        for p in self.players:
            total += p.action_cubes_remaining
        # Add actions from future rounds (using effective actions)
        for r in range(self.current_round + 1, ROUNDS + 1):
            total += self._effective_actions(r) * self.num_players
        return total

    def current_round_goal(self) -> Goal | None:
        """The goal for the current round."""
        if 1 <= self.current_round <= len(self.round_goals):
            return self.round_goals[self.current_round - 1]
        return None

    def advance_turn(self) -> None:
        """Move to the next player's turn within the current round."""
        self.current_player.action_cubes_remaining -= 1
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        # Skip players who have no cubes left
        skipped = 0
        while (self.current_player.action_cubes_remaining <= 0
               and skipped < self.num_players):
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            skipped += 1

        # If all players exhausted, advance round
        if all(p.action_cubes_remaining <= 0 for p in self.players):
            self.advance_round()

    def advance_round(self) -> None:
        """End the current round and start the next.

        On Oceania boards, all nectar in player supplies is discarded.
        """
        # 1) Resolve round-end powers first (teal / explicit end-of-round text).
        from backend.engine.timed_powers import trigger_end_of_round_powers
        trigger_end_of_round_powers(self, self.current_round)

        # 2) On Oceania boards, discard unspent nectar before scoring round goal.
        if self.board_type == BoardType.OCEANIA:
            for p in self.players:
                p.food_supply.nectar = 0

        # 3) Score the round goal for this round.
        if 1 <= self.current_round <= len(self.round_goals):
            from backend.engine.scoring import compute_round_goal_scores
            scores = compute_round_goal_scores(self, self.current_round)
            if scores:
                self.round_goal_scores[self.current_round] = scores

        # 4) Refresh the tray at end of round (discard all, then refill to 3).
        discarded = self.card_tray.clear()
        self.discard_pile_count += len(discarded)

        def _draw_card_for_tray():
            deck_cards = getattr(self, "_deck_cards", None)
            if isinstance(deck_cards, list) and deck_cards:
                card = deck_cards.pop()
                self.deck_remaining = max(0, self.deck_remaining - 1)
                if self.deck_tracker is not None:
                    self.deck_tracker.mark_drawn(card.name)
                return card
            if self.deck_remaining <= 0:
                return None
            from backend.data.registries import get_bird_registry
            import random

            pool = list(get_bird_registry().all_birds)
            if not pool:
                return None
            self.deck_remaining = max(0, self.deck_remaining - 1)
            card = random.choice(pool)
            if self.deck_tracker is not None:
                self.deck_tracker.mark_drawn(card.name)
            return card

        for _ in range(3):
            card = _draw_card_for_tray()
            if card is None:
                break
            self.card_tray.add_card(card)

        # 5) Reset per-round action trackers after round cleanup.
        for p in self.players:
            p.play_bird_actions_this_round = 0
            p.gain_food_actions_this_round = 0
            p.lay_eggs_actions_this_round = 0
            p.draw_cards_actions_this_round = 0
            p.action_types_used_this_round.clear()

        # 6) Advance to next round and reset action cubes.
        self.current_round += 1
        self.turn_in_round = 1
        self.current_player_idx = 0

        if self.current_round <= ROUNDS:
            cubes = self._effective_actions(self.current_round)
            for p in self.players:
                p.action_cubes_remaining = cubes
        else:
            # Resolve end-of-game powers once when the game ends.
            from backend.engine.timed_powers import trigger_end_of_game_powers
            trigger_end_of_game_powers(self)

    def get_player(self, name: str) -> Player | None:
        """Find a player by name."""
        for p in self.players:
            if p.name == name:
                return p
        return None

    def other_players(self, player: Player) -> list[Player]:
        """All players except the given one."""
        return [p for p in self.players if p.name != player.name]

    def player_to_left(self, player: Player) -> Player | None:
        """The player seated to the left (next index, wrapping)."""
        if self.num_players < 2:
            return None
        idx = next((i for i, p in enumerate(self.players) if p.name == player.name), 0)
        return self.players[(idx + 1) % self.num_players]

    def player_to_right(self, player: Player) -> Player | None:
        """The player seated to the right (previous index, wrapping)."""
        if self.num_players < 2:
            return None
        idx = next((i for i, p in enumerate(self.players) if p.name == player.name), 0)
        return self.players[(idx - 1) % self.num_players]

    def record_power_event(
        self,
        *,
        timing: str,
        color: PowerColor | str,
        player_name: str,
        bird_name: str,
        executed: bool,
    ) -> None:
        """Append a normalized power event record for validation/debug tests."""
        color_value = color.value if isinstance(color, PowerColor) else str(color)
        self.power_events.append(
            {
                "timing": str(timing),
                "color": color_value,
                "player": str(player_name),
                "bird": str(bird_name),
                "round": int(self.current_round),
                "turn": int(self.turn_in_round),
                "executed": bool(executed),
            }
        )


def create_new_game(player_names: list[str],
                    round_goals: list[Goal] | None = None,
                    board_type: BoardType = BoardType.BASE,
                    strict_rules_mode: bool = False) -> GameState:
    """Create a fresh game state with players and initial setup."""
    players = [Player(name=name) for name in player_names]

    # Round 1 starts with max actions
    cubes = ACTIONS_PER_ROUND[0]
    for p in players:
        p.action_cubes_remaining = cubes
        # Oceania: each player starts with 1 free nectar
        if board_type == BoardType.OCEANIA:
            from backend.models.enums import FoodType
            p.food_supply.add(FoodType.NECTAR, 1)

    from backend.solver.deck_tracker import DeckTracker

    return GameState(
        players=players,
        board_type=board_type,
        birdfeeder=Birdfeeder(board_type=board_type),
        round_goals=round_goals or [],
        strict_rules_mode=strict_rules_mode,
        deck_tracker=DeckTracker(),
    )
