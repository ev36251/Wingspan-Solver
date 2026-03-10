"""Monte Carlo Tree Search for Wingspan self-play.

Implements UCB-PUCT search with NN value + fast rollout blend.
Opponent is modeled as a greedy NN (no tree expansion for opponent turns).

Usage (inference):
    mcts = MCTS(model, encoder, num_sims=150)
    best_move = mcts.get_best_move(game, player_idx, temperature=0.0)

Usage (training data generation):
    moves, probs = mcts.get_policy(game, player_idx, temperature=1.0)
    chosen = random.choices(moves, weights=probs, k=1)[0]
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from backend.engine.scoring import calculate_score
from backend.solver.move_generator import Move, generate_all_moves
from backend.solver.simulation import (
    deep_copy_game,
    execute_move_on_sim,
    fast_rollout_move,
    simulate_playout,
    _refill_tray,
    _score_round_goal,
)

if TYPE_CHECKING:
    from backend.ml.factorized_inference import FactorizedPolicyModel
    from backend.ml.state_encoder import StateEncoder
    from backend.models.game_state import GameState


def _move_sig(move: Move) -> str:
    """Stable string key for a move (uses description field)."""
    return move.description


class MCTSNode:
    """A node in the MCTS tree.

    Each node represents a game state where it is the 'tracked' player's turn.
    The tree only expands nodes for the tracked player's turns; opponent turns
    are resolved greedily with the NN outside the tree.
    """

    __slots__ = ("parent", "action", "prior", "children", "N", "W")

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: Move | None = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action = action   # our Move that led FROM parent TO this node
        self.prior = prior
        self.children: dict[str, MCTSNode] = {}
        self.N: int = 0
        self.W: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, c_puct: float) -> float:
        parent_n = self.parent.N if self.parent else 1
        return self.Q + c_puct * self.prior * math.sqrt(parent_n) / (1 + self.N)


class MCTS:
    """UCB-PUCT Monte Carlo Tree Search guided by a factorized policy model.

    Algorithm per simulation:
      1. Selection   — follow UCB-PUCT from root to an unexpanded leaf.
      2. Expansion   — generate legal moves, compute softmax priors from NN,
                       create child MCTSNodes.
      3. Evaluation  — blend NN value + fast rollout from the leaf state.
      4. Backprop    — update N, W for all nodes on the selection path.

    Opponent handling:
      Between our moves, opponents play the greedy NN (single forward pass,
      argmax over legal moves). No tree nodes are created for opponent turns.
      This cuts simulation cost roughly in half while keeping playouts realistic.
    """

    def __init__(
        self,
        model: FactorizedPolicyModel,
        encoder: StateEncoder,
        num_sims: int = 150,
        c_puct: float = 1.5,
        value_blend: float = 0.5,
        rollout_policy: str = "fast",
        root_dirichlet_epsilon: float = 0.0,
        root_dirichlet_alpha: float = 0.3,
        backup_win_weight: float = 0.8,
        backup_score_weight: float = 0.2,
        tie_value_target: float = 0.5,
    ):
        self.model = model
        self.encoder = encoder
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.value_blend = value_blend
        self.rollout_policy = rollout_policy
        self.root_dirichlet_epsilon = max(0.0, min(1.0, float(root_dirichlet_epsilon)))
        self.root_dirichlet_alpha = max(1e-6, float(root_dirichlet_alpha))
        self.backup_win_weight = float(backup_win_weight)
        self.backup_score_weight = float(backup_score_weight)
        self.tie_value_target = max(0.0, min(1.0, float(tie_value_target)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_policy(
        self,
        game: GameState,
        player_idx: int,
        temperature: float = 1.0,
    ) -> tuple[list[Move], list[float]]:
        """Run MCTS and return (moves, visit-count probabilities).

        temperature=1.0  → sample proportionally to visit^(1/τ)  [training]
        temperature=0.0  → argmax visits                           [inference]
        """
        root = MCTSNode()
        for _ in range(self.num_sims):
            self._run_simulation(root, game, player_idx)

        if not root.children:
            # No legal moves found — return uniform fallback
            player = game.players[player_idx]
            moves = generate_all_moves(game, player)
            if not moves:
                return [], []
            n = len(moves)
            return moves, [1.0 / n] * n

        moves = []
        visits = []
        for child in root.children.values():
            moves.append(child.action)
            visits.append(child.N)

        visits_arr = np.array(visits, dtype=np.float64)

        if temperature < 1e-8:
            # Argmax: concentrate all probability on most-visited move
            best = int(np.argmax(visits_arr))
            probs = [0.0] * len(moves)
            probs[best] = 1.0
        else:
            powered = np.power(visits_arr + 1e-10, 1.0 / temperature)
            probs = list(powered / powered.sum())

        return moves, probs

    def get_best_move(
        self,
        game: GameState,
        player_idx: int,
        temperature: float = 0.0,
    ) -> Move | None:
        """Return the single best move from MCTS.

        Visits are summed to num_sims; temperature=0 → argmax (exploitation).
        """
        moves, probs = self.get_policy(game, player_idx, temperature=temperature)
        if not moves:
            return None
        if temperature < 1e-8:
            return moves[int(np.argmax(probs))]
        return random.choices(moves, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        root: MCTSNode,
        root_state: GameState,
        player_idx: int,
    ) -> None:
        """Run one complete MCTS simulation from the root."""
        sim = deep_copy_game(root_state)
        node = root
        path: list[MCTSNode] = [root]

        # ---- Selection ------------------------------------------------
        while node.children and not sim.is_game_over:
            best_child = max(
                node.children.values(), key=lambda c: c.ucb(self.c_puct)
            )
            # Execute our chosen move on the sim
            player = sim.players[player_idx]
            if player.action_cubes_remaining <= 0:
                break  # our cubes ran out mid-search; stop selection
            if not execute_move_on_sim(sim, player, best_child.action):
                break  # move failed (stale legal-move set); stop
            _refill_tray(sim)
            sim.advance_turn()
            # Let opponents play their turn(s) greedily
            self._advance_through_opponents(sim, player_idx)
            node = best_child
            path.append(node)

        # ---- Expansion -----------------------------------------------
        # Capture value estimates here so backup utility can skip redundant forward passes.
        precomputed_score_raw: float | None = None
        precomputed_win_prob: float | None = None
        if not sim.is_game_over and not node.children:
            if sim.current_player_idx == player_idx:
                player = sim.players[player_idx]
                if player.action_cubes_remaining > 0:
                    moves = generate_all_moves(sim, player)
                    if moves:
                        state_vec = np.asarray(
                            self.encoder.encode(sim, player_idx), dtype=np.float32
                        )
                        if hasattr(self.model, "forward_with_values"):
                            logits, score_raw, win_prob = self.model.forward_with_values(state_vec)
                        else:
                            logits, score_raw = self.model.forward(state_vec)
                            win_prob = None
                        if score_raw is not None:
                            precomputed_score_raw = score_raw
                        if win_prob is not None:
                            precomputed_win_prob = float(win_prob)
                        scores = np.array(
                            [
                                self.model.score_move(
                                    state_vec, m, player, logits=logits
                                )
                                for m in moves
                            ],
                            dtype=np.float64,
                        )
                        # Softmax → priors
                        exp_s = np.exp(scores - scores.max())
                        priors = exp_s / max(1e-12, exp_s.sum())
                        if (
                            node.parent is None
                            and self.root_dirichlet_epsilon > 0.0
                            and len(priors) > 1
                        ):
                            noise = np.random.dirichlet(
                                [self.root_dirichlet_alpha] * len(priors)
                            )
                            eps = self.root_dirichlet_epsilon
                            priors = (1.0 - eps) * priors + eps * noise
                            priors = priors / max(1e-12, priors.sum())
                        for m, p in zip(moves, priors):
                            node.children[_move_sig(m)] = MCTSNode(
                                parent=node, action=m, prior=float(p)
                            )

        # ---- Evaluation -----------------------------------------------
        utility = self._evaluate_utility(
            sim,
            player_idx,
            precomputed_score_raw=precomputed_score_raw,
            precomputed_win_prob=precomputed_win_prob,
        )

        # ---- Backprop -------------------------------------------------
        for n in path:
            n.N += 1
            n.W += utility

    def _evaluate(
        self,
        sim: GameState,
        player_idx: int,
        precomputed_value_raw: float | None = None,
    ) -> float:
        """Evaluate the leaf state: blend NN value with a fast rollout.

        Returns the tracked player's expected absolute final score.
        This matches value_target_score semantics used in self-play training.

        precomputed_value_raw: if provided, skip an extra model.forward() call by
        reusing the value_raw already obtained during expansion.
        """
        player_name = sim.players[player_idx].name

        if sim.is_game_over:
            scores = {p.name: calculate_score(sim, p).total for p in sim.players}
            my_score = float(scores.get(player_name, 0))
            return my_score

        # NN value estimate (absolute score after value_to_expected_score)
        nn_val = 0.0
        nn_available = self.model.has_value_head and self.value_blend > 0.0
        if nn_available:
            try:
                if precomputed_value_raw is not None:
                    # Reuse value_raw from expansion — saves one model.forward() call
                    nn_val = self.model.value_to_expected_score(precomputed_value_raw)
                else:
                    state_vec = np.asarray(
                        self.encoder.encode(sim, player_idx), dtype=np.float32
                    )
                    _, value_raw = self.model.forward(state_vec)
                    nn_val = self.model.value_to_expected_score(value_raw)
            except Exception:
                nn_available = False

        # Fast rollout estimate
        rollout_val = 0.0
        if self.value_blend < 1.0 or not nn_available:
            try:
                playout = simulate_playout(sim, rollout_policy=self.rollout_policy)
                rollout_val = float(playout.get(player_name, 0))
            except Exception:
                rollout_val = 0.0

        if nn_available:
            return float(
                self.value_blend * nn_val + (1.0 - self.value_blend) * rollout_val
            )
        return rollout_val

    def _score_to_norm(self, score: float) -> float:
        scale = float(getattr(self.model, "value_score_scale", 120.0) or 120.0)
        bias = float(getattr(self.model, "value_score_bias", 0.0) or 0.0)
        return float(np.clip((float(score) - bias) / max(1e-9, scale), 0.0, 1.0))

    def _win_target_from_scores(self, my_score: float, other_scores: list[float]) -> float:
        best_other = max(other_scores) if other_scores else 0.0
        if my_score > best_other:
            return 1.0
        if my_score < best_other:
            return 0.0
        return self.tie_value_target

    def _utility_from_win_score(self, win_prob: float, score_norm: float) -> float:
        return float(
            self.backup_win_weight * float(win_prob)
            + self.backup_score_weight * float(score_norm)
        )

    def _evaluate_utility(
        self,
        sim: GameState,
        player_idx: int,
        precomputed_score_raw: float | None = None,
        precomputed_win_prob: float | None = None,
    ) -> float:
        """Evaluate leaf utility in [0,1] for MCTS backups.

        Utility blends win probability (primary) and normalized score (auxiliary):
        `0.8 * win_prob + 0.2 * normalized_score` by default.
        """
        player_name = sim.players[player_idx].name

        if sim.is_game_over:
            final_scores = {
                p.name: float(calculate_score(sim, p).total) for p in sim.players
            }
            my_score = float(final_scores.get(player_name, 0.0))
            others = [s for n, s in final_scores.items() if n != player_name]
            win_t = self._win_target_from_scores(my_score, others)
            return self._utility_from_win_score(win_t, self._score_to_norm(my_score))

        nn_utility: float | None = None
        if self.value_blend > 0.0:
            try:
                score_est: float | None = None
                win_prob_est: float | None = None
                if precomputed_score_raw is not None:
                    score_est = self.model.value_to_expected_score(precomputed_score_raw)
                if precomputed_win_prob is not None:
                    win_prob_est = float(precomputed_win_prob)
                if score_est is None or win_prob_est is None:
                    state_vec = np.asarray(
                        self.encoder.encode(sim, player_idx), dtype=np.float32
                    )
                    if hasattr(self.model, "forward_with_values"):
                        _, score_raw, win_prob = self.model.forward_with_values(state_vec)
                    else:
                        _, score_raw = self.model.forward(state_vec)
                        win_prob = None
                    if score_est is None and score_raw is not None:
                        score_est = self.model.value_to_expected_score(score_raw)
                    if win_prob_est is None and win_prob is not None:
                        win_prob_est = float(win_prob)
                if score_est is not None:
                    score_norm = self._score_to_norm(score_est)
                    if win_prob_est is None:
                        win_prob_est = score_norm
                    nn_utility = self._utility_from_win_score(win_prob_est, score_norm)
            except Exception:
                nn_utility = None

        rollout_utility: float | None = None
        if self.value_blend < 1.0 or nn_utility is None:
            try:
                playout = simulate_playout(sim, rollout_policy=self.rollout_policy)
                my_score = float(playout.get(player_name, 0.0))
                other_scores = [
                    float(playout.get(p.name, 0.0))
                    for i, p in enumerate(sim.players)
                    if i != player_idx
                ]
                win_t = self._win_target_from_scores(my_score, other_scores)
                rollout_utility = self._utility_from_win_score(
                    win_t, self._score_to_norm(my_score)
                )
            except Exception:
                rollout_utility = None

        if nn_utility is not None and rollout_utility is not None:
            return float(
                self.value_blend * nn_utility + (1.0 - self.value_blend) * rollout_utility
            )
        if nn_utility is not None:
            return float(nn_utility)
        if rollout_utility is not None:
            return float(rollout_utility)
        return 0.0

    def _advance_through_opponents(
        self, sim: GameState, player_idx: int
    ) -> None:
        """Execute opponent moves greedily until it's player_idx's turn or game over.

        Uses a single NN forward pass + argmax per opponent turn for speed.
        """
        max_steps = 200
        steps = 0
        while (
            not sim.is_game_over
            and sim.current_player_idx != player_idx
            and steps < max_steps
        ):
            opp_idx = sim.current_player_idx
            opp_player = sim.players[opp_idx]

            if opp_player.action_cubes_remaining <= 0:
                # All players might be exhausted — advance_turn handles round end
                if all(p.action_cubes_remaining <= 0 for p in sim.players):
                    _score_round_goal(sim, sim.current_round)
                    sim.advance_round()
                else:
                    sim.current_player_idx = (
                        (sim.current_player_idx + 1) % sim.num_players
                    )
                steps += 1
                continue

            opp_moves = generate_all_moves(sim, opp_player)
            if not opp_moves:
                opp_player.action_cubes_remaining = 0
                sim.advance_turn()
                steps += 1
                continue

            # Fast heuristic for opponents — avoids a full NN forward pass per step,
            # saving ~1-2 model.forward() calls per simulation.
            opp_move = fast_rollout_move(opp_moves, sim, opp_player)

            execute_move_on_sim(sim, opp_player, opp_move)
            _refill_tray(sim)
            sim.advance_turn()
            steps += 1
