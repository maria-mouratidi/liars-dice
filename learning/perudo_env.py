"""
Perudo (Liar's Dice) Environment for Reinforcement Learning.

This module implements a complete Perudo environment compatible with
the standard Gymnasium interface, designed for self-play training.

Features:
- Multiplayer support (2-6 players)
- Calza rule for advanced play
- Probability-based observation features to help the agent understand bid likelihood
- Reward shaping to provide learning signal for good/bad decisions
- Full game rules including palifico rounds
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum


class ActionType(IntEnum):
    """Action types in Perudo."""
    CHALLENGE = 0  # Call "dudo" - challenge the current bid
    CALZA = 1      # Call "calza" - declare bid is exactly right
    BID = 2        # Make a new bid


@dataclass
class Bid:
    """Represents a bid in the game."""
    quantity: int
    face_value: int  # 1-6, where 1 is ace (wild)
    bidder: int = 0  # Player who made this bid

    def __eq__(self, other):
        if not isinstance(other, Bid):
            return False
        return self.quantity == other.quantity and self.face_value == other.face_value

    def __hash__(self):
        return hash((self.quantity, self.face_value))

    def is_higher_than(self, other: "Bid") -> bool:
        """Check if this bid is higher than another (Perudo rules)."""
        if self.quantity > other.quantity:
            return True
        if self.quantity == other.quantity and self.face_value > other.face_value:
            return True
        return False

    def is_ace_bid(self) -> bool:
        """Check if this is an ace (wild) bid."""
        return self.face_value == 1


def binomial_at_least(k: int, n: int, p: float) -> float:
    """
    Calculate probability of getting at least k successes in n trials with probability p.
    
    P(X >= k) = sum_{i=k}^{n} C(n,i) * p^i * (1-p)^{n-i}
    
    Uses a simple implementation to avoid scipy dependency.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if n == 0:
        return 0.0 if k > 0 else 1.0
    
    # Calculate using cumulative sum
    prob = 0.0
    for i in range(k, n + 1):
        # Calculate binomial coefficient C(n, i)
        coef = 1.0
        for j in range(i):
            coef = coef * (n - j) / (j + 1)
        prob += coef * (p ** i) * ((1 - p) ** (n - i))
    
    return min(1.0, max(0.0, prob))


def binomial_exactly(k: int, n: int, p: float) -> float:
    """
    Calculate probability of getting exactly k successes in n trials with probability p.
    
    P(X = k) = C(n,k) * p^k * (1-p)^{n-k}
    """
    if k < 0 or k > n:
        return 0.0
    if n == 0:
        return 1.0 if k == 0 else 0.0
    
    # Calculate binomial coefficient C(n, k)
    coef = 1.0
    for j in range(k):
        coef = coef * (n - j) / (j + 1)
    
    prob = coef * (p ** k) * ((1 - p) ** (n - k))
    return min(1.0, max(0.0, prob))


class PerudoEnv:
    """
    Perudo (Liar's Dice) environment for multiplayer self-play.
    
    Supports 2-6 players with full Perudo rules including:
    - Aces (1s) are wild
    - Palifico rounds
    - Calza calls (optional, for 3+ players only)
    
    Observation Space (per player):
        - Own dice counts per face: 6 values
        - Current bid quantity (normalized): 1 value
        - Current bid face value (one-hot): 6 values
        - Number of own dice (normalized): 1 value
        - Number of each opponent's dice (normalized): (num_players-1) values
        - Total dice in play (normalized): 1 value
        - Is first bid of round: 1 value
        - Is palifico round: 1 value
        - Current player indicator: 1 value
        - Am I the bidder: 1 value
        - Can I calza: 1 value
        - Probability current bid is true: 1 value
        - Probability current bid is exactly right: 1 value
        - My matching dice for current bid: 1 value
        - Needed from others for bid to be true: 1 value
        - Expected matches from others: 1 value
        
    Action Space:
        - Action 0: Challenge (dudo)
        - Action 1: Calza (declare bid is exactly right)
        - Actions 2 to max_bid_actions+1: Bid (quantity, face_value)
    """
    
    # Maximum dice per player
    MAX_DICE = 5
    
    # Maximum players
    MAX_PLAYERS = 6
    
    # Number of face values (1-6)
    NUM_FACES = 6
    
    def __init__(
        self,
        num_players: int = 2,
        starting_dice: int = 5,
        joker_mode: bool = True,
        enable_calza: bool = True,
        reward_win_game: float = 1.0,
        reward_lose_game: float = -1.0,
        reward_win_round: float = 0.1,
        reward_lose_round: float = -0.1,
        reward_calza_success: float = 0.2,
        reward_calza_fail: float = -0.2,
        reward_invalid: float = -0.5,
        reward_shaping: bool = True,
        shaping_scale: float = 0.05,
    ):
        """
        Initialize the Perudo environment.
        
        Args:
            num_players: Number of players (2-6)
            starting_dice: Number of dice each player starts with
            joker_mode: Whether aces are wild (count as any value)
            enable_calza: Whether to allow calza calls
            reward_win_game: Reward for winning the game
            reward_lose_game: Reward for losing the game
            reward_win_round: Reward for winning a round (challenge)
            reward_lose_round: Reward for losing a round
            reward_calza_success: Reward for successful calza
            reward_calza_fail: Penalty for failed calza
            reward_invalid: Penalty for invalid actions
            reward_shaping: Whether to use reward shaping
            shaping_scale: Scale factor for shaping rewards
        """
        self.num_players = max(2, min(num_players, self.MAX_PLAYERS))
        self.starting_dice = min(starting_dice, self.MAX_DICE)
        self.joker_mode = joker_mode
        self.enable_calza = enable_calza
        self.reward_win_game = reward_win_game
        self.reward_lose_game = reward_lose_game
        self.reward_win_round = reward_win_round
        self.reward_lose_round = reward_lose_round
        self.reward_calza_success = reward_calza_success
        self.reward_calza_fail = reward_calza_fail
        self.reward_invalid = reward_invalid
        self.reward_shaping = reward_shaping
        self.shaping_scale = shaping_scale
        
        # Maximum quantity that can be bid (use max possible for consistent action space)
        self.max_quantity = self.MAX_PLAYERS * self.MAX_DICE  # Always 30
        
        # Calculate observation and action space sizes
        self.obs_size = self._calculate_obs_size()
        # Actions: challenge + calza + (max_quantity * 6 faces)
        # This is fixed at 2 + 30 * 6 = 182 actions regardless of num_players
        self.action_space_n = 2 + self.max_quantity * self.NUM_FACES
        
        # Game state
        self.player_dice: List[np.ndarray] = [np.array([]) for _ in range(self.num_players)]
        self.player_num_dice: List[int] = [0] * self.num_players
        self.player_active: List[bool] = [True] * self.num_players
        self.current_player: int = 0
        self.current_bid: Optional[Bid] = None
        self.round_number: int = 0
        self.is_palifico_round: bool = False
        self.palifico_player: Optional[int] = None
        self.players_had_palifico: set = set()
        self.game_over: bool = False
        self.winner: Optional[int] = None
        
    def _calculate_obs_size(self) -> int:
        """Calculate the size of the observation vector."""
        # Dice counts per face
        dice_counts = self.NUM_FACES  # 6 values: count of 1s, 2s, ..., 6s
        # Current bid info
        bid_quantity_size = 1  # Normalized quantity
        bid_face_size = self.NUM_FACES  # One-hot face value
        # Game state info
        own_dice_count = 1
        # Use MAX_PLAYERS - 1 slots for opponent dice counts (pad with 0 for missing players)
        other_dice_counts = self.MAX_PLAYERS - 1  # Always 5 slots for opponents
        total_dice_count = 1
        is_first_bid = 1
        is_palifico = 1
        current_player_indicator = 1
        am_i_bidder = 1
        can_calza = 1
        
        # Probability features
        prob_bid_true = 1
        prob_bid_exact = 1
        my_matches = 1
        needed_from_others = 1
        expected_others_matches = 1
        
        return (dice_counts + bid_quantity_size + bid_face_size + 
                own_dice_count + other_dice_counts + total_dice_count +
                is_first_bid + is_palifico + current_player_indicator +
                am_i_bidder + can_calza +
                prob_bid_true + prob_bid_exact + my_matches + 
                needed_from_others + expected_others_matches)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new game."""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset dice counts
        self.player_num_dice = [self.starting_dice] * self.num_players
        self.player_active = [True] * self.num_players
        
        # Roll dice for all players
        self._roll_dice()
        
        # Reset game state
        self.current_player = np.random.randint(0, self.num_players)
        self.current_bid = None
        self.round_number = 1
        self.is_palifico_round = False
        self.palifico_player = None
        self.players_had_palifico = set()
        self.game_over = False
        self.winner = None
        
        obs = self._get_observation(self.current_player)
        info = {"current_player": self.current_player, "num_active_players": self._count_active_players()}
        
        return obs, info
    
    def _count_active_players(self) -> int:
        """Count players still in the game."""
        return sum(1 for active in self.player_active if active)
    
    def _roll_dice(self):
        """Roll dice for all players."""
        for player in range(self.num_players):
            if self.player_num_dice[player] > 0:
                self.player_dice[player] = np.random.randint(
                    1, 7, size=self.player_num_dice[player]
                )
            else:
                self.player_dice[player] = np.array([])
    
    def _get_total_dice(self) -> int:
        """Get total dice in play."""
        return sum(self.player_num_dice)
    
    def _count_my_matches(self, player: int, face_value: int) -> int:
        """Count how many of player's dice match the face value."""
        dice = self.player_dice[player]
        if len(dice) == 0:
            return 0
        count = np.sum(dice == face_value)
        
        # Aces are wild (unless palifico or counting aces)
        if self.joker_mode and not self.is_palifico_round and face_value != 1:
            count += np.sum(dice == 1)
        
        return int(count)
    
    def _calculate_bid_probability(self, player: int, bid: Bid) -> float:
        """
        Calculate the probability that a bid is true from a player's perspective.
        
        Uses binomial distribution for unknown dice.
        """
        if bid is None:
            return 0.5  # No bid, neutral
        
        # How many of my dice match?
        my_matches = self._count_my_matches(player, bid.face_value)
        
        # How many more needed from others?
        needed = max(0, bid.quantity - my_matches)
        
        # Count unknown dice (all dice not belonging to this player)
        unknown_dice = sum(self.player_num_dice[p] for p in range(self.num_players) if p != player)
        
        if needed == 0:
            return 1.0  # Already have enough
        if needed > unknown_dice:
            return 0.0  # Impossible
        
        # Probability of a single die matching
        if self.joker_mode and not self.is_palifico_round and bid.face_value != 1:
            p_match = 2 / 6  # 1/6 for target + 1/6 for ace
        else:
            p_match = 1 / 6
        
        # Probability of at least 'needed' matches from unknown dice
        return binomial_at_least(needed, unknown_dice, p_match)
    
    def _calculate_calza_probability(self, player: int, bid: Bid) -> float:
        """
        Calculate the probability that a bid is exactly right.
        """
        if bid is None:
            return 0.0
        
        # How many of my dice match?
        my_matches = self._count_my_matches(player, bid.face_value)
        
        # How many exactly needed from others?
        needed = bid.quantity - my_matches
        
        # Count unknown dice
        unknown_dice = sum(self.player_num_dice[p] for p in range(self.num_players) if p != player)
        
        if needed < 0 or needed > unknown_dice:
            return 0.0
        
        # Probability of a single die matching
        if self.joker_mode and not self.is_palifico_round and bid.face_value != 1:
            p_match = 2 / 6
        else:
            p_match = 1 / 6
        
        # Probability of exactly 'needed' matches
        return binomial_exactly(needed, unknown_dice, p_match)
    
    def _can_calza(self, player: int) -> bool:
        """Check if a player can call calza."""
        if not self.enable_calza:
            return False
        if self.current_bid is None:
            return False
        # Cannot calza in palifico round
        if self.is_palifico_round:
            return False
        # Cannot calza with only 2 players left
        if self._count_active_players() <= 2:
            return False
        # Cannot calza if you're the next player (current player)
        if player == self.current_player:
            return False
        # Cannot calza your own bid
        if self.current_bid.bidder == player:
            return False
        
        return True
    
    def _get_observation(self, player: int) -> np.ndarray:
        """Get the observation for a specific player."""
        obs = []
        
        # Dice counts per face (normalized by max dice)
        dice_counts = np.zeros(self.NUM_FACES)
        for die in self.player_dice[player]:
            dice_counts[int(die) - 1] += 1
        obs.extend(dice_counts / self.MAX_DICE)
        
        # Current bid info
        if self.current_bid is not None:
            # Normalized quantity
            obs.append(self.current_bid.quantity / self.max_quantity)
            # One-hot face value
            face_one_hot = np.zeros(self.NUM_FACES)
            face_one_hot[self.current_bid.face_value - 1] = 1.0
            obs.extend(face_one_hot)
        else:
            obs.append(0.0)
            obs.extend([0.0] * self.NUM_FACES)
        
        # Own dice count (normalized)
        obs.append(self.player_num_dice[player] / self.MAX_DICE)
        
        # Other players' dice counts (normalized) - always MAX_PLAYERS-1 slots
        other_counts = []
        for other in range(self.num_players):
            if other != player:
                other_counts.append(self.player_num_dice[other] / self.MAX_DICE)
        # Pad with zeros if fewer than MAX_PLAYERS-1 opponents
        while len(other_counts) < self.MAX_PLAYERS - 1:
            other_counts.append(0.0)
        obs.extend(other_counts)
        
        # Total dice (normalized by max possible for 6 players)
        total_dice = self._get_total_dice()
        obs.append(total_dice / (self.MAX_PLAYERS * self.MAX_DICE))
        
        # Flags
        obs.append(1.0 if self.current_bid is None else 0.0)  # First bid
        obs.append(1.0 if self.is_palifico_round else 0.0)    # Palifico
        obs.append(1.0 if player == self.current_player else 0.0)  # My turn
        obs.append(1.0 if self.current_bid and self.current_bid.bidder == player else 0.0)  # Am I bidder
        obs.append(1.0 if self._can_calza(player) else 0.0)  # Can calza
        
        # PROBABILITY FEATURES
        if self.current_bid is not None:
            # Probability current bid is true
            prob_true = self._calculate_bid_probability(player, self.current_bid)
            obs.append(prob_true)
            
            # Probability bid is exactly right (for calza)
            prob_exact = self._calculate_calza_probability(player, self.current_bid)
            obs.append(prob_exact)
            
            # My matches for current bid (normalized)
            my_matches = self._count_my_matches(player, self.current_bid.face_value)
            obs.append(my_matches / self.max_quantity)
            
            # Needed from others (normalized)
            needed = max(0, self.current_bid.quantity - my_matches)
            obs.append(needed / self.max_quantity)
            
            # Expected matches from others
            unknown_dice = sum(self.player_num_dice[p] for p in range(self.num_players) if p != player)
            if self.joker_mode and not self.is_palifico_round and self.current_bid.face_value != 1:
                p_match = 2 / 6
            else:
                p_match = 1 / 6
            expected = unknown_dice * p_match
            obs.append(expected / self.max_quantity)
        else:
            obs.extend([0.5, 0.0, 0.0, 0.0, 0.0])  # Neutral values when no bid
        
        return np.array(obs, dtype=np.float32)
    
    def _decode_action(self, action: int) -> Tuple[ActionType, Optional[Bid]]:
        """Decode an action index into action type and bid."""
        if action == 0:
            return ActionType.CHALLENGE, None
        if action == 1:
            return ActionType.CALZA, None
        
        bid_idx = action - 2
        quantity = (bid_idx // self.NUM_FACES) + 1
        face_value = (bid_idx % self.NUM_FACES) + 1
        
        return ActionType.BID, Bid(quantity, face_value, self.current_player)
    
    def _encode_bid(self, bid: Bid) -> int:
        """Encode a bid into an action index."""
        return 2 + (bid.quantity - 1) * self.NUM_FACES + (bid.face_value - 1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action in the environment."""
        if self.game_over:
            obs = self._get_observation(self.current_player)
            return obs, 0.0, True, False, {"winner": self.winner}
        
        action_type, bid = self._decode_action(action)
        info = {"action_type": action_type.name, "acting_player": self.current_player}
        
        # Validate action
        if not self._is_valid_action(action):
            obs = self._get_observation(self.current_player)
            return obs, self.reward_invalid, False, False, {"error": "invalid_action"}
        
        if action_type == ActionType.CHALLENGE:
            reward, terminated = self._handle_challenge()
            info["challenge_result"] = "success" if reward > 0 else "failure"
        elif action_type == ActionType.CALZA:
            reward, terminated = self._handle_calza()
            info["calza_result"] = "success" if reward > 0 else "failure"
        else:
            reward, terminated = self._handle_bid(bid)
            info["bid"] = f"{bid.quantity}x{bid.face_value}"
        
        obs = self._get_observation(self.current_player)
        info["current_player"] = self.current_player
        info["num_active_players"] = self._count_active_players()
        
        return obs, reward, terminated, False, info
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid in the current state."""
        action_type, bid = self._decode_action(action)
        
        if action_type == ActionType.CHALLENGE:
            return self.current_bid is not None
        
        if action_type == ActionType.CALZA:
            return self._can_calza(self.current_player)
        
        return self._is_valid_bid(bid)
    
    def _is_valid_bid(self, bid: Bid) -> bool:
        """Check if a bid is valid in the current state."""
        if bid.quantity <= 0 or bid.quantity > self.max_quantity:
            return False
        if bid.face_value < 1 or bid.face_value > 6:
            return False
        
        total_dice = self._get_total_dice()
        if bid.quantity > total_dice:
            return False
        
        # First bid of round
        if self.current_bid is None:
            if bid.is_ace_bid():
                if not self.is_palifico_round:
                    return False
                if self.current_player != self.palifico_player:
                    return False
            return True
        
        # Palifico round rules
        if self.is_palifico_round:
            player_had_palifico = (
                self.current_player in self.players_had_palifico and
                self.current_player != self.palifico_player
            )
            if not player_had_palifico:
                if bid.face_value != self.current_bid.face_value:
                    return False
                if bid.quantity <= self.current_bid.quantity:
                    return False
                return True
        
        # Normal bidding rules
        if bid.is_ace_bid() and not self.current_bid.is_ace_bid():
            min_quantity = (self.current_bid.quantity + 1) // 2
            return bid.quantity >= min_quantity
        elif not bid.is_ace_bid() and self.current_bid.is_ace_bid():
            min_quantity = self.current_bid.quantity * 2 + 1
            return bid.quantity >= min_quantity
        else:
            return bid.is_higher_than(self.current_bid)
    
    def _get_next_active_player(self, from_player: int) -> int:
        """Get the next active player after from_player."""
        next_p = (from_player + 1) % self.num_players
        while not self.player_active[next_p]:
            next_p = (next_p + 1) % self.num_players
        return next_p
    
    def _count_all_matching_dice(self, face_value: int) -> int:
        """Count all dice matching the face value across all players."""
        all_dice = np.concatenate([d for d in self.player_dice if len(d) > 0])
        count = np.sum(all_dice == face_value)
        
        # IMPORTANT: In palifico round, aces are NOT wild
        if self.joker_mode and not self.is_palifico_round and face_value != 1:
            count += np.sum(all_dice == 1)
        
        return int(count)
    
    def _handle_challenge(self) -> Tuple[float, bool]:
        """Handle a challenge action with reward shaping."""
        challenger = self.current_player
        bidder = self.current_bid.bidder
        
        # Count all matching dice
        actual_count = self._count_all_matching_dice(self.current_bid.face_value)
        
        # Calculate how good the decision was (for shaping)
        prob_bid_true = self._calculate_bid_probability(challenger, self.current_bid)
        
        # Determine winner
        if actual_count >= self.current_bid.quantity:
            # Bid was valid, challenger loses
            loser = challenger
            is_win = False
            base_reward = self.reward_lose_round
        else:
            # Bid was invalid, bidder loses
            loser = bidder
            is_win = True
            base_reward = self.reward_win_round
        
        # Reward shaping
        shaping_reward = 0.0
        if self.reward_shaping:
            if is_win:
                shaping_reward = self.shaping_scale * (1.0 - prob_bid_true)
            else:
                shaping_reward = -self.shaping_scale * prob_bid_true
        
        # Loser loses a die
        return self._player_loses_die(loser, base_reward + shaping_reward, is_win)
    
    def _handle_calza(self) -> Tuple[float, bool]:
        """Handle a calza action."""
        caller = self.current_player
        
        # Count all matching dice
        actual_count = self._count_all_matching_dice(self.current_bid.face_value)
        
        # Check if exactly right
        if actual_count == self.current_bid.quantity:
            # Calza successful! Caller gains a die (max 5)
            if self.player_num_dice[caller] < self.MAX_DICE:
                self.player_num_dice[caller] += 1
            
            # Start new round with caller
            self._start_new_round(caller)
            
            return self.reward_calza_success, False
        else:
            # Calza failed, caller loses a die
            return self._player_loses_die(caller, self.reward_calza_fail, False)
    
    def _player_loses_die(self, player: int, base_reward: float, is_win: bool) -> Tuple[float, bool]:
        """Handle a player losing a die."""
        self.player_num_dice[player] -= 1
        
        # Check if player is out
        if self.player_num_dice[player] <= 0:
            self.player_active[player] = False
            
            # Check if game is over
            if self._count_active_players() <= 1:
                self.game_over = True
                # Find the winner
                for p in range(self.num_players):
                    if self.player_active[p]:
                        self.winner = p
                        break
                
                # Return appropriate reward
                if is_win:
                    return base_reward + self.reward_win_game, True
                else:
                    return base_reward + self.reward_lose_game, True
        
        # Start new round with loser (or next active player if loser is out)
        starting_player = player if self.player_active[player] else self._get_next_active_player(player)
        self._start_new_round(starting_player)
        
        return base_reward, False
    
    def _handle_bid(self, bid: Bid) -> Tuple[float, bool]:
        """Handle a bid action with reward shaping."""
        shaping_reward = 0.0
        
        if self.reward_shaping:
            prob_new_bid = self._calculate_bid_probability(self.current_player, bid)
            if prob_new_bid < 0.1:
                shaping_reward = -self.shaping_scale * 0.5
            elif prob_new_bid > 0.7:
                shaping_reward = self.shaping_scale * 0.2
        
        self.current_bid = bid
        self.current_player = self._get_next_active_player(self.current_player)
        
        return shaping_reward, False
    
    def _start_new_round(self, starting_player: int):
        """Start a new round after a challenge or calza."""
        self.current_bid = None
        self.current_player = starting_player
        self.round_number += 1
        
        # Check for palifico (only with 3+ active players)
        self.is_palifico_round = False
        self.palifico_player = None
        
        if self._count_active_players() > 2:
            for player in range(self.num_players):
                if (self.player_active[player] and
                    self.player_num_dice[player] == 1 and 
                    player not in self.players_had_palifico):
                    self.is_palifico_round = True
                    self.palifico_player = player
                    self.players_had_palifico.add(player)
                    break
        
        self._roll_dice()
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """Get a mask of legal actions."""
        mask = np.zeros(self.action_space_n, dtype=np.bool_)
        
        for action in range(self.action_space_n):
            if self._is_valid_action(action):
                mask[action] = True
        
        return mask
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices."""
        return list(np.where(self.get_legal_actions_mask())[0])
    
    def render(self) -> str:
        """Render the current game state as a string."""
        lines = [
            f"=== Round {self.round_number} ===",
        ]
        
        for p in range(self.num_players):
            status = "OUT" if not self.player_active[p] else ""
            lines.append(f"Player {p}: {self.player_dice[p]} ({self.player_num_dice[p]} dice) {status}")
        
        lines.append(f"Current player: {self.current_player}")
        
        if self.current_bid:
            prob = self._calculate_bid_probability(self.current_player, self.current_bid)
            prob_exact = self._calculate_calza_probability(self.current_player, self.current_bid)
            lines.append(f"Current bid: {self.current_bid.quantity}x{self.current_bid.face_value} by P{self.current_bid.bidder} (P>={prob:.1%}, P=={prob_exact:.1%})")
        else:
            lines.append("Current bid: None (first bid)")
        
        if self.is_palifico_round:
            lines.append(f"PALIFICO ROUND (triggered by player {self.palifico_player})")
        
        if self.game_over:
            lines.append(f"GAME OVER - Winner: Player {self.winner}")
        
        return "\n".join(lines)


class SelfPlayEnv:
    """
    Wrapper for self-play training with alternating perspectives.
    
    In 2-player mode, this wraps a PerudoEnv and provides a simple
    interface where the agent always sees the game from the current
    player's perspective.
    """
    
    def __init__(self, num_players: int = 2, **env_kwargs):
        """Initialize the self-play environment."""
        self.env = PerudoEnv(num_players=num_players, **env_kwargs)
        self.obs_size = self.env.obs_size
        self.action_space_n = self.env.action_space_n
        self.num_players = num_players
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset(seed)
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        acting_player = self.env.current_player
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["acting_player"] = acting_player
        return obs, reward, terminated, truncated, info
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """Get legal actions mask."""
        return self.env.get_legal_actions_mask()
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal actions."""
        return self.env.get_legal_actions()
    
    def render(self) -> str:
        """Render the environment."""
        return self.env.render()
    
    @property
    def current_player(self) -> int:
        """Get the current player."""
        return self.env.current_player
    
    @property
    def game_over(self) -> bool:
        """Check if game is over."""
        return self.env.game_over
    
    @property 
    def is_palifico_round(self) -> bool:
        """Check if this is a palifico round."""
        return self.env.is_palifico_round
    
    @property
    def joker_mode(self) -> bool:
        """Check if joker mode is on."""
        return self.env.joker_mode
