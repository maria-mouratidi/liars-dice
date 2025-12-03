"""
Perudo (Liar's Dice) Environment for Reinforcement Learning.

This module implements a complete Perudo environment compatible with
the standard Gymnasium interface, designed for self-play training.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum


class ActionType(IntEnum):
    """Action types in Perudo."""
    CHALLENGE = 0  # Call "dudo" - challenge the current bid
    BID = 1        # Make a new bid


@dataclass
class Bid:
    """Represents a bid in the game."""
    quantity: int
    face_value: int  # 1-6, where 1 is ace (wild)

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


class PerudoEnv:
    """
    Perudo (Liar's Dice) environment for 2-player self-play.
    
    Observation Space:
        - Own dice (one-hot encoded): 6 * max_dice values
        - Current bid quantity (normalized): 1 value
        - Current bid face value (one-hot): 6 values
        - Number of own dice (normalized): 1 value
        - Number of opponent dice (normalized): 1 value
        - Total dice in play (normalized): 1 value
        - Is first bid of round: 1 value
        - Is palifico round: 1 value
        - Current player (binary): 1 value
        
    Action Space:
        - Action 0: Challenge (dudo)
        - Actions 1 to max_bid_actions: Bid (quantity, face_value)
        
    The bid actions are encoded as:
        action = 1 + (quantity - 1) * 6 + (face_value - 1)
    """
    
    # Maximum dice per player
    MAX_DICE = 5
    
    # Maximum quantity that can be bid (2 players * 5 dice)
    MAX_QUANTITY = 10
    
    # Number of face values (1-6)
    NUM_FACES = 6
    
    def __init__(
        self,
        starting_dice: int = 5,
        joker_mode: bool = True,
        reward_win: float = 1.0,
        reward_lose: float = -1.0,
        reward_invalid: float = -0.5,
    ):
        """
        Initialize the Perudo environment.
        
        Args:
            starting_dice: Number of dice each player starts with
            joker_mode: Whether aces are wild (count as any value)
            reward_win: Reward for winning a challenge
            reward_lose: Reward for losing a challenge
            reward_invalid: Penalty for invalid actions
        """
        self.starting_dice = min(starting_dice, self.MAX_DICE)
        self.joker_mode = joker_mode
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_invalid = reward_invalid
        
        # Calculate observation and action space sizes
        self.obs_size = self._calculate_obs_size()
        self.action_space_n = 1 + self.MAX_QUANTITY * self.NUM_FACES  # Challenge + all possible bids
        
        # Game state
        self.player_dice: List[np.ndarray] = [np.array([]), np.array([])]
        self.player_num_dice: List[int] = [0, 0]
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
        # Own dice one-hot (MAX_DICE * 6 for one-hot encoding of each die)
        own_dice_size = self.MAX_DICE * self.NUM_FACES
        # Current bid info
        bid_quantity_size = 1  # Normalized quantity
        bid_face_size = self.NUM_FACES  # One-hot face value
        # Game state info
        own_dice_count = 1
        opponent_dice_count = 1
        total_dice_count = 1
        is_first_bid = 1
        is_palifico = 1
        current_player_indicator = 1
        
        return (own_dice_size + bid_quantity_size + bid_face_size + 
                own_dice_count + opponent_dice_count + total_dice_count +
                is_first_bid + is_palifico + current_player_indicator)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial observation for the starting player
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Reset dice counts
        self.player_num_dice = [self.starting_dice, self.starting_dice]
        
        # Roll dice for both players
        self._roll_dice()
        
        # Reset game state
        self.current_player = np.random.randint(0, 2)  # Random first player
        self.current_bid = None
        self.round_number = 1
        self.is_palifico_round = False
        self.palifico_player = None
        self.players_had_palifico = set()
        self.game_over = False
        self.winner = None
        
        obs = self._get_observation(self.current_player)
        info = {"current_player": self.current_player}
        
        return obs, info
    
    def _roll_dice(self):
        """Roll dice for all players."""
        for player in range(2):
            if self.player_num_dice[player] > 0:
                self.player_dice[player] = np.random.randint(
                    1, 7, size=self.player_num_dice[player]
                )
            else:
                self.player_dice[player] = np.array([])
    
    def _get_observation(self, player: int) -> np.ndarray:
        """
        Get the observation for a specific player.
        
        Args:
            player: Player index (0 or 1)
            
        Returns:
            Observation vector
        """
        obs = []
        
        # Own dice one-hot encoding
        dice_one_hot = np.zeros(self.MAX_DICE * self.NUM_FACES)
        for i, die in enumerate(self.player_dice[player]):
            if i < self.MAX_DICE:
                dice_one_hot[i * self.NUM_FACES + int(die) - 1] = 1.0
        obs.extend(dice_one_hot)
        
        # Current bid info
        if self.current_bid is not None:
            # Normalized quantity (0 to 1 range)
            obs.append(self.current_bid.quantity / self.MAX_QUANTITY)
            # One-hot face value
            face_one_hot = np.zeros(self.NUM_FACES)
            face_one_hot[self.current_bid.face_value - 1] = 1.0
            obs.extend(face_one_hot)
        else:
            obs.append(0.0)  # No current bid
            obs.extend([0.0] * self.NUM_FACES)
        
        # Dice counts (normalized)
        opponent = 1 - player
        total_dice = self.player_num_dice[0] + self.player_num_dice[1]
        obs.append(self.player_num_dice[player] / self.MAX_DICE)
        obs.append(self.player_num_dice[opponent] / self.MAX_DICE)
        obs.append(total_dice / (2 * self.MAX_DICE))
        
        # Is first bid of round
        obs.append(1.0 if self.current_bid is None else 0.0)
        
        # Is palifico round
        obs.append(1.0 if self.is_palifico_round else 0.0)
        
        # Current player indicator
        obs.append(1.0 if player == self.current_player else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _decode_action(self, action: int) -> Tuple[ActionType, Optional[Bid]]:
        """
        Decode an action index into action type and bid.
        
        Args:
            action: Action index
            
        Returns:
            (action_type, bid) where bid is None for challenge
        """
        if action == 0:
            return ActionType.CHALLENGE, None
        
        # Decode bid: action = 1 + (quantity - 1) * 6 + (face_value - 1)
        bid_idx = action - 1
        quantity = (bid_idx // self.NUM_FACES) + 1
        face_value = (bid_idx % self.NUM_FACES) + 1
        
        return ActionType.BID, Bid(quantity, face_value)
    
    def _encode_bid(self, bid: Bid) -> int:
        """Encode a bid into an action index."""
        return 1 + (bid.quantity - 1) * self.NUM_FACES + (bid.face_value - 1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.
        
        Args:
            action: Action index
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether the game is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.game_over:
            # Game already over, return terminal state
            obs = self._get_observation(self.current_player)
            return obs, 0.0, True, False, {"winner": self.winner}
        
        action_type, bid = self._decode_action(action)
        info = {"action_type": action_type.name}
        
        # Validate action
        if not self._is_valid_action(action):
            # Invalid action - penalize and let the player try again
            # In training, we mask invalid actions, so this shouldn't happen often
            obs = self._get_observation(self.current_player)
            return obs, self.reward_invalid, False, False, {"error": "invalid_action"}
        
        if action_type == ActionType.CHALLENGE:
            reward, terminated = self._handle_challenge()
            info["challenge_result"] = "success" if reward > 0 else "failure"
        else:
            reward, terminated = self._handle_bid(bid)
            info["bid"] = f"{bid.quantity}x{bid.face_value}"
        
        obs = self._get_observation(self.current_player)
        info["current_player"] = self.current_player
        
        return obs, reward, terminated, False, info
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid in the current state."""
        action_type, bid = self._decode_action(action)
        
        if action_type == ActionType.CHALLENGE:
            # Can only challenge if there's a current bid
            return self.current_bid is not None
        
        # Validate bid
        return self._is_valid_bid(bid)
    
    def _is_valid_bid(self, bid: Bid) -> bool:
        """Check if a bid is valid in the current state."""
        # Basic validation
        if bid.quantity <= 0 or bid.quantity > self.MAX_QUANTITY:
            return False
        if bid.face_value < 1 or bid.face_value > 6:
            return False
        
        total_dice = self.player_num_dice[0] + self.player_num_dice[1]
        if bid.quantity > total_dice:
            return False
        
        # First bid of round
        if self.current_bid is None:
            # Cannot bid aces on first bid (unless palifico player)
            if bid.is_ace_bid():
                if not self.is_palifico_round:
                    return False
                if self.current_player != self.palifico_player:
                    return False
            return True
        
        # Palifico round rules
        if self.is_palifico_round:
            # During palifico, must keep same face value (only raise quantity)
            # Exception: players who already had palifico can change
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
            # Switching to aces: quantity must be at least half (rounded up)
            min_quantity = (self.current_bid.quantity + 1) // 2
            return bid.quantity >= min_quantity
        elif not bid.is_ace_bid() and self.current_bid.is_ace_bid():
            # Switching from aces: quantity must be at least 2 * ace_qty + 1
            min_quantity = self.current_bid.quantity * 2 + 1
            return bid.quantity >= min_quantity
        else:
            # Normal raise
            return bid.is_higher_than(self.current_bid)
    
    def _handle_challenge(self) -> Tuple[float, bool]:
        """
        Handle a challenge action.
        
        Returns:
            (reward, terminated)
        """
        challenger = self.current_player
        bidder = 1 - challenger
        
        # Count all dice
        all_dice = np.concatenate([self.player_dice[0], self.player_dice[1]])
        actual_count = self._count_matching_dice(all_dice, self.current_bid.face_value)
        
        # Determine winner
        if actual_count >= self.current_bid.quantity:
            # Bid was valid, challenger loses
            loser = challenger
            reward = self.reward_lose
        else:
            # Bid was invalid, bidder loses
            loser = bidder
            reward = self.reward_win
        
        # Loser loses a die
        self.player_num_dice[loser] -= 1
        
        # Check if game is over
        if self.player_num_dice[loser] <= 0:
            self.game_over = True
            self.winner = 1 - loser
            return reward, True
        
        # Start new round
        self._start_new_round(loser)
        
        # The reward is from the perspective of the current player who challenged
        return reward, False
    
    def _count_matching_dice(self, dice: np.ndarray, face_value: int) -> int:
        """Count dice matching the face value (including wilds)."""
        count = np.sum(dice == face_value)
        
        # In joker mode, aces are wild (unless during palifico or counting aces)
        if self.joker_mode and not self.is_palifico_round and face_value != 1:
            count += np.sum(dice == 1)
        
        return int(count)
    
    def _handle_bid(self, bid: Bid) -> Tuple[float, bool]:
        """
        Handle a bid action.
        
        Returns:
            (reward, terminated)
        """
        self.current_bid = bid
        
        # Advance to next player
        self.current_player = 1 - self.current_player
        
        # No immediate reward for bidding
        return 0.0, False
    
    def _start_new_round(self, starting_player: int):
        """Start a new round after a challenge."""
        self.current_bid = None
        self.current_player = starting_player
        self.round_number += 1
        
        # Check for palifico
        self.is_palifico_round = False
        self.palifico_player = None
        
        for player in range(2):
            if (self.player_num_dice[player] == 1 and 
                player not in self.players_had_palifico):
                self.is_palifico_round = True
                self.palifico_player = player
                self.players_had_palifico.add(player)
                break
        
        # Roll new dice
        self._roll_dice()
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """
        Get a mask of legal actions.
        
        Returns:
            Boolean mask where True indicates a legal action
        """
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
            f"Player 0: {self.player_dice[0]} ({self.player_num_dice[0]} dice)",
            f"Player 1: {self.player_dice[1]} ({self.player_num_dice[1]} dice)",
            f"Current player: {self.current_player}",
        ]
        
        if self.current_bid:
            lines.append(f"Current bid: {self.current_bid.quantity}x{self.current_bid.face_value}")
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
    
    This environment wraps PerudoEnv and provides a consistent interface
    where each step returns the observation from the current player's
    perspective, making it suitable for training a single policy that
    plays as both players.
    """
    
    def __init__(self, **env_kwargs):
        """Initialize the self-play environment."""
        self.env = PerudoEnv(**env_kwargs)
        self.obs_size = self.env.obs_size
        self.action_space_n = self.env.action_space_n
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset(seed)
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Get the current player before the step
        acting_player = self.env.current_player
        
        # Take the action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # The reward is from the acting player's perspective
        # For self-play, we want the new current player's observation
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
