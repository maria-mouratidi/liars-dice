"""
Vectorized Environment for parallel simulation.

This module provides a vectorized wrapper for running multiple
Perudo environments in parallel, enabling efficient data collection
for PPO training.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional

# Support both relative and direct imports
try:
    from .perudo_env import SelfPlayEnv
except ImportError:
    from perudo_env import SelfPlayEnv


class VecEnv:
    """
    Vectorized environment that runs multiple environments in parallel.
    
    This enables efficient batch data collection for PPO training by
    running multiple independent game instances simultaneously.
    """
    
    def __init__(
        self,
        env_fns: List[Callable[[], SelfPlayEnv]],
    ):
        """
        Initialize the vectorized environment.
        
        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get observation and action space info from first env
        self.obs_size = self.envs[0].obs_size
        self.action_space_n = self.envs[0].action_space_n
        
        # Track which environments are waiting for actions
        self._dones = np.zeros(self.num_envs, dtype=np.bool_)
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset all environments.
        
        Args:
            seed: Base random seed (each env gets seed + env_idx)
            
        Returns:
            Stacked observations of shape (num_envs, obs_size)
        """
        observations = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            observations.append(obs)
        
        self._dones = np.zeros(self.num_envs, dtype=np.bool_)
        return np.stack(observations)
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Take a step in all environments.
        
        Args:
            actions: Array of actions, one per environment
            
        Returns:
            observations: Stacked observations (num_envs, obs_size)
            rewards: Rewards for each environment (num_envs,)
            terminateds: Whether each env terminated (num_envs,)
            truncateds: Whether each env was truncated (num_envs,)
            infos: List of info dicts from each environment
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._dones[i]:
                # Environment already done, reset it
                obs, info = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(int(action))
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
            # Auto-reset terminated environments
            if terminated or truncated:
                self._dones[i] = True
                # Get fresh observation from reset
                reset_obs, _ = env.reset()
                observations[-1] = reset_obs
        
        # Reset done flags for next iteration
        self._dones = np.array(terminateds) | np.array(truncateds)
        
        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )
    
    def get_legal_actions_masks(self) -> np.ndarray:
        """
        Get legal action masks for all environments.
        
        Returns:
            Stacked masks of shape (num_envs, action_space_n)
        """
        masks = []
        for env in self.envs:
            masks.append(env.get_legal_actions_mask())
        return np.stack(masks)
    
    def close(self):
        """Close all environments."""
        pass  # No cleanup needed for our simple envs


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.
    
    Stores observations, actions, rewards, values, log probabilities,
    and advantage estimates collected during environment rollouts.
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_size: int,
        action_space_n: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size: Number of steps per rollout
            num_envs: Number of parallel environments
            obs_size: Size of observation vector
            action_space_n: Number of possible actions
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size, num_envs, obs_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.bool_)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, num_envs, action_space_n), dtype=np.bool_)
        
        # Computed during finalization
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        self.ptr = 0
        self.full = False
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        action_mask: np.ndarray,
    ):
        """Add a step of experience to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.action_masks[self.ptr] = action_mask
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute returns and GAE advantages.
        
        Args:
            last_values: Value estimates for the final observation
            last_dones: Whether the final state is terminal
        """
        # GAE computation
        last_gae = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].astype(np.float32)
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """
        Get mini-batches for PPO training.
        
        Args:
            batch_size: Size of each mini-batch
            
        Returns:
            List of dictionaries containing batch data
        """
        # Flatten the buffer
        total_size = self.buffer_size * self.num_envs
        
        flat_obs = self.observations.reshape(total_size, self.obs_size)
        flat_actions = self.actions.reshape(total_size)
        flat_log_probs = self.log_probs.reshape(total_size)
        flat_advantages = self.advantages.reshape(total_size)
        flat_returns = self.returns.reshape(total_size)
        flat_masks = self.action_masks.reshape(total_size, self.action_space_n)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # Generate random indices
        indices = np.random.permutation(total_size)
        
        batches = []
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batches.append({
                "observations": flat_obs[batch_indices],
                "actions": flat_actions[batch_indices],
                "old_log_probs": flat_log_probs[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
                "action_masks": flat_masks[batch_indices],
            })
        
        return batches
    
    def reset(self):
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.full = False
