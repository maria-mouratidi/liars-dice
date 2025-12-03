"""
Perudo Reinforcement Learning Package.

This package contains modules for training a PPO agent to play
Perudo (Liar's Dice) through self-play.

Modules:
    perudo_env: Complete Perudo environment implementation
    vec_env: Vectorized environment wrapper for parallel simulation
    ppo_agent: PPO agent with neural network policy/value functions
    train: Main training script
"""

from .perudo_env import PerudoEnv, SelfPlayEnv, Bid, ActionType
from .vec_env import VecEnv, RolloutBuffer
from .ppo_agent import PPOAgent, ActorCritic

__all__ = [
    "PerudoEnv",
    "SelfPlayEnv",
    "Bid",
    "ActionType",
    "VecEnv",
    "RolloutBuffer",
    "PPOAgent",
    "ActorCritic",
]
