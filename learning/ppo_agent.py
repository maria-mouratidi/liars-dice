"""
PPO Agent implementation for Perudo.

This module implements a PPO (Proximal Policy Optimization) agent
with neural network policy and value functions, designed for
self-play training in the Perudo environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.
    
    Uses shared feature extraction layers followed by separate
    policy (actor) and value (critic) heads.
    """
    
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        """
        Initialize the Actor-Critic network.
        
        Args:
            obs_size: Size of observation vector
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers in shared network
        """
        super().__init__()
        
        self.obs_size = obs_size
        self.action_size = action_size
        
        # Shared feature extraction network
        layers = []
        in_size = obs_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.shared = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller initialization for policy output
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
        # Even smaller for value output
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_size)
            action_mask: Optional mask of valid actions (batch_size, action_size)
            
        Returns:
            policy_logits: Logits for each action (batch_size, action_size)
            value: State value estimate (batch_size, 1)
        """
        features = self.shared(obs)
        
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        # Apply action mask (set invalid actions to very negative logits)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, float('-inf'))
        
        return policy_logits, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor
            action_mask: Optional mask of valid actions
            action: Optional action (for evaluation during training)
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: State value estimate
        """
        policy_logits, value = self.forward(obs, action_mask)
        
        # Create distribution over actions
        probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get just the value estimate."""
        features = self.shared(obs)
        value = self.value_head(features)
        return value.squeeze(-1)


class PPOAgent:
    """
    PPO Agent for Perudo self-play training.
    
    Implements the PPO-Clip algorithm with advantages computed using
    Generalized Advantage Estimation (GAE).
    """
    
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the PPO agent.
        
        Args:
            obs_size: Size of observation vector
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (cuda/cpu)
        """
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.network = ActorCritic(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Training statistics
        self.train_stats = {}
    
    def act(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for a batch of observations.
        
        Args:
            obs: Observations of shape (batch_size, obs_size)
            action_mask: Optional mask of valid actions
            deterministic: Whether to select actions deterministically
            
        Returns:
            actions: Selected actions (batch_size,)
            log_probs: Log probabilities (batch_size,)
            values: Value estimates (batch_size,)
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        if action_mask is not None:
            mask_tensor = torch.BoolTensor(action_mask).to(self.device)
        else:
            mask_tensor = None
        
        with torch.no_grad():
            if deterministic:
                policy_logits, values = self.network(obs_tensor, mask_tensor)
                actions = policy_logits.argmax(dim=-1)
                probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(actions)
            else:
                actions, log_probs, _, values = self.network.get_action_and_value(
                    obs_tensor, mask_tensor
                )
        
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )
    
    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Get value estimates for observations."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            values = self.network.get_value(obs_tensor)
        return values.cpu().numpy()
    
    def update(self, batch: Dict[str, np.ndarray], num_epochs: int = 4) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Args:
            batch: Dictionary containing:
                - observations: (batch_size, obs_size)
                - actions: (batch_size,)
                - old_log_probs: (batch_size,)
                - advantages: (batch_size,)
                - returns: (batch_size,)
                - action_masks: (batch_size, action_size)
            num_epochs: Number of PPO epochs
            
        Returns:
            Dictionary of training statistics
        """
        # Convert to tensors
        obs = torch.FloatTensor(batch["observations"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["old_log_probs"]).to(self.device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self.device)
        returns = torch.FloatTensor(batch["returns"]).to(self.device)
        action_masks = torch.BoolTensor(batch["action_masks"]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Get current policy outputs
            _, log_probs, entropy, values = self.network.get_action_and_value(
                obs, action_masks, actions
            )
            
            # Policy loss (PPO-Clip)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track statistics
            with torch.no_grad():
                approx_kl = (old_log_probs - log_probs).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                num_updates += 1
        
        self.train_stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
        }
        
        return self.train_stats
    
    def save(self, path: str):
        """Save the agent to a file."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "hidden_size": self.network.shared[0].out_features,
            "num_layers": len([m for m in self.network.shared if isinstance(m, nn.Linear)]),
        }, path)
    
    def load(self, path: str):
        """Load the agent from a file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Try to get architecture from checkpoint, or infer from state_dict
        state_dict = checkpoint["network_state_dict"]
        
        if "hidden_size" in checkpoint:
            hidden_size = checkpoint["hidden_size"]
            num_layers = checkpoint["num_layers"]
        else:
            # Infer from state_dict
            hidden_size = state_dict["shared.0.weight"].shape[0]
            # Count Linear layers in shared network (every 3rd key starting from 0: Linear, LayerNorm, ReLU)
            num_layers = sum(1 for k in state_dict.keys() if k.startswith("shared.") and k.endswith(".weight") and "LayerNorm" not in k.replace(k.split(".")[1], "")) // 2
            # Actually just count the linear layers by checking weights shape
            num_layers = len([k for k in state_dict.keys() if k.startswith("shared.") and k.endswith(".weight") and len(state_dict[k].shape) == 2])
        
        # Check if we need to rebuild the network
        current_hidden = self.network.shared[0].out_features
        current_layers = len([m for m in self.network.shared if isinstance(m, nn.Linear)])
        
        if hidden_size != current_hidden or num_layers != current_layers:
            self.network = ActorCritic(
                obs_size=self.obs_size,
                action_size=self.action_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4, eps=1e-5)
        
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
