# Perudo PPO Self-Play Training

This module implements a reinforcement learning agent for Perudo (Liar's Dice) using Proximal Policy Optimization (PPO) with self-play training.

## Overview

The implementation trains an agent to play Perudo by playing against copies of itself. As the agent improves, its opponents (past versions of itself) also improve, creating a curriculum that drives continuous learning.

## Module Structure

```
learning/
├── __init__.py          # Package exports
├── perudo_env.py        # Game environment (Gym-compatible)
├── ppo_agent.py         # PPO agent with actor-critic network
├── vec_env.py           # Vectorized environment wrapper
├── train.py             # Main training script
├── play.py              # Interactive play against trained agent
└── README.md            # This documentation
```

## Components

### 1. Environment (`perudo_env.py`)

The `PerudoEnv` class implements a Gym-compatible environment for 2-player Perudo.

#### State Space

The observation is a 39-dimensional vector:

| Indices | Size | Description |
|---------|------|-------------|
| 0-5 | 6 | One-hot encoding of own dice counts (dice showing 1-6) |
| 6 | 1 | Own dice count (normalized by max dice) |
| 7 | 1 | Opponent dice count (normalized by max dice) |
| 8 | 1 | Current bid quantity (normalized) |
| 9-14 | 6 | One-hot encoding of current bid face value |
| 15 | 1 | Flag: is there a current bid? |
| 16 | 1 | Flag: is it our turn? |
| 17 | 1 | Round number (normalized) |
| 18-38 | 21 | Bid history (last 7 bids × 3 features each) |

#### Action Space

The action space has 38 discrete actions:

| Action | Description |
|--------|-------------|
| 0 | Challenge (Dudo) - call the opponent's bluff |
| 1-36 | Bid actions: `(quantity-1) * 6 + (face-1)` |
| 37 | Calza (exact match) - claim bid is exactly correct |

**Bid Encoding:**
- Actions 1-6: Bid 1 of face 1-6
- Actions 7-12: Bid 2 of face 1-6
- Actions 13-18: Bid 3 of face 1-6
- ... and so on up to 6 dice

#### Reward Structure

| Event | Reward |
|-------|--------|
| Win the game | +1.0 |
| Lose the game | -1.0 |
| Win a round | +0.1 |
| Lose a round | -0.1 |
| Invalid action | -0.5 |

#### Key Methods

```python
env = PerudoEnv(max_dice=3)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
valid_mask = env.get_valid_actions()  # Boolean mask of valid actions
```

### 2. PPO Agent (`ppo_agent.py`)

The `PPOAgent` class implements the PPO algorithm with an actor-critic architecture.

#### Network Architecture

```
Input (39) → Linear(128) → ReLU → Linear(128) → ReLU
                                       ↓
                              ┌────────┴────────┐
                              ↓                 ↓
                        Actor Head         Critic Head
                        Linear(38)         Linear(1)
                              ↓                 ↓
                    Action Logits           Value
```

#### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.2 | PPO clip parameter |
| `value_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

#### Key Methods

```python
agent = PPOAgent(obs_dim=39, action_dim=38)

# Get action with exploration
action, log_prob, value = agent.get_action(obs, valid_mask)

# Get deterministic action for evaluation
action = agent.get_deterministic_action(obs, valid_mask)

# Update policy using collected trajectories
metrics = agent.update(obs, actions, old_log_probs, returns, advantages, valid_masks)

# Save/load models
agent.save("checkpoint.pt")
agent.load("checkpoint.pt")
```

### 3. Vectorized Environment (`vec_env.py`)

The `VecEnv` class runs multiple environments in parallel for efficient data collection.

```python
vec_env = VecEnv(num_envs=8, max_dice=3)

# Reset all environments
obs = vec_env.reset()  # Shape: (num_envs, obs_dim)

# Step all environments
obs, rewards, dones, infos = vec_env.step(actions)

# Get valid action masks
masks = vec_env.get_valid_masks()  # Shape: (num_envs, action_dim)
```

### 4. Training Script (`train.py`)

The main training loop with self-play.

#### Self-Play Mechanism

1. The agent plays against a frozen copy of itself (the opponent)
2. Every N updates, the opponent is replaced with the current agent
3. This creates an evolving curriculum as both players improve

#### Training Configuration

```python
config = TrainingConfig(
    num_envs=16,           # Parallel environments
    max_dice=3,            # Starting dice per player
    total_timesteps=100000,# Total training steps
    steps_per_update=256,  # Steps before each PPO update
    num_epochs=4,          # PPO epochs per update
    batch_size=64,         # Minibatch size
    opponent_update_freq=10,# Updates between opponent refresh
    save_freq=50,          # Updates between checkpoints
    log_freq=10,           # Updates between logging
)
```

#### Running Training

```bash
# Basic training
python -m learning.train

# With custom parameters
python -m learning.train --total-timesteps 500000 --num-envs 32
```

#### Output

Training creates a timestamped checkpoint directory:
```
learning/checkpoints/ppo_perudo_YYYYMMDD_HHMMSS/
├── model_update_50.pt
├── model_update_100.pt
├── ...
└── final_model.pt
```

### 5. Interactive Play (`play.py`)

Play against a trained agent interactively.

```bash
python -m learning.play

# Or specify a checkpoint
python -m learning.play --model learning/checkpoints/ppo_perudo_xxx/final_model.pt
```

## Game Rules Reference

### Basic Rules

- Each player starts with a set number of dice (default: 3)
- Players take turns making bids about the total dice on the table
- A bid consists of a quantity and a face value (e.g., "3 fives")
- Each subsequent bid must be higher than the previous
- **Ones (aces) are wild** and count as any face value

### Valid Bid Raises

A bid is higher if:
1. **Same face, higher quantity**: "3 fives" → "4 fives"
2. **Higher face, same or higher quantity**: "3 fives" → "3 sixes"
3. **Lower face, higher quantity**: "3 fives" → "4 threes"

**Special rules for ones:**
- Bidding ones requires at least ⌈previous_quantity / 2⌉
- Switching from ones to another face requires 2× the ones quantity + 1

### Actions

- **Dudo (Challenge)**: Call the previous bid a lie
  - If total matching dice < bid quantity: Challenger wins, bidder loses a die
  - If total matching dice ≥ bid quantity: Bidder wins, challenger loses a die

- **Calza (Exact)**: Claim the bid is exactly correct
  - If total matching dice == bid quantity: Caller wins, opponent loses a die
  - Otherwise: Caller loses a die

### Winning

The last player with dice remaining wins the game.

## Implementation Details

### Action Masking

Invalid actions are masked during both training and inference:
- Cannot challenge if no bid has been made
- Cannot make a bid that isn't higher than the current bid
- Cannot bid more dice than exist on the table

The agent's policy outputs are masked before sampling:
```python
logits[~valid_mask] = -1e8  # Mask invalid actions
probs = softmax(logits)
action = sample(probs)
```

### GAE (Generalized Advantage Estimation)

Advantages are computed using GAE for reduced variance:

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

Default: $\lambda = 0.95$, $\gamma = 0.99$

### PPO Clipped Objective

The policy is updated using the clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

## Extending the Implementation

### Adding More Players

The current implementation supports 2 players. To extend to N players:

1. Modify `PerudoEnv` to track N players' dice
2. Expand the observation space to include all opponents' dice counts
3. Update the turn logic to cycle through N players

### Custom Reward Shaping

Modify the reward structure in `PerudoEnv.step()`:

```python
# Example: Add reward for making good bids
if action_is_valid_bid:
    # Reward based on probability of bid being true
    reward += 0.01 * self._bid_probability(quantity, face)
```

### Curriculum Learning

Implement progressive difficulty:

```python
# Start with fewer dice, gradually increase
if avg_win_rate > 0.6:
    config.max_dice += 1
```

## Troubleshooting

### Agent Makes Invalid Actions

Ensure `get_valid_actions()` is correctly called and the mask is applied:
```python
valid_mask = env.get_valid_actions()
action = agent.get_action(obs, valid_mask)
```

### Training Doesn't Converge

- Increase `entropy_coef` for more exploration
- Reduce learning rate
- Increase `steps_per_update` for more stable gradients
- Check that rewards are properly normalized

### Memory Issues

- Reduce `num_envs`
- Reduce `steps_per_update`
- Use smaller network architecture

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Gymnasium (optional, for Gym compatibility)

## License

See the main project LICENSE file.
