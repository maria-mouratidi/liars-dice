"""
PPO Self-Play Training for Perudo.

This script implements the complete training loop for training a
PPO agent to play Perudo through self-play.

Usage:
    python train.py [options]

Options:
    --num-envs: Number of parallel environments (default: 16)
    --total-timesteps: Total training timesteps (default: 1_000_000)
    --rollout-steps: Steps per rollout before update (default: 256)
    --batch-size: Mini-batch size for PPO updates (default: 64)
    --num-epochs: PPO epochs per update (default: 4)
    --lr: Learning rate (default: 3e-4)
    --gamma: Discount factor (default: 0.99)
    --gae-lambda: GAE lambda (default: 0.95)
    --clip-epsilon: PPO clip parameter (default: 0.2)
    --hidden-size: Hidden layer size (default: 256)
    --save-freq: Save frequency in updates (default: 100)
    --eval-freq: Evaluation frequency in updates (default: 50)
    --seed: Random seed (default: 42)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from perudo_env import SelfPlayEnv
from vec_env import VecEnv, RolloutBuffer
from ppo_agent import PPOAgent


def make_env(num_players: int = 2, seed: int = 0):
    """Create a Perudo environment factory."""
    def _init():
        env = SelfPlayEnv(
            num_players=num_players,
            starting_dice=5,
            joker_mode=True,
            enable_calza=True,
            reward_win_game=1.0,
            reward_lose_game=-1.0,
            reward_win_round=0.1,
            reward_lose_round=-0.1,
            reward_calza_success=0.2,
            reward_calza_fail=-0.2,
            reward_shaping=True,
            shaping_scale=0.05,
        )
        return env
    return _init


def evaluate_agent(
    agent: PPOAgent,
    num_players: int = 2,
    num_games: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Evaluate the agent by playing games against itself.
    
    Returns statistics about the games played.
    """
    env = SelfPlayEnv(num_players=num_players)
    
    stats = {
        "total_games": num_games,
        "wins_by_player": {p: 0 for p in range(num_players)},
        "avg_rounds": 0,
        "avg_actions_per_game": 0,
        "challenge_rate": 0,
        "calza_rate": 0,
    }
    
    total_rounds = 0
    total_actions = 0
    total_challenges = 0
    total_calzas = 0
    
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        game_actions = 0
        game_challenges = 0
        game_calzas = 0
        
        while not done:
            action_mask = env.get_legal_actions_mask()
            action, _, _ = agent.act(
                obs[np.newaxis, :],
                action_mask[np.newaxis, :],
                deterministic=True,
            )
            action = action[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            game_actions += 1
            if action == 0:  # Challenge
                game_challenges += 1
            elif action == 1:  # Calza
                game_calzas += 1
        
        # Record winner
        if env.env.winner is not None:
            stats["wins_by_player"][env.env.winner] += 1
        
        total_rounds += env.env.round_number
        total_actions += game_actions
        total_challenges += game_challenges
        total_calzas += game_calzas
    
    stats["avg_rounds"] = total_rounds / num_games
    stats["avg_actions_per_game"] = total_actions / num_games
    stats["challenge_rate"] = total_challenges / total_actions if total_actions > 0 else 0
    stats["calza_rate"] = total_calzas / total_actions if total_actions > 0 else 0
    
    # For backwards compatibility
    stats["player0_wins"] = stats["wins_by_player"].get(0, 0)
    stats["player1_wins"] = stats["wins_by_player"].get(1, 0)
    
    if verbose:
        print(f"  Games played: {num_games}")
        for p in range(num_players):
            wins = stats["wins_by_player"][p]
            print(f"  Player {p} wins: {wins} ({100*wins/num_games:.1f}%)")
        print(f"  Avg rounds per game: {stats['avg_rounds']:.1f}")
        print(f"  Avg actions per game: {stats['avg_actions_per_game']:.1f}")
        print(f"  Challenge rate: {100*stats['challenge_rate']:.1f}%")
        print(f"  Calza rate: {100*stats['calza_rate']:.1f}%")
    
    return stats


def train(args):
    """Main training loop."""
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"ppo_perudo_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PPO Self-Play Training for Perudo")
    print("=" * 60)
    print(f"Save directory: {save_dir}")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create vectorized environment
    env_fns = [make_env(num_players=args.num_players, seed=args.seed + i) for i in range(args.num_envs)]
    vec_env = VecEnv(env_fns)
    
    print(f"\nEnvironment created:")
    print(f"  Number of players: {args.num_players}")
    print(f"  Observation size: {vec_env.obs_size}")
    print(f"  Action space size: {vec_env.action_space_n}")
    print(f"  Number of parallel envs: {vec_env.num_envs}")
    
    # Create agent
    agent = PPOAgent(
        obs_size=vec_env.obs_size,
        action_size=vec_env.action_space_n,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
    )
    
    print(f"\nAgent created on device: {agent.device}")
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        num_envs=args.num_envs,
        obs_size=vec_env.obs_size,
        action_space_n=vec_env.action_space_n,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    
    # Training loop
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (args.num_envs * args.rollout_steps)
    
    print(f"\nTraining for {total_timesteps:,} timesteps")
    print(f"  Updates: {num_updates:,}")
    print(f"  Samples per update: {args.num_envs * args.rollout_steps:,}")
    print("=" * 60)
    
    # Initialize
    obs = vec_env.reset(seed=args.seed)
    timesteps_done = 0
    start_time = time.time()
    
    # Track rewards for logging
    episode_rewards = []
    episode_lengths = []
    current_rewards = np.zeros(args.num_envs)
    current_lengths = np.zeros(args.num_envs)
    
    for update in range(1, num_updates + 1):
        update_start = time.time()
        
        # Collect rollout
        buffer.reset()
        
        for step in range(args.rollout_steps):
            # Get action masks for all environments
            action_masks = vec_env.get_legal_actions_masks()
            
            # Select actions
            actions, log_probs, values = agent.act(obs, action_masks)
            
            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            dones = terminateds | truncateds
            
            # Store experience
            buffer.add(
                obs=obs,
                action=actions,
                reward=rewards,
                done=dones,
                value=values,
                log_prob=log_probs,
                action_mask=action_masks,
            )
            
            # Track episode statistics
            current_rewards += rewards
            current_lengths += 1
            
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    current_rewards[i] = 0
                    current_lengths[i] = 0
            
            obs = next_obs
            timesteps_done += args.num_envs
        
        # Compute returns and advantages
        final_values = agent.get_value(obs)
        buffer.compute_returns_and_advantages(final_values, dones)
        
        # PPO Update
        batches = buffer.get_batches(args.batch_size)
        
        all_stats = []
        for batch in batches:
            stats = agent.update(batch, num_epochs=args.num_epochs)
            all_stats.append(stats)
        
        # Average statistics
        avg_stats = {
            key: np.mean([s[key] for s in all_stats])
            for key in all_stats[0].keys()
        }
        
        update_time = time.time() - update_start
        total_time = time.time() - start_time
        fps = timesteps_done / total_time
        
        # Logging
        if update % args.log_freq == 0:
            print(f"\n[Update {update}/{num_updates}]")
            print(f"  Timesteps: {timesteps_done:,} / {total_timesteps:,}")
            print(f"  FPS: {fps:.0f}")
            print(f"  Update time: {update_time:.2f}s")
            
            if episode_rewards:
                recent_rewards = episode_rewards[-100:]
                print(f"  Episodes completed: {len(episode_rewards)}")
                print(f"  Avg reward (last 100): {np.mean(recent_rewards):.3f}")
                print(f"  Avg length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
            
            print(f"  Policy loss: {avg_stats['policy_loss']:.4f}")
            print(f"  Value loss: {avg_stats['value_loss']:.4f}")
            print(f"  Entropy: {avg_stats['entropy']:.4f}")
            print(f"  Approx KL: {avg_stats['approx_kl']:.4f}")
        
        # Evaluation
        if update % args.eval_freq == 0:
            print(f"\n[Evaluation at update {update}]")
            eval_stats = evaluate_agent(agent, num_players=args.num_players, num_games=100, verbose=True)
        
        # Save checkpoint
        if update % args.save_freq == 0:
            checkpoint_path = save_dir / f"checkpoint_{update}.pt"
            agent.save(str(checkpoint_path))
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = save_dir / "final_model.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    evaluate_agent(agent, num_players=args.num_players, num_games=500, verbose=True)
    
    # Clean up
    vec_env.close()
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="PPO Self-Play Training for Perudo")
    
    # Environment settings
    parser.add_argument("--num-players", type=int, default=2,
                        help="Number of players (2-6)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--rollout-steps", type=int, default=256,
                        help="Steps per rollout before update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO epochs per update")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clipping parameter")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    
    # Network architecture
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of hidden layers")
    
    # Logging and saving
    parser.add_argument("--log-freq", type=int, default=10,
                        help="Logging frequency (in updates)")
    parser.add_argument("--eval-freq", type=int, default=50,
                        help="Evaluation frequency (in updates)")
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Save frequency (in updates)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Run training
    train(args)


if __name__ == "__main__":
    main()
