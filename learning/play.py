"""
Interactive Play Against Trained Agent.

This script allows you to play Perudo against a trained PPO agent.

Usage:
    python play.py --model checkpoints/ppo_perudo_*/final_model.pt
"""

import argparse
import numpy as np
from pathlib import Path

# Support both relative and direct imports
try:
    from .perudo_env import PerudoEnv, Bid
    from .ppo_agent import PPOAgent
except ImportError:
    from perudo_env import PerudoEnv, Bid
    from ppo_agent import PPOAgent


def print_game_state(env: PerudoEnv, show_opponent_dice: bool = False):
    """Print the current game state."""
    print("\n" + "=" * 50)
    print(f"Round {env.round_number}")
    print("=" * 50)
    
    # Your dice
    your_dice = env.player_dice[0]
    print(f"\nYour dice ({len(your_dice)}): {sorted(your_dice.tolist())}")
    
    # Opponent info
    if show_opponent_dice:
        opp_dice = env.player_dice[1]
        print(f"Opponent dice ({len(opp_dice)}): {sorted(opp_dice.tolist())}")
    else:
        print(f"Opponent has {env.player_num_dice[1]} dice")
    
    # Total dice
    total = env.player_num_dice[0] + env.player_num_dice[1]
    print(f"Total dice in play: {total}")
    
    # Current bid
    if env.current_bid:
        face_names = {1: "aces", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
        bid_str = f"{env.current_bid.quantity} {face_names[env.current_bid.face_value]}"
        print(f"\nCurrent bid: {bid_str}")
    else:
        print("\nNo current bid (first bid of round)")
    
    # Palifico
    if env.is_palifico_round:
        print("** PALIFICO ROUND - Aces are NOT wild, only quantity can change **")


def get_human_action(env: PerudoEnv) -> int:
    """Get action from human player."""
    legal_actions = env.get_legal_actions()
    
    # Action encoding:
    # 0 = Challenge (dudo)
    # 1 = Calza (declare bid is exactly right)
    # 2+ = Bids
    
    face_names = {1: "aces", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
    
    while True:
        try:
            print("\n  Commands: 'challenge' (or 'c'), 'calza' (if available)")
            print("  Bids: 'quantity face', e.g., '3 fours'")
            if 1 in legal_actions:
                print("  ** CALZA is available! **")
            user_input = input("\nYour action: ").strip().lower()
            
            if user_input in ["challenge", "dudo", "d", "c"]:
                if 0 in legal_actions:
                    return 0
                else:
                    print("Cannot challenge - no current bid!")
                    continue
            
            if user_input in ["calza", "cal", "z"]:
                if 1 in legal_actions:
                    return 1
                else:
                    print("Cannot calza - not available (palifico, 2 players, or you're next)!")
                    continue
            
            # Parse bid
            parts = user_input.split()
            if len(parts) == 2:
                quantity = int(parts[0])
                face_str = parts[1]
                
                face_map = {
                    "aces": 1, "ace": 1, "1": 1, "ones": 1,
                    "twos": 2, "two": 2, "2": 2,
                    "threes": 3, "three": 3, "3": 3,
                    "fours": 4, "four": 4, "4": 4,
                    "fives": 5, "five": 5, "5": 5,
                    "sixes": 6, "six": 6, "6": 6,
                }
                
                if face_str in face_map:
                    face_value = face_map[face_str]
                    # Bids now start at action 2
                    action = 2 + (quantity - 1) * 6 + (face_value - 1)
                    
                    if action in legal_actions:
                        return action
                    else:
                        print("Invalid bid - not a legal raise!")
                        continue
            
            print("Invalid input. Enter 'challenge', 'calza', or a bid like '3 fours'")
            
        except (ValueError, IndexError):
            print("Invalid input. Try again.")


def play_game(agent: PPOAgent, show_opponent_dice: bool = False):
    """Play a game against the agent."""
    env = PerudoEnv()
    obs, _ = env.reset()
    
    print("\n" + "=" * 60)
    print("   PERUDO - Play against the AI!")
    print("=" * 60)
    print("\nYou are Player 0 (human)")
    print("Player 1 is the AI opponent")
    print("\nRemember: Aces (1s) are wild and count as any value!")
    print("The goal is to be the last player with dice.")
    
    while not env.game_over:
        print_game_state(env, show_opponent_dice)
        
        if env.current_player == 0:
            # Human's turn
            print("\n>>> YOUR TURN <<<")
            action = get_human_action(env)
        else:
            # AI's turn
            print("\n>>> AI's TURN <<<")
            action_mask = env.get_legal_actions_mask()
            action, _, _ = agent.act(
                obs[np.newaxis, :],
                action_mask[np.newaxis, :],
                deterministic=True,
            )
            action = action[0]
            
            # Show what AI did
            if action == 0:
                print("AI calls: CHALLENGE!")
            elif action == 1:
                print("AI calls: CALZA! (declares bid is exactly right)")
            else:
                # Bids start at action 2
                bid_idx = action - 2
                quantity = (bid_idx // 6) + 1
                face_value = (bid_idx % 6) + 1
                face_names = {1: "aces", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
                print(f"AI bids: {quantity} {face_names[face_value]}")
        
        # Save state BEFORE step (for challenge resolution display)
        # step() will re-roll dice after a challenge, so we need to capture them now
        dice_before_step = [d.copy() for d in env.player_dice]
        bid_before_step = env.current_bid
        was_palifico = env.is_palifico_round  # Capture palifico state too
        
        # Take the action
        obs, reward, terminated, _, info = env.step(action)
        
        # Handle challenge resolution
        if "challenge_result" in info:
            print("\n" + "-" * 40)
            print("CHALLENGE RESOLUTION")
            print("-" * 40)
            
            # Show dice as they were BEFORE the challenge was resolved
            for p in range(env.num_players):
                player_name = "You" if p == 0 else f"Player {p}"
                print(f"{player_name}: {sorted(dice_before_step[p].tolist())}")
            
            # Count actual matches using the dice from before
            # In palifico round, aces are NOT wild
            bid = bid_before_step
            if bid:
                all_dice = np.concatenate([d for d in dice_before_step if len(d) > 0])
                # Count correctly based on palifico state at time of challenge
                actual = int(np.sum(all_dice == bid.face_value))
                if env.joker_mode and not was_palifico and bid.face_value != 1:
                    actual += int(np.sum(all_dice == 1))
                
                face_names = {1: "aces", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
                print(f"Bid was: {bid.quantity} {face_names[bid.face_value]}")
                print(f"Actual count: {actual}")
                if was_palifico:
                    print("(Palifico round - aces were NOT wild)")
            
            if info["challenge_result"] == "success":
                print("Challenge SUCCESSFUL! Bidder loses a die.")
            else:
                print("Challenge FAILED! Challenger loses a die.")
        
        # Handle calza resolution
        if "calza_result" in info:
            print("\n" + "-" * 40)
            print("CALZA RESOLUTION")
            print("-" * 40)
            
            # Show dice as they were BEFORE the calza was resolved
            for p in range(env.num_players):
                player_name = "You" if p == 0 else f"Player {p}"
                print(f"{player_name}: {sorted(dice_before_step[p].tolist())}")
            
            bid = bid_before_step
            if bid:
                all_dice = np.concatenate([d for d in dice_before_step if len(d) > 0])
                actual = int(np.sum(all_dice == bid.face_value))
                if env.joker_mode and not was_palifico and bid.face_value != 1:
                    actual += int(np.sum(all_dice == 1))
                
                face_names = {1: "aces", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
                print(f"Bid was: {bid.quantity} {face_names[bid.face_value]}")
                print(f"Actual count: {actual}")
            
            if info["calza_result"] == "success":
                print("CALZA SUCCESSFUL! Caller gains a die!")
            else:
                print("Calza FAILED! Caller loses a die.")
    
    # Game over
    print("\n" + "=" * 60)
    if env.winner == 0:
        print("   🎉 CONGRATULATIONS! YOU WIN! 🎉")
    else:
        print("   💔 Game Over - AI Wins")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Play Perudo against a trained agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--show-opponent", action="store_true",
                        help="Show opponent's dice (for debugging)")
    args = parser.parse_args()
    
    # Find or use default model
    if args.model:
        model_path = Path(args.model)
    else:
        # Look for any checkpoint
        checkpoints = list(Path("checkpoints").glob("**/final_model.pt"))
        if checkpoints:
            model_path = checkpoints[-1]
            print(f"Using latest checkpoint: {model_path}")
        else:
            print("No model found. Training a quick model...")
            # Create untrained agent for demo
            model_path = None
    
    # Create environment to get sizes
    env = PerudoEnv()
    
    # Create agent
    agent = PPOAgent(
        obs_size=env.obs_size,
        action_size=env.action_space_n,
        hidden_size=256,
        num_layers=3,
    )
    
    if model_path and model_path.exists():
        print(f"Loading model from: {model_path}")
        agent.load(str(model_path))
    else:
        print("Note: Using untrained agent (random behavior)")
    
    # Play games
    while True:
        play_game(agent, show_opponent_dice=args.show_opponent)
        
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again not in ["y", "yes"]:
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
