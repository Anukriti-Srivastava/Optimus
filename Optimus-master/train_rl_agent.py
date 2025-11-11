import os
import argparse
from rl_optimizer.optimizer_env import MLGOEnvironment
from rl_optimizer.agent import MLGOAgent
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLGO agent for LLVM optimization')
    parser.add_argument('--input-ir', type=str, default='test4.ll', help='Path to input LLVM IR file')
    parser.add_argument('--output-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of neural network')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward-weights', type=str, default='runtime=1.0,size=0.5,memory=0.3',
                       help='Comma-separated list of reward weights')
    return parser.parse_args()

def parse_reward_weights(weights_str):
    weights = {}
    for item in weights_str.split(','):
        key, value = item.split('=')
        weights[key.strip()] = float(value.strip())
    return weights

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse reward weights
    reward_weights = parse_reward_weights(args.reward_weights)
    print(f"Using reward weights: {reward_weights}")
    
    # Define enhanced optimization passes
    passes = [
        "mem2reg",         # Promote memory to register
        "sroa",           # Scalar replacement of aggregates
        "early-cse",      # Early common subexpression elimination
        "simplifycfg",    # Simplify the CFG
        "instcombine",    # Combine redundant instructions
        "reassociate",    # Reassociate expressions
        "gvn",            # Global value numbering
        "sccp",           # Sparse conditional constant propagation
        "licm",           # Loop invariant code motion
        "loop-unroll",    # Unroll loops
        "loop-vectorize", # Loop vectorization
        "slp-vectorize",  # SLP vectorization
        "adce",           # Aggressive dead code elimination
        "inline",         # Function inlining
        "dse",            # Dead store elimination
        "constprop",      # Constant propagation
    ]
    
    # Create environment
    env = MLGOEnvironment(args.input_ir, passes, reward_weights=reward_weights)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MLGOAgent(state_dim, action_dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma)
    
    # Training loop
    episode_rewards = []
    best_reward = -float('inf')
    best_sequence = []
    
    print(f"Starting training for {args.episodes} episodes...")
    
    for episode in range(1, args.episodes + 1):
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Run episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store reward
            agent.rewards.append(reward)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # If done, update policy
            if done or truncated:
                agent.update_policy()
        
        # Track episode rewards
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{args.episodes} | Avg Reward: {avg_reward:.4f}")
            
            # Track best model but don't save due to disk space issues
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_sequence = env.get_optimal_pass_sequence()
                print(f"New best model found with reward: {best_reward:.4f}")
                print(f"Best pass sequence: {best_sequence}")
        
        # Skip checkpoint saving due to disk space issues
        if episode % 100 == 0:
            print(f"Checkpoint at episode {episode} (not saved due to disk space constraints)")
    
    # Skip final model saving due to disk space issues
    print("Training completed. Final model not saved due to disk space constraints.")
    
    # Print best pass sequence instead of saving to file due to disk space issues
    print("\nFinal best pass sequence:")
    for i, pass_name in enumerate(best_sequence):
        print(f"{i+1}. {pass_name}")
    
    # Skip plotting learning curve due to disk space issues
    print("\nSkipping learning curve plot due to disk space constraints.")
    print(f"Final statistics:")
    print(f"- Total episodes trained: {len(episode_rewards)}")
    print(f"- Average reward over all episodes: {np.mean(episode_rewards):.4f}")
    print(f"- Best reward achieved: {best_reward:.4f}")
    print("Training completed successfully.")
    print("Note: Models were not saved due to disk space constraints.")

if __name__ == "__main__":
    main()
