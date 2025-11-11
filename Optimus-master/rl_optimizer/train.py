import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom MLGO components
from optimizer_env import MLGOEnvironment
from agent import MLGOAgent
from reward import calculate_combined_reward

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLGO agent for LLVM optimization')
    parser.add_argument('--input-ir', type=str, required=True, help='Path to input LLVM IR file')
    parser.add_argument('--output-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of neural network')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=100, help='Save interval')
    parser.add_argument('--reward-weights', type=str, default='runtime=1.0,size=0.5,memory=0.3',
                       help='Comma-separated list of reward weights')
    return parser.parse_args()

def setup_logging(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=run_dir)
    
    return run_dir, writer

def parse_reward_weights(weights_str):
    weights = {}
    for item in weights_str.split(','):
        key, value = item.split('=')
        weights[key.strip()] = float(value.strip())
    return weights

def train(args):
    # Setup logging
    run_dir, writer = setup_logging(args.output_dir)
    
    # Parse reward weights
    reward_weights = parse_reward_weights(args.reward_weights)
    print(f"Using reward weights: {reward_weights}")
    
    # Define optimization passes
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
        "jump-threading", # Jump threading
        "tailcallelim",   # Tail call elimination
        "constprop",      # Constant propagation
        "sink",           # Code sinking
        "argpromotion",   # Argument promotion
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
        
        # Log to TensorBoard
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        
        # Log metrics if available
        if 'metrics' in info and 'improvement' in info:
            for metric, value in info['metrics'].items():
                writer.add_scalar(f'Metrics/{metric}', value, episode)
            for metric, improvement in info['improvement'].items():
                writer.add_scalar(f'Improvement/{metric}', improvement, episode)
        
        # Print progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            print(f"Episode {episode}/{args.episodes} | Avg Reward: {avg_reward:.4f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(run_dir, 'best_model.pt'))
                print(f"New best model saved with reward: {best_reward:.4f}")
        
        # Save checkpoint
        if episode % args.save_interval == 0:
            agent.save(os.path.join(run_dir, f'checkpoint_{episode}.pt'))
    
    # Save final model
    agent.save(os.path.join(run_dir, 'final_model.pt'))
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(run_dir, 'learning_curve.png'))
    
    print(f"Training completed. Models saved to {run_dir}")
    return run_dir

if __name__ == "__main__":
    args = parse_args()
    train(args)
