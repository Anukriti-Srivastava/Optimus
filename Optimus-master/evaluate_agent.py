import os
import argparse
import json
import numpy as np
import torch
from rl_optimizer.optimizer_env import MLGOEnvironment
from rl_optimizer.agent import MLGOAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MLGO agent for LLVM optimization')
    parser.add_argument('--input-ir', type=str, default='test4.ll', help='Path to input LLVM IR file')
    parser.add_argument('--output-ir', type=str, default='optimized.ll', help='Path to save optimized IR')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt', help='Path to trained model')
    parser.add_argument('--report-path', type=str, default='evaluation_report.json', help='Path to save evaluation report')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Define optimization passes (must match those used in training)
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
    env = MLGOEnvironment(args.input_ir, passes)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MLGOAgent(state_dim, action_dim)
    
    # Load trained model
    try:
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}. Using random policy.")
    
    # Reset environment
    state, _ = env.reset()
    done = False
    total_reward = 0
    applied_passes = []
    metrics_history = []
    
    # Store initial metrics
    metrics_history.append(env.current_metrics.copy())
    
    # Run agent until the environment says "done"
    step = 0
    max_steps = 50  # Prevent infinite loops
    
    print("Starting evaluation...")
    
    while not done and step < max_steps:
        # Select action
        action = agent.select_action(state)
        
        # Take action
        next_state, reward, done, truncated, info = env.step(action)
        
        # Update tracking
        state = next_state
        total_reward += reward
        applied_passes.append(info.get("applied_pass", "unknown"))
        metrics_history.append(info.get("metrics", {}).copy())
        
        # Print step info
        print(f"Step {step+1}: Applied {info.get('applied_pass', 'unknown')} | Reward: {reward:.4f}")
        
        step += 1
        if truncated:
            break
    
    # Save optimized IR
    env.save_ir(args.output_ir)
    
    # Calculate improvements
    initial_metrics = metrics_history[0]
    final_metrics = metrics_history[-1]
    
    improvements = {}
    for key in initial_metrics:
        if key in final_metrics:
            abs_change = initial_metrics[key] - final_metrics[key]
            rel_change = abs_change / max(1, abs(initial_metrics[key])) * 100
            improvements[key] = {
                "initial": initial_metrics[key],
                "final": final_metrics[key],
                "absolute_change": abs_change,
                "relative_change_percent": rel_change
            }
    
    # Create evaluation report
    report = {
        "input_ir": args.input_ir,
        "output_ir": args.output_ir,
        "model_path": args.model_path,
        "total_reward": total_reward,
        "applied_passes": applied_passes,
        "improvements": improvements
    }
    
    # Save report
    with open(args.report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n✅ Evaluation completed!")
    print(f"👉 Applied Passes: {applied_passes}")
    print(f"🎯 Total Reward: {total_reward:.4f}")
    print("\nPerformance Improvements:")
    
    for metric, data in improvements.items():
        print(f"  {metric}: {data['relative_change_percent']:.2f}% improvement")
    
    print(f"\nOptimized IR saved to: {args.output_ir}")
    print(f"Evaluation report saved to: {args.report_path}")

if __name__ == "__main__":
    main()
