import numpy as np
from typing import Dict, List, Tuple, Union, Any

def calculate_runtime_reward(old_runtime: float, new_runtime: float, weight: float = 1.0) -> float:
    """
    Calculate reward based on runtime improvement
    Lower runtime is better, so reward is positive if new_runtime < old_runtime
    
    Args:
        old_runtime: Previous runtime measurement
        new_runtime: New runtime measurement after optimization
        weight: Weight factor for this reward component
        
    Returns:
        Reward value for runtime improvement
    """
    # Avoid division by zero
    old_runtime = max(old_runtime, 0.001)
    
    # Calculate relative improvement
    improvement = (old_runtime - new_runtime) / old_runtime
    
    # Apply weight and return
    return weight * improvement

def calculate_size_reward(old_size: int, new_size: int, weight: float = 0.5) -> float:
    """
    Calculate reward based on code size improvement
    Lower size is better, so reward is positive if new_size < old_size
    
    Args:
        old_size: Previous code size (in bytes or instructions)
        new_size: New code size after optimization
        weight: Weight factor for this reward component
        
    Returns:
        Reward value for size improvement
    """
    # Avoid division by zero
    old_size = max(old_size, 1)
    
    # Calculate relative improvement
    improvement = (old_size - new_size) / old_size
    
    # Apply weight and return
    return weight * improvement

def calculate_memory_reward(old_memory: int, new_memory: int, weight: float = 0.3) -> float:
    """
    Calculate reward based on memory usage improvement
    Lower memory usage is better, so reward is positive if new_memory < old_memory
    
    Args:
        old_memory: Previous memory usage estimate
        new_memory: New memory usage estimate after optimization
        weight: Weight factor for this reward component
        
    Returns:
        Reward value for memory improvement
    """
    # Avoid division by zero
    old_memory = max(old_memory, 1)
    
    # Calculate relative improvement
    improvement = (old_memory - new_memory) / old_memory
    
    # Apply weight and return
    return weight * improvement

def calculate_combined_reward(
    old_metrics: Dict[str, Union[float, int]],
    new_metrics: Dict[str, Union[float, int]],
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate combined reward based on multiple metrics
    
    Args:
        old_metrics: Dictionary of metrics before optimization
        new_metrics: Dictionary of metrics after optimization
        weights: Dictionary of weights for each metric
        
    Returns:
        Combined reward value
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'runtime': 1.0,
            'size': 0.5,
            'memory': 0.3
        }
    
    total_reward = 0.0
    
    # Runtime reward
    if 'runtime' in old_metrics and 'runtime' in new_metrics:
        runtime_reward = calculate_runtime_reward(
            old_metrics['runtime'],
            new_metrics['runtime'],
            weights.get('runtime', 1.0)
        )
        total_reward += runtime_reward
    
    # Size reward
    if 'size' in old_metrics and 'size' in new_metrics:
        size_reward = calculate_size_reward(
            old_metrics['size'],
            new_metrics['size'],
            weights.get('size', 0.5)
        )
        total_reward += size_reward
    
    # Memory reward
    if 'memory' in old_metrics and 'memory' in new_metrics:
        memory_reward = calculate_memory_reward(
            old_metrics['memory'],
            new_metrics['memory'],
            weights.get('memory', 0.3)
        )
        total_reward += memory_reward
    
    # Add penalties for compilation or execution failures
    if new_metrics.get('runtime', 0) >= 1e6:  # Our error code for failures
        total_reward -= 10.0  # Heavy penalty
    
    return total_reward

def calculate_sequence_reward(metrics_history: List[Dict[str, Union[float, int]]]) -> List[float]:
    """
    Calculate rewards for a sequence of optimization steps
    
    Args:
        metrics_history: List of metrics dictionaries for each step
        
    Returns:
        List of reward values for each step
    """
    rewards = []
    
    for i in range(1, len(metrics_history)):
        old_metrics = metrics_history[i-1]
        new_metrics = metrics_history[i]
        reward = calculate_combined_reward(old_metrics, new_metrics)
        rewards.append(reward)
    
    return rewards

def normalize_rewards(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Normalize rewards using discounted returns and standardization
    
    Args:
        rewards: List of raw rewards
        gamma: Discount factor
        
    Returns:
        List of normalized rewards
    """
    # Calculate discounted returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    # Convert to numpy array
    returns = np.array(returns)
    
    # Normalize
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    return returns.tolist()
