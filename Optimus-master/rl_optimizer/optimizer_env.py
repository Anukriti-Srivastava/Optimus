import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import tempfile
import os
import re
import sys
import platform
from typing import List, Dict, Tuple, Optional, Any

# Add parent directory to path to import llvm_wrapper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from llvm_wrapper.llvm_tools import LLVMTools
    LLVM_WRAPPER_AVAILABLE = True
except ImportError:
    LLVM_WRAPPER_AVAILABLE = False
    print("Warning: LLVM wrapper not available. Using fallback methods.")

class MLGOEnvironment(gym.Env):
    """
    Enhanced environment for Machine Learning Guided Optimization (MLGO)
    This environment provides more detailed code features and supports multiple optimization objectives
    """
    def __init__(self, input_ir: str, passes: List[str], feature_extractors=None, reward_weights=None):
        super().__init__()

        self.input_ir = input_ir
        self.passes = passes
        self.current_ir = None
        self.pass_history = []
        
        # Define action space: each pass is one discrete action
        self.action_space = spaces.Discrete(len(passes))
        
        # Define observation space with enhanced features
        # Features include: pass history, code size metrics, and extracted IR features
        self.feature_dim = len(passes) + 10  # Pass history + additional features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)
        
        # Feature extractors for code analysis
        self.feature_extractors = feature_extractors or self._default_feature_extractors()
        
        # Reward weights for multi-objective optimization
        self.reward_weights = reward_weights or {'runtime': 1.0, 'size': 0.5, 'memory': 0.3}
        
        # State tracking
        self.current_state = np.zeros(self.feature_dim, dtype=np.float32)
        self.baseline_metrics = None
        self.current_metrics = None

    def _default_feature_extractors(self):
        return {
            'instruction_count': lambda ir: len(re.findall(r'\s+%\d+\s+=', ir)),
            'basic_block_count': lambda ir: len(re.findall(r'\blabel\b', ir)),
            'function_count': lambda ir: len(re.findall(r'define\s+.*@', ir)),
            'memory_ops': lambda ir: len(re.findall(r'\balloca\b|\bload\b|\bstore\b', ir)),
            'branch_ops': lambda ir: len(re.findall(r'\bbr\b|\bswitch\b', ir)),
            'call_ops': lambda ir: len(re.findall(r'\bcall\b', ir)),
            'loop_markers': lambda ir: len(re.findall(r'\bloop\b', ir)),
            'code_size': lambda ir: len(ir.splitlines()),
            'phi_nodes': lambda ir: len(re.findall(r'\bphi\b', ir)),
            'vector_ops': lambda ir: len(re.findall(r'\bvector\b', ir))
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load the initial IR code
        with open(self.input_ir, 'r') as f:
            self.current_ir = f.read()

        # Reset state tracking
        self.pass_history = []
        self.current_state = self._extract_state_features(self.current_ir)
        
        # Compute baseline metrics for relative improvement calculation
        self.baseline_metrics = self._compute_metrics(self.current_ir)
        self.current_metrics = self.baseline_metrics.copy()
        
        info = {}
        return self.current_state, info

    def step(self, action):
        pass_name = self.passes[action]
        
        # Skip if pass already applied recently
        if pass_name in self.pass_history[-3:]:
            reward = -0.1  # Small penalty for redundant pass selection
            info = {"applied_pass": pass_name, "skipped": True}
            return self.current_state, reward, False, False, info

        # Apply optimization pass (simplified for this example)
        # In a real implementation, this would use LLVM tools to apply the pass
        self.pass_history.append(pass_name)
        
        # Compute metrics for the new IR (simplified)
        new_metrics = self._compute_metrics(self.current_ir)
        
        # Calculate reward
        reward = 0.1  # Simplified reward
        
        # Update current state
        self.current_metrics = new_metrics
        self.current_state = self._extract_state_features(self.current_ir)

        # Check termination conditions
        done = len(self.pass_history) >= 20  # Limit sequence length
        truncated = False
        
        info = {"applied_pass": pass_name}
        return self.current_state, reward, done, truncated, info

    def _extract_state_features(self, ir_code):
        # Initialize state with zeros
        state = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Set some basic features (simplified)
        state[0] = len(ir_code) / 1000.0  # Code size normalized
        
        return state

    def _compute_metrics(self, ir_code):
        # Simplified metrics computation
        metrics = {
            'size': len(ir_code),
            'runtime': self._measure_runtime(ir_code),
            'memory': 100.0  # Placeholder
        }
        return metrics

    def _measure_runtime(self, ir_code):
        # Simplified runtime measurement
        if LLVM_WRAPPER_AVAILABLE:
            try:
                runtime, error = LLVMTools.measure_runtime(ir_code)
                if error:
                    print(f"Warning: Error during runtime measurement: {error}")
                    return self._estimate_performance_statically(ir_code)
                return runtime
            except Exception as e:
                print(f"Error measuring runtime: {e}")
                return self._estimate_performance_statically(ir_code)
        else:
            return self._estimate_performance_statically(ir_code)

    def _estimate_performance_statically(self, ir_code):
        # Simplified static analysis
        instruction_count = len(re.findall(r'\s+%\d+\s+=', ir_code))
        memory_ops = len(re.findall(r'\balloca\b|\bload\b|\bstore\b', ir_code))
        branch_ops = len(re.findall(r'\bbr\b|\bswitch\b', ir_code))
        call_ops = len(re.findall(r'\bcall\b', ir_code))
        loop_markers = len(re.findall(r'\bloop\b', ir_code))
        
        # Simple performance model
        estimated_runtime = (
            0.01 * instruction_count +
            0.05 * memory_ops +
            0.02 * branch_ops +
            0.1 * call_ops +
            0.5 * loop_markers
        )
        
        # Add some randomness
        noise = np.random.normal(0, 0.05) * estimated_runtime
        estimated_runtime += noise
        
        return max(0.001, estimated_runtime)

    def save_ir(self, filename):
        with open(filename, 'w') as f:
            f.write(self.current_ir)

    def get_optimal_pass_sequence(self):
        return self.pass_history.copy()
