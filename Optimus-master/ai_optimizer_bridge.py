import os
import sys
import json
import torch
import numpy as np
import subprocess
import tempfile
import re
from typing import List, Dict, Tuple, Optional, Any

# Import our custom MLGO components
from rl_optimizer.agent import MLGOAgent
from rl_optimizer.optimizer_env import MLGOEnvironment

class AIOptimizerBridge:
    """
    Bridge between the LLVM C++ optimizer and our AI model
    This class provides an interface for the C++ optimizer to use our trained AI model
    """
    def __init__(self, model_path: str, passes: List[str] = None):
        """
        Initialize the bridge with a trained model
        
        Args:
            model_path: Path to the trained model
            passes: List of optimization passes (must match those used during training)
        """
        # Default passes if none provided
        if passes is None:
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
        
        self.passes = passes
        self.feature_extractors = self._create_feature_extractors()
        
        # Create agent
        self.state_dim = len(passes) + 10  # Pass history + additional features
        self.action_dim = len(passes)
        self.agent = MLGOAgent(self.state_dim, self.action_dim)
        
        # Load model if it exists
        try:
            self.agent.load(model_path)
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Model not found at {model_path}. Using default policy.")
    
    def _create_feature_extractors(self) -> Dict[str, callable]:
        """
        Create feature extractors for LLVM IR code
        """
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
    
    def _extract_features(self, ir_code: str, pass_history: List[str]) -> np.ndarray:
        """
        Extract features from IR code to represent the state
        
        Args:
            ir_code: LLVM IR code
            pass_history: List of previously applied passes
            
        Returns:
            Feature vector
        """
        # Initialize state vector
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Set pass history features
        for pass_name in pass_history:
            if pass_name in self.passes:
                pass_idx = self.passes.index(pass_name)
                state[pass_idx] = 1.0
        
        # Extract code features
        feature_values = []
        for extractor_name, extractor_fn in self.feature_extractors.items():
            try:
                value = extractor_fn(ir_code)
                feature_values.append(float(value))
            except Exception as e:
                feature_values.append(0.0)
        
        # Normalize feature values
        if feature_values:
            feature_array = np.array(feature_values, dtype=np.float32)
            # Simple normalization to prevent extremely large values
            feature_array = np.clip(feature_array / 1000.0, -100, 100)
            state[len(self.passes):len(self.passes)+len(feature_values)] = feature_array
        
        return state
    
    def get_optimal_pass_sequence(self, ir_code: str, max_passes: int = 10) -> List[str]:
        """
        Get the optimal sequence of optimization passes for the given IR code
        
        Args:
            ir_code: LLVM IR code
            max_passes: Maximum number of passes to apply
            
        Returns:
            List of optimization passes to apply
        """
        pass_history = []
        current_ir = ir_code
        
        for _ in range(max_passes):
            # Extract features
            state = self._extract_features(current_ir, pass_history)
            
            # Get action from agent
            action = self.agent.select_action(state)
            pass_name = self.passes[action]
            
            # Skip if pass already applied recently
            if pass_name in pass_history[-3:]:
                continue
            
            # Apply pass
            new_ir = self._apply_pass(current_ir, pass_name)
            if new_ir is None:
                continue
            
            # Update state
            current_ir = new_ir
            pass_history.append(pass_name)
        
        return pass_history
    
    def _apply_pass(self, ir_code: str, pass_name: str) -> Optional[str]:
        """
        Apply an optimization pass to the IR code
        
        Args:
            ir_code: LLVM IR code
            pass_name: Name of the optimization pass
            
        Returns:
            Optimized IR code, or None if optimization failed
        """
        # Save IR to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ll', delete=False) as ir_file:
            ir_file.write(ir_code)
            ir_file.flush()
            ir_path = ir_file.name
        
        # Prepare output path
        with tempfile.NamedTemporaryFile(mode='r', suffix='.ll', delete=False) as out_file:
            out_path = out_file.name
        
        # Apply LLVM optimization pass
        result = subprocess.run(['opt', '-S', ir_path, f'-{pass_name}', '-o', out_path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if optimization was successful
        if result.returncode != 0:
            os.remove(ir_path)
            try:
                os.remove(out_path)
            except:
                pass
            return None
        
        # Load optimized IR
        with open(out_path, 'r') as f:
            optimized_ir = f.read()
        
        # Cleanup
        os.remove(ir_path)
        os.remove(out_path)
        
        return optimized_ir
    
    def optimize_ir(self, input_ir_path: str, output_ir_path: str) -> Dict[str, Any]:
        """
        Optimize IR code using our trained model
        
        Args:
            input_ir_path: Path to input IR file
            output_ir_path: Path to output IR file
            
        Returns:
            Dictionary with optimization results
        """
        # Load input IR
        with open(input_ir_path, 'r') as f:
            ir_code = f.read()
        
        # Get optimal pass sequence
        pass_sequence = self.get_optimal_pass_sequence(ir_code)
        
        # Apply passes sequentially
        current_ir = ir_code
        for pass_name in pass_sequence:
            new_ir = self._apply_pass(current_ir, pass_name)
            if new_ir is not None:
                current_ir = new_ir
        
        # Save optimized IR
        with open(output_ir_path, 'w') as f:
            f.write(current_ir)
        
        # Return results
        return {
            "input_ir": input_ir_path,
            "output_ir": output_ir_path,
            "applied_passes": pass_sequence
        }

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-driven LLVM optimizer')
    parser.add_argument('--input-ir', type=str, required=True, help='Path to input LLVM IR file')
    parser.add_argument('--output-ir', type=str, required=True, help='Path to output LLVM IR file')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt', help='Path to trained model')
    parser.add_argument('--max-passes', type=int, default=10, help='Maximum number of optimization passes')
    args = parser.parse_args()
    
    # Create bridge
    bridge = AIOptimizerBridge(args.model_path)
    
    # Optimize IR
    result = bridge.optimize_ir(args.input_ir, args.output_ir)
    
    # Print results
    print(f"Applied {len(result['applied_passes'])} optimization passes:")
    for i, pass_name in enumerate(result['applied_passes']):
        print(f"  {i+1}. {pass_name}")
    
    print(f"\nOptimized IR saved to: {args.output_ir}")

if __name__ == "__main__":
    main()
