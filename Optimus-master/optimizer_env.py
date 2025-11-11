import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import tempfile
import os
class LLVMEnv(gym.Env):
    def __init__(self, input_ir, passes):
        super().__init__()

        self.input_ir = input_ir
        self.passes = passes
        self.current_ir = None

        # Define action space: each pass is one discrete action
        self.action_space = spaces.Discrete(len(passes))

        # Dummy observation space: binary vector of which passes have been applied
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(passes),), dtype=np.float32)

        self.current_state = np.zeros(len(passes), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load the initial IR code
        with open(self.input_ir, 'r') as f:
            self.current_ir = f.read()

        # Reset pass tracking
        self.current_state = np.zeros(len(self.passes), dtype=np.float32)

        info = {}
        return self.current_state, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        pass_name = self.passes[action]

        # Save current IR to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ll', delete=False) as ir_file:
            ir_file.write(self.current_ir)
            ir_file.flush()
            ir_path = ir_file.name

        # Prepare output path
        with tempfile.NamedTemporaryFile(mode='r', suffix='.ll', delete=False) as out_file:
            out_path = out_file.name

        # Apply LLVM optimization pass
        result = subprocess.run(['opt', '-S', ir_path, f'-{pass_name}', '-o', out_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Load updated IR
        with open(out_path, 'r') as f:
            self.current_ir = f.read()

        # Update state: mark that this pass was applied
        self.current_state[action] = 1.0

        # Calculate reward
        runtime = self.run_ir_and_measure_runtime(self.current_ir)
        reward = -runtime

        # Cleanup
        os.remove(ir_path)
        os.remove(out_path)

        # Mark done if all passes applied
        done = bool(np.all(self.current_state == 1.0))
        truncated = False
        info = {"runtime": runtime}

        return self.current_state, reward, done, truncated, info

    def run_ir_and_measure_runtime(self, ir_code):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ll', delete=False) as ir_file:
            ir_file.write(ir_code)
            ir_file.flush()
            ir_path = ir_file.name

        exec_path = ir_path + '.out'

        # Compile IR to binary
        compile = subprocess.run(['clang', ir_path, '-o', exec_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if compile.returncode != 0:
            os.remove(ir_path)
            return 1e6  # heavy penalty for failure

        import time
        start = time.time()
        run = subprocess.run([exec_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()

        os.remove(ir_path)
        os.remove(exec_path)

        if run.returncode != 0:
            return 1e6  # heavy penalty for runtime failure
        def save_ir(self, filename):
          with open(filename, 'w') as f:
            f.write(self.current_ir)

        return end - start

