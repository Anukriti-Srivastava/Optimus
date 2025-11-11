import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MLGOEnvironment(gym.Env):
    def __init__(self, input_ir: str, passes: list):
        super().__init__()

        self.input_ir = input_ir
        self.passes = passes
        self.current_ir = None
        self.pass_history = []

        self.action_space = spaces.Discrete(len(passes))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_ir = self.input_ir
        self.pass_history = []
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        pass_name = self.passes[action]
        self.pass_history.append(pass_name)
        reward = 0.1
        done = len(self.pass_history) >= 20
        truncated = False
        return np.zeros(1, dtype=np.float32), reward, done, truncated, {}
