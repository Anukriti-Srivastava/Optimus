import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class MLGOPolicy(nn.Module):
    """
    Neural network policy for Machine Learning Guided Optimization (MLGO)
    Inspired by Google's MLGO framework for LLVM
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLGOPolicy, self).__init__()
        
        # Feature extraction layers
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.feature_network(x)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        return action_probs, state_value

class MLGOAgent:
    """
    Agent that uses reinforcement learning to make optimization decisions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99):
        self.policy = MLGOPolicy(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        # Storage for episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.policy(state_tensor)
        
        # Sample action from probability distribution
        m = Categorical(action_probs)
        action = m.sample()
        
        # Store episode data
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(m.log_prob(action))
        self.values.append(state_value)

        return action.item()
    
    def update_policy(self):
        # Convert lists to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        
        # Calculate returns (discounted rewards)
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # Calculate advantage (returns - values)
        values = torch.cat(self.values).squeeze()
        advantage = returns - values.detach()
        # Get log probabilities for taken actions
        log_probs = torch.cat(self.log_probs)
        # Calculate losses
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()
        # Clear episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        return total_loss.item()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
