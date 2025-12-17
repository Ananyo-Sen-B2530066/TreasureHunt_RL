import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256, 128, 64], dropout=0.2):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, level="medium", hyperparams_path='hyperparams.json'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.level = level
        
        # Load hyperparameters based on level
        self._load_hyperparameters(hyperparams_path, level)
        
        # Initialize replay memory
        self.memory = deque(maxlen=self.memory_capacity)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine network architecture based on level
        if level == "easy":
            hidden_layers = [128, 128, 64]
            dropout = 0.1
        elif level == "medium":
            hidden_layers = [256, 256, 128, 64]
            dropout = 0.2
        else:  # hard
            hidden_layers = [512, 512, 256, 128, 64]
            dropout = 0.3
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim, hidden_layers, dropout).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_layers, dropout).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        self.steps = 0
        
        print(f"DQN Agent initialized for {level.upper()} level:")
        print(f"  Device: {self.device}")
        print(f"  State dim: {state_dim}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Memory capacity: {self.memory_capacity}")
        print(f"  Epsilon decay: {self.epsilon_decay}")
        print(f"  Target update: Every {self.update_target_every} episodes")
    
    def _load_hyperparameters(self, path, level):
        """Load hyperparameters from JSON file based on level."""
        try:
            with open(path, 'r') as f:
                all_params = json.load(f)
            
            # Check if file has level-specific configs
            if level in all_params:
                params = all_params[level]
                print(f"Loaded {level}-specific hyperparameters from {path}")
            elif isinstance(all_params, dict) and 'alpha' in all_params:
                # Old format - single config
                params = all_params
                print(f"Warning: Using single config from {path} (not level-specific)")
            else:
                raise ValueError(f"Invalid hyperparameter file format")
            
            # Load parameters
            self.lr = params.get('learning_rate', params.get('alpha', 0.001))
            self.gamma = params.get('gamma', 0.99)
            self.epsilon = params.get('epsilon', 1.0)
            self.epsilon_min = params.get('epsilon_min', 0.01)
            self.epsilon_decay = params.get('epsilon_decay', 0.995)
            self.batch_size = params.get('batch_size', 64)
            self.memory_capacity = params.get('memory_capacity', 10000)
            self.update_target_every = params.get('update_target_every', 100)
            
        except FileNotFoundError:
            print(f"Warning: Hyperparameter file not found at {path}")
            print("Using default values")
            self._set_defaults(level)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Error reading {path}: {e}")
            print("Using default values")
            self._set_defaults(level)
    
    def _set_defaults(self, level):
        """Set default hyperparameters based on level."""
        defaults = {
            'easy': {
                'lr': 0.001, 'gamma': 0.95, 'epsilon': 1.0,
                'epsilon_min': 0.05, 'epsilon_decay': 0.996,
                'batch_size': 64, 'memory_capacity': 15000,
                'update_target_every': 50
            },
            'medium': {
                'lr': 0.0005, 'gamma': 0.97, 'epsilon': 1.0,
                'epsilon_min': 0.03, 'epsilon_decay': 0.9975,
                'batch_size': 128, 'memory_capacity': 50000,
                'update_target_every': 100
            },
            'hard': {
                'lr': 0.0002, 'gamma': 0.99, 'epsilon': 1.0,
                'epsilon_min': 0.02, 'epsilon_decay': 0.9990,
                'batch_size': 256, 'memory_capacity': 100000,
                'update_target_every': 150
            }
        }
        
        params = defaults.get(level, defaults['medium'])
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.memory_capacity = params['memory_capacity']
        self.update_target_every = params['update_target_every']
    
    def get_state(self, observation):
        """Convert observation dictionary to normalized state vector."""
        local_view = observation['local_view']
        agent_pos = observation['agent_pos']
        score = observation['score']
        lives = observation['lives']
        treasures_left = observation['treasures_left']
        hint_direction = observation['hint_direction']
        
        # Convert to numpy arrays
        local_view = np.array(local_view).flatten()
        agent_pos = np.array(agent_pos)
        score = np.array(score).flatten()
        lives = np.array(lives).flatten()
        treasures_left = np.array(treasures_left).flatten()
        hint_direction = np.array(hint_direction).flatten()
        
        # CRITICAL: Normalize all features to [0, 1] range
        local_view = local_view / 6.0  # Cell types 0-6
        agent_pos = agent_pos / 30.0   # Max 30x30 grid
        score = score / 1000.0         # Normalize score
        lives = lives / 5.0            # Max ~5 lives
        treasures_left = treasures_left / 20.0  # Max ~20 treasures
        # hint_direction already in [0, 1]
        
        # Concatenate all components
        state = np.concatenate([
            local_view,
            agent_pos,
            score,
            lives,
            treasures_left,
            hint_direction
        ])
        
        return torch.FloatTensor(state).to(self.device)
    
    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        state_cpu = state.cpu() if state.is_cuda else state
        next_state_cpu = next_state.cpu() if next_state.is_cuda else next_state
        self.memory.append((state_cpu, action, reward, next_state_cpu, done))
    
    def train(self):
        """Train on a batch from replay memory using Double DQN."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack tensors
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy net to select, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            print(f"Model loaded from {path}, epsilon: {self.epsilon}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            print("Initializing new model...")