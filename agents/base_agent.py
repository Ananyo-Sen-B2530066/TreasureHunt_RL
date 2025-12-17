import numpy as np
from collections import defaultdict
import json

class BaseAgent:
    """
    Base class for all Reinforcement Learning agents (Q-Learning, SARSA)
    in the Treasure Hunt environment. Handles common utilities like 
    initialization, parameter loading, and epsilon-greedy action selection.
    
    The state is represented by the agent's position (r, c), and potentially 
    other game stats like (r, c, lives, score), which is simplified here 
    to just the agent_pos for table lookup flexibility.
    """
    
    def __init__(self, action_space, obs_space, hyperparams_path='../training/hyperparams.json'):
        """
        Initializes the agent with the environment's action and observation spaces.
        Loads hyperparameters from a JSON file.
        
        Args:
            action_space: The action space of the Gymnasium environment.
            obs_space: The observation space of the Gymnasium environment.
            hyperparams_path (str): Path to the hyperparams.json file.
        """
        # Store environment spaces (needed for action selection/Q-table sizing)
        self.action_space = action_space
        self.obs_space = obs_space
        
        # Q-table initialization: Using defaultdict for sparse Q-tables
        # Maps (state_key) -> (action_index: Q-value)
        # We rely on the observation being hashable (e.g., a tuple of position or features)
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        
        # Load hyperparameters (alpha, gamma, epsilon)
        self._load_hyperparameters(hyperparams_path)
        
        print(f"Agent initialized with: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}")

    def _load_hyperparameters(self, path):
        """Loads hyperparameters from a JSON file."""
        try:
            with open(path, 'r') as f:
                params = json.load(f)
                # Assuming the JSON has top-level keys for the common parameters
                self.alpha = params.get('alpha', 0.1)      # Learning rate
                self.gamma = params.get('gamma', 0.99)     # Discount factor
                self.epsilon = params.get('epsilon', 1.0)  # Exploration rate (starts high)
                self.epsilon_min = params.get('epsilon_min', 0.01) # Minimum exploration rate
                self.epsilon_decay = params.get('epsilon_decay', 0.9995) # Decay factor
        except FileNotFoundError:
            print(f"Warning: Hyperparameter file not found at {path}. Using defaults.")
            self.alpha = 0.1
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.9995
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON file at {path}. Using defaults.")
            self.alpha = 0.1
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.9995


    def get_action(self, state_key):
        """
        Selects an action using the epsilon-greedy policy.
        
        Args:
            state_key (tuple): The current state observation (must be hashable).
        
        Returns:
            int: The selected action index.
        """
        if np.random.random() < self.epsilon:
            # Explore: Choose a random action
            return self.action_space.sample()
        else:
            # Exploit: Choose the action with the maximum Q-value
            q_values = self.Q[state_key]
            # Use argmax to get the index (action) of the max Q-value
            return np.argmax(q_values)

    def update_epsilon(self):
        """Decays the epsilon value for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def update_q_value(self, state, action, reward, next_state, next_action=None):
        """
        Abstract method for updating the Q-table. Must be implemented by 
        the specific algorithm (Q-Learning or SARSA).
        """
        raise NotImplementedError("The update_q_value method must be implemented by the subclass.")