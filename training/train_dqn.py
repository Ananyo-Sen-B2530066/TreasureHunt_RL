import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import torch
import csv
import json
from env.treasure_env import TreasureHuntEnv
from agents.dqn_agent import DQNAgent

def create_log_file(level):
    """Create CSV log file with headers."""
    logs_dir = 'analysis/logs'
    abs_logs_dir = os.path.abspath(logs_dir)
    os.makedirs(abs_logs_dir, exist_ok=True)
    
    filename = f'dqn_training_{level}.csv'
    filepath = os.path.join(abs_logs_dir, filename)
    
    headers = [
        'episode', 'level', 'total_reward', 'steps_taken', 
        'treasures_collected', 'final_score', 'lives_remaining',
        'epsilon', 'duration_sec', 'avg_reward_100', 'avg_treasures_100',
        'walls_hit', 'explored_cells', 'hint_accuracy'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.flush()
        os.fsync(f.fileno())
    
    print(f"✓ Log file created: {filepath}")
    return filepath

def write_log_entry(filepath, data):
    """Write single entry to CSV with verification."""
    try:
        row_data = [
            int(data['episode']),
            str(data['level']),
            round(float(data['total_reward']), 2),
            int(data['steps_taken']),
            int(data['treasures_collected']),
            int(data['final_score']),
            int(data['lives_remaining']),
            round(float(data['epsilon']), 6),
            round(float(data['duration_sec']), 2),
            round(float(data['avg_reward_100']), 2),
            round(float(data['avg_treasures_100']), 2),
            int(data['walls_hit']),
            int(data['explored_cells']),
            round(float(data['hint_accuracy']), 2)
        ]
        
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
            f.flush()
            os.fsync(f.fileno())
        
        return True
    except Exception as e:
        print(f"✗ CSV write error: {e}")
        return False

def calculate_hint_reward(env, action):
    """Calculate reward for following hint direction."""
    hint = env.get_hint()
    
    if hint in ["Here!", "No treasure left!"]:
        return 0
    
    hint_map = {
        "N": 0, "S": 1, "W": 2, "E": 3,
        "NE": [0, 3], "NW": [0, 2],
        "SE": [1, 3], "SW": [1, 2]
    }
    
    correct_actions = hint_map.get(hint, [])
    if isinstance(correct_actions, int):
        correct_actions = [correct_actions]
    
    return 1.0 if action in correct_actions else -0.5

def load_config(level, hyperparams_path='hyperparams.json'):
    """Load configuration for the level."""
    try:
        with open(hyperparams_path, 'r') as f:
            all_configs = json.load(f)
        
        if level in all_configs:
            config = all_configs[level]
            print(f"✓ Loaded {level}-specific config from {hyperparams_path}")
        else:
            print(f"⚠ No config for {level}, using defaults")
            config = get_default_config(level)
    except:
        print(f"⚠ Could not load {hyperparams_path}, using defaults")
        config = get_default_config(level)
    
    return config

def get_default_config(level):
    """Get default configuration for level."""
    configs = {
        'easy': {
            'episodes': 1500,
            'max_steps_per_episode': 500,
            'early_stop_patience': 150,
            'early_stop_min_episodes': 500
        },
        'medium': {
            'episodes': 3000,
            'max_steps_per_episode': 1500,
            'early_stop_patience': 200,
            'early_stop_min_episodes': 1000
        },
        'hard': {
            'episodes': 5000,
            'max_steps_per_episode': 3000,
            'early_stop_patience': 300,
            'early_stop_min_episodes': 2000
        }
    }
    return configs.get(level, configs['medium'])

def train_dqn_on_level(level="medium", episodes=None, save_model=True):
    """Train DQN on a specific level with corrected parameters."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING DQN ON {level.upper()} LEVEL")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config(level)
    if episodes is None:
        episodes = config.get('episodes', 3000)
    
    # Initialize environment
    env = TreasureHuntEnv(level=level)
    obs = env.reset()[0]
    
    state_dim = len(obs['local_view']) + 2 + 1 + 1 + 1 + 4
    action_dim = env.action_space.n
    
    # Initialize agent (will load level-specific hyperparams)
    agent = DQNAgent(state_dim, action_dim, level=level, hyperparams_path='hyperparams.json')
    agent.max_rows = env.rows
    agent.max_cols = env.cols
    
    print(f"\nEnvironment:")
    print(f"  Map size: {env.rows}x{env.cols}")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Total episodes: {episodes}")
    print()
    
    # Create log file
    log_path = create_log_file(level)
    
    # Training metrics
    episode_rewards = []
    episode_treasures = []
    episode_steps = []
    
    best_avg_reward = -np.inf
    patience_counter = 0
    max_steps = config.get('max_steps_per_episode', 1500)
    patience_limit = config.get('early_stop_patience', 200)
    min_episodes = config.get('early_stop_min_episodes', 1000)
    
    # Warmup
    print("Warming up replay buffer...")
    obs, _ = env.reset()
    state = agent.get_state(obs)
    warmup_steps = agent.batch_size * 2
    
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = agent.get_state(next_obs)
        agent.remember(state, action, reward, next_state, done or truncated)
        
        if done or truncated:
            obs, _ = env.reset()
            state = agent.get_state(obs)
        else:
            state = next_state
    
    print(f"✓ Replay buffer initialized with {len(agent.memory)} experiences\n")
    print("="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    # Training loop
    for episode in range(1, episodes + 1):
        episode_start = time.time()
        
        obs, _ = env.reset()
        state = agent.get_state(obs)
        
        total_reward = 0
        steps = 0
        treasures_collected = 0
        walls_hit = 0
        hint_follows = 0
        hint_decisions = 0
        done = False
        
        last_treasure_count = sum(row.count(2) for row in env.map)
        
        # Episode loop
        while not done and steps < max_steps:
            action = agent.get_action(state)
            hint_reward = calculate_hint_reward(env, action)
            
            if abs(hint_reward) > 0:
                hint_decisions += 1
                if hint_reward > 0:
                    hint_follows += 1
            
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = agent.get_state(next_obs)
            
            # Reward shaping
            shaped_reward = env_reward + hint_reward
            
            # Track treasure collection
            current_treasure_count = sum(row.count(2) for row in env.map)
            if current_treasure_count < last_treasure_count:
                treasures_collected += 1
                efficiency_bonus = max(20, 50 - steps * 0.1)
                shaped_reward += efficiency_bonus
                last_treasure_count = current_treasure_count
            
            walls_hit += info.get('walls_hit', 0)
            
            # Penalty for being stuck
            if steps > 50 and treasures_collected == 0:
                shaped_reward -= 0.1
            
            # Store and train
            agent.remember(state, action, shaped_reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                agent.train()
            
            state = next_state
            total_reward += shaped_reward
            steps += 1
        
        # Episode end
        episode_duration = time.time() - episode_start
        episode_rewards.append(total_reward)
        episode_treasures.append(treasures_collected)
        episode_steps.append(steps)
        
        avg_reward_100 = np.mean(episode_rewards[-100:])
        avg_treasures_100 = np.mean(episode_treasures[-100:])
        hint_accuracy = (hint_follows / hint_decisions * 100) if hint_decisions > 0 else 0
        
        # Log data
        log_data = {
            'episode': episode,
            'level': level,
            'total_reward': total_reward,
            'steps_taken': steps,
            'treasures_collected': treasures_collected,
            'final_score': env.score,
            'lives_remaining': env.lifelines,
            'epsilon': agent.epsilon,
            'duration_sec': episode_duration,
            'avg_reward_100': avg_reward_100,
            'avg_treasures_100': avg_treasures_100,
            'walls_hit': walls_hit,
            'explored_cells': len(env.visited_cells),
            'hint_accuracy': hint_accuracy
        }
        
        write_log_entry(log_path, log_data)
        
        # Save best model
        if avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100
            patience_counter = 0
            if save_model:
                model_dir = 'analysis/models'
                os.makedirs(model_dir, exist_ok=True)
                agent.save(f'{model_dir}/dqn_model_{level}_best.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if episode % 50 == 0 or episode == 1:
            print(f"Ep {episode:4d}/{episodes} | "
                  f"R: {total_reward:7.1f} | "
                  f"T: {treasures_collected:2d} | "
                  f"Steps: {steps:4d} | "
                  f"Eps: {agent.epsilon:.4f} | "
                  f"Hint: {hint_accuracy:5.1f}% | "
                  f"Avg100: {avg_reward_100:7.1f}")
        
        # Early stopping
        if patience_counter >= patience_limit and episode >= min_episodes:
            print(f"\n✓ Early stopping at episode {episode}")
            print(f"  Best 100-ep avg reward: {best_avg_reward:.1f}")
            break
    
    # Save final model
    if save_model:
        model_dir = 'analysis/models'
        os.makedirs(model_dir, exist_ok=True)
        agent.save(f'{model_dir}/dqn_model_{level}_final.pth')
    
    env.close()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - {level.upper()}")
    print(f"{'='*70}")
    print(f"Episodes trained: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")
    print(f"Average treasures: {np.mean(episode_treasures):.1f}")
    print(f"Best 100-ep avg: {best_avg_reward:.1f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Log file: {log_path}")
    print(f"{'='*70}\n")
    
    return agent

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN on Treasure Hunt')
    parser.add_argument('--level', type=str, default='medium', 
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (default: from config)')
    
    args = parser.parse_args()
    
    train_dqn_on_level(level=args.level, episodes=args.episodes)