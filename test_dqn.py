import sys
import os
import numpy as np
import pygame
import torch
import matplotlib.pyplot as plt
import random
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.treasure_env import TreasureHuntEnv, ROAD, WALL, TREASURE, STATIC_TRAP, LIFELINE, DYNAMIC_TRAP, FINAL_TREASURE
from agents.dqn_agent import DQNAgent

def test_dqn_with_analysis(env_level="easy", episodes=5, render=True, max_steps=1000):
    """Test DQN with comprehensive analysis of agent behavior."""
    
    env = TreasureHuntEnv(level=env_level)
    obs = env.reset()[0]
    
    # UPDATED: State dimension now includes hint_direction (4 values)
    # local_view (25) + agent_pos (2) + score (1) + lives (1) + treasures_left (1) + hint_direction (4) = 34
    state_dim = len(obs['local_view']) + 2 + 1 + 1 + 1 + 4
    action_dim = env.action_space.n
    
    # Initialize agent with level-specific hyperparameters
    agent = DQNAgent(state_dim, action_dim, level=env_level, hyperparams_path='hyperparams.json')
    agent.max_rows = env.rows
    agent.max_cols = env.cols
    
    # Load trained model
    model_paths = [
        f'analysis/models/dqn_model_{env_level}_best.pth',
        f'analysis/models/dqn_model_{env_level}_final.pth'
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            agent.load(model_path)
            agent.epsilon = 0.0  # Pure exploitation for testing
            print(f"✅ Model loaded from: {model_path}")
            model_loaded = True
            break
    
    if not model_loaded:
        print("⚠️ No trained model found. Testing with random actions.")
        agent.epsilon = 1.0  # Pure exploration
    
    print(f"\n{'='*60}")
    print(f"TESTING DQN AGENT ON {env_level.upper()} LEVEL")
    print(f"Map size: {env.rows}x{env.cols}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"{'='*60}\n")
    
    # Analysis tracking
    episode_stats = []
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    for episode in range(episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {episode+1}/{episodes}")
        print(f"{'='*40}")
        
        obs, _ = env.reset()
        state = agent.get_state(obs)
        
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        positions = []
        treasures_collected = 0
        walls_hit = 0
        trap_encounters = 0
        action_counts = defaultdict(int)
        stuck_counter = 0
        consecutive_wall_hits = 0
        hint_following_decisions = 0
        correct_hint_follows = 0
        
        while not done and not truncated and steps < max_steps:
            if render:
                env.render()
                pygame.time.delay(50)  # Slower for observation
            
            # Get current hint
            current_hint = env.get_hint()
            
            # Get action and Q-values
            with torch.no_grad():
                q_values = agent.policy_net(state).cpu().numpy()
            
            action = agent.get_action(state) if agent.epsilon > 0 else np.argmax(q_values)
            action_counts[action] += 1
            
            # UPDATED: Check if action follows hint (improved logic)
            if current_hint not in ["Here!", "No treasure left!"]:
                hint_following_decisions += 1
                
                # Map hint directions to actions
                hint_map = {
                    "N": [0],      # Up
                    "S": [1],      # Down
                    "W": [2],      # Left
                    "E": [3],      # Right
                    "NE": [0, 3],  # North-East (up or right)
                    "NW": [0, 2],  # North-West (up or left)
                    "SE": [1, 3],  # South-East (down or right)
                    "SW": [1, 2]   # South-West (down or left)
                }
                
                correct_actions = hint_map.get(current_hint, [])
                if action in correct_actions:
                    correct_hint_follows += 1
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_state(next_obs)
            
            # Track position for stuck detection
            positions.append(env.agent_pos)
            
            # Check for wall hits
            if info.get('walls_hit', 0) > 0:
                walls_hit += 1
                consecutive_wall_hits += 1
                if consecutive_wall_hits > 3:
                    print(f"  🔴 Consecutive wall hits: {consecutive_wall_hits}")
                    q_str = " | ".join([f"{name}:{q:.2f}" for name, q in zip(action_names, q_values)])
                    print(f"     Current Q-values: [{q_str}]")
            else:
                consecutive_wall_hits = 0
            
            # UPDATED: Track trap encounters from environment
            if info.get('lives_lost', 0) > 0 or 'trap' in info.get('reason', '').lower():
                trap_encounters += 1
            
            # Check for treasure collection
            if info.get('treasure_collected', False):
                treasures_collected += 1
                print(f"  ✅ Treasure collected! Total: {treasures_collected}")
                q_str = " | ".join([f"{name}:{q:.2f}" for name, q in zip(action_names, q_values)])
                print(f"     Q-values at treasure: [{q_str}]")
            
            # Print detailed info every 50 steps
            if steps % 50 == 0 and steps > 0:
                remaining = info.get('treasures_remaining', 0)
                print(f"  Step {steps}: Reward={total_reward:.1f}, Treasures={treasures_collected}, "
                      f"Walls={walls_hit}, Hint={current_hint}, Remaining={remaining}")
            
            # Check if stuck (repeating positions)
            if len(positions) > 15:
                recent_positions = positions[-15:]
                unique_positions = len(set(recent_positions))
                if unique_positions < 4:
                    stuck_counter += 1
                    if stuck_counter == 3:
                        print(f"  ⚠️ Agent might be stuck! Unique positions in last 15 steps: {unique_positions}")
                    if stuck_counter > 5:
                        print(f"  🔀 Forcing random action to break loop")
                        action = random.randint(0, 3)
                else:
                    stuck_counter = 0
            
            # Update
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
        
        # Episode summary
        remaining_treasures = info.get('treasures_remaining', 
                                       sum(row.count(TREASURE) for row in env.map))
        
        # Calculate hint following accuracy
        hint_accuracy = (correct_hint_follows / hint_following_decisions * 100) if hint_following_decisions > 0 else 0
        
        print(f"\n📊 EPISODE {episode+1} SUMMARY:")
        print(f"   Steps taken: {steps}")
        print(f"   Total Reward: {total_reward:.1f}")
        print(f"   Treasures Collected: {treasures_collected}")
        print(f"   Treasures Remaining: {remaining_treasures}")
        print(f"   Walls Hit: {walls_hit}")
        print(f"   Trap Encounters: {trap_encounters}")
        print(f"   Final Score: {env.score}")
        print(f"   Lives Remaining: {env.lifelines}")
        print(f"   Hint Following Accuracy: {hint_accuracy:.1f}%")
        print(f"   Termination Reason: {info.get('reason', 'Max steps reached')}")
        
        # Action distribution
        print(f"\n   Action Distribution:")
        for action_idx in range(action_dim):
            count = action_counts[action_idx]
            percentage = (count / steps * 100) if steps > 0 else 0
            print(f"     {action_names[action_idx]}: {count} ({percentage:.1f}%)")
        
        # Path efficiency analysis
        if len(positions) > 0:
            unique_positions = len(set(positions))
            efficiency = (unique_positions / len(positions)) * 100 if len(positions) > 0 else 0
            print(f"   Path Efficiency: {efficiency:.1f}% ({unique_positions} unique positions out of {len(positions)})")
        
        # Store episode stats
        episode_stats.append({
            'episode': episode + 1,
            'steps': steps,
            'reward': total_reward,
            'treasures': treasures_collected,
            'walls': walls_hit,
            'traps': trap_encounters,
            'efficiency': efficiency if 'efficiency' in locals() else 0,
            'hint_accuracy': hint_accuracy,
            'positions': positions.copy(),
            'action_counts': dict(action_counts),
            'final_lives': env.lifelines,
            'final_score': env.score
        })
    
    env.close()
    
    # Overall analysis
    print(f"\n{'='*60}")
    print(f"OVERALL TESTING SUMMARY")
    print(f"{'='*60}")
    
    if episode_stats:
        avg_steps = np.mean([s['steps'] for s in episode_stats])
        avg_reward = np.mean([s['reward'] for s in episode_stats])
        avg_treasures = np.mean([s['treasures'] for s in episode_stats])
        avg_walls = np.mean([s['walls'] for s in episode_stats])
        avg_traps = np.mean([s['traps'] for s in episode_stats])
        avg_hint_accuracy = np.mean([s['hint_accuracy'] for s in episode_stats])
        avg_efficiency = np.mean([s['efficiency'] for s in episode_stats])
        
        print(f"  Average Steps per Episode: {avg_steps:.1f}")
        print(f"  Average Reward per Episode: {avg_reward:.1f}")
        print(f"  Average Treasures Collected: {avg_treasures:.1f}")
        print(f"  Average Walls Hit: {avg_walls:.1f}")
        print(f"  Average Trap Encounters: {avg_traps:.1f}")
        print(f"  Average Hint Following Accuracy: {avg_hint_accuracy:.1f}%")
        print(f"  Average Path Efficiency: {avg_efficiency:.1f}%")
        
        # Calculate success metrics
        success_rate = (sum(1 for s in episode_stats if s['treasures'] > 0) / len(episode_stats)) * 100
        survival_rate = (sum(1 for s in episode_stats if s['final_lives'] > 0) / len(episode_stats)) * 100
        
        print(f"  Success Rate (found at least 1 treasure): {success_rate:.1f}%")
        print(f"  Survival Rate (ended with lives remaining): {survival_rate:.1f}%")
        
        # Best episode
        best_episode = max(episode_stats, key=lambda s: s['treasures'])
        print(f"\n  Best Episode: #{best_episode['episode']}")
        print(f"    Treasures Collected: {best_episode['treasures']}")
        print(f"    Reward: {best_episode['reward']:.1f}")
        print(f"    Steps: {best_episode['steps']}")
        
        # Visualize path of best episode
        if render and best_episode['positions']:
            visualize_agent_path(env_level, best_episode['positions'], env.rows, env.cols)
    
    return episode_stats


def visualize_agent_path(level, positions, rows, cols):
    """Visualize the agent's path on the grid."""
    try:
        # Create grid
        grid = np.zeros((rows, cols))
        
        # Mark visited cells
        for r, c in positions:
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] += 1
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Visit Count')
        plt.title(f'Agent Path Heatmap - {level.upper()} Level')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # Mark start and end
        if positions:
            start_r, start_c = positions[0]
            end_r, end_c = positions[-1]
            plt.scatter(start_c, start_r, c='green', s=200, marker='o', label='Start', edgecolors='black')
            plt.scatter(end_c, end_r, c='red', s=200, marker='X', label='End', edgecolors='black')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        os.makedirs('analysis/plots', exist_ok=True)
        plt.savefig(f'analysis/plots/path_heatmap_{level}.png', dpi=150)
        print(f"  📈 Path heatmap saved to: analysis/plots/path_heatmap_{level}.png")
        
        # Show if possible
        try:
            plt.show()
        except:
            pass
            
        plt.close()
        
    except Exception as e:
        print(f"  ⚠️ Could not generate path visualization: {e}")


def test_all_levels():
    """Test agent on all difficulty levels."""
    levels = ["easy", "medium", "hard"]
    
    print(f"\n{'#'*70}")
    print(f"COMPREHENSIVE AGENT TESTING ON ALL LEVELS")
    print(f"{'#'*70}")
    
    results = {}
    for level in levels:
        print(f"\n>>> Testing {level.upper()} level")
        try:
            stats = test_dqn_with_analysis(env_level=level, episodes=3, render=False, max_steps=1000)
            results[level] = stats
        except Exception as e:
            print(f"  ❌ Error testing {level}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparative summary
    print(f"\n{'='*70}")
    print(f"COMPARATIVE SUMMARY ACROSS LEVELS")
    print(f"{'='*70}")
    
    for level, stats in results.items():
        if stats:
            avg_treasures = np.mean([s['treasures'] for s in stats])
            avg_reward = np.mean([s['reward'] for s in stats])
            print(f"\n{level.upper()}:")
            print(f"  Avg Treasures: {avg_treasures:.2f}")
            print(f"  Avg Reward: {avg_reward:.1f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN agent')
    parser.add_argument('--level', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level to test')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--all-levels', action='store_true',
                       help='Test on all difficulty levels')
    
    args = parser.parse_args()
    
    if args.all_levels:
        test_all_levels()
    else:
        test_dqn_with_analysis(
            env_level=args.level,
            episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps
        )