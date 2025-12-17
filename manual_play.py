import pygame
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.treasure_env import TreasureHuntEnv

# --- Key to Action Mapping ---
KEY_ACTIONS = {
    pygame.K_UP: 0,      # Up
    pygame.K_DOWN: 1,    # Down
    pygame.K_LEFT: 2,    # Left
    pygame.K_RIGHT: 3,   # Right
    pygame.K_w: 0,       # W = Up
    pygame.K_s: 1,       # S = Down
    pygame.K_a: 2,       # A = Left
    pygame.K_d: 3        # D = Right
}

def print_game_state(env, reward, info):
    """Print current game state information."""
    print(f"\n{'='*50}")
    print(f"Score: {env.score} | Lives: {env.lifelines} | Steps: {env.steps}")
    print(f"Reward: {reward:.1f}")
    print(f"Treasures Remaining: {info.get('treasures_remaining', 0)}")
    print(f"Hint: {env.get_hint()}")
    print(f"Position: {env.agent_pos}")
    print(f"Visited Cells: {len(env.visited_cells)}")
    
    if info.get('treasure_collected'):
        print("✅ TREASURE COLLECTED!")
    if info.get('walls_hit', 0) > 0:
        print("🧱 Hit a wall!")
    if info.get('new_cell_visited'):
        print("🔍 New area explored!")
    print(f"{'='*50}")

def main():
    """Main function to run manual gameplay."""
    pygame.init()
    
    # --- Level Selection ---
    print("\n" + "="*60)
    print("TREASURE HUNT - MANUAL PLAY MODE")
    print("="*60)
    print("\nSelect Difficulty Level:")
    print("  1 - Easy   (10x10 map, slower traps)")
    print("  2 - Medium (16x16 map, balanced)")
    print("  3 - Hard   (30x30 map, faster traps)")
    
    while True:
        choice = input("\nEnter choice (1-3) or press Enter for Medium: ").strip()
        if choice == "1":
            level = "easy"
            break
        elif choice == "2" or choice == "":
            level = "medium"
            break
        elif choice == "3":
            level = "hard"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\n✓ Starting {level.upper()} level...")
    print("\nControls:")
    print("  Arrow Keys or WASD - Move")
    print("  ESC or Q - Quit")
    print("  R - Restart Episode")
    print("\nGoal: Collect all treasures and find the final treasure!")
    print("="*60 + "\n")
    
    # --- Initialize Environment ---
    env = TreasureHuntEnv(level=level)
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    
    # Game state
    running = True
    paused = False
    total_episodes = 0
    episode_rewards = []
    current_episode_reward = 0
    
    # Initial render
    env.render()
    print_game_state(env, 0, {'treasures_remaining': sum(row.count(2) for row in env.map)})
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            elif event.type == pygame.KEYDOWN:
                # Quit keys
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                    break
                
                # Restart key
                elif event.key == pygame.K_r:
                    print("\n🔄 Restarting episode...")
                    if current_episode_reward != 0:
                        episode_rewards.append(current_episode_reward)
                        total_episodes += 1
                    current_episode_reward = 0
                    obs, _ = env.reset()
                    env.render()
                    print_game_state(env, 0, {'treasures_remaining': sum(row.count(2) for row in env.map)})
                
                # Movement keys
                elif event.key in KEY_ACTIONS:
                    action = KEY_ACTIONS[event.key]
                    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                    
                    # Take step
                    obs, reward, done, truncated, info = env.step(action)
                    current_episode_reward += reward
                    
                    # Render and display info
                    env.render()
                    print(f"\nAction: {action_names[action]}")
                    print_game_state(env, reward, info)
                    
                    # Check if episode ended
                    if done or truncated:
                        total_episodes += 1
                        episode_rewards.append(current_episode_reward)
                        
                        print("\n" + "🎉"*20)
                        print("EPISODE FINISHED!")
                        print("🎉"*20)
                        print(f"\nReason: {info.get('reason', 'Unknown')}")
                        print(f"Final Score: {env.score}")
                        print(f"Final Lives: {env.lifelines}")
                        print(f"Total Steps: {env.steps}")
                        print(f"Episode Reward: {current_episode_reward:.1f}")
                        print(f"Treasures Collected: {sum(row.count(2) for row in env.map) == 0}")
                        
                        if total_episodes > 0:
                            print(f"\nSession Stats:")
                            print(f"  Episodes Completed: {total_episodes}")
                            print(f"  Average Reward: {sum(episode_rewards)/len(episode_rewards):.1f}")
                        
                        print("\nPress R to restart or ESC to quit.")
                        current_episode_reward = 0
        
        clock.tick(30)  # 30 FPS for smooth rendering
    
    # Cleanup
    env.close()
    pygame.quit()
    
    # Final statistics
    if total_episodes > 0:
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total Episodes: {total_episodes}")
        print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.1f}")
        print(f"Best Episode: {max(episode_rewards):.1f}")
        print(f"Worst Episode: {min(episode_rewards):.1f}")
        print("="*60)
    
    print("\nThanks for playing! 👋\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Exiting...")
        pygame.quit()
        sys.exit(0)