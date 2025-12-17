# ============================================================
# File: test_env_run.py
# Description: Enhanced test script for Treasure Hunt Environment
#              Tests environment initialization, rendering, and random actions
# ============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.treasure_env import TreasureHuntEnv
import pygame
import random
import time

def test_environment_basic(level="medium"):
    """Test basic environment functionality."""
    print(f"\n{'='*60}")
    print(f"TESTING TREASURE HUNT ENVIRONMENT - {level.upper()} LEVEL")
    print(f"{'='*60}\n")
    
    # Initialize environment
    print("1. Initializing environment...")
    env = TreasureHuntEnv(level=level)
    print(f"   ✓ Environment created")
    print(f"   Map size: {env.rows}x{env.cols}")
    print(f"   Tile size: {env.tile_size}px")
    print(f"   Initial position: {env.agent_pos}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset")
    print(f"   Initial lives: {env.lifelines}")
    print(f"   Initial score: {env.score}")
    
    # Check observation structure
    print("\n3. Checking observation space...")
    print(f"   Local view shape: {obs['local_view'].shape}")
    print(f"   Agent position: {obs['agent_pos']}")
    print(f"   Score: {obs['score']}")
    print(f"   Lives: {obs['lives']}")
    print(f"   Treasures left: {obs['treasures_left']}")
    print(f"   Hint direction: {obs['hint_direction']}")
    print(f"   Current hint: {env.get_hint()}")
    
    # Test rendering
    print("\n4. Testing rendering...")
    env.render()
    print(f"   ✓ Rendering works")
    
    # Test action space
    print(f"\n5. Testing action space...")
    print(f"   Action space: {env.action_space}")
    print(f"   Number of actions: {env.action_space.n}")
    print(f"   Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT")
    
    # Count treasures
    treasure_count = sum(row.count(2) for row in env.map)
    static_trap_count = sum(row.count(3) for row in env.map)
    lifeline_count = sum(row.count(4) for row in env.map)
    
    print(f"\n6. Map statistics:")
    print(f"   Total treasures: {treasure_count}")
    print(f"   Static traps: {static_trap_count}")
    print(f"   Lifelines: {lifeline_count}")
    print(f"   Final treasure position: {env.final_treasure_pos}")
    
    print(f"\n{'='*60}")
    print("BASIC TESTS PASSED ✓")
    print(f"{'='*60}\n")
    
    return env

def test_random_episode(env, max_steps=100, render_delay=50):
    """Run a test episode with random actions."""
    print(f"\n{'='*60}")
    print("RUNNING RANDOM EPISODE TEST")
    print(f"{'='*60}\n")
    
    obs, _ = env.reset()
    
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    total_reward = 0
    treasures_collected = 0
    walls_hit = 0
    steps = 0
    
    running = True
    while running and steps < max_steps:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Update statistics
        total_reward += reward
        if info.get('treasure_collected'):
            treasures_collected += 1
            print(f"  ✅ Step {steps}: Treasure collected! Total: {treasures_collected}")
        
        walls_hit += info.get('walls_hit', 0)
        steps += 1
        
        # Render
        env.render()
        pygame.time.delay(render_delay)
        
        # Print progress every 20 steps
        if steps % 20 == 0:
            print(f"  Step {steps}: Reward={total_reward:.1f}, "
                  f"Lives={env.lifelines}, Score={env.score}, "
                  f"Treasures={treasures_collected}")
        
        # Check if episode ended
        if done or truncated:
            print(f"\n  Episode ended at step {steps}")
            print(f"  Reason: {info.get('reason', 'Unknown')}")
            break
    
    # Episode summary
    print(f"\n{'='*60}")
    print("EPISODE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Treasures Collected: {treasures_collected}")
    print(f"Walls Hit: {walls_hit}")
    print(f"Final Score: {env.score}")
    print(f"Final Lives: {env.lifelines}")
    print(f"Visited Cells: {len(env.visited_cells)}")
    print(f"Exploration Rate: {len(env.visited_cells)/(env.rows*env.cols)*100:.1f}%")
    print(f"{'='*60}\n")

def test_all_levels(quick=True):
    """Test all difficulty levels."""
    levels = ["easy", "medium", "hard"]
    
    print(f"\n{'#'*60}")
    print("TESTING ALL DIFFICULTY LEVELS")
    print(f"{'#'*60}\n")
    
    for level in levels:
        try:
            env = test_environment_basic(level)
            
            if not quick:
                test_random_episode(env, max_steps=50, render_delay=20)
            
            env.close()
            print(f"✓ {level.upper()} level test completed\n")
            
        except Exception as e:
            print(f"✗ Error testing {level} level: {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print("ALL LEVEL TESTS COMPLETED")
    print(f"{'#'*60}\n")

def interactive_test():
    """Interactive test mode with menu."""
    pygame.init()
    
    while True:
        print("\n" + "="*60)
        print("TREASURE HUNT ENVIRONMENT TEST MENU")
        print("="*60)
        print("\n1. Test Easy Level")
        print("2. Test Medium Level")
        print("3. Test Hard Level")
        print("4. Test All Levels (Quick)")
        print("5. Run Random Episode on Medium")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            env = test_environment_basic("easy")
            test_random_episode(env, max_steps=100, render_delay=50)
            env.close()
        elif choice == "2":
            env = test_environment_basic("medium")
            test_random_episode(env, max_steps=150, render_delay=30)
            env.close()
        elif choice == "3":
            env = test_environment_basic("hard")
            test_random_episode(env, max_steps=200, render_delay=20)
            env.close()
        elif choice == "4":
            test_all_levels(quick=True)
        elif choice == "5":
            env = test_environment_basic("medium")
            test_random_episode(env, max_steps=300, render_delay=20)
            env.close()
        elif choice == "6":
            print("\nExiting... Goodbye! 👋\n")
            break
        else:
            print("\n⚠️ Invalid choice. Please enter 1-6.")
    
    pygame.quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Treasure Hunt Environment')
    parser.add_argument('--level', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level to test')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum steps for random episode')
    parser.add_argument('--delay', type=int, default=50,
                       help='Render delay in milliseconds')
    parser.add_argument('--all-levels', action='store_true',
                       help='Test all difficulty levels')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive menu mode')
    
    args = parser.parse_args()
    
    pygame.init()
    
    try:
        if args.interactive:
            interactive_test()
        elif args.all_levels:
            test_all_levels(quick=False)
        else:
            # Test single level
            env = test_environment_basic(args.level)
            test_random_episode(env, max_steps=args.steps, render_delay=args.delay)
            env.close()
            pygame.quit()
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user. Exiting...")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)

