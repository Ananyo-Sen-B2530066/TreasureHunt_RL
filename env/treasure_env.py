# ============================================================
# Project Title   : Gamma Force — From Curiosity to Discovery
# File Name       : treasure_env.py
# Description     : Reinforcement Learning-based Treasure Hunt Environment
# Author(s)       : Team Gamma Force
# Team Members    : Ananyo Sen (Leader, Env Designer),
#                   Arkadip Kansabanik (RL Implementation),
#                   Subhajit Paul (Analysis & Report)
# Date Created    : November 2025
# Last Modified   : —
# ============================================================
# Notes:
#   - This module implements the Treasure Hunt Environment using
#     Gymnasium + Pygame.
#   - The environment supports multiple difficulty levels (easy,
#     medium, hard) with dynamic trap spawning and partial observability.
# ============================================================


import os
import json
import random
import pygame
import gymnasium as gym
import numpy as np

# ==========================
# CONSTANT DEFINITIONS
# ==========================
ROAD = 0
WALL = 1
TREASURE = 2
STATIC_TRAP = 3
LIFELINE = 4
DYNAMIC_TRAP = 5
FINAL_TREASURE = 6


class TreasureHuntEnv(gym.Env):
    """
    Treasure Hunt RL Environment
    ----------------------------
    A grid-based treasure hunt game environment where the agent collects treasures,
    avoids traps, and manages lifelines. Built with Gymnasium + Pygame.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, level="easy", spawn_corner="bottom-left"):
        super().__init__()
        self.level = level.lower()

        # ==========================
        # DIFFICULTY SETTINGS
        # ==========================
        if self.level == "easy":
            self.spawn_interval = 12     # traps appear slower
        elif self.level == "medium":
            self.spawn_interval = 9      # balanced
        elif self.level == "hard":
            self.spawn_interval = 7      # frequent traps
        else:
            raise ValueError("Invalid level! Choose from ['easy', 'medium', 'hard']")

        # ==========================
        # CORE INITIALIZATION
        # ==========================
        # --- Dynamic Tile Size Based on Map Level ---
        if level == "easy":
           self.tile_size = 48       # fits 10x10 nicely
        elif level == "medium":
            self.tile_size = 36       # fits 16x16 maps
        elif level == "hard":
            self.tile_size = 24       # fits 30x30 maps
        else:
            self.tile_size = 32       # fallback
            
        # --- Load Map and Get Grid Size ---
        self.map = self.load_map(level)
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.dynamic_traps = []
        
        # --- Final treasure setup ---
        self.final_treasure_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map[r][c] == FINAL_TREASURE:
                    self.final_treasure_pos = (r, c)
                    self.map[r][c] = ROAD   # hide initially
        self.final_treasure_unlocked = False
        
        # --- Dynamic trap setup ---
        self.active_traps = []
        if self.level == "easy":
            self.spawn_interval = 12
            self.trap_move_limit = 5
            self.max_dynamic_traps = 1
        elif self.level == "medium":
            self.spawn_interval = 9
            self.trap_move_limit = 7
            self.max_dynamic_traps = 2
        else:  # hard
            self.spawn_interval = 5
            self.trap_move_limit = 12
            self.max_dynamic_traps = 4

        # Find initial spawn (any safe cell)
        self.agent_pos = self._find_start_position()
        
        # --- Initialize Gameplay Stats ---
        self.score = 0
        self.lifelines = 3  # starting number of lives
        self.steps = 0
        self.last_min_distance = None
        self.wall_hit_count = 0
        self.exploration_rate = 1.0
        
        # --- Auto Resize to Fit Screen if Needed ---
        max_width, max_height = 1280, 720  # safe defaults for most laptops
        
        window_width = self.cols * self.tile_size
        window_height = self.rows * self.tile_size
        
        if window_width > max_width or window_height > max_height:
            scale_w = max_width / window_width
            scale_h = max_height / window_height
            scale_factor = min(scale_w, scale_h)
            self.tile_size = int(self.tile_size * scale_factor)
            window_width = int(self.cols * self.tile_size)
            window_height = int(self.rows * self.tile_size)
            
            
        # ==========================
        # GYMNASIUM SPACE DEFINITIONS
        # ==========================
        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        
        # Observation space: agent position (row, col), score, lives
        self.observation_space = gym.spaces.Dict({
            "local_view": gym.spaces.Box(
                low=0, high=6, shape=(25,), dtype=np.int32
            ),
            "agent_pos": gym.spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self.rows - 1, self.cols - 1]), 
                dtype=np.int32
            ),
            "score": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
            "lives": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
            "treasures_left": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "hint_direction": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # New: hint vector
        })


        # ==========================
        # PYGAME INITIALIZATION
        # ==========================
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Treasure Hunt RL — {level.title()} Level")
        self.clock = pygame.time.Clock()

        # ==========================
        # LOAD GRAPHICS
        # ==========================
        # --- Load Assets (from asset_loader.py) ---
        self.assets = self.load_assets(tile_size=self.tile_size)

    # ============================================================
    # MAP LOADING
    # ============================================================
    def load_map(self, level):
        """Load the JSON map layout for the chosen difficulty."""
        map_path = os.path.join(os.path.dirname(__file__), "map_layouts", f"map_{level}.json")
        print("Looking for map at:", map_path)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found for level '{level}'")
        with open(map_path, "r") as f:
            grid = json.load(f)
        return grid

    # ============================================================
    # ASSET LOADING
    # ============================================================
    def load_assets(self, tile_size):
        """Load all game assets or create fallback colors if missing."""
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        assets = {}
    
        def load_image(name, fallback_color):
            path = os.path.join(asset_dir, name)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, (tile_size, tile_size))
            else:
                # fallback colored surface
                surf = pygame.Surface((tile_size, tile_size))
                surf.fill(fallback_color)
                return surf
    
        # Use CONSTANTS as keys, not strings
        assets[ROAD] = load_image("road.png", (60, 40, 20))
        assets[WALL] = load_image("wall.png", (80, 80, 80))
        assets[TREASURE] = load_image("treasure.png", (255, 215, 0))
        assets[STATIC_TRAP] = load_image("trap_static.png", (200, 0, 0))
        assets[DYNAMIC_TRAP] = load_image("trap_dynamic.png", (255, 80, 0))
        assets[LIFELINE] = load_image("heart.png", (255, 0, 100))
        assets[FINAL_TREASURE] = load_image("final_treasure.png", (255, 255, 0))
        assets["agent"] = load_image("agent.png", (0, 200, 255))  # keep this separate
    
        print("[Assets] Loaded successfully from:", asset_dir)
        return assets


    # ============================================================
    # HINT ENCODING
    # ============================================================
    def _encode_hint(self, hint_str):
        """Convert hint string to numerical vector."""
        encoding = {
            "N": [1, 0, 0, 0],   # North
            "S": [0, 1, 0, 0],   # South  
            "E": [0, 0, 1, 0],   # East
            "W": [0, 0, 0, 1],   # West
            "NE": [1, 0, 1, 0],  # North-East
            "NW": [1, 0, 0, 1],  # North-West
            "SE": [0, 1, 1, 0],  # South-East
            "SW": [0, 1, 0, 1],  # South-West
            "Here!": [0.5, 0.5, 0.5, 0.5],  # On treasure
            "No treasure left!": [0, 0, 0, 0]
        }
        return encoding.get(hint_str, [0, 0, 0, 0])


    # ============================================================
    # START POSITION FINDER
    # ============================================================
    def _find_start_position(self):
        """Find the first available road starting from the chosen corner."""
        if getattr(self, "spawn_corner", "bottom-left") == "top-left":
           row_range = range(self.rows)  # top to bottom
        else:
           row_range = range(self.rows - 1, -1, -1)  # bottom to top

        for r in row_range:
           for c in range(self.cols):
              if self.map[r][c] == ROAD:
                return (r, c)
        raise ValueError("No valid starting position found in the map.")

    # ============================================================
    # RESET FUNCTION
    # ============================================================
    def reset(self, *, seed=None, options=None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        self.map = self.load_map(self.level)
        self.agent_pos = self._find_start_position()
        self.lifelines = 3
        self.score = 0
        self.steps = 0
        self.active_traps.clear()
        self.wall_hit_count = 0
        self.exploration_rate = 1.0
    
        # --- Initialize tracking variables ---
        self.visited_cells = set()
        self.visited_cells.add(self.agent_pos)  # Add starting position
        self.last_min_distance = None  # Initialize for distance tracking
        
        # --- Hide the final treasure again ---
        self.final_treasure_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map[r][c] == FINAL_TREASURE:
                    self.final_treasure_pos = (r, c)
                    self.map[r][c] = ROAD
        self.final_treasure_unlocked = False
    
        obs = self.get_observation()
        return obs, {}
    
    # ============================================================
    # HINT FUNCTION
    # ============================================================
    def get_hint(self):
        """Return direction of nearest uncollected treasure."""
        treasures = [
            (r, c) for r in range(self.rows)
            for c in range(self.cols)
            if self.map[r][c] == TREASURE or (self.final_treasure_unlocked and self.map[r][c] == FINAL_TREASURE)
        ]
        if not treasures:
            return "No treasure left!"
    
        ax, ay = self.agent_pos
        nearest = min(treasures, key=lambda t: abs(t[0]-ax) + abs(t[1]-ay))
        dx, dy = nearest[0] - ax, nearest[1] - ay
    
        # Convert (dx, dy) into direction string
        dir_x = "N" if dx < 0 else ("S" if dx > 0 else "")
        dir_y = "W" if dy < 0 else ("E" if dy > 0 else "")
        return dir_x + dir_y if (dir_x + dir_y) else "Here!"
    
    # ============================================================
    # DISTANCE REWARD CALCULATION
    # ============================================================
    def calculate_distance_reward(self):
        """Calculate reward based on distance to nearest treasure with bonus for progress."""
        agent_r, agent_c = self.agent_pos
        
        # Find all treasures
        treasures = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map[r][c] == TREASURE:
                    treasures.append((r, c))
        
        if not treasures:
            return 0
        
        # Calculate distances to all treasures
        distances = [abs(tr - agent_r) + abs(tc - agent_c) for tr, tc in treasures]
        min_distance = min(distances)
        
        if self.last_min_distance is None:
            self.last_min_distance = min_distance
            return 0
        
        # Calculate improvement
        improvement = self.last_min_distance - min_distance
        self.last_min_distance = min_distance
        
        # Enhanced reward calculation
        if improvement > 0:
            # Bigger reward for getting closer
            return improvement * 2.0 + 0.5  # Base reward + bonus
        elif improvement < 0:
            # Smaller penalty for moving away (might be exploring)
            return improvement * 0.2
        else:
            # Small penalty for no progress
            return -0.1
        
        
    def _calculate_trap_danger(self):
        """Calculate danger level from nearby traps (0 = safe, 1 = high danger)."""
        agent_r, agent_c = self.agent_pos
        danger_scores = []
        
        # Check for traps in 3x3 immediate vicinity
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r, c = agent_r + dr, agent_c + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    cell = self.map[r][c]
                    if cell == STATIC_TRAP:
                        distance = max(abs(dr), abs(dc))
                        danger_scores.append(0.5 / distance)  # Static trap danger
                    elif cell == DYNAMIC_TRAP:
                        distance = max(abs(dr), abs(dc))
                        danger_scores.append(1.0 / distance)  # Dynamic trap MORE dangerous
        
        # Return aggregated danger (max of all trap dangers)
        return [min(1.0, sum(danger_scores))] if danger_scores else [0.0]
    
    
    def is_cornered(self, agent_pos=None):
        """Check if agent is cornered with limited safe escape routes."""
        if agent_pos is None:
            agent_pos = self.agent_pos
        
        agent_r, agent_c = agent_pos
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        safe_moves = 0
        for dr, dc in moves:
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cell = self.map[nr][nc]
                # Count as safe if it's road, treasure, lifeline, or final treasure
                if cell in (ROAD, TREASURE, LIFELINE, FINAL_TREASURE):
                    safe_moves += 1
        
        # Cornered if 1 or fewer safe moves
        return safe_moves <= 1
    
    
    def get_best_cornered_action(self, agent_pos=None):
        """When cornered, choose least dangerous move."""
        if agent_pos is None:
            agent_pos = self.agent_pos
            
        agent_r, agent_c = agent_pos
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        move_dangers = []
        for i, (dr, dc) in enumerate(moves):
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cell = self.map[nr][nc]
                # Calculate danger score
                if cell == WALL:
                    danger = 100  # Can't move here
                elif cell == DYNAMIC_TRAP:
                    danger = 80   # Most dangerous
                elif cell == STATIC_TRAP:
                    danger = 50   # Less dangerous
                elif cell == ROAD:
                    danger = 0    # Safe
                elif cell == TREASURE:
                    danger = -10  # Actually beneficial
                elif cell == LIFELINE:
                    danger = -20  # Very beneficial
                elif cell == FINAL_TREASURE:
                    danger = -50  # Win condition
                else:
                    danger = 10   # Unknown, slightly risky
                move_dangers.append((i, danger))
            else:
                move_dangers.append((i, 100))  # Out of bounds
        
        # Choose move with lowest danger
        best_action, _ = min(move_dangers, key=lambda x: x[1])
        return best_action

    # ============================================================
    # UPDATED STEP FUNCTION
    # ============================================================
    def step(self, action):
        """Performs one step in the environment based on the agent's action."""
        self.last_action = action
        self.steps += 1
        reward = 0
        done = False
        truncated = False
        info = {}
        
        # Initialize info with tracking
        info["treasure_collected"] = False
        info["agent_row"], info["agent_col"] = self.agent_pos
        info["walls_hit"] = 0
        info["treasures_remaining"] = sum(row.count(TREASURE) for row in self.map)
        info["new_cell_visited"] = False
        old_danger = self._calculate_trap_danger()[0]
        
        # Track visited cells for exploration reward
        if not hasattr(self, 'visited_cells'):
            self.visited_cells = set()
        
        # Track current position
        current_pos = self.agent_pos
        
        # --- Movement Directions ---
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        dr, dc = moves.get(action, (0, 0))
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc
        new_danger = self._calculate_trap_danger()[0]
        
        if new_danger < old_danger:
            danger_reduction = old_danger - new_danger
            reward += danger_reduction * 10.0  # Significant reward for moving away
        elif new_danger > old_danger:
            danger_increase = new_danger - old_danger
            reward -= danger_increase * 15.0  # Penalty for moving closer
    
        
        # === WALL COLLISION HANDLING ===
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Out of bounds - moderate penalty
            reward -= 2
            new_r, new_c = r, c
            info["walls_hit"] = 1
            self.wall_hit_count += 1
        else:
            target_tile = self.map[new_r][new_c]
            
            if target_tile == WALL:
                # Adaptive wall penalty: decreases as agent learns
                adaptive_penalty = max(0.1, 3.0 / (1 + self.wall_hit_count * 0.1))
                reward -= adaptive_penalty
                info["walls_hit"] = 1
                self.wall_hit_count += 1
                # Don't move into wall
                new_r, new_c = r, c
                
            elif target_tile == ROAD:
                # Check if this is a new cell
                if (new_r, new_c) not in self.visited_cells:
                    info["new_cell_visited"] = True
                    self.visited_cells.add((new_r, new_c))
                    reward += 2.0
                    if len(self.visited_cells) % 5 == 0:
                        reward += 5.0
                else:
                    reward -= 0.05
                self.agent_pos = (new_r, new_c)
                
            elif target_tile == TREASURE:
                # Treasure reward with efficiency bonus
                base_reward = 50
                steps_penalty = self.steps * 0.01
                efficiency_bonus = max(0, 30 - steps_penalty)
                reward += base_reward + efficiency_bonus
                
                exploration_bonus = len(self.visited_cells) * 0.1
                reward += exploration_bonus
                
                self.score += 50
                self.map[new_r][new_c] = ROAD
                self.agent_pos = (new_r, new_c)
                
                remaining_treasures = sum(row.count(TREASURE) for row in self.map)
                info["treasure_collected"] = True
                info["treasures_remaining"] = remaining_treasures
    
            elif target_tile == FINAL_TREASURE:
                reward += 100
                self.score += 100
                done = True
                self.agent_pos = (new_r, new_c)
                info["reason"] = "All treasures collected - Victory!"
    
            elif target_tile == LIFELINE:
                reward += 25
                self.lifelines += 1
                self.map[new_r][new_c] = ROAD
                self.agent_pos = (new_r, new_c)
    
            elif target_tile == STATIC_TRAP:
                reward -= 50
                self.score -= 20
                
                self.map[new_r][new_c] = ROAD
                self.agent_pos = (new_r, new_c)
                
                # Difficulty-based life loss probability
                if self.level == "easy":
                    lose_life = random.random() < 0.3
                elif self.level == "medium":
                    lose_life = random.random() < 0.5
                else:  # hard
                    lose_life = random.random() < 0.7
                    
                if lose_life:
                    self.lifelines -= 1
                    reward -= 30  # Additional penalty for losing life
                    print("💥 Lost a life from static trap!")
                    if self.lifelines <= 0:
                        done = True
                        reward -= 100  # Game over penalty
                        info["reason"] = "Game Over - No lives left"
                else:
                    print("😮 Escaped static trap safely!")
    
            elif target_tile == DYNAMIC_TRAP:
                reward -= 80
                self.score -= 10
                self.lifelines -= 1
                reward -= 40  # Additional penalty for losing life
                self.map[new_r][new_c] = ROAD
                info["lives_lost"] = 1
    
                # Clear all dynamic traps (trap reset)
                for trap in self.active_traps:
                    tr, tc = trap["pos"]
                    self.map[tr][tc] = ROAD
                self.active_traps.clear()
    
                print("💀 Agent caught by dynamic trap! All traps cleared.")
    
                if self.lifelines <= 0:
                    done = True
                    reward -= 100  # Game over penalty
                    info["reason"] = "Game Over - Caught by dynamic trap"
                else:
                    self.agent_pos = (new_r, new_c)
                    
        if not done:
            next_positions = []
            for move_action in range(4):
                dr, dc = moves.get(move_action, (0, 0))
                nr, nc = new_r + dr, new_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    next_positions.append((nr, nc, self.map[nr][nc]))
            
            # Check if any adjacent cell is a trap
            adjacent_trap_count = sum(1 for _, _, cell in next_positions 
                                      if cell in (STATIC_TRAP, DYNAMIC_TRAP))
            
            if adjacent_trap_count > 0:
                # Small penalty for being near traps
                reward -= adjacent_trap_count * 2.0
        
        # === CRITICAL BUG FIX: Check if agent is on same cell as dynamic trap ===
        # This happens when trap moves onto agent's position
        agent_r, agent_c = self.agent_pos
        for trap in self.active_traps[:]:  # Use [:] to iterate over copy
            trap_r, trap_c = trap["pos"]
            if (agent_r, agent_c) == (trap_r, trap_c):
                # Agent caught by dynamic trap!
                reward -= 20
                self.score -= 15
                self.lifelines -= 1
                
                print("💀 Agent caught by dynamic trap! All traps cleared.")
                
                # Clear all dynamic traps
                for t in self.active_traps:
                    tr, tc = t["pos"]
                    if (tr, tc) != (agent_r, agent_c):
                        self.map[tr][tc] = ROAD
                self.active_traps.clear()
                
                # Clear current cell
                self.map[agent_r][agent_c] = ROAD
                
                if self.lifelines <= 0:
                    done = True
                    reward -= 100  # Game over penalty
                    info["reason"] = "Game Over - Caught by dynamic trap"
                
                break  # Only process one trap collision per step
        
        # --- Dynamic Trap Spawning & Movement ---
        if self.steps % self.spawn_interval == 0:
            self.spawn_dynamic_trap()
        self.move_dynamic_traps()
        
        # === CHECK AGAIN AFTER TRAP MOVEMENT ===
        # Traps might move onto agent after movement
        agent_r, agent_c = self.agent_pos
        for trap in self.active_traps[:]:
            trap_r, trap_c = trap["pos"]
            if (agent_r, agent_c) == (trap_r, trap_c):
                # Agent caught after trap movement!
                reward -= 20
                self.score -= 15
                self.lifelines -= 1
                
                print("💀 Agent caught by dynamic trap! All traps cleared.")
                
                # Clear all dynamic traps
                for t in self.active_traps:
                    tr, tc = t["pos"]
                    if (tr, tc) != (agent_r, agent_c):
                        self.map[tr][tc] = ROAD
                self.active_traps.clear()
                
                self.map[agent_r][agent_c] = ROAD
                
                if self.lifelines <= 0:
                    done = True
                    reward -= 100  # Game over penalty
                    info["reason"] = "Game Over - Caught by moving trap"
                
                break
        
        # --- Reveal Final Treasure ---
        if not self.final_treasure_unlocked:
            remaining = sum(row.count(TREASURE) for row in self.map)
            if remaining == 0 and self.final_treasure_pos:
                r, c = self.final_treasure_pos
                self.map[r][c] = FINAL_TREASURE
                self.final_treasure_unlocked = True
                print("✨ Final Treasure Revealed!")
                
        # Calculate distance-based reward
        distance_reward = self.calculate_distance_reward()
        reward += distance_reward
    
        # --- Check Lifelines ---
        if self.lifelines <= 0 and not done:
            done = True
            info["reason"] = "No lifelines left - Game Over"
    
        # --- Build Observation ---
        obs = self.get_observation()
        return obs, reward, done, truncated, info
    
    # ============================================================
    # DYNAMIC TRAP FUNCTION
    # ============================================================
    def spawn_dynamic_trap(self):
        """Spawn a dynamic trap on a random road tile if below max limit."""
    
        if len(self.active_traps) >= self.max_dynamic_traps:
            return
    
        road_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.map[r][c] == ROAD
        ]
        if not road_cells:
            return
    
        r, c = random.choice(road_cells)
        self.map[r][c] = DYNAMIC_TRAP
        self.active_traps.append({
            "pos": (r, c),
            "remaining_moves": self.trap_move_limit
        })
        
    # ============================================================
    # MOVE DYNAMIC TRAP FUNCTION
    # ============================================================    
    def move_dynamic_traps(self):
        """Dynamic traps that chase the agent (smartest on hard level)."""
        new_active = []
    
        for trap in self.active_traps:
            if trap["remaining_moves"] <= 0:
                # Trap disappears after move limit
                r, c = trap["pos"]
                self.map[r][c] = ROAD
                continue
    
            r, c = trap["pos"]
            ar, ac = self.agent_pos
    
            # --- HARD MODE: Predictive Movement ---
            if self.level == "hard":
                # Predict agent's next likely move direction
                pr, pc = ar, ac
                if self.last_action == 0:  # up
                    pr -= 1
                elif self.last_action == 1:  # down
                    pr += 1
                elif self.last_action == 2:  # left
                    pc -= 1
                elif self.last_action == 3:  # right
                    pc += 1
    
                # Clamp prediction within map boundaries
                pr = max(0, min(pr, self.rows - 1))
                pc = max(0, min(pc, self.cols - 1))
    
                # Target the predicted position
                target_r, target_c = pr, pc
            else:
                # Normal chase behavior
                target_r, target_c = ar, ac
    
            # --- Compute desired direction (toward target) ---
            dir_r = 1 if target_r > r else -1 if target_r < r else 0
            dir_c = 1 if target_c > c else -1 if target_c < c else 0
    
            # --- Primary & Secondary move preferences ---
            possible_moves = []
            if abs(target_r - r) > abs(target_c - c):
                possible_moves = [(dir_r, 0), (0, dir_c)]
            else:
                possible_moves = [(0, dir_c), (dir_r, 0)]
    
            # Add fallback directions for smoothness
            possible_moves += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
            moved = False
            for dr, dc in possible_moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.map[nr][nc] in (ROAD, LIFELINE):
                        # Move to new tile
                        self.map[r][c] = ROAD
                        self.map[nr][nc] = DYNAMIC_TRAP
                        trap["pos"] = (nr, nc)
                        trap["remaining_moves"] -= 1
                        moved = True
                        break
    
            # If trapped, reduce lifetime but stay
            if not moved:
                trap["remaining_moves"] -= 1
    
            new_active.append(trap)
    
        self.active_traps = new_active

      
    # ============================================================
    # MANUAL OBSERVE FUNCTION
    # ============================================================    
    def get_obs(self):
        """Return a minimal observation (agent position, score, lives)."""
        return {
            "agent_pos": tuple(self.agent_pos),
            "score": self.score,
            "lives": self.lifelines
        }

    # ============================================================
    # RENDER FUNCTION
    # ============================================================
    def render(self):
        """Render Treasure Hunt environment with fog and line-of-sight treasure visibility."""
        if self.screen is None:
            return
    
        self.screen.fill((0, 0, 0))
        VISION_RADIUS = 4
        agent_r, agent_c = self.agent_pos
    
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                cell = self.map[r][c]
                dist = abs(r - agent_r) + abs(c - agent_c)
    
                # --- Visibility Logic ---
                visible = True
                # 1. Outside vision radius → dark fog
                if dist > VISION_RADIUS:
                    visible = False
    
                # 2. Treasure / Trap hidden behind walls
                if cell in (TREASURE, FINAL_TREASURE, STATIC_TRAP):
                    # only show if within range and clear path exists
                    if dist <= VISION_RADIUS and self._has_line_of_sight(agent_r, agent_c, r, c):
                        visible = True
                    else:
                        visible = False
    
                # --- Draw the Tile ---
                if cell in self.assets:
                    # Always draw walls, roads, hearts even under fog
                    if cell in (ROAD, WALL, LIFELINE):
                        self.screen.blit(self.assets[cell], rect)
                    elif visible:
                        self.screen.blit(self.assets[cell], rect)
                    else:
                        # Hide treasure/trap as road
                        self.screen.blit(self.assets[ROAD], rect)
                else:
                    # fallback for missing assets
                    color = (100, 100, 100) if cell == ROAD else (50, 50, 50)
                    pygame.draw.rect(self.screen, color, rect)
    
                # Darken far-away regions slightly (fog layer)
                if dist > VISION_RADIUS:
                    s = pygame.Surface((self.tile_size, self.tile_size))
                    s.set_alpha(160)
                    s.fill((20, 20, 20))
                    self.screen.blit(s, rect.topleft)
    
        # --- Draw Agent ---
        agent_rect = pygame.Rect(
            agent_c * self.tile_size,
            agent_r * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        if "agent" in self.assets:
            self.screen.blit(self.assets["agent"], agent_rect)
        else:
            pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

    
        # --- Display Hint ---
        hint = self.get_hint()
        font = pygame.font.SysFont(None, 28)
        text_surface = font.render(f"Hint: {hint}", True, (255, 255, 0))
        self.screen.blit(text_surface, (10, 10))
    
        # --- HUD ---
        hud_font = pygame.font.SysFont(None, 24)
        # --- HUD (Heads-Up Display) ---
        remaining_treasures = sum(row.count(TREASURE) for row in self.map)
        hud_text = hud_font.render(f"Score: {self.score}   Lives: {self.lifelines}   Treasures Left: {remaining_treasures}", True, (255, 255, 255))
        self.screen.blit(hud_text, (10, 35))
    
        # Display wall hit count for debugging
        debug_text = hud_font.render(f"Wall Hits: {self.wall_hit_count}   Visited Cells: {len(self.visited_cells)}", True, (200, 200, 200))
        self.screen.blit(debug_text, (10, 60))
    
        pygame.display.flip()
        
        
    def _has_line_of_sight(self, ar, ac, tr, tc):
        """Check if a clear path (no wall) exists in a straight line between agent and target."""
        if ar == tr:  # same row
            step = 1 if tc > ac else -1
            for c in range(ac + step, tc, step):
                if self.map[ar][c] == WALL:
                    return False
            return True
        elif ac == tc:  # same column
            step = 1 if tr > ar else -1
            for r in range(ar + step, tr, step):
                if self.map[r][ac] == WALL:
                    return False
            return True
        else:
            # diagonal or offset line of sight = blocked
            return False

    # ============================================================
    # UPDATED OBSERVATION FUNCTION
    # ============================================================
    def get_observation(self):
        """Enhanced observation with trap awareness."""
        vision_radius = 2
        local_view = []
        
        agent_r, agent_c = self.agent_pos
        
        # Build local view
        for dr in range(-vision_radius, vision_radius + 1):
            row = []
            for dc in range(-vision_radius, vision_radius + 1):
                r, c = agent_r + dr, agent_c + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    row.append(self.map[r][c])
                else:
                    row.append(WALL)
            local_view.append(row)
        
        flattened = np.array(local_view).flatten()
        
        # NEW: Calculate trap proximity features
        trap_danger = self._calculate_trap_danger()
        
        # Count remaining treasures
        remaining_treasures = sum(row.count(TREASURE) for row in self.map)
        
        # Get and encode hint
        hint_str = self.get_hint()
        hint_encoded = self._encode_hint(hint_str)
        
        # NEW: Calculate trap danger
        trap_danger = self._calculate_trap_danger()
        
        return {
            "local_view": flattened,
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "score": np.array(self.score, dtype=np.int32),
            "lives": np.array(self.lifelines, dtype=np.int32),
            "treasures_left": np.array([remaining_treasures], dtype=np.int32),
            "hint_direction": np.array(hint_encoded, dtype=np.float32),
            "trap_danger": np.array(trap_danger, dtype=np.float32)  # NEW
        }

    # ============================================================
    # CLOSE FUNCTION
    # ============================================================
    def close(self):
        pygame.quit()