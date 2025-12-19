# ============================================================
# Asset Loader for Treasure Hunt Environment
# Loads all graphical elements used by the environment.
# ============================================================

import os
import pygame

def load_assets(tile_size=48):
    """
    Loads and scales all game assets (images). 
    Falls back to colored surfaces if files are missing.
    """
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    assets = {}

    def load_image(name, fallback_color):
        path = os.path.join(asset_dir, name)
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (tile_size, tile_size))
        else:
            surf = pygame.Surface((tile_size, tile_size))
            surf.fill(fallback_color)
            return surf

    # Load icons / placeholders
    assets["road"] = load_image("road.png", (60, 40, 20))
    assets["wall"] = load_image("wall.png", (80, 80, 80))
    assets["treasure"] = load_image("treasure.png", (255, 215, 0))
    assets["static_trap"] = load_image("trap_static.png", (200, 0, 0))
    assets["dynamic_trap"] = load_image("trap_dynamic.png", (255, 80, 0))
    assets["lifeline"] = load_image("heart.png", (255, 0, 100))
    assets["final_treasure"] = load_image("final_treasure.png", (255, 255, 0))
    assets["agent"] = load_image("agent.png", (0, 200, 255))

    return assets
