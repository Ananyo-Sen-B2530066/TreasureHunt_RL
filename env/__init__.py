# ============================================================
# Project Title   : From Curiosity to Discovery
# File Name       : __init__.py
# Description     : Initializes the 'env' package for the Treasure Hunt RL Project.
# Team       : Gamma Force
# Team Members    : Ananyo Sen (Leader, Env Designer),
#                   Arkadip Kansabanik (RL Implementation),
#                   Subhajit Paul (Analysis & Report)
# Date Created    : November 2025
# ============================================================
# Notes:
#   - This package contains all environment-related modules.
#   - It includes:
#       • treasure_env.py    : The main Gymnasium + Pygame environment.
#       • asset_loader.py    : Loads graphical assets for the environment.
#       • map_layouts/       : JSON map files for all difficulty levels.
#       • assets/            : Visual icons for roads, traps, treasures, etc.
# ============================================================

from .treasure_env import TreasureHuntEnv
from .asset_loader import load_assets

__all__ = ["TreasureHuntEnv", "load_assets"]
