# 🧭 From Curiosity to Discovery: A Reinforcement Learning Approach to Treasure Hunting

---

## 🏆 Project Overview

**From Curiosity to Discovery** is an interactive AI project that demonstrates how a reinforcement learning (RL) agent learns to survive and succeed in a **Treasure Hunt environment** filled with static and dynamic traps, lifelines, and hidden treasures.  

Built using **Gymnasium** and **Pygame**, this simulation bridges **game design** and **artificial intelligence** — showing how agents can adapt in uncertain and partially observable worlds.

---

## 🎮 Environment Description

The environment is a **2D grid world** where the agent explores to collect treasures while avoiding traps and managing lifelines.

| Element | Description | Reward / Penalty |
|----------|--------------|------------------|
| 🧱 **Wall** | Blocks movement | –2 |
| 🟫 **Road** | Safe path (pitch brown) | –1 per move |
| 💎 **Treasure (♢)** | Increases score | +30 |
| 🎁 **Final Treasure (♢)** | Increases score | +70 |
| ☠️ **Static Trap (⊗)** | Old hidden traps, 50% chance of life loss | –6 |
| ⚡ **Dynamic Trap (Hunters)** | Appears randomly; always decreases life | –12 |
| ❤️ **Lifeline (♡)** | Restores 1 life | +5 |
| 🧍 **Agent** | Learns via RL | — |

- **Visibility:** limited to 4 cells; walls block view.  
- **Dynamic traps** appear every few steps to simulate unpredictable danger.  
- The agent starts with **3 lifelines** and can gain up to **5**.  

---

## 🧠 Algorithms Implemented

| Algorithm | Type | Description |
|------------|------|--------------|
| **DQ-N** | - | Learns from expected future rewards (exploitative) |

Algorithms are trained and evaluated under the same environment for performance comparison.

---

## ⚙️ Project Structure


TreasureHuntRL/
│
├── env/                                # 🌍 Environment Module
│   ├── treasure_env.py                 # Main Pygame + Gymnasium environment
│   ├── map_layouts/                    # Different maps or levels
│   │   ├── map_easy.json
│   │   ├── map_medium.json
│   │   └── map_hard.json
│   ├── assets/                         # Game icons and visuals
│   │   ├── wall.png
│   │   ├── road.png
│   │   ├── treasure.png
│   │   ├── trap_static.png
│   │   ├── trap_dynamic.png
│   │   ├── heart.png
│   │   └── agent.png
│   └── __init__.py
│
├── agents/                             # 🧠 RL Agents
│   ├── dqn_agent.py                   # Only agent
│   ├── base_agent.py                   # Common utilities (optional)
│   └── __init__.py
│
├── training/                           # ⚙️ Training Scripts
│   ├── train_dqn.py
│   └── hyperparams.json                # Tunable parameters (alpha, gamma, epsilon)
│
├── analysis/                           # 📊 Result Analysis
│   ├── evaluation.py                  # Graphs and metrics comparison
│   ├── logs/                           # Episode logs (rewards, steps, lifelines)
│   │   ├── dqn_training_hard.csv
│   │   ├── dqn_training_medium.csv
│   │   └── dqn_training_easy.csv
│   └── plots/                          # Generated plots
│       ├── rewards_vs_episodes.png
│       ├── steps_vs_episodes.png
│       └── traps_vs_treasures.png
│
├── test_env_run.py
├── manual_run.py
├── test_dqn.py                         # 🚀 Central launcher for the project
├── requirements.txt                    # Dependencies (Gymnasium, Pygame, Numpy, Matplotlib)
├── README.md                           # Project overview + usage guide
└── .gitignore                          # To ignore unnecessary files in repo


---

