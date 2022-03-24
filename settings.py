import logging
from pathlib import Path

from fallbacks import pygame

# Game properties
COLS = 17
ROWS = 17
SCENARIOS = {
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    # This is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.5,
        "COIN_COUNT": 25
    }, 
    "clean": {
        "CRATE_DENSITY": 0.2,
        "COIN_COUNT": 0
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 40 #400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3 # explosion extends three tiles up, down, left and right
BOMB_TIMER = 4 # detonate after four steps
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around # remains dangerous for one more round before it vanishes in smoke

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.INFO
LOG_AGENT_WRAPPER = logging.INFO
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
