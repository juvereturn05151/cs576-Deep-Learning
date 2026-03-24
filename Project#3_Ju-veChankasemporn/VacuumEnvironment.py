"""
File Name:    VacuumEnvironment.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import random

# ----------------------------
# Configuration
# ----------------------------

CELL_SIZE = 100
GRID_ROWS = 5
GRID_COLS = 5
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 60

FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (180, 180, 180)
LIGHT_GRAY = (220, 220, 220)
GREEN = (100, 220, 120)
RED = (220, 90, 90)
BLUE = (80, 140, 255)
YELLOW = (240, 220, 90)


# ----------------------------
# Environment
# ----------------------------
class VacuumEnvironment:
    def __init__(self, rows=5, cols=5, dirt_probability=0.35):
        self.rows = rows
        self.cols = cols
        self.dirt_probability = dirt_probability
        self.reset()

    def reset(self):
        self.grid = []
        for _ in range(self.rows):
            row = []
            for _ in range(self.cols):
                row.append(1 if random.random() < self.dirt_probability else 0)
            self.grid.append(row)

        self.agent_row = 0
        self.agent_col = 0

        # Ensure starting cell is visible and reasonable
        self.steps = 0
        self.cleaned_tiles = 0

    def move_up(self):
        if self.agent_row > 0:
            self.agent_row -= 1
            self.steps += 1

    def move_down(self):
        if self.agent_row < self.rows - 1:
            self.agent_row += 1
            self.steps += 1

    def move_left(self):
        if self.agent_col > 0:
            self.agent_col -= 1
            self.steps += 1

    def move_right(self):
        if self.agent_col < self.cols - 1:
            self.agent_col += 1
            self.steps += 1

    def suck(self):
        if self.grid[self.agent_row][self.agent_col] == 1:
            self.grid[self.agent_row][self.agent_col] = 0
            self.cleaned_tiles += 1
        self.steps += 1

    def is_dirty(self, row, col):
        return self.grid[row][col] == 1

    def all_clean(self):
        for row in self.grid:
            for cell in row:
                if cell == 1:
                    return False
        return True
