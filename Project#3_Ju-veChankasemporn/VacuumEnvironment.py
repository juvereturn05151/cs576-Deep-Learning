"""
File Name:    VacuumEnvironment.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import random

FPS = 60
UI_HEIGHT = 80

# Limit window so larger grids still fit on screen better
MAX_WINDOW_WIDTH = 900
MAX_WINDOW_HEIGHT = 900

# Actions
ACTION_UP = "UP"
ACTION_DOWN = "DOWN"
ACTION_LEFT = "LEFT"
ACTION_RIGHT = "RIGHT"
ACTION_SUCK = "SUCK"

# Colors
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (180, 180, 180)
LIGHT_GRAY = (220, 220, 220)
GREEN = (100, 220, 120)
RED = (220, 90, 90)
BLUE = (80, 140, 255)
YELLOW = (240, 220, 90)
CYAN = (90, 220, 240)


class VacuumEnvironment:
    def __init__(self, rows=5, cols=5, dirt_probability=0.35):
        self.dirt_probability = dirt_probability
        self.resize(rows, cols)

    def resize(self, rows, cols):
        self.rows = rows
        self.cols = cols
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

    def apply_action(self, action):
        if action == ACTION_UP:
            self.move_up()
        elif action == ACTION_DOWN:
            self.move_down()
        elif action == ACTION_LEFT:
            self.move_left()
        elif action == ACTION_RIGHT:
            self.move_right()
        elif action == ACTION_SUCK:
            self.suck()

    def is_dirty(self, row, col):
        return self.grid[row][col] == 1

    def all_clean(self):
        for row in self.grid:
            for cell in row:
                if cell == 1:
                    return False
        return True

    def get_remaining_dirt(self):
        return sum(sum(row) for row in self.grid)

    def get_dirty_positions(self):
        dirty_positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 1:
                    dirty_positions.append((row, col))
        return dirty_positions

    def get_nearest_dirty_tile(self):
        dirty_positions = self.get_dirty_positions()
        if not dirty_positions:
            return None

        best_tile = None
        best_distance = float("inf")

        for row, col in dirty_positions:
            distance = abs(self.agent_row - row) + abs(self.agent_col - col)
            if distance < best_distance:
                best_distance = distance
                best_tile = (row, col)

        return best_tile

    def get_ai_action(self):
        if self.all_clean():
            return None

        if self.is_dirty(self.agent_row, self.agent_col):
            return ACTION_SUCK

        target = self.get_nearest_dirty_tile()
        if target is None:
            return None

        target_row, target_col = target

        row_diff = target_row - self.agent_row
        col_diff = target_col - self.agent_col

        # Greedy movement toward nearest dirty tile
        if abs(row_diff) > abs(col_diff):
            if row_diff < 0:
                return ACTION_UP
            return ACTION_DOWN

        if col_diff != 0:
            if col_diff < 0:
                return ACTION_LEFT
            return ACTION_RIGHT

        if row_diff < 0:
            return ACTION_UP
        if row_diff > 0:
            return ACTION_DOWN

        return ACTION_SUCK

    def step_ai(self):
        action = self.get_ai_action()
        if action is not None:
            self.apply_action(action)
        return action

    def get_cell_size(self):
        usable_height = MAX_WINDOW_HEIGHT - UI_HEIGHT
        cell_width = MAX_WINDOW_WIDTH // self.cols
        cell_height = usable_height // self.rows
        return max(25, min(cell_width, cell_height, 120))

    def get_window_size(self):
        cell_size = self.get_cell_size()
        width = self.cols * cell_size
        height = self.rows * cell_size + UI_HEIGHT
        return width, height