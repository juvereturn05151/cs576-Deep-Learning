"""
File Name:    environment.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np


class VacuumEnvironment:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
        4: (0, 0),   # clean
    }

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.robot_position = (0, 0)
        self.dirty_tiles = self._create_fixed_dirty_tiles(grid_size)
        self.max_steps = grid_size * grid_size * 4
        self.steps_taken = 0

    def _create_fixed_dirty_tiles(self, grid_size: int) -> set[tuple[int, int]]:
        presets = {
            5: {(1, 1), (2, 3), (4, 2)},
            10: {(1, 1), (2, 6), (3, 8), (5, 4), (6, 1), (7, 7)},
            15: {(1, 2), (2, 12), (4, 6), (6, 10), (7, 3), (9, 13), (10, 5), (12, 8), (13, 1), (14, 11)},
        }
        return set(presets.get(grid_size, set()))

    def reset(self) -> np.ndarray:
        self.robot_position = (0, 0)
        self.dirty_tiles = self._create_fixed_dirty_tiles(self.grid_size)
        self.steps_taken = 0
        return self.get_state_vector()

    def set_grid_size(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.max_steps = grid_size * grid_size * 4
        self.reset()

    def get_state_vector(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for row, col in self.dirty_tiles:
            grid[row, col] = 1.0

        robot_row, robot_col = self.robot_position
        grid[robot_row, robot_col] = 2.0
        return grid.flatten()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.steps_taken += 1
        reward = -0.1
        robot_row, robot_col = self.robot_position

        if action == 4:
            if self.robot_position in self.dirty_tiles:
                self.dirty_tiles.remove(self.robot_position)
                reward = 10.0
            else:
                reward = -1.0
        else:
            delta_row, delta_col = self.ACTIONS[action]
            next_row = robot_row + delta_row
            next_col = robot_col + delta_col

            if 0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size:
                self.robot_position = (next_row, next_col)
            else:
                reward = -2.0

        done = len(self.dirty_tiles) == 0 or self.steps_taken >= self.max_steps
        if len(self.dirty_tiles) == 0:
            reward += 20.0

        return self.get_state_vector(), reward, done

    def action_count(self) -> int:
        return len(self.ACTIONS)