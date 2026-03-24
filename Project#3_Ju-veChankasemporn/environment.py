"""
File Name:    environment.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

class VacuumEnvironment:
    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.robot_position = (0, 0)
        self.dirty_tiles = self._create_fixed_dirty_tiles(grid_size)

    def _create_fixed_dirty_tiles(self, grid_size: int) -> set[tuple[int, int]]:
        presets = {
            5: {(1, 1), (2, 3), (4, 2)},
            10: {(1, 1), (2, 6), (3, 8), (5, 4), (6, 1), (7, 7)},
            15: {(1, 2), (2, 12), (4, 6), (6, 10), (7, 3), (9, 13), (10, 5), (12, 8), (13, 1), (14, 11)},
        }
        return set(presets.get(grid_size, set()))

    def reset(self) -> None:
        self.robot_position = (0, 0)
        self.dirty_tiles = self._create_fixed_dirty_tiles(self.grid_size)

    def set_grid_size(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.reset()
