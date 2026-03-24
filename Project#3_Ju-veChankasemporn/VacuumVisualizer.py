"""
File Name:    VacuumVisualizer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import pygame
import sys

from VacuumEnvironment import (
    VacuumEnvironment,
    CELL_SIZE,
    GRID_ROWS,
    GRID_COLS,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    WHITE,
    BLACK,
    GRAY,
    LIGHT_GRAY,
    GREEN,
    RED,
    BLUE,
    YELLOW
)

class VacuumVisualizer:
    def __init__(self, env: VacuumEnvironment):
        pygame.init()
        pygame.display.set_caption("Vacuum Cleaner Environment")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.small_font = pygame.font.SysFont(None, 24)

        self.env = env
        self.running = True

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.env.reset()
                elif event.key == pygame.K_UP:
                    self.env.move_up()
                elif event.key == pygame.K_DOWN:
                    self.env.move_down()
                elif event.key == pygame.K_LEFT:
                    self.env.move_left()
                elif event.key == pygame.K_RIGHT:
                    self.env.move_right()
                elif event.key == pygame.K_SPACE:
                    self.env.suck()

    def draw_grid(self):
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x = col * CELL_SIZE
                y = row * CELL_SIZE

                # Tile background
                tile_color = LIGHT_GRAY
                pygame.draw.rect(self.screen, tile_color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 2)

                # Dirt
                if self.env.is_dirty(row, col):
                    dirt_margin = 25
                    pygame.draw.circle(
                        self.screen,
                        RED,
                        (x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                        CELL_SIZE // 4
                    )
                    dirt_text = self.small_font.render("DIRT", True, WHITE)
                    dirt_rect = dirt_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    self.screen.blit(dirt_text, dirt_rect)
                else:
                    clean_text = self.small_font.render("Clean", True, GREEN)
                    clean_rect = clean_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    self.screen.blit(clean_text, clean_rect)

    def draw_agent(self):
        x = self.env.agent_col * CELL_SIZE
        y = self.env.agent_row * CELL_SIZE

        agent_margin = 15
        agent_rect = pygame.Rect(
            x + agent_margin,
            y + agent_margin,
            CELL_SIZE - 2 * agent_margin,
            CELL_SIZE - 2 * agent_margin
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect, border_radius=12)

        eye_radius = 5
        pygame.draw.circle(self.screen, WHITE, (x + 35, y + 35), eye_radius)
        pygame.draw.circle(self.screen, WHITE, (x + 65, y + 35), eye_radius)

        mouth_rect = pygame.Rect(x + 35, y + 55, 30, 10)
        pygame.draw.arc(self.screen, WHITE, mouth_rect, 3.14, 0, 2)

        label = self.small_font.render("VAC", True, WHITE)
        label_rect = label.get_rect(center=(x + CELL_SIZE // 2, y + 78))
        self.screen.blit(label, label_rect)

    def draw_ui(self):
        ui_y = GRID_ROWS * CELL_SIZE
        pygame.draw.rect(self.screen, BLACK, (0, ui_y, WINDOW_WIDTH, 60))

        dirty_count = sum(sum(row) for row in self.env.grid)

        status_text = (
            f"Steps: {self.env.steps}   "
            f"Cleaned: {self.env.cleaned_tiles}   "
            f"Remaining Dirt: {dirty_count}"
        )
        text_surface = self.font.render(status_text, True, WHITE)
        self.screen.blit(text_surface, (15, ui_y + 10))

        help_text = "Arrows = Move | SPACE = Suck | R = Reset | ESC = Quit"
        help_surface = self.small_font.render(help_text, True, YELLOW)
        self.screen.blit(help_surface, (15, ui_y + 35))

    def draw_win_message(self):
        if self.env.all_clean():
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))

            msg = self.font.render("All tiles are clean!", True, WHITE)
            msg_rect = msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 15))
            self.screen.blit(msg, msg_rect)

            msg2 = self.small_font.render("Press R to reset", True, YELLOW)
            msg2_rect = msg2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            self.screen.blit(msg2, msg2_rect)

    def render(self):
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_agent()
        self.draw_ui()
        self.draw_win_message()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_input()
            self.render()

        pygame.quit()
        sys.exit()
