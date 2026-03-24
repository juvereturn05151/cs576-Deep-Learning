"""
File Name:    VacuumVisualizer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import sys
import pygame

from VacuumEnvironment import (
    VacuumEnvironment,
    FPS,
    UI_HEIGHT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SUCK,
    WHITE,
    BLACK,
    GRAY,
    LIGHT_GRAY,
    GREEN,
    RED,
    BLUE,
    YELLOW,
    CYAN,
)


class VacuumVisualizer:
    def __init__(self, env: VacuumEnvironment):
        pygame.init()
        pygame.display.set_caption("Vacuum Cleaner Environment")

        self.env = env
        self.running = True
        self.ai_enabled = False
        self.ai_move_delay_ms = 180
        self.last_ai_move_time = 0
        self.last_ai_action = "None"

        self.screen = None
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 30)
        self.small_font = pygame.font.SysFont(None, 24)

        self.resize_window()

    def resize_window(self):
        window_width, window_height = self.env.get_window_size()
        self.screen = pygame.display.set_mode((window_width, window_height))

    def change_grid_size(self, rows, cols):
        self.env.resize(rows, cols)
        self.ai_enabled = False
        self.last_ai_action = "None"
        self.resize_window()

    def do_ai_step(self):
        action = self.env.step_ai()
        self.last_ai_action = action if action is not None else "None"

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_r:
                    self.env.reset()
                    self.ai_enabled = False
                    self.last_ai_action = "None"

                elif event.key == pygame.K_UP:
                    self.ai_enabled = False
                    self.env.move_up()

                elif event.key == pygame.K_DOWN:
                    self.ai_enabled = False
                    self.env.move_down()

                elif event.key == pygame.K_LEFT:
                    self.ai_enabled = False
                    self.env.move_left()

                elif event.key == pygame.K_RIGHT:
                    self.ai_enabled = False
                    self.env.move_right()

                elif event.key == pygame.K_SPACE:
                    self.ai_enabled = False
                    self.env.suck()

                elif event.key == pygame.K_1:
                    self.change_grid_size(5, 5)

                elif event.key == pygame.K_2:
                    self.change_grid_size(10, 10)

                elif event.key == pygame.K_3:
                    self.change_grid_size(15, 15)

                elif event.key == pygame.K_a:
                    self.ai_enabled = not self.ai_enabled

                elif event.key == pygame.K_n:
                    self.ai_enabled = False
                    self.do_ai_step()

    def update_ai(self):
        if not self.ai_enabled:
            return

        if self.env.all_clean():
            self.ai_enabled = False
            return

        current_time = pygame.time.get_ticks()
        if current_time - self.last_ai_move_time >= self.ai_move_delay_ms:
            self.do_ai_step()
            self.last_ai_move_time = current_time

    def draw_grid(self):
        cell_size = self.env.get_cell_size()

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x = col * cell_size
                y = row * cell_size

                pygame.draw.rect(self.screen, LIGHT_GRAY, (x, y, cell_size, cell_size))
                pygame.draw.rect(self.screen, GRAY, (x, y, cell_size, cell_size), 2)

                if self.env.is_dirty(row, col):
                    pygame.draw.circle(
                        self.screen,
                        RED,
                        (x + cell_size // 2, y + cell_size // 2),
                        max(4, cell_size // 4)
                    )

                    if cell_size >= 45:
                        dirt_text = self.small_font.render("DIRT", True, WHITE)
                        dirt_rect = dirt_text.get_rect(
                            center=(x + cell_size // 2, y + cell_size // 2)
                        )
                        self.screen.blit(dirt_text, dirt_rect)
                else:
                    if cell_size >= 50:
                        clean_text = self.small_font.render("Clean", True, GREEN)
                        clean_rect = clean_text.get_rect(
                            center=(x + cell_size // 2, y + cell_size // 2)
                        )
                        self.screen.blit(clean_text, clean_rect)

    def draw_agent(self):
        cell_size = self.env.get_cell_size()

        x = self.env.agent_col * cell_size
        y = self.env.agent_row * cell_size

        agent_margin = max(4, cell_size // 6)
        border_radius = max(6, cell_size // 8)

        agent_rect = pygame.Rect(
            x + agent_margin,
            y + agent_margin,
            cell_size - 2 * agent_margin,
            cell_size - 2 * agent_margin
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect, border_radius=border_radius)

        eye_radius = max(2, cell_size // 18)
        left_eye = (x + cell_size // 3, y + cell_size // 3)
        right_eye = (x + (2 * cell_size) // 3, y + cell_size // 3)

        pygame.draw.circle(self.screen, WHITE, left_eye, eye_radius)
        pygame.draw.circle(self.screen, WHITE, right_eye, eye_radius)

        if cell_size >= 40:
            mouth_rect = pygame.Rect(
                x + cell_size // 3,
                y + cell_size // 2,
                cell_size // 3,
                max(6, cell_size // 10)
            )
            pygame.draw.arc(self.screen, WHITE, mouth_rect, 3.14, 0, 2)

        if cell_size >= 55:
            label = self.small_font.render("VAC", True, WHITE)
            label_rect = label.get_rect(center=(x + cell_size // 2, y + cell_size - 15))
            self.screen.blit(label, label_rect)

    def draw_target_hint(self):
        target = self.env.get_nearest_dirty_tile()
        if target is None:
            return

        cell_size = self.env.get_cell_size()
        target_row, target_col = target

        x = target_col * cell_size
        y = target_row * cell_size

        hint_rect = pygame.Rect(x + 3, y + 3, cell_size - 6, cell_size - 6)
        pygame.draw.rect(self.screen, CYAN, hint_rect, 3)

    def draw_ui(self):
        cell_size = self.env.get_cell_size()
        window_width, _ = self.env.get_window_size()
        ui_y = self.env.rows * cell_size

        pygame.draw.rect(self.screen, BLACK, (0, ui_y, window_width, UI_HEIGHT))

        ai_status = "ON" if self.ai_enabled else "OFF"
        status_text = (
            f"Grid: {self.env.rows}x{self.env.cols}   "
            f"Steps: {self.env.steps}   "
            f"Cleaned: {self.env.cleaned_tiles}   "
            f"Remaining Dirt: {self.env.get_remaining_dirt()}   "
            f"AI: {ai_status}"
        )
        text_surface = self.small_font.render(status_text, True, WHITE)
        self.screen.blit(text_surface, (15, ui_y + 8))

        action_text = f"Last AI Action: {self.last_ai_action}"
        action_surface = self.small_font.render(action_text, True, CYAN)
        self.screen.blit(action_surface, (15, ui_y + 30))

        help_text = (
            "Arrows=Move | SPACE=Suck | R=Reset | "
            "1=5x5 | 2=10x10 | 3=15x15 | A=Toggle AI | N=AI Step | ESC=Quit"
        )
        help_surface = self.small_font.render(help_text, True, YELLOW)
        self.screen.blit(help_surface, (15, ui_y + 52))

    def draw_win_message(self):
        if self.env.all_clean():
            window_width, window_height = self.env.get_window_size()

            overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))

            msg = self.font.render("All tiles are clean!", True, WHITE)
            msg_rect = msg.get_rect(center=(window_width // 2, window_height // 2 - 15))
            self.screen.blit(msg, msg_rect)

            msg2 = self.small_font.render("Press R to reset", True, YELLOW)
            msg2_rect = msg2.get_rect(center=(window_width // 2, window_height // 2 + 20))
            self.screen.blit(msg2, msg2_rect)

    def render(self):
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_target_hint()
        self.draw_agent()
        self.draw_ui()
        self.draw_win_message()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_input()
            self.update_ai()
            self.render()

        pygame.quit()
        sys.exit()