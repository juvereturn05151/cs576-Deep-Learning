"""
File Name:    app.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import sys
from enum import Enum

import pygame

from config import (
    AVAILABLE_GRID_SIZES,
    BACKGROUND_COLOR,
    DEFAULT_GRID_SIZE,
    DIRTY_CELL_COLOR,
    EMPTY_CELL_COLOR,
    FPS,
    GRID_LINE_COLOR,
    GRID_PANEL_HEIGHT,
    GRID_PANEL_WIDTH,
    MARGIN,
    PANEL_BORDER,
    PANEL_COLOR,
    RIGHT_PANEL_WIDTH,
    ROBOT_COLOR,
    SUBTEXT_COLOR,
    TEXT_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

# Support both the packaged project layout and flat local files.
try:
    from model.dqn_wrapper import DQNWrapper, DQNConfig
except ImportError:
    from model.dqn_wrapper import DQNWrapper, DQNConfig

try:
    from app.ui_components import Button
except ImportError:
    from ui_components import Button

try:
    from vacuum_environment.environment import VacuumEnvironment
except ImportError:
    from vacuum_environment.environment import VacuumEnvironment


class AppMode(Enum):
    IDLE = "Idle"
    TRAINING = "Training"
    DEPLOYED = "Deployed"


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Vacuum Cleaner DQN Environment")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("arial", 30, bold=True)
        self.heading_font = pygame.font.SysFont("arial", 22, bold=True)
        self.body_font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 16)

        self.environment = VacuumEnvironment(DEFAULT_GRID_SIZE)
        self.mode = AppMode.IDLE
        self.status_message = "Ready. Select a grid size and train a model."
        self.last_training_summary = ""
        self.trainers: dict[int, DQNWrapper] = {}
        self.training_histories: dict[int, dict] = {}
        self.deployment_timer_ms = 0
        self.deployment_interval_ms = 250

        self.grid_origin_x = MARGIN
        self.grid_origin_y = MARGIN
        self.grid_area_size = min(GRID_PANEL_WIDTH, GRID_PANEL_HEIGHT)

        self.buttons: dict[str, Button] = {}
        self._create_buttons()
        self._refresh_button_states()

    def _create_buttons(self) -> None:
        panel_x = WINDOW_WIDTH - RIGHT_PANEL_WIDTH - MARGIN
        current_y = MARGIN + 340
        button_width = RIGHT_PANEL_WIDTH
        button_height = 48
        spacing = 14

        for size in AVAILABLE_GRID_SIZES:
            self.buttons[f"grid_{size}"] = Button(
                rect=pygame.Rect(panel_x, current_y, button_width, button_height),
                text=f"Use {size} x {size} Grid",
            )
            current_y += button_height + spacing

        current_y += 16

        self.buttons["train"] = Button(
            rect=pygame.Rect(panel_x, current_y, button_width, 52),
            text="Train Model",
        )
        current_y += 52 + spacing

        self.buttons["deploy"] = Button(
            rect=pygame.Rect(panel_x, current_y, button_width, 52),
            text="Deploy Model",
        )
        current_y += 52 + spacing

        self.buttons["reset"] = Button(
            rect=pygame.Rect(panel_x, current_y, button_width, 48),
            text="Reset Environment",
        )

    def _refresh_button_states(self) -> None:
        model_ready = self.environment.grid_size in self.trainers
        busy = self.mode == AppMode.TRAINING

        self.buttons["deploy"].enabled = model_ready and not busy
        self.buttons["train"].enabled = not busy
        self.buttons["reset"].enabled = not busy

        for size in AVAILABLE_GRID_SIZES:
            self.buttons[f"grid_{size}"].enabled = not busy

    def set_grid_size(self, size: int) -> None:
        self.environment.set_grid_size(size)
        self.mode = AppMode.IDLE
        self.status_message = f"Environment updated to fixed {size} x {size} map."
        self.last_training_summary = self._build_training_summary(size)
        self._refresh_button_states()

    def reset_environment(self) -> None:
        self.environment.reset()
        self.mode = AppMode.IDLE
        self.status_message = (
            f"Environment reset for {self.environment.grid_size} x {self.environment.grid_size}."
        )
        self._refresh_button_states()

    def handle_click(self, pos: tuple[int, int]) -> None:
        for key, button in self.buttons.items():
            if button.enabled and button.contains(pos):
                if key.startswith("grid_"):
                    size = int(key.split("_")[1])
                    self.set_grid_size(size)
                elif key == "train":
                    self.train_model()
                elif key == "deploy":
                    self.deploy_model()
                elif key == "reset":
                    self.reset_environment()
                break

        self._refresh_button_states()

    def update_hover_states(self, mouse_pos: tuple[int, int]) -> None:
        for button in self.buttons.values():
            button.hovered = button.enabled and button.contains(mouse_pos)

    def draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_grid_panel()
        self._draw_right_panel()
        pygame.display.flip()

    def _draw_grid_panel(self) -> None:
        grid_rect = pygame.Rect(
            self.grid_origin_x,
            self.grid_origin_y,
            self.grid_area_size,
            self.grid_area_size,
        )
        pygame.draw.rect(self.screen, PANEL_COLOR, grid_rect, border_radius=14)
        pygame.draw.rect(self.screen, PANEL_BORDER, grid_rect, width=2, border_radius=14)

        cell_size = self.grid_area_size / self.environment.grid_size

        for row in range(self.environment.grid_size):
            for col in range(self.environment.grid_size):
                x = self.grid_origin_x + col * cell_size
                y = self.grid_origin_y + row * cell_size
                cell_rect = pygame.Rect(round(x), round(y), round(cell_size), round(cell_size))
                color = DIRTY_CELL_COLOR if (row, col) in self.environment.dirty_tiles else EMPTY_CELL_COLOR
                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, GRID_LINE_COLOR, cell_rect, width=1)

        robot_row, robot_col = self.environment.robot_position
        robot_center_x = self.grid_origin_x + robot_col * cell_size + cell_size / 2
        robot_center_y = self.grid_origin_y + robot_row * cell_size + cell_size / 2
        robot_radius = max(10, int(cell_size * 0.25))
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            (round(robot_center_x), round(robot_center_y)),
            robot_radius,
        )

        legend_y = self.grid_origin_y + self.grid_area_size + 12
        self._draw_legend(legend_y)

    def _draw_legend(self, y: int) -> None:
        items = [
            (ROBOT_COLOR, "Robot start / current position"),
            (DIRTY_CELL_COLOR, "Fixed dirty tile"),
            (EMPTY_CELL_COLOR, "Clean tile"),
        ]

        x = self.grid_origin_x
        for color, label in items:
            swatch = pygame.Rect(x, y, 18, 18)
            pygame.draw.rect(self.screen, color, swatch, border_radius=4)
            pygame.draw.rect(self.screen, PANEL_BORDER, swatch, width=1, border_radius=4)
            label_surface = self.small_font.render(label, True, SUBTEXT_COLOR)
            self.screen.blit(label_surface, (x + 26, y))
            x += 210

    def _draw_right_panel(self) -> None:
        panel_x = WINDOW_WIDTH - RIGHT_PANEL_WIDTH - MARGIN
        panel_rect = pygame.Rect(panel_x, MARGIN, RIGHT_PANEL_WIDTH, WINDOW_HEIGHT - MARGIN * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect, border_radius=14)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=2, border_radius=14)

        self._draw_info_block(panel_x + 18, MARGIN + 10)

        for button in self.buttons.values():
            button.draw(self.screen, self.body_font)

        status_box = pygame.Rect(panel_x + 18, 250, RIGHT_PANEL_WIDTH - 36, 100)
        pygame.draw.rect(self.screen, (38, 42, 53), status_box, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, status_box, width=1, border_radius=12)

        status_title = self.heading_font.render("Status", True, TEXT_COLOR)
        self.screen.blit(status_title, (status_box.x + 12, status_box.y + 10))
        self._draw_wrapped_text(
            self.status_message,
            self.small_font,
            SUBTEXT_COLOR,
            pygame.Rect(status_box.x + 12, status_box.y + 40, status_box.width - 24, status_box.height - 48),
        )

    def _draw_info_block(self, x: int, y: int) -> None:
        info_box = pygame.Rect(x, y, RIGHT_PANEL_WIDTH - 36, 210)
        pygame.draw.rect(self.screen, (38, 42, 53), info_box, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, info_box, width=1, border_radius=12)

        info_title = self.heading_font.render("Environment", True, TEXT_COLOR)
        self.screen.blit(info_title, (info_box.x + 12, info_box.y + 10))

        lines = [
            f"Grid size: {self.environment.grid_size} x {self.environment.grid_size}",
            f"Robot pos: {self.environment.robot_position}",
            f"Steps: {self.environment.steps_taken}",
            f"Mode: {self.mode.value}",
            f"Model ready: {'Yes' if self.environment.grid_size in self.trainers else 'No'}",
        ]

        for index, line in enumerate(lines):
            text = self.body_font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (info_box.x + 12, info_box.y + 42 + index * 24))

        if self.last_training_summary:
            self._draw_wrapped_text(
                self.last_training_summary,
                self.small_font,
                SUBTEXT_COLOR,
                pygame.Rect(info_box.x + 12, info_box.bottom - 56, info_box.width - 24, 42),
            )

    def _draw_wrapped_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        rect: pygame.Rect,
        line_spacing: int = 4,
    ) -> None:
        words = text.split()
        lines: list[str] = []
        current_line = ""

        for word in words:
            test_line = word if current_line == "" else f"{current_line} {word}"
            if font.size(test_line)[0] <= rect.width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        y = rect.y
        for line in lines:
            rendered = font.render(line, True, color)
            self.screen.blit(rendered, (rect.x, y))
            y += font.get_height() + line_spacing
            if y > rect.bottom:
                break

    def run(self) -> None:
        running = True
        while running:
            dt_ms = self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()
            self.update_hover_states(mouse_pos)
            self.update_deployment(dt_ms)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.draw()

        pygame.quit()
        sys.exit()

    def _build_training_summary(self, grid_size: int) -> str:
        history = self.training_histories.get(grid_size)
        if not history or not history.get("rewards"):
            return ""

        window = min(20, len(history["rewards"]))
        avg_reward = sum(history["rewards"][-window:]) / window
        avg_steps = sum(history["steps"][-window:]) / window
        avg_loss = sum(history["losses"][-window:]) / max(1, window)
        final_eps = history["epsilons"][-1] if history.get("epsilons") else 0.0
        return (
            f"Avg reward: {avg_reward:.2f} | Avg steps: {avg_steps:.1f} | "
            f"Avg loss: {avg_loss:.4f} | Eps: {final_eps:.3f}"
        )

    def train_model(self) -> None:
        self.mode = AppMode.TRAINING
        self._refresh_button_states()
        self.status_message = (
            f"Training DQN for {self.environment.grid_size} x {self.environment.grid_size}..."
        )
        self.draw()

        wrapper = DQNWrapper(
            DQNConfig(
                episodes=500,
                gamma=0.99,
                lr=1e-3,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=1e-3,
                mem_size=10000,
                batch_size=64,
                target_replace=100,
            )
        )

        history = wrapper.train(self.environment)
        grid_size = self.environment.grid_size
        self.trainers[grid_size] = wrapper
        self.training_histories[grid_size] = history

        self.mode = AppMode.IDLE
        self.last_training_summary = self._build_training_summary(grid_size)
        self.status_message = (
            f"Training finished for {grid_size} x {grid_size}. {self.last_training_summary}"
        )

        self.environment.reset()
        self._refresh_button_states()

    def deploy_model(self) -> None:
        trainer = self.trainers.get(self.environment.grid_size)
        if trainer is None:
            self.status_message = "No trained model exists for this grid size yet."
            return

        self.environment.reset()
        self.mode = AppMode.DEPLOYED
        self.deployment_timer_ms = 0
        self.status_message = (
            f"Deploying trained model on {self.environment.grid_size} x {self.environment.grid_size}."
        )
        self._refresh_button_states()

    def update_deployment(self, dt_ms: int) -> None:
        if self.mode != AppMode.DEPLOYED:
            return

        trainer = self.trainers.get(self.environment.grid_size)
        if trainer is None:
            self.mode = AppMode.IDLE
            self.status_message = "Deployment stopped because no model is available."
            self._refresh_button_states()
            return

        self.deployment_timer_ms += dt_ms
        if self.deployment_timer_ms < self.deployment_interval_ms:
            return

        self.deployment_timer_ms = 0
        state = self.environment.get_state_vector()
        action = trainer.act(state)
        _, _, done = self.environment.step(action)

        if done:
            self.mode = AppMode.IDLE
            if len(self.environment.dirty_tiles) == 0:
                self.status_message = (
                    f"Deployment finished successfully in {self.environment.steps_taken} steps."
                )
            else:
                self.status_message = (
                    f"Deployment stopped after {self.environment.steps_taken} steps "
                    f"without fully cleaning the map."
                )
            self._refresh_button_states()
