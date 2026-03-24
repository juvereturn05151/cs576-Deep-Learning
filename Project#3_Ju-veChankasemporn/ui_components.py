"""
File Name:    ui_components.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from dataclasses import dataclass

import pygame

from config import BUTTON_COLOR, BUTTON_DISABLED_COLOR, BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR, PANEL_BORDER


@dataclass
class Button:
    rect: pygame.Rect
    text: str
    enabled: bool = True
    hovered: bool = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            color = BUTTON_DISABLED_COLOR
        elif self.hovered:
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR

        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=2, border_radius=10)

        text_surface = font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)
