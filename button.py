import pygame


class Button:
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.idle_color = (92,76,164)
        self.clicked_color = (147,139,197)
        self.clicked = False


    def draw(self):
        pygame.draw.rect(self.screen, self.clicked_color if self.clicked else self.idle_color,
                         (self.x, self.y, self.width, self.height),
                         border_radius=12)