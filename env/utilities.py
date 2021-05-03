import pygame

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)

def pygame_text(x,y,text,display_surface):
    font = pygame.font.Font('freesansbold.ttf', 18)
    text = font.render(text, True, green, blue)
    text_rect = text.get_rect()
    text_rect.center = (x, y)
    display_surface.blit(text, text_rect)

class actions(object):
    ACCELERATE = 1
    DECELERATE = -1
    STEER_LEFT = 1
    STEER_RIGHT = -1
    REMAIN = 0
    def __init__(self):
        self.longitudinal = self.REMAIN
        self.lateral = self.REMAIN


