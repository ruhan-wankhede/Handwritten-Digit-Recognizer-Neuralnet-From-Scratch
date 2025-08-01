import pygame
import numpy as np
import model

pygame.init()

WIDTH, HEIGHT = 784, 784

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwritten Digit Recognizer")

grid = [[0 for i in range(28)] for j in range(28)]

def clear_grid():
    global grid
    grid = [[0 for i in range(28)] for j in range(28)]

def draw_gridlines():
    """
    draw the gridlines
    """
    for i in range(28):
        pygame.draw.line(WIN, (0, 0, 0), (i * 28, 0), (i * 28, HEIGHT))
        pygame.draw.line(WIN, (0, 0, 0), (0, i * 28), (WIDTH, i * 28))

def draw_pixels():
    """
    draw the pixels of the digit
    """
    for x in range(28):
        for y in range(28):
            if grid[y][x] == 1:
                pygame.draw.rect(WIN, (0, 0, 0), (x * 28, y * 28, 28, 28))


def update_pixels(mouse_x: float, mouse_y: float):
    """
    update the grid
    """
    if not int(mouse_y // 28) > 27 and not int(mouse_x // 28) > 27:
        grid[int(mouse_y // 28)][int(mouse_x // 28)] = 1


def guess():
    global grid
    formatted_grid = np.array(grid).flatten()

    try:
        with open("model.json", "r") as f:
            net = model.load_model("model.json")
            print(net.predict(formatted_grid))
    except FileNotFoundError:
        model.train_model("model.json")
        net = model.load_model("model.json")
        print(net.predict(formatted_grid))



def main():
    run = True
    drawing = False
    clock = pygame.time.Clock()
    while run:
        clock.tick(240)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    clear_grid()
                elif event.key == pygame.K_p:
                    guess()

        WIN.fill((255, 255, 255))

        draw_gridlines()

        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            update_pixels(mouse_x, mouse_y)

        draw_pixels()
        pygame.display.flip()

if __name__ == "__main__":
    main()