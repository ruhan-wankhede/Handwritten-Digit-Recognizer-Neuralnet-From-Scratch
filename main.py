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


def preprocess_grid(grid_2d: list[list]) -> np.ndarray:
    """
    preprocess the drawn image to make it similar to the MNIST dataset
    """

    # Cropping the grid using a bounding box

    arr = np.array(grid_2d)

    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)

    # in case of a blank input
    if not rows.any() or not cols.any():
        return np.zeros(784)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    digit = arr[y_min:y_max + 1, x_min:x_max + 1]

    # resizing
    src_h, src_w = digit.shape
    target_h, target_w = 20, 20
    resized = np.zeros((target_h, target_w), dtype=digit.dtype)
    
    for i in range(target_h):
        for j in range(target_w):
            src_y = int(i * src_h / target_h)
            src_x = int(j * src_w / target_w)
            resized[i, j] = digit[src_y, src_x]

    # center resized digit on a 28 x 28 canvas
    canvas = np.zeros((28, 28), dtype=digit.dtype)
    y_offset = (28 - target_h) // 2
    x_offset = (28 - target_w) // 2
    canvas[y_offset: y_offset + target_h, x_offset: x_offset + target_w] = resized

    # Normalize, Flatten and return the formatted grid
    return canvas.flatten()


def guess():
    global grid
    formatted_grid = preprocess_grid(grid)

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