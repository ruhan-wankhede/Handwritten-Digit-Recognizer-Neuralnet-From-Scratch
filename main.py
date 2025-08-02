import pygame
import numpy as np
import model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from button import Button

pygame.init()

WIDTH, HEIGHT = 784, 784

WIN = pygame.display.set_mode((WIDTH + 400, HEIGHT))
pygame.display.set_caption("Handwritten Digit Recognizer")

BUTTON_WIDTH, BUTTON_HEIGHT = 200, 60

predict_button = Button(WIN, WIDTH + 100, 100, BUTTON_WIDTH, BUTTON_HEIGHT)
clear_button = Button(WIN, WIDTH + 100, 200, BUTTON_WIDTH, BUTTON_HEIGHT)

prediction_text = "Prediction: ?"

preview_surface = None

grid = [[0 for i in range(28)] for j in range(28)]

def render_prediction_preview(img: np.ndarray) -> pygame.Surface:
    """
        Show user the digit as fed to the AI
    """
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    fig.tight_layout(pad=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()

    # Convert to pygame surface
    surf = pygame.image.frombuffer(raw_data, size, "RGBA").convert_alpha()

    plt.close(fig)
    return surf


def clear_grid():
    global grid
    grid = [[0 for i in range(28)] for j in range(28)]

def draw_gridlines():
    """
    draw the gridlines
    """
    for i in range(29):
        pygame.draw.line(WIN, (0, 0, 0), (i * 28, 0), (i * 28, HEIGHT))
        pygame.draw.line(WIN, (0, 0, 0), (0, i * 28), (WIDTH, i * 28))

def draw_pixels():
    """
    draw the pixels of the digit
    """
    for x in range(28):
        for y in range(28):
            if grid[y][x] == 255:
                pygame.draw.rect(WIN, (0, 0, 0), (x * 28, y * 28, 28, 28))


def update_pixels(mouse_x: float, mouse_y: float):
    """
    update the grid
    """
    if not int(mouse_y // 28) > 27 and not int(mouse_x // 28) > 27:
        grid[int(mouse_y // 28)][int(mouse_x // 28)] = 255


def draw_ui():
    """
    Draw UI to the right of the user canvas
    """
    # UI Panel
    pygame.draw.rect(WIN, (240, 240, 240), (WIDTH, 0, 400, HEIGHT))

    # Predict Button
    font = pygame.font.SysFont("Roboto", 36)

    predict_button.draw()
    text = font.render("Predict", True, (255, 255, 255))
    WIN.blit(text, (predict_button.x + 56, predict_button.y + 15))

    # Clear Button
    clear_button.draw()
    text = font.render("Clear", True, (255, 255, 255))
    WIN.blit(text, (clear_button.x + 65, clear_button.y + 16))

    # Show image fed to AI
    if preview_surface:
        WIN.blit(preview_surface, (WIDTH + 100, 500))

        # Preview text
        prev_surface = font.render("Image fed to AI", True, (40, 40, 40))
        WIN.blit(prev_surface, (WIDTH + 97, 750))

    # Prediction text
    font = pygame.font.SysFont("Roboto", 44)
    pred_surface = font.render(prediction_text, True, (40, 40, 40))
    WIN.blit(pred_surface, (WIDTH + 100, 350))



    pygame.display.flip()


def blur(canvas: np.ndarray) -> np.ndarray:
    """
    add box blur to replicate antialiasing in MNIST dataset
    """

    # the value of a certain pixel will be the average of all the pixels in a 3x3 square
    blurred = np.zeros((28, 28))
    for y in range(1, 27):
        for x in range(1, 27):
            region = np.array(canvas[y - 1:y + 2, x - 1:x + 2])
            blurred[y][x] = np.mean(region)
    return blurred


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
    img = Image.fromarray(arr.astype(np.uint8))
    resized = np.array(img.resize((20, 20), resample=Image.BILINEAR))

    # center resized digit on a 28 x 28 canvas
    canvas = np.zeros((28, 28), dtype=digit.dtype)
    y_offset = (28 - 20) // 2
    x_offset = (28 - 20) // 2
    canvas[y_offset: y_offset + 20, x_offset: x_offset + 20] = resized
    canvas = blur(canvas)
    # Normalize, Flatten and return the formatted grid
    return (canvas / 255.0).flatten()


def guess():
    global grid
    formatted_grid = preprocess_grid(grid)

    try:
        with open("model.json", "r") as f:
            net = model.load_model("model.json")


    except FileNotFoundError:
        font = pygame.font.SysFont("Roboto", 36)
        text = font.render("Training...", True, (0, 0, 0))
        WIN.blit(text, (WIDTH + 130, 50))
        pygame.display.flip()
        model.train_model("model.json")
        net = model.load_model("model.json")
    pygame.draw.rect(WIN, (240, 240, 240), (WIDTH + 100, 100, 50, 10))
    pred = net.predict(formatted_grid)

    global prediction_text
    prediction_text = f"Prediction: {pred}"

    global preview_surface
    canvas_img = formatted_grid.reshape(28, 28) * 255  # convert back to 2D image
    preview_surface = render_prediction_preview(canvas_img)



def main():
    global preview_surface
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
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if predict_button.x <= mouse_x <= predict_button.x + predict_button.width and predict_button.y <= mouse_y <= predict_button.y + predict_button.height:
                    predict_button.clicked = True
                    predict_button.draw()
                elif clear_button.x <= mouse_x <= clear_button.x + clear_button.width and clear_button.y <= mouse_y <= clear_button.y + clear_button.height:
                    clear_button.clicked = True
                    clear_button.draw()
                elif 0 <= mouse_x <= 784 and 0 <= mouse_y <= 784:
                    drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                if predict_button.clicked:
                    guess()
                    predict_button.clicked = False

                if clear_button.clicked:
                    clear_grid()
                    global prediction_text
                    prediction_text = "Prediction: ?"
                    clear_button.clicked = False
                    preview_surface = None

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
        draw_ui()
        pygame.display.flip()

if __name__ == "__main__":

    main()