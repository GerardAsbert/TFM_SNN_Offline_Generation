import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


def load_stroke_file(file_path):
    """
    Load a txt file with format:
    dx dy pen_state
    """
    data = np.loadtxt(file_path)

    dx = data[:, 0]
    dy = data[:, 1]
    pen = data[:, 2]

    return dx, dy, pen


def reconstruct_points(dx, dy):
    """
    Convert relative movements into absolute coordinates.
    """
    x = [0]
    y = [0]

    for i in range(len(dx)):
        x.append(x[-1] + dx[i])
        y.append(y[-1] + dy[i])

    return np.array(x), np.array(y)


def draw_symbol(x, y, pen):
    """
    Draw the traced symbol.
    pen == 1 -> pen down (draw)
    pen == 0 -> pen up (move without drawing)
    """

    plt.figure(figsize=(6, 6))

    for i in range(len(pen)):
        if pen[i] == 1:
            plt.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                'k-',
                linewidth=2
            )

    plt.gca().invert_yaxis()   # Optional: handwriting-style coordinates
    plt.axis('equal')
    plt.axis('off')

    plt.show()


def main():
    # Open file picker
    Tk().withdraw()

    file_path = filedialog.askopenfilename(
        title="Select stroke txt file",
        filetypes=[("Text files", "*.txt")]
    )

    if not file_path:
        print("No file selected.")
        return

    dx, dy, pen = load_stroke_file(file_path)

    x, y = reconstruct_points(dx, dy)

    draw_symbol(x, y, pen)


if __name__ == "__main__":
    main()