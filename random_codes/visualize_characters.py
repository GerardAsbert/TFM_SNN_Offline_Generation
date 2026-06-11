import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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


# ── batch trajectory → image conversion ───────────────────────────────────────

def trajectories_to_images(
    input_dir: str,
    output_dir: str,
    base_px: int = 128,
    line_width: int = 2,
    margin_frac: float = 0.05,
) -> None:
    """
    Recursively convert every .txt trajectory file under input_dir to a binary
    PNG image and write it to the mirrored path under output_dir.

    Image dimensions reflect each symbol's aspect ratio: the larger bounding-box
    dimension maps to base_px pixels; the shorter one is scaled proportionally,
    so a tall symbol produces a tall image.

    Parameters
    ----------
    input_dir    : root of the source tree (any depth of subdirectories)
    output_dir   : root of the destination tree (created if absent)
    base_px      : pixel size of the longer side of each image
    line_width   : stroke width in pixels
    margin_frac  : fractional padding added around the bounding box on each side
    """
    input_dir  = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    converted = skipped = 0

    for dirpath, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue

            src = os.path.join(dirpath, fname)

            # Mirror directory structure in output tree
            rel_dir = os.path.relpath(dirpath, input_dir)
            dst_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png")

            try:
                data = np.loadtxt(src)
            except Exception as exc:
                print(f"  [skip] {src}: {exc}")
                skipped += 1
                continue

            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[0] < 2 or data.shape[1] < 2:
                skipped += 1
                continue

            dx  = data[:, 0].astype(float)
            dy  = data[:, 1].astype(float)
            pen = data[:, 2].astype(float) if data.shape[1] >= 3 else np.ones(len(dx))

            # Reconstruct absolute positions; x/y have one extra point (origin)
            x = np.concatenate([[0.0], np.cumsum(dx)])
            y = np.concatenate([[0.0], np.cumsum(dy)])

            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            span_x = x_max - x_min
            span_y = y_max - y_min

            if span_x == 0 and span_y == 0:
                skipped += 1
                continue

            # Padding proportional to each axis span (fall back to a fixed amount
            # for degenerate dimensions, e.g. purely horizontal strokes)
            pad_x = span_x * margin_frac if span_x > 0 else span_y * margin_frac
            pad_y = span_y * margin_frac if span_y > 0 else span_x * margin_frac

            padded_w = span_x + 2 * pad_x
            padded_h = span_y + 2 * pad_y

            # Scale so the longer side = base_px
            scale = base_px / max(padded_w, padded_h)

            img_w = max(1, round(padded_w * scale))
            img_h = max(1, round(padded_h * scale))

            img  = Image.new("L", (img_w, img_h), color=255)
            draw = ImageDraw.Draw(img)

            def to_px(xi, yi):
                px = (xi - x_min + pad_x) * scale
                py = (yi - y_min + pad_y) * scale
                return (px, py)

            # Draw only pen-down segments
            for i in range(len(pen)):
                if pen[i] >= 0.5:
                    draw.line([to_px(x[i], y[i]), to_px(x[i + 1], y[i + 1])],
                              fill=0, width=line_width)

            # Threshold to strict binary (0 or 255)
            img.point(lambda v: 0 if v < 128 else 255).save(dst)
            converted += 1

    print(f"trajectories_to_images: {converted} converted, {skipped} skipped → {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise a single trajectory or batch-convert a directory tree to images."
    )
    subparsers = parser.add_subparsers(dest="cmd")

    # single-file viewer (original behaviour)
    subparsers.add_parser("view", help="Pick a single .txt file and display it")

    # batch conversion
    conv = subparsers.add_parser(
        "convert", help="Recursively convert .txt trajectories to binary PNGs"
    )
    conv.add_argument("input_dir",  help="Root of the source directory tree")
    conv.add_argument("output_dir", help="Root of the destination directory tree")
    conv.add_argument("--base-px",    type=int,   default=128,
                      help="Pixel size of the longer image side (default: 128)")
    conv.add_argument("--line-width", type=int,   default=2,
                      help="Stroke width in pixels (default: 2)")
    conv.add_argument("--margin",     type=float, default=0.05,
                      help="Fractional padding around bounding box (default: 0.05)")

    args = parser.parse_args()

    if args.cmd == "convert":
        trajectories_to_images(
            args.input_dir,
            args.output_dir,
            base_px=args.base_px,
            line_width=args.line_width,
            margin_frac=args.margin,
        )

    else:
        # default: interactive single-file viewer
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
