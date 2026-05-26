import os
import shutil
from pathlib import Path


def parse_stroke_line(line):
    """
    Parse a stroke line like:
    '13,120;14,122;15,126;'
    into [(13,120), (14,122), (15,126)]
    """
    points = []

    chunks = line.strip().split(";")

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        try:
            x_str, y_str = chunk.split(",")
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue

    return points


def normalize_coordinates(all_points):
    """
    Normalize coordinates to roughly [-1, 1] scale.
    Uses global bounding box.
    """
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    scale = max(width, height)

    normalized = []

    for x, y in all_points:
        nx = (x - min_x) / scale - 0.5
        ny = (y - min_y) / scale - 0.5

        normalized.append((nx, ny))

    return normalized


def convert_file(input_file, output_file):
    """
    Convert one TXT file from absolute stroke format
    to relative coordinate + pen state format.
    """

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 2:
        print(f"Skipping empty/invalid file: {input_file}")
        return

    # First line becomes suffix in filename
    header = lines[0]

    strokes = []

    for line in lines[1:]:
        stroke = parse_stroke_line(line)

        if len(stroke) > 0:
            strokes.append(stroke)

    if not strokes:
        print(f"No valid strokes in: {input_file}")
        return

    # Flatten all points for normalization
    all_points = [p for stroke in strokes for p in stroke]

    normalized_points = normalize_coordinates(all_points)

    # Rebuild normalized strokes
    idx = 0
    normalized_strokes = []

    for stroke in strokes:
        nstroke = normalized_points[idx: idx + len(stroke)]
        normalized_strokes.append(nstroke)
        idx += len(stroke)

    output_lines = []

    prev_x = None
    prev_y = None

    for stroke in normalized_strokes:

        for i, (x, y) in enumerate(stroke):

            if prev_x is None:
                dx = 0.0
                dy = 0.0
            else:
                dx = x - prev_x
                dy = y - prev_y

            # Pen state:
            # 1.0 = pen down
            # 0.0 = last point of stroke (pen up)
            pen = 1.0

            if i == len(stroke) - 1:
                pen = 0.0

            output_lines.append(f"{dx:.4f} {dy:.4f} {pen:.4f}")

            prev_x = x
            prev_y = y

    # Write transformed file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))


def sanitize_filename(name):
    """
    Remove problematic filename characters.
    """
    invalid = '<>:"/\\|?*'

    for ch in invalid:
        name = name.replace(ch, "_")

    return name


def process_directory(input_dir, output_dir):
    """
    Recursively process all TXT files while preserving structure.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for root, dirs, files in os.walk(input_dir):

        root_path = Path(root)

        # Relative path from input root
        rel_path = root_path.relative_to(input_dir)

        # Create corresponding output directory
        target_dir = output_dir / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:

            input_file = root_path / file

            # Copy non-txt files directly
            if input_file.suffix.lower() != ".txt":
                shutil.copy2(input_file, target_dir / file)
                continue

            try:
                # Read first line for filename suffix
                with open(input_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                suffix = sanitize_filename(first_line)

                new_filename = (
                    input_file.stem + "_" + suffix + ".txt"
                )

                output_file = target_dir / new_filename

                convert_file(input_file, output_file)

                print(f"Converted: {input_file} -> {output_file}")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":

    INPUT_DIR = "/data/113-2/users/gasbert/HOMUS"
    OUTPUT_DIR = "/data/113-2/users/gasbert/HOMUS_PROCESSED"

    process_directory(INPUT_DIR, OUTPUT_DIR)

    print("Done.")