from pathlib import Path

import numpy as np


INPUT_PATH = Path(__file__).resolve().parent / "dataset" / "mnist_sequence.npz"
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "dataset" / "mnist_sequence_neuromorphic.npz"
)

# N-MNIST-inspired conversion using three saccades that trace an isosceles
# triangle. The paper specifies the trajectory in degrees; here we map it to a
# small pixel motion so it fits the 28x140 sequential-MNIST canvas.
PIXELS_PER_DEGREE = 6.0
CONTRAST_THRESHOLD = 0.2
TIMESTEPS_PER_SACCADE = 35
BATCH_SIZE = 256
LOG_EPS = 1e-3

SACCADE_POINTS_DEG = [
    (-0.5, 0.5),
    (0.0, -0.5),
    (0.5, 0.5),
    (-0.5, 0.5),
]


def build_integer_saccade_path():
    points_px = [
        (
            int(round(x * PIXELS_PER_DEGREE)),
            int(round(y * PIXELS_PER_DEGREE)),
        )
        for x, y in SACCADE_POINTS_DEG
    ]

    path = []
    for start, end in zip(points_px[:-1], points_px[1:]):
        xs = np.linspace(start[0], end[0], TIMESTEPS_PER_SACCADE, endpoint=False)
        ys = np.linspace(start[1], end[1], TIMESTEPS_PER_SACCADE, endpoint=False)
        for x, y in zip(xs, ys):
            path.append((int(round(x)), int(round(y))))
    return path


def shift_batch(images, dx, dy, fill_value=1.0):
    batch, height, width = images.shape
    shifted = np.full((batch, height, width), fill_value, dtype=images.dtype)

    src_x0 = max(0, dx)
    src_x1 = min(width, width + dx)
    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)

    src_y0 = max(0, dy)
    src_y1 = min(height, height + dy)
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)

    if src_x0 < src_x1 and src_y0 < src_y1:
        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, src_y0:src_y1, src_x0:src_x1]

    return shifted


def append_events(events, sample_ids, ys, xs, t, polarity):
    if sample_ids.size == 0:
        return
    block = np.empty((sample_ids.size, 4), dtype=np.int32)
    block[:, 0] = t
    block[:, 1] = ys
    block[:, 2] = xs
    block[:, 3] = polarity
    for sample_id, row in zip(sample_ids, block):
        events[sample_id].append(row)


def convert_batch(images, path):
    batch = images.shape[0]
    log_reference = np.log(np.clip(shift_batch(images, *path[0]), LOG_EPS, 1.0))
    event_lists = [[] for _ in range(batch)]

    for t, (dx, dy) in enumerate(path[1:], start=1):
        current = shift_batch(images, dx, dy)
        log_current = np.log(np.clip(current, LOG_EPS, 1.0))
        delta = log_current - log_reference

        on_mask = delta >= CONTRAST_THRESHOLD
        off_mask = delta <= -CONTRAST_THRESHOLD

        if np.any(on_mask):
            sample_ids, ys, xs = np.nonzero(on_mask)
            append_events(event_lists, sample_ids, ys, xs, t, polarity=1)
        if np.any(off_mask):
            sample_ids, ys, xs = np.nonzero(off_mask)
            append_events(event_lists, sample_ids, ys, xs, t, polarity=-1)

        log_reference = log_reference.copy()
        log_reference[on_mask] += CONTRAST_THRESHOLD
        log_reference[off_mask] -= CONTRAST_THRESHOLD

    return event_lists


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    data = np.load(INPUT_PATH)
    inputs = data["inputs"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    num_samples, height, width = inputs.shape

    path = build_integer_saccade_path()
    all_event_blocks = []
    offsets = [0]

    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        batch_events = convert_batch(inputs[start:end], path)
        for sample_events in batch_events:
            if sample_events:
                sample_array = np.stack(sample_events, axis=0)
            else:
                sample_array = np.empty((0, 4), dtype=np.int32)
            all_event_blocks.append(sample_array)
            offsets.append(offsets[-1] + len(sample_array))
        print(f"Converted samples {start:05d}..{end - 1:05d}")

    events = np.concatenate(all_event_blocks, axis=0) if all_event_blocks else np.empty((0, 4), dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    path_array = np.asarray(path, dtype=np.int32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        events=events,
        offsets=offsets,
        labels=labels,
        sensor_path=path_array,
        timesteps=np.array(len(path), dtype=np.int64),
        contrast_threshold=np.array(CONTRAST_THRESHOLD, dtype=np.float32),
        pixels_per_degree=np.array(PIXELS_PER_DEGREE, dtype=np.float32),
        height=np.array(height, dtype=np.int64),
        width=np.array(width, dtype=np.int64),
    )

    print(f"Saved neuromorphic dataset to {OUTPUT_PATH}")
    print(f"events shape: {events.shape}")
    print(f"offsets shape: {offsets.shape}")
    print(f"labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
