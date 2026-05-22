from pathlib import Path

import numpy as np

from mnist_sequence import MNIST_Sequence


SEQ_LEN = 5
NUM_SAMPLES = 10000
MIN_SPACING = 0
MAX_SPACING = 0
OUTPUT_NAME = "mnist_sequence.npz"
SEED = 42


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_path = script_dir / "dataset" / OUTPUT_NAME

    images_file = "t10k-images.idx3-ubyte"
    labels_file = "t10k-labels.idx1-ubyte"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    rng = np.random.default_rng(SEED)
    generator = MNIST_Sequence(
        path=str(data_dir),
        name_img=images_file,
        name_lbl=labels_file,
    )

    width = 28 * SEQ_LEN
    inputs = np.empty((NUM_SAMPLES, 28, width), dtype=np.float32)
    labels = np.empty((NUM_SAMPLES, SEQ_LEN), dtype=np.int64)

    for idx in range(NUM_SAMPLES):
        seq = rng.integers(0, 10, size=SEQ_LEN, endpoint=False)
        image = generator.generate_image_sequence(
            seq.tolist(),
            MIN_SPACING,
            MAX_SPACING,
            width,
        )
        inputs[idx] = image
        labels[idx] = seq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs,
        labels=labels,
    )
    print(f"Saved dataset to {output_path}")
    print(f"inputs shape: {inputs.shape}")
    print(f"labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
