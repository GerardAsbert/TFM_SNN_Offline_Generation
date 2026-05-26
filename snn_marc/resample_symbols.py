"""
Resample all hiragana trajectory files to a common length.
This ensures the NEST code can process them without errors.
"""

import numpy as np
import os

dataset_path = "/data/113-2/users/gasbert/HOMUS_PROCESSED_mini"
target_length = 200  # resample all to this many points (use the longest)

# Collect all .txt files recursively
all_txt_files = [
    os.path.join(root, fname)
    for root, _, files in os.walk(dataset_path)
    for fname in files
    if fname.endswith(".txt")
]

# Find the maximum length across all files
max_length = 0
for path in all_txt_files:
    data = np.loadtxt(path)
    max_length = max(max_length, data.shape[0])

print(f"Max trajectory length: {max_length}")
target_length = max_length

# Resample all files
for path in all_txt_files:
    data = np.loadtxt(path)
    n_points = data.shape[0]
    rel_path = os.path.relpath(path, dataset_path)

    if n_points == target_length:
        print(f"{rel_path:40s}: {n_points:3d} points (no change)")
        continue

    x_old = np.linspace(0, 1, n_points)
    x_new = np.linspace(0, 1, target_length)

    data_resampled = np.zeros((target_length, data.shape[1]))
    for col in range(data.shape[1]):
        data_resampled[:, col] = np.interp(x_new, x_old, data[:, col])

    np.savetxt(path, data_resampled, fmt="%.4f")
    print(f"{rel_path:40s}: {n_points:3d} → {target_length:3d} points")

print(f"\n✓ All files resampled to {target_length} points")
