"""
Resample all hiragana trajectory files to a common length.
This ensures the NEST code can process them without errors.
"""

import numpy as np
import os

dataset_path = "/home/gerardasbert/Desktop/Master/TFM/snn_marc/input_characters"
target_length = 200  # resample all to this many points (use the longest)

# First, find the maximum length
max_length = 0
for fname in os.listdir(dataset_path):
    if fname.endswith(".txt"):
        data = np.loadtxt(os.path.join(dataset_path, fname))
        max_length = max(max_length, data.shape[0])

print(f"Max trajectory length: {max_length}")
target_length = max_length

# Resample all files
for fname in os.listdir(dataset_path):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(dataset_path, fname)
    data = np.loadtxt(path)
    n_points = data.shape[0]

    if n_points == target_length:
        print(f"{fname:15s}: {n_points:3d} points (no change)")
        continue

    # Interpolate to target length
    x_old = np.linspace(0, 1, n_points)
    x_new = np.linspace(0, 1, target_length)

    # Interpolate each column
    data_resampled = np.zeros((target_length, data.shape[1]))
    for col in range(data.shape[1]):
        data_resampled[:, col] = np.interp(x_new, x_old, data[:, col])

    # Save back
    np.savetxt(path, data_resampled, fmt="%.4f")
    print(f"{fname:15s}: {n_points:3d} → {target_length:3d} points")

print(f"\n✓ All files resampled to {target_length} points")
