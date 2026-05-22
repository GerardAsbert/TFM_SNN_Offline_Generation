import os
import torch
from torch.utils.data import Dataset
import numpy as np


class NMNISTDataset(Dataset):
    def __init__(
        self,
        root_dir,
        labels_to_use=range(10),
        seq_len=100,
        cache_in_memory=False,
    ):
        self.samples = []
        self.seq_len = seq_len
        self.cache_in_memory = cache_in_memory
        self._cache = {}

        for label in labels_to_use:
            label_path = os.path.join(root_dir, str(label))
            if os.path.isdir(label_path):
                for f in sorted(os.listdir(label_path)):
                    if f.endswith(".bin"):
                        self.samples.append((os.path.join(label_path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        file_path, label = self.samples[idx]
        spike_cube = load_nmnist_as_tensor(file_path, seq_len=self.seq_len)
        sample = (spike_cube, torch.tensor(label, dtype=torch.long))
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


def load_nmnist_as_tensor(file_path, seq_len=100, n_x=34, n_y=34):
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    n_events = raw_data.size // 5
    if n_events == 0:
        return torch.zeros((seq_len, n_x * n_y * 2), dtype=torch.float32)

    raw_data = raw_data[: n_events * 5].reshape(-1, 5)
    x = raw_data[:, 0].astype(np.int64)
    y = raw_data[:, 1].astype(np.int64)
    p = (raw_data[:, 2] >> 7).astype(np.int64)

    t = ((raw_data[:, 2] & 0x7F).astype(np.int64) << 16 | 
         raw_data[:, 3].astype(np.int64) << 8 | 
         raw_data[:, 4].astype(np.int64))


    if t.max() > 0:
        t_scaled = (t / t.max() * (seq_len - 1)).astype(np.int64)
    else:
        t_scaled = t.astype(np.int64)

    neuron_idx = p * (n_x * n_y) + y * n_x + x

    feature_size = n_x * n_y * 2
    flat_index = torch.from_numpy(t_scaled * feature_size + neuron_idx)
    spike_cube = torch.zeros(seq_len * feature_size, dtype=torch.float32)
    spike_cube[flat_index] = 1.0

    spike_cube = spike_cube.view(seq_len, feature_size)
    return spike_cube
