from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "mnist_sequence.npz"
NEUROMORPHIC_DATASET_PATH = (
    Path(__file__).resolve().parent / "dataset" / "mnist_sequence_neuromorphic.npz"
)


class SequentialMNISTDataset(Dataset):
    def __init__(self, path=DATASET_PATH):
        data = np.load(path)
        self.inputs = data["inputs"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).unsqueeze(0)
        y = torch.from_numpy(self.labels[idx])
        return x, y


class NeuromorphicSequentialMNISTDataset(Dataset):
    def __init__(self, path=NEUROMORPHIC_DATASET_PATH, seq_len=None, cache_in_memory=False):
        data = np.load(path)
        self.events = data["events"].astype(np.int32)
        self.offsets = data["offsets"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)
        self.height = int(data["height"])
        self.width = int(data["width"])
        self.default_seq_len = int(data["timesteps"])
        self.seq_len = self.default_seq_len if seq_len is None else seq_len
        self.cache_in_memory = cache_in_memory
        self._cache = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        sample_events = self.events[start:end]
        spike_cube = self.events_to_tensor(sample_events)
        label = torch.from_numpy(self.labels[idx])
        sample = (spike_cube, label)

        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample

    def events_to_tensor(self, events):
        feature_size = self.height * self.width * 2
        spike_cube = torch.zeros(self.seq_len * feature_size, dtype=torch.float32)
        if len(events) == 0:
            return spike_cube.view(self.seq_len, feature_size)

        t = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        x = events[:, 2].astype(np.int64)
        p = (events[:, 3] > 0).astype(np.int64)

        if t.max() > 0 and self.seq_len != self.default_seq_len:
            t = np.rint((t / t.max()) * (self.seq_len - 1)).astype(np.int64)
        else:
            t = np.clip(t, 0, self.seq_len - 1)

        neuron_idx = p * (self.height * self.width) + y * self.width + x
        flat_index = torch.from_numpy(t * feature_size + neuron_idx)
        spike_cube[flat_index] = 1.0
        return spike_cube.view(self.seq_len, feature_size)
