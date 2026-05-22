import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset/mnist_sequence_neuromorphic.npz", allow_pickle=True)

events = data["events"]
offsets = data["offsets"]
labels = data["labels"]

i = 0
start = offsets[i]
end = offsets[i + 1]
sample = events[start:end]
label = labels[i]

height = int(data["height"])
width = int(data["width"])
img = np.zeros((height, width))

for e in sample:
    t, y, x, p = e
    img[int(y), int(x)] += 1  

plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.colorbar()
plt.show()