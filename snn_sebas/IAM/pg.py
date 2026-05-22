import os

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, load_from_disk


cache_dir = "./IAM/iam_local"

if os.path.exists(cache_dir):
    ds = load_from_disk(cache_dir)
else:
    ds = load_dataset(
        "Teklia/IAM-line",
        split="train",
        download_mode="reuse_dataset_if_exists",
        verification_mode="no_checks"
    )
    
    ds.save_to_disk(cache_dir)

#ds[0] es la primera muestra
#tiene las keys "image" y "text"

#la convierto en numpy pa manejarla mejor
#shape[0] siempre es 128 pero shape[1] siempre varía
#esto quiere decir que siempre tienen la misma altura pero el ancho puede cambiar

#img = np.array(ds[100]["image"])
#print(img.shape)



max_idx = max(range(len(ds)), key=lambda i: np.array(ds[i]["image"]).shape[1])
min_idx = min(range(len(ds)), key=lambda i: np.array(ds[i]["image"]).shape[1])

img1 = np.array(ds[min_idx]["image"])
img2 = np.array(ds[max_idx]["image"])

"""plt.subplot(2, 1, 1)
plt.imshow(img1, cmap="gray")
plt.title(img1.shape)
plt.axis("off")"""

#plt.subplot(2,1,2)
plt.imshow(img2, cmap="gray")
plt.title(img2.shape)
plt.axis("off")

plt.show()




