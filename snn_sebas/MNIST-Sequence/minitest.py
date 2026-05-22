import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

repo = Path(r"C:\Users\Usuario\Desktop\cvc\pytorchV0\MNIST-Sequence")
sys.path.insert(0, str(repo))

from models import CNNBaseline, CRNNBaseline, SNN

model = SNN()
state_dict = torch.load(repo / "dataset" / "snn_baseline.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

data = np.load(repo / "dataset" / "mnist_sequence.npz")


for i in range(5):
    x_np = data["inputs"][i]
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        pred = model(x).argmax(dim=-1).squeeze(0).tolist()

    true = data["labels"][i].tolist()
    print("pred:", pred, "true:", true)

    plt.imshow(x_np, cmap="gray")
    plt.title(f"pred={''.join(map(str, pred))} true={''.join(map(str, true))}")
    plt.axis("off")
    plt.show()