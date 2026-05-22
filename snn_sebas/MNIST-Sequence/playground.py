import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\Usuario\Desktop\cvc\pytorchV0\MNIST-Sequence\dataset\mnist_sequence.npz")

print(data.files)
print(data["inputs"].shape)
print(data["labels"].shape)


for i in range(1):
    x = data["inputs"][i]
    y = data["labels"][i]

    print(y)
    print(x.min(), x.max())


    plt.imshow(x, cmap="gray")
    plt.title("Label: " + "".join(map(str, y)))
    plt.axis("off")
    plt.show()