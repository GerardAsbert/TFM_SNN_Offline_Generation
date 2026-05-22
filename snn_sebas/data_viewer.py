import numpy as np
import matplotlib.pyplot as plt

def load_nmnist_file(path):

    data = np.fromfile(path, dtype=np.uint8)

    data = data.reshape(-1, 5)

    x = data[:, 0]
    y = data[:, 1]
    polarity = (data[:, 2] >> 7) & 1
    t = ((data[:, 2] & 0x7F) << 16) | (data[:, 3] << 8) | data[:, 4]

    return x, y, t, polarity

def events_to_image_fast(x, y, polarity, size=(34, 34)):
    img = np.zeros(size)

    values = np.where(polarity == 1, 1, -1)
    np.add.at(img, (y, x), values)

    return img

def plot_image(img):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.title("Reconstructed NMNIST image")
    plt.show()


file_path = "468j46mzdv-1/Train/0/00002.bin" 

x, y, t, p = load_nmnist_file(file_path)
img = events_to_image_fast(x, y, p)

plot_image(img)