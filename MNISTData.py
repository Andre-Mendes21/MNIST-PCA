import struct
import numpy as np
import matplotlib.pyplot as plt


def loadMNIST(images_path, labels_path):
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def showMNIST(X, y, n_samples):
    images = X[:n_samples]
    labels = y[:n_samples]

    num_row = 2
    num_col = n_samples//2
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    for i in range(n_samples):
        ax = axes[i//num_col, i % num_col]
        ax.imshow(images[i].reshape(28, 28), cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()
