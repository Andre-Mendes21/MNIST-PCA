import struct
import noise
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


def noisy_MNIST(images_path, labels_path):
    X, y = loadMNIST(images_path, labels_path)
    noise_X = np.array(noise.add_noise(X))
    return noise_X, y


def showMNIST(X, y, n_samples):
    images = X[:n_samples]
    labels = y[:n_samples]

    num_col = 3
    num_row = n_samples // num_col + 1
    fig = plt.figure()
    for i in range(1, n_samples + 1):
        ax = fig.add_subplot(num_row, num_col, i)
        plt.imshow(images[i - 1].reshape(28, 28), cmap='gray')
        ax.set_title('Label: {}'.format(labels[i - 1]))
    plt.tight_layout()
    plt.show()


def showDigit(X, y, window_name):
    plt.figure()
    plt.imshow(X.reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.title('{} label: {}'.format(window_name, y))
