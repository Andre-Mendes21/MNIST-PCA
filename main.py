import MNISTData
import numpy as np

if __name__ == '__main__':
    X, y = MNISTData.loadMNIST('data/training/train-images.idx3-ubyte',
                                'data/training/train-labels.idx1-ubyte')
    MNISTData.showMNIST(X, y, 14)
