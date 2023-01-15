import numpy as np
from skimage import util


def add_noise(X, seed=None):
    noisy_images = []
    for i in range(len(X)):
        noisy = util.random_noise(X[i], mode='gaussian', seed=seed,
                                  clip=True, var=0.035)
        noisy = np.floor(np.clip(noisy * 255, 0, 255))
        noisy_images.append(noisy)
    return noisy_images
