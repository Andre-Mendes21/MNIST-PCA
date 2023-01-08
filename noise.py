import numpy as np


def add_noise(X, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mu, sigma = 0, 1
    noisy_images = []
    noise = np.random.normal(mu, sigma, size=X[0].shape)
    for i in range(len(X)):
        noisy = np.clip((X[i] + noise * 0.2), 0, 1)
        noisy_images.append(noisy)
    return noisy_images
