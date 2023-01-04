import MNISTData
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

GOAL_CONFIDENCE = 0.95


def show_eigen_vals(X_len, eigen_vals, k):
    plt.figure(figsize=(10, 10))
    t = np.arange(0, X_len, 1)
    plt.plot(t, eigen_vals, 'x')
    plt.plot(k, eigen_vals[k], 'o')
    plt.xlabel('K')
    plt.ylabel('Eigen values')
    plt.show()


def find_elbow(eigen_vals, trace, goal_confidence):
    confidence = 0.0
    k = 0
    while confidence < goal_confidence:
        confidence += eigen_vals[k] / trace
        k += 1
    return k, confidence


def calc_confidence(eigen_vals, trace, k):
    confidence = 0
    for i in range(k):
        confidence += eigen_vals[k] / trace
    return confidence


def calc_coeficiente_proj(e_digits, centred_X, X_len):
    coef_proj = [np.dot(centred_X[i], e_digits) for i in range(X_len)]
    return coef_proj


def PCA(X, goal_confidence, k=None):
    mean_X = np.mean(X, 0)
    centred_X = X - mean_X
    e_digits, singular_values, V = linalg.svd(centred_X.transpose(), full_matrices=False)
    eigen_vals = singular_values * singular_values
    trace = sum(eigen_vals)

    if k == None:
        k, confidence = find_elbow(eigen_vals, trace, goal_confidence)
    else:
        confidence = calc_confidence(eigen_vals, trace, k)
    print(f'k: {k}\nconfidence: {confidence}\neigen_vals len: {len(eigen_vals)}')

    coef_proj = calc_coeficiente_proj(e_digits, centred_X, len(X))
    return mean_X, e_digits, eigen_vals[:k], coef_proj


def euclidean_dist(coef_proj, test_coef_proj, X_len, eigen_vals=None):
    dist = [linalg.norm(coef_proj[i] - test_coef_proj) for i in range(X_len)]
    return dist


def mahalanobis_dist(coef_proj, test_coef_proj, X_len, eigen_vals):
    dist = []
    for i in range(len(eigen_vals)):
        inv_eigen_val = 1/eigen_vals[i]
        diff_sqr = (coef_proj[i] - test_coef_proj[i])**2
        dist = np.append(dist, inv_eigen_val * diff_sqr)
    return dist


def identify(test_X, mean, e_digits, eigen_vals, coef_proj, X_len, dist_func):
    centred_test_X = test_X - mean
    test_coef_prof = np.dot(centred_test_X, e_digits)
    dist = dist_func(coef_proj, test_coef_prof, X_len, eigen_vals)
    d_min = np.min(dist)
    d_arg_min = np.argmin(dist)
    return d_min, d_arg_min


if __name__ == '__main__':
    X, y = MNISTData.loadMNIST('data/training/train-images.idx3-ubyte',
                            'data/training/train-labels.idx1-ubyte')
    test_X, test_y = MNISTData.loadMNIST('data/test/t10k-images.idx3-ubyte', 
                                        'data/test/t10k-labels.idx1-ubyte')
    mean_X, e_digits, eigen_vals, coef_proj = PCA(X, GOAL_CONFIDENCE, 3)
    print(f'eigen_vals len: {len(eigen_vals)}')
    tests_passed = 0
    for i in range(len(test_X)):
        d_min, d_arg_min = identify(test_X[i], mean_X, e_digits, eigen_vals, coef_proj, len(X), euclidean_dist)
        if test_y[i] != y[d_arg_min]:
            print(f'i: {i} expected: {test_y[i]} got: {y[d_arg_min]}')
        else:
            tests_passed += 1
            current_percent = tests_passed / (len(test_X))
            print(f'Current Percent: {current_percent * 100}')

    print(f'Passed Tests: {current_percent}')
