import MNISTData
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

GOAL_CONFIDENCE = 0.95


class PCA:

    def __init__(self, X, goal_confidence, k=None):
        self.X = X
        self.X_len = len(self.X)
        self.goal_confidence = goal_confidence
        self.k = k

    def __find_elbow(self, trace):
        confidence = 0.0
        k = 0
        while confidence < self.goal_confidence:
            confidence += self.all_eigen_vals[k] / trace
            k += 1
        return k, confidence

    def __calc_confidence(self, trace):
        confidence = 0
        for i in range(self.k):
            confidence += self.all_eigen_vals[i] / trace
        return confidence

    def __calc_coeficiente_proj(self, centred_X):
        return [np.dot(centred_X[i], self.e_digits) for i in range(self.X_len)]

    def pca(self):
        self.mean_X = np.mean(self.X, 0)
        centred_X = self.X - self.mean_X
        self.e_digits, singular_values, V = linalg.svd(centred_X.transpose(), full_matrices=False)
        self.all_eigen_vals = singular_values * singular_values
        trace = sum(self.all_eigen_vals)

        if self.k is None:
            self.k, self.confidence = self.__find_elbow(trace)
        else:
            self.confidence = self.__calc_confidence(trace)
        print(f'k: {self.k}\nconfidence: {self.confidence}\n')

        self.eigen_vals = self.all_eigen_vals[:self.k]
        self.coef_proj = self.__calc_coeficiente_proj(centred_X)


def show_eigen_vals(pca: PCA):
    plt.figure(figsize=(10, 10))
    t = np.arange(0, pca.X_len, 1)
    plt.plot(t, pca._eigen_vals, 'x')
    plt.plot(pca.k, pca.all_eigen_vals[pca.k], 'o')
    plt.xlabel('K')
    plt.ylabel('Eigen values')
    plt.show()


def euclidean_dist(pca: PCA, test_coef_proj):
    return [linalg.norm(pca.coef_proj[i] - test_coef_proj) for i in range(pca.X_len)]


# Does not work as intended
def mahalanobis_dist(pca: PCA, test_coef_proj):
    dist = []
    for i in range(pca.k):
        inv_eigen_val = 1/pca.eigen_vals[i]
        diff_sqr = np.square(pca.coef_proj[i] - test_coef_proj[i])
        dist = np.append(dist, inv_eigen_val * diff_sqr)
    return dist


def identify(pca: PCA, test_X, dist_func):
    centred_test_X = test_X - pca.mean_X
    test_coef_prof = np.dot(centred_test_X, pca.e_digits)
    dist = dist_func(pca, test_coef_prof)
    d_min = np.min(dist)
    d_arg_min = np.argmin(dist)
    return d_min, d_arg_min


if __name__ == '__main__':
    train_X, train_y = MNISTData.loadMNIST('data/training/train-images.idx3-ubyte',
                                        'data/training/train-labels.idx1-ubyte')
    test_X, test_y = MNISTData.loadMNIST('data/test/t10k-images.idx3-ubyte', 
                                        'data/test/t10k-labels.idx1-ubyte')
    test_len = len(test_X)

    pca = PCA(train_X, GOAL_CONFIDENCE, 3)
    pca.pca()
    dists = []
    tests_passed = 0
    for i in range(test_len):
        d_min, d_arg_min = identify(pca, test_X[i], euclidean_dist)
        if test_y[i] != train_y[d_arg_min]:
            dists = np.append(dists, d_min)
            avg_dist = np.mean(dists)
            print(f'dist: {d_min} avg_dist: {avg_dist} i: {i} expected: {test_y[i]} got: {train_y[d_arg_min]}')
            MNISTData.showDigit(test_X, test_y, i, "Test X\nExpected")
            MNISTData.showDigit(pca.X, train_y, d_arg_min, "Training X\nGot")
            plt.show()
        else:
            tests_passed += 1
            current_percent = tests_passed / test_len
            # MNISTData.showDigit(test_X, i)
            # MNISTData.showDigit(train_X, d_arg_min)
            print(f'i: {i} Current Percent: {current_percent * 100}')

    print(f'Passed Tests: {current_percent * 100}%')
