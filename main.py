import MNISTData
import noise
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

GOAL_CONFIDENCE = 0.75


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

    def pca(self):
        self.mean_X = np.mean(self.X, 0)
        centred_X = self.X - self.mean_X
        _, singular_values, self.v = linalg.svd(centred_X, full_matrices=False)
        self.all_eigen_vals = singular_values * singular_values
        trace = sum(self.all_eigen_vals)

        if self.k is None:
            self.k, self.confidence = self.__find_elbow(trace)
        else:
            self.confidence = self.__calc_confidence(trace)
        print(f'k: {self.k}\nconfidence: {self.confidence}')

        self.eigen_vals = self.all_eigen_vals[:self.k]
        self.inv_eigen_val = [np.double(1/val) for val in self.eigen_vals]
        self.diag_inv_eigen_val = np.diag(self.inv_eigen_val)
        self.eigen_vecs = self.v[:self.k]
        self.coef_proj = np.matmul(centred_X, self.eigen_vecs.T)


def show_eigen_vals(pca: PCA):
    plt.figure(figsize=(10, 10))
    t = np.arange(0, len(pca.all_eigen_vals), 1)
    plt.plot(t, pca.all_eigen_vals, 'x')
    plt.plot(pca.k, pca.all_eigen_vals[pca.k], 'o')
    plt.xlabel('K')
    plt.ylabel('Eigen values')
    plt.show()


def show_mean(pca: PCA):
    plt.imshow(np.reshape(pca.mean_X, (28, 28)), cmap=plt.get_cmap('gray'))
    plt.show()


def show_eigen_vecs(pca: PCA):
    fig = plt.figure()
    cols = 3
    rows = pca.k // cols + 1
    for i in range(1, pca.k + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(np.reshape(pca.eigen_vecs[i - 1], (28, 28)), cmap=plt.get_cmap('gray'))
    plt.show()


def plot_digits_3D(pca: PCA, y):
    limit = 2500
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        xs=pca.coef_proj[:limit, 0],
        ys=pca.coef_proj[:limit, 1],
        zs=pca.coef_proj[:limit, 2],
        c=y[:limit],
        linewidths=0.3,
        cmap='tab10')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(*scatter.legend_elements(), loc='center left', bbox_to_anchor=(1.1, 0.5))
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


def euclidean_dist(pca: PCA, coef_proj, test_coef_proj):
    return linalg.norm(coef_proj - test_coef_proj)


def mahalanobis_dist(pca: PCA, coef_proj, test_coef_proj):
    diff = coef_proj - test_coef_proj
    return np.matmul((np.matmul(pca.diag_inv_eigen_val, diff)).T, diff)


def identify(pca: PCA, test_X, train_y, dist_func):
    centred_test_X = test_X - pca.mean_X
    test_coef_prof = np.dot(centred_test_X, pca.eigen_vecs.T)
    dist = [dist_func(pca, pca.coef_proj[i], test_coef_prof) for i in range(pca.X_len)]
    d_min = np.min(dist)
    d_arg_min = np.argmin(dist)
    return train_y[d_arg_min], d_min


def scorer(test_X, test_y, dist_func):
    preds = []; dists = []
    tests_passed = 0
    for i in range(len(test_y)):
        pred, d_min = identify(pca, test_X[i], train_y, dist_func)
        preds.append(pred)
        if test_y[i] != pred:
            dists.append(d_min)
            avg_dist = np.mean(dists)
            print(f'i: {i} dist: {d_min} avg_dist: {avg_dist} expected: {test_y[i]} got: {pred}')
        else:
            tests_passed += 1
    score = (tests_passed / len(test_y)) * 100
    return score, preds


def confusion_matrix(test_y, predictions):
    conf_mat = np.zeros((10, 10), np.uint32)
    for i in range(len(predictions)):
        conf_mat[test_y[i]][predictions[i]] += 1
    return conf_mat


if __name__ == '__main__':
    train_X, train_y = MNISTData.loadMNIST('data/training/train-images.idx3-ubyte',
                                        'data/training/train-labels.idx1-ubyte')
    test_X, test_y = MNISTData.loadMNIST('data/test/t10k-images.idx3-ubyte', 
                                        'data/test/t10k-labels.idx1-ubyte')
    noisy_X, noisy_y = noise.noisey_MNIST('data/test/t10k-images.idx3-ubyte', 
                                            'data/test/t10k-labels.idx1-ubyte')
    MNISTData.showMNIST(noisy_X[10:], noisy_y[10:], 10)
    pca = PCA(train_X, GOAL_CONFIDENCE, 10)
    pca.pca()
    show_mean(pca)
    show_eigen_vecs(pca)
    plot_digits_3D(pca, train_y)

    euc_score, euc_preds = scorer(test_X, test_y, euclidean_dist)
    euc_confusion = confusion_matrix(test_y, euc_preds)
    euc_noise_score, euc_noise_preds = scorer(noisy_X, noisy_y, euclidean_dist)
    euc_noise_confusion = confusion_matrix(noisy_y, euc_noise_preds)

    mah_score, mah_preds = scorer(test_X, test_y, mahalanobis_dist)
    mah_confusion = confusion_matrix(test_y, mah_preds)
    mah_noise_score, mah_noise_preds = scorer(noisy_X, noisy_y, mahalanobis_dist)
    mah_noise_confusion = confusion_matrix(noisy_y, mah_noise_preds)

    print(f'Euclidean Dist Score: {euc_score}%\n{euc_confusion}\n',
          f'Noise Euclidean Dist Score: {euc_noise_score}%\n{euc_noise_confusion}\n')
    print(f'Mahalanobis Dist Score: {mah_score}%\n{mah_confusion}\n',
          f'Noise Mahalanobis Dist Score: {mah_noise_score}%\n{mah_noise_confusion}')
