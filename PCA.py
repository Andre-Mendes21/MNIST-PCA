import numpy as np
import numpy.linalg as linalg


class PCA:

    def __init__(self, X, goal_confidence, k=None):
        self.X = X
        self.X_len = len(self.X)
        self.goal_confidence = goal_confidence
        self.k = k

    def __find_elbow(self, trace):
        confidence = 0.0; k = 0
        while confidence < self.goal_confidence:
            confidence += self.all_eigen_vals[k] / trace
            k += 1
        return k, confidence

    def __calc_confidence(self, trace):
        eigen_k_sum = sum(self.all_eigen_vals[:self.k])
        confidence = eigen_k_sum / trace
        return confidence

    def pca(self):
        self.mean_X = np.mean(self.X, 0)
        centred_X = self.X - self.mean_X
        _, singular_values, self.v = linalg.svd(centred_X, full_matrices=False)
        self.all_eigen_vals = singular_values * singular_values
        trace = sum(self.all_eigen_vals)

        if self.k is None or self.k == 0:
            self.k, self.confidence = self.__find_elbow(trace)
        else:
            self.confidence = self.__calc_confidence(trace)
        print(f'k: {self.k}\nconfidence: {self.confidence}')

        self.eigen_vals = self.all_eigen_vals[:self.k]
        self.eigen_vecs = self.v[:self.k]
        self.coef_proj = np.matmul(centred_X, self.eigen_vecs.T)

        self.inv_eigen_val = [np.double(1/val) for val in self.eigen_vals]
        self.diag_inv_eigen_val = np.diag(self.inv_eigen_val)
