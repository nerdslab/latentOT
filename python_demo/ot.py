import numpy as np

from .utils import compute_cost_matrix, solve_ot


class OT:
    def __init__(self, norm=2):
        self.p = norm

    def fit(self, source, target):
        n = source.shape[0]
        m = target.shape[0]
        mu = 1 / n * np.ones([n, 1])
        nu = 1 / m * np.ones([m, 1])
        cost_xy = compute_cost_matrix(source, target, p=self.p)

        # optimal transport
        P, _ = solve_ot(mu, nu, cost_xy)
        self.P_ = P

    def transport(self, source, target):
        n = source.shape[0]
        transported = np.dot(self.P_, target) * n
        return transported
