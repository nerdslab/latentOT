
import numpy as np
from numpy import linalg as LA

from .utils import compute_kmeans_centroids, compute_cost_matrix


class FC:
    def __init__(self, n_anchors, epsilon=1, floyditer=50, tolratio=1e-7, norm=2, random_state=None):
        self.n_anchors = n_anchors
        self.epsilon = epsilon
        self.niter = floyditer
        self.tolratio = tolratio
        self.p = norm
        self.random_state = random_state

    def fit(self, source, target):
        # initialize centroids using kmeans
        Cx = compute_kmeans_centroids(source, n_clusters=self.n_anchors, random_state=self.random_state)

        Px, Py, P = None, None, None
        for t in range(0, self.niter):
            # compute cost matrices
            cost_x = compute_cost_matrix(source, Cx, p=self.p)
            cost_y = compute_cost_matrix(Cx, target, p=self.p)

            Kx = np.exp(-cost_x / self.epsilon)
            Ky = np.exp(-cost_y / self.epsilon)
            
            # optimal transport with fixed anchors
            Px, Py, P = self.update_transport_plan(Kx, Ky)  # update trans. plan

            # check for convergence
            if t and LA.norm(P - Pt1) / LA.norm(Pt1) < self.tolratio:
                break
            Pt1 = P.copy()

            # update anchors
            if t < self.niter - 1:
                Cx = self.update_anchors(Px, Py, source, target)

        self.Px_, self.Py_, self.P_ = Px, Py, P

    def update_transport_plan(self, Kx, Ky, niter=100, tol=1e-20):
        dimx, k = Kx.shape
        dimy = Ky.shape[1]

        mu = 1 / dimx * np.ones([dimx, 1])
        nu = 1 / dimy * np.ones([dimy, 1])

        ax = np.ones([dimx, 1])
        bx = np.ones([k, 1])
        ay = np.ones([k, 1])
        by = np.ones([dimy, 1])
        w = np.ones([k, 1])
        stablecons = 0
        stablecons2 = np.inf
        
        for i in range(1,niter + 1):
            ax = np.minimum(mu / ((Kx.dot(bx)) + stablecons),stablecons2)
            err1x = LA.norm(bx * Kx.T.dot(ax) - w, ord=1)
            by = np.minimum(nu / (Ky.T.dot(ay) + stablecons),stablecons2)
            err2y = LA.norm(ay * (Ky.dot(by)) - w, ord=1)
            w = ((bx * (Kx.T.dot(ax))) * (ay * (Ky.dot(by))))**(1/2)
            bx = np.minimum(w / (Kx.T.dot(ax) + stablecons),stablecons2)
            err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord=1)
            ay = np.minimum(w / (Ky.dot(by) + stablecons),stablecons2)
            err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord=1)
            if max(err1x, err2x, err1y, err2y) < tol:
                break
            
        Px = np.diagflat(ax).dot(Kx.dot(np.diagflat(bx)))
        Py = np.diagflat(ay).dot(Ky.dot(np.diagflat(by)))
        P  = np.dot(Px,np.dot(LA.inv(np.diagflat(w)),Py))
        
        return Px, Py, P

    def update_anchors(self, Px, Py, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Cx = (Px.T.dot(source) + Py.dot(target)) / (Py.dot(np.ones([m, 1])) + Px.T.dot(np.ones([n, 1])) + 10**-20)
        return Cx

    def transport(self, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Cx_fc = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_fc = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)
        transported = source + np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cy_fc - Cx_fc)
        return transported
