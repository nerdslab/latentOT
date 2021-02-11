import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA


class FC:
    def __init__(self, n_cluster_source=5, epsilon=1, floyditer=50,
         tolratio=1e-7, normNum=2, init='kmeans', random_state=None):
        self.n_cluster_source = n_cluster_source
        self.epsilon = epsilon
        self.niter = floyditer
        self.tolratio = tolratio
        self.p = normNum
        self.init = init
        self.random_state = random_state

        if not self.init == 'kmeans':
            raise NotImplementedError

    def __call__(self, source, target):
        # centroid initialized by K-means
        Cx = self._compute_kmeans_centroids(source, n_clusters=self.n_cluster_source, random_state=self.random_state)
        for t in range(0, self.niter):
            
            # compute cost matrices
            cost_x = self._compute_cost_matrix(source, Cx, p=self.p)
            cost_y = self._compute_cost_matrix(Cx, target, p=self.p)
            Kx = np.exp(-cost_x / self.epsilon)
            Ky = np.exp(-cost_y / self.epsilon)
            
            # optimal transport
            Px, Py, P = self._latent_ot_known_anchors(Kx, Ky)  # update trans. plan

            # check for convergence
            if t and LA.norm(P - Pt1) / LA.norm(Pt1) < self.tolratio:
                break
            Pt1 = P.copy()

            # update anchors
            if t < self.niter - 1:
                Cx = self._compute_anchors(Px, Py, source, target)
        return Px, Py, P, Cx, t

    def _compute_cost_matrix(self, source, target, p=2):
        cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                      target.reshape([1, target.shape[0], target.shape[1]]),
                                      p), axis=-1)
        return cost_matrix
    

    def _compute_kmeans_centroids(self, data, **kwargs):
        kmeans = KMeans(**kwargs).fit(data)
        return kmeans.cluster_centers_

    def _latent_ot_known_anchors(self, Kx, Ky, niter=100, tol=1e-20, epsilon=0, clip_val=np.inf):
        dimx, k = Kx.shape
        dimy = Ky.shape[1]

        mu = 1 / dimx * np.ones([dimx, 1])
        nu = 1 / dimy * np.ones([dimy, 1])

        ax = np.ones([dimx,1])
        bx = np.ones([k,1])
        ay = np.ones([k,1])
        by = np.ones([dimy,1])
        w = np.ones([k,1])
        stablecons = 0
        stablecons2 = np.inf
        
        for i in range(1,niter + 1):
            ax = np.minimum(mu / ((Kx.dot(bx)) + stablecons),stablecons2)
            err1x = LA.norm(bx * Kx.T.dot(ax) - w, ord =1)
            by = np.minimum(nu / (Ky.T.dot(ay) + stablecons),stablecons2)
            err2y = LA.norm(ay * (Ky.dot(by)) - w, ord =1)
            w = ((bx * (Kx.T.dot(ax))) * (ay * (Ky.dot(by))))**(1/2)
            bx = np.minimum(w / (Kx.T.dot(ax) + stablecons),stablecons2)
            err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord =1)
            ay = np.minimum(w / (Ky.dot(by) + stablecons),stablecons2)
            err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord =1)
            if max(err1x,err2x,err1y,err2y) < tol:
                break        
            
        Px = np.diagflat(ax).dot(Kx.dot(np.diagflat(bx)))
        Py = np.diagflat(ay).dot(Ky.dot(np.diagflat(by)))
        P =  np.dot(Px,np.dot(LA.inv(np.diagflat(w)),Py))  
        
        return Px, Py, P

    def _compute_anchors(self, Px, Py, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Cx = (Px.T.dot(source) + Py.dot(target)) / (Py.dot(np.ones([m,1])) + Px.T.dot(np.ones([n,1])) + 10**-20)
        return Cx



def get_transport_plan(ot_matrix):
    """Return max correspondence"""
    map_source2target = np.argmax(ot_matrix, axis=1)
    map_target2source = np.argmax(ot_matrix, axis=0)
    return map_source2target, map_target2source



