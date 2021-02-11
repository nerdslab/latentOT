import ot
from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA


class LOT:
    def __init__(self, n_cluster_source=5, n_cluster_target=20, epsilon=1, epsilon_z=1, intensity=[50, 10, 50], floyditer=50,
         tolratio=1e-7, normNum=2, init='kmeans', random_state=None):
        self.n_cluster_source, self.n_cluster_target = n_cluster_source, n_cluster_target
        self.epsilon = epsilon
        self.epsilon_z = epsilon_z
        self.intensity = intensity
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
        Cy = self._compute_kmeans_centroids(target, n_clusters=self.n_cluster_target, random_state=self.random_state)
        # Px, Py initialized by K-means and one-sided OT
        n = source.shape[0]
        m = target.shape[0]
        mu = 1 / n * np.ones([n, 1])
        nu = 1 / m * np.ones([m, 1])
        cost_xy = self._compute_cost_matrix(source, target, p=self.p)
        P = np.zeros([n,m]) + 1 / n / m
        converr = np.inf
        converrlist = np.zeros(self.niter) + np.inf
        for t in range(0, self.niter):
            
            # compute cost matrices
            cost_x = self._compute_cost_matrix(source, Cx, p=self.p)
            cost_z = self._compute_cost_matrix(Cx, Cy, p=self.p)
            cost_y = self._compute_cost_matrix(Cy, target, p=self.p)
            Kx = np.exp(-self.intensity[0] * cost_x / self.epsilon)
            Kz = np.exp(-self.intensity[1] * cost_z / self.epsilon_z)
            Ky = np.exp(-self.intensity[2] * cost_y / self.epsilon)
            
            Pt1 = P
            Px, Py, Pz, P = self._latent_ot_known_anchors(Kx, Kz, Ky)  # update trans. plan

            # check for convergence
            converr = LA.norm(P - Pt1) / LA.norm(Pt1)
            converrlist[t] = converr
            if converr < self.tolratio:
                break
            

            # update anchors
            if t < self.niter - 1:
                Cx, Cy = self._compute_anchors(Px, Py, Pz, source, target)
        # optimal transport
        Pot, _ = solve_ot(mu,nu,cost_xy)
        return Px, Py, Pz, P, Pot, Cx, Cy, t, converrlist

    def _compute_cost_matrix(self, source, target, p=2):
        cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                      target.reshape([1, target.shape[0], target.shape[1]]),
                                      p), axis=-1) 
        return cost_matrix
    

    def _compute_kmeans_centroids(self, data, **kwargs):
        kmeans = KMeans(**kwargs).fit(data)
        return kmeans.cluster_centers_

    def _latent_ot_known_anchors(self, Kx, Kz, Ky, niter=100, tol=1e-20, epsilon=0, clip_val=np.inf, epsilon1 = 0):
        dimx = Kx.shape[0]
        dimy = Ky.shape[1]
        dimz1, dimz2 = Kz.shape

        mu = 1 / dimx * np.ones([dimx, 1])
        nu = 1 / dimy * np.ones([dimy, 1])

        ax = np.ones([dimx, 1])
        bx = np.ones([dimz1, 1])
        ay = np.ones([dimz2, 1])
        by = np.ones([dimy, 1])
        az = np.ones([dimz1, 1])
        bz = np.ones([dimz2, 1])
        wxz = np.ones([dimz1, 1])
        wzy = np.ones([dimz2, 1])
        for i in range(1, niter + 1):
            
            ax = np.exp(np.minimum(np.log(np.maximum(mu,epsilon1)) - np.log(np.maximum(Kx.dot(bx), epsilon1)), clip_val))
            err1x = LA.norm(bx * Kx.T.dot(ax) - wxz, ord=1)
            

            by = np.exp(np.minimum(np.log(np.maximum(nu,epsilon1)) - np.log(np.maximum(Ky.T.dot(ay), epsilon1)), clip_val))
            err2y = LA.norm(ay * (Ky.dot(by)) - wzy, ord=1)
            
               
            wxz = ((az * (Kz.dot(bz))) * (bx * (Kx.T.dot(ax)))) ** (1 / 2)
            bx = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log( np.maximum(Kx.T.dot(ax),epsilon)), clip_val))
            err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord=1)

            az = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log(np.maximum(Kz.dot(bz), epsilon)), clip_val))
            err1z = LA.norm(bz * Kz.T.dot(az) - wzy, ord=1)
            wzy = ((ay * (Ky.dot(by))) * (bz * (Kz.T.dot(az)))) ** (1 / 2)
            bz = np.exp(np.minimum(np.log(np.maximum(wzy,epsilon)) - np.log(np.maximum(Kz.T.dot(az), epsilon)), clip_val))
            err2z = LA.norm(az * (Kz.dot(bz)) - wxz, ord=1)

            ay = np.exp(np.minimum(np.log(np.maximum(wzy, epsilon)) - np.log(np.maximum(Ky.dot(by), epsilon)), clip_val))
            err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord=1)
            if max(err1x, err2x, err1z, err2z, err1y, err2y) < tol:
                break

        Px = np.diagflat(ax).dot(Kx.dot(np.diagflat(bx)))
        Pz = np.diagflat(az).dot(Kz.dot(np.diagflat(bz)))
        Py = np.diagflat(ay).dot(Ky.dot(np.diagflat(by)))
        const = 0
        z1 = Px.T.dot(np.ones([dimx, 1])) + const
        z2 = Py.dot(np.ones([dimy, 1])) + const
        P = np.dot(Px / z1.T, np.dot(Pz, Py / z2))
        return Px, Py, Pz, P

    def _compute_anchors(self, Px, Py, Pz, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Px = self.intensity[0] * Px
        Pz = self.intensity[1] * Pz
        Py = self.intensity[2] * Py

        temp = np.concatenate((np.diagflat(Px.T.dot(np.ones([n, 1])) + Pz.dot(np.ones([self.n_cluster_target, 1]))),
                               -Pz),
                              axis=1)
        temp1 = np.concatenate((-Pz.T,
                                np.diagflat(Py.dot(np.ones([m, 1])) + Pz.T.dot(np.ones([self.n_cluster_source, 1])))),
                               axis=1)
        temp = np.concatenate((temp, temp1), axis=0)
        sol = np.concatenate((source.T.dot(Px), target.T.dot(Py.T)), axis=1).dot(LA.inv(temp))
        Cx = sol[:, 0:self.n_cluster_source].T
        Cy = sol[:, self.n_cluster_source:self.n_cluster_source + self.n_cluster_target].T
        return Cx, Cy



def solve_ot(mu,nu,cost_matrix):
    n, m = cost_matrix.shape
    ot_matrix = ot.emd(mu.reshape(-1), nu.reshape(-1), cost_matrix)
    ot_value = ot.emd2(mu.reshape(-1), nu.reshape(-1), cost_matrix)
    return ot_matrix, ot_value




def get_transport_plan(ot_matrix):
    """Return max correspondence"""
    map_source2target = np.argmax(ot_matrix, axis=1)
    map_target2source = np.argmax(ot_matrix, axis=0)
    return map_source2target, map_target2source




