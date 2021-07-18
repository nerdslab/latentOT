from sklearn.cluster import KMeans
import numpy as np
import ot

def compute_kmeans_centroids(X, **kwargs):
    kmeans = KMeans(**kwargs).fit(X)
    return kmeans.cluster_centers_


def compute_cost_matrix(source, target, p=2):
    cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                  target.reshape([1, target.shape[0], target.shape[1]]),
                                  p), axis=-1)
    return cost_matrix

def get_transport_plan(ot_matrix):
    """Return max correspondence"""
    map_source2target = np.argmax(ot_matrix, axis=1)
    map_target2source = np.argmax(ot_matrix, axis=0)
    return map_source2target, map_target2source

def solve_ot(mu,nu,cost_matrix):
    ot_matrix = ot.emd(mu.reshape(-1), nu.reshape(-1), cost_matrix)
    ot_value = ot.emd2(mu.reshape(-1), nu.reshape(-1), cost_matrix)
    return ot_matrix, ot_value




