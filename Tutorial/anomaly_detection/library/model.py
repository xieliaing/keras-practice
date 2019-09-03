import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph


def density_last(X, n_neighbors, metric):
    if (X.shape[0] <= n_neighbors):
        return 1.0 / n_neighbors
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric, mode='distance', n_jobs=-1)
    t1 = graph[-1].sum()
    t2 = graph[graph[-1].indices].sum()
    
    return 1.0 / n_neighbors if t1 * t2 == 0 else t1 / t2


def proximity_last(X, n_neighbors, metric):
    if (X.shape[0] <= n_neighbors):
        return 1.0 / n_neighbors
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric, mode='distance', n_jobs=-1)
    return np.max(graph[-1])


def martingale(m, p, epsilon):
    return m * epsilon * np.power(p, epsilon - 1.0)


def p_value(A, a):
    return (float(np.sum(A > a)) + np.random.uniform() * float(np.sum(A == a))) / float(A.size)


class KNNAnomalyDetector():
    def __init__(self, threshold=2.0, epsilon=0.92, n_neighbors=3, metric='euclidean', method='density',
                 anomaly='level'):
        self.threshold = threshold
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.metric = metric
        assert method == 'density' or method == 'proximity'
        self.method = method
        assert anomaly == 'level' or anomaly == 'change'
        self.anomaly = anomaly
        self.def_score = 1.0 / self.n_neighbors
        self.observations = []
        self.A = []
        self.M = 1.0

    def score_last(self, X):
        if self.method == 'density':
            return density_last(X, self.n_neighbors, self.metric)
        return proximity_last(X, self.n_neighbors, self.metric)

    def observe(self, x):
        self.observations.append(x)
        X = StandardScaler().fit_transform(np.array(self.observations))
        a = self.score_last(X)
        self.A.append(a)
        A = np.array(self.A)        
        p = p_value(A, a)        
        m = martingale(self.M, p, self.epsilon)
        if self.anomaly == 'level':
            is_anomaly = (m > self.threshold)
        else:
            is_anomaly = ((m - self.M) > self.threshold)
        if is_anomaly:
            self.M = 1.0
            self.observations = []
        else:
            self.M = m
        return [a, p, m, is_anomaly]
