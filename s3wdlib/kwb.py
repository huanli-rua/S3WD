# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KWBProbEstimator:
    def __init__(self, k=6, metric="euclidean", eps=1e-6, min_prob=0.01):
        self.k = int(k); self.metric = metric; self.eps = float(eps); self.min_prob = float(min_prob)
        self.nn = None; self.y_tr = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, int)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn.fit(X); self.y_tr = y
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        _, idx = self.nn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        probs = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            nbr_y = self.y_tr[idx[i]]
            pos = int((nbr_y == 1).sum()); neg = int((nbr_y == 0).sum())
            p_hat = (pos + 1) / (pos + neg + 2)  # Laplace smoothing
            p_hat = max(self.min_prob, min(1.0 - self.min_prob, p_hat))
            probs[i] = p_hat
        return probs
