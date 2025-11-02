# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def rank_features_mi(X: pd.DataFrame, y, method="mi"):
    Xv = X.values; yv = np.asarray(y, int)
    mi = mutual_info_classif(Xv, yv, discrete_features=False, random_state=42)
    order = np.argsort(-mi)
    feat_names = X.columns.to_list()
    ranked = [feat_names[i] for i in order]
    return ranked, mi[order]

def make_levels(feat_rank, level_pcts=(0.6,0.8,1.0)):
    n = len(feat_rank)
    l1 = max(1, int(round(level_pcts[0]*n)))
    l2 = max(l1+1, int(round(level_pcts[1]*n)))
    L1 = feat_rank[:l1]; L2 = feat_rank[:l2]; L3 = feat_rank
    return L1, L2, L3
