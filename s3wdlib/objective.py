# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class S3WDParams:
    c1: float = 0.37
    c2: float = 0.63
    xi_min: float = 0.10
    theta_pos: float = 2.0
    theta_neg: float = 1.0
    penalty_large: float = 1e6
    gamma_last: bool = True
    gap: float = 0.02

def _entropy_w(y, w):
    y = (np.asarray(y) > 0).astype(int)
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return 0.0
    p1 = (w[y==1].sum()) / (w.sum())
    p1 = min(max(p1, 1e-12), 1-1e-12)
    return -(p1*np.log2(p1) + (1-p1)*np.log2(1-p1))

def _info_gain_threeway(prob, y, a, b, c1, c2):
    pos = prob >= a
    neg = prob <= b
    bnd = (~pos) & (~neg)
    w = np.where(y>0, c1, c2)
    H_parent = _entropy_w(y, w)
    IG = 0.0
    for m in (pos, bnd, neg):
        if m.sum() == 0:
            continue
        wm = w[m]; ym = y[m]
        IG += (wm.sum()/w.sum()) * _entropy_w(ym, wm)
    IG = H_parent - IG
    return IG, pos, bnd, neg

def regret_layer(prob, y, pos_m, neg_m, theta_pos, theta_neg):
    n_pos = max(1, int(pos_m.sum()))
    fp = int(((y==0) & pos_m).sum())
    r_pos = theta_pos * (fp / n_pos)
    n_neg = max(1, int(neg_m.sum()))
    fn = int(((y==1) & neg_m).sum())
    r_neg = theta_neg * (fn / n_neg)
    return r_pos + r_neg

def thresholds_penalty_monotonic(alphas, betas, gamma_last, gap, penalty_large):
    pen = 0.0
    n = len(alphas)
    for i in range(n):
        if alphas[i] < betas[i] + gap:
            pen += penalty_large
    for i in range(1, n):
        if betas[i] < betas[i-1]:
            pen += penalty_large
    for i in range(1, n):
        if alphas[i] > alphas[i-1]:
            pen += penalty_large
    if gamma_last and (alphas[-1] < 0.5 or betas[-1] > 0.5):
        pen += penalty_large
    return pen

def s3wd_objective(prob_levels, y, alphas, betas, gamma_last, params: S3WDParams):
    y = (np.asarray(y)>0).astype(int)
    nL = len(prob_levels)
    assert nL == len(alphas) == len(betas)

    total_ig = 0.0
    total_regret = 0.0
    bnd_shortfall_pen = 0.0

    for li in range(nL-1):
        IG_l, pos_m, bnd_m, neg_m = _info_gain_threeway(
            prob_levels[li], y, alphas[li], betas[li], params.c1, params.c2
        )
        total_ig += IG_l
        total_regret += regret_layer(prob_levels[li], y, pos_m, neg_m, params.theta_pos, params.theta_neg)
        denom = max(1, int((pos_m | bnd_m | neg_m).sum()))
        bnd_ratio = int(bnd_m.sum()) / denom
        if bnd_ratio < params.xi_min:
            bnd_shortfall_pen += params.penalty_large

    if params.gamma_last is True:
        g = 0.5
    elif isinstance(params.gamma_last, (float, int)):
        g = float(params.gamma_last)
    else:
        g = float(gamma_last) if gamma_last is not None else 0.5

    prob_n = prob_levels[-1]
    pos_m = (prob_n >= g)
    neg_m = ~pos_m
    total_regret += regret_layer(prob_n, y, pos_m, neg_m, params.theta_pos, params.theta_neg)

    pen_mono = thresholds_penalty_monotonic(np.asarray(alphas), np.asarray(betas), True, params.gap, params.penalty_large)
    f = (-total_ig) + total_regret + bnd_shortfall_pen + pen_mono
    details = {"ig": float(total_ig), "regret": float(total_regret), "pen_bnd": float(bnd_shortfall_pen), "pen_mono": float(pen_mono)}
    return f, details
