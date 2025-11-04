"""扩展评估工具：批量指标统计和层级流量分析。"""
from __future__ import annotations

from typing import List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Union[Sequence[int], Sequence[float], np.ndarray, pd.Series]


def _flatten_array(values: ArrayLike) -> np.ndarray:
    """将输入转换为一维 numpy 数组。"""
    if isinstance(values, pd.Series):
        array = values.to_numpy()
    else:
        array = np.asarray(values)
    return array.ravel()


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: Optional[ArrayLike] = None,
    average: str = "binary",
) -> MutableMapping[str, float]:
    """计算单次分类结果的核心指标。

    Parameters
    ----------
    y_true, y_pred : array-like
        真实标签与预测标签。
    y_score : array-like, optional
        用于计算 AUC 的概率或评分，缺失时退化为使用预测标签。
    average : str, default "binary"
        传递给 sklearn 指标的 `average` 参数，便于多分类兼容。

    Returns
    -------
    dict
        包含 F1/BAC/AUC/MCC/Kappa/Prec/Rec 的字典。
    """

    yt = _flatten_array(y_true)
    yp = _flatten_array(y_pred)
    score = _flatten_array(y_score) if y_score is not None else yp

    metrics = {
        "F1": float(f1_score(yt, yp, average=average)),
        "BAC": float(balanced_accuracy_score(yt, yp)),
        "Prec": float(precision_score(yt, yp, average=average, zero_division=0)),
        "Rec": float(recall_score(yt, yp, average=average, zero_division=0)),
        "MCC": float(matthews_corrcoef(yt, yp)),
        "Kappa": float(cohen_kappa_score(yt, yp)),
    }
    try:
        metrics["AUC"] = float(roc_auc_score(yt, score))
    except ValueError:
        metrics["AUC"] = float("nan")
    return metrics


def batch_metrics(
    truths: Sequence[ArrayLike],
    preds: Sequence[ArrayLike],
    scores: Optional[Sequence[ArrayLike]] = None,
    average: str = "binary",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """批量计算指标并输出明细与汇总表。

    Parameters
    ----------
    truths, preds : sequence of array-like
        多次实验的真实值与预测值列表（长度一致）。
    scores : sequence of array-like, optional
        与 `truths` 对齐的概率或评分，可缺失。
    average : str, default "binary"
        多分类场景下转发给 sklearn 指标的 `average` 参数。

    Returns
    -------
    detail : pandas.DataFrame
        每次实验的指标明细。
    summary : pandas.DataFrame
        明细的均值/标准差汇总。
    """

    if len(truths) != len(preds):
        raise ValueError("`truths` 与 `preds` 长度不一致")
    if scores is not None and len(scores) != len(truths):
        raise ValueError("`scores` 长度必须与 truths 对齐")

    rows: List[Mapping[str, float]] = []
    for idx, (yt, yp) in enumerate(zip(truths, preds)):
        sc = scores[idx] if scores is not None else None
        rows.append(classification_metrics(yt, yp, sc, average=average))

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "mean": detail.mean(numeric_only=True),
        "std": detail.std(numeric_only=True),
    })
    return detail, summary


def layer_stats(flows: Sequence[Mapping[str, Sequence[int]]]) -> pd.DataFrame:
    """根据分层流量统计均值与标准差。

    Parameters
    ----------
    flows : sequence of mapping
        每次实验的层级流量，如 {"L1": (pos, bnd, neg), "L3": (pos, neg)}。

    Returns
    -------
    pandas.DataFrame
        MultiIndex（层级 × 指标）的均值/标准差表。
    """

    records: List[Mapping[str, Union[int, float, str]]] = []
    for run_idx, flow in enumerate(flows):
        for layer, counts in flow.items():
            if len(counts) not in (2, 3):
                raise ValueError(f"层 {layer} 的计数长度异常: {counts}")
            pos = int(counts[0])
            if len(counts) == 3:
                bnd = int(counts[1])
                neg = int(counts[2])
            else:
                bnd = 0
                neg = int(counts[1])
            total = pos + bnd + neg
            record = {
                "run": run_idx,
                "layer": layer,
                "POS": pos,
                "BND": bnd,
                "NEG": neg,
                "TOTAL": total,
            }
            if total > 0:
                record.update(
                    {
                        "POS_RATE": pos / total,
                        "BND_RATE": bnd / total,
                        "NEG_RATE": neg / total,
                    }
                )
            else:
                record.update({"POS_RATE": np.nan, "BND_RATE": np.nan, "NEG_RATE": np.nan})
            records.append(record)

    df = pd.DataFrame(records)
    grouped = df.groupby("layer").agg(["mean", "std"])
    return grouped.round(4)


__all__ = [
    "classification_metrics",
    "batch_metrics",
    "layer_stats",
]
