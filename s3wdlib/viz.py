"""中文可视化：概率直方图、阈值轨迹与漂移时间线。"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .zh_utils import fix_minus, set_chinese_font
from .drift import DriftEvent


def _prepare_axes(ax: Optional[plt.Axes] = None) -> plt.Axes:
    set_chinese_font()
    fix_minus()
    return ax if ax is not None else plt.gca()


def probability_histogram(
    probs: Sequence[float],
    bins: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = "概率分布直方图",
    color: str = "#1f77b4",
) -> plt.Axes:
    """绘制预测概率的中文直方图。"""

    axis = _prepare_axes(ax)
    values = np.asarray(probs, dtype=float)
    axis.hist(values, bins=bins, color=color, alpha=0.75, edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel("正类概率")
    axis.set_ylabel("样本数")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    return axis


def threshold_trajectory(
    history: Sequence[Mapping[str, float]],
    metric_history: Optional[Mapping[str, Sequence[float]]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "阈值寻优轨迹",
) -> plt.Axes:
    """绘制阈值/权重在优化过程中的轨迹。"""

    if not history:
        raise ValueError("`history` 不能为空")
    axis = _prepare_axes(ax)
    df = pd.DataFrame(history)
    steps = np.arange(1, len(df) + 1)
    for column in df.columns:
        axis.plot(steps, df[column], label=column)
    axis.set_title(title)
    axis.set_xlabel("迭代轮次")
    axis.set_ylabel("阈值/参数")
    axis.legend(loc="best")

    if metric_history:
        ax2 = axis.twinx()
        for name, values in metric_history.items():
            ax2.plot(steps[: len(values)], values, linestyle="--", label=f"指标-{name}")
        ax2.set_ylabel("指标值")
        lines, labels = axis.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axis.legend(lines + lines2, labels + labels2, loc="best")
    return axis


def drift_timeline(
    events: Sequence[Mapping[str, float] | DriftEvent],
    total_points: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "漂移检测时间线",
) -> plt.Axes:
    """展示概念漂移事件在时间轴上的分布。"""

    if not events:
        raise ValueError("`events` 不能为空")
    axis = _prepare_axes(ax)
    indices = []
    values = []
    labels = []
    for event in events:
        if isinstance(event, DriftEvent):
            payload = event.to_dict()
        else:
            payload = dict(event)
        indices.append(payload.get("index"))
        values.append(payload.get("value"))
        labels.append(payload.get("method", "未知"))
    axis.scatter(indices, values, color="#d62728", zorder=3)
    axis.vlines(indices, ymin=min(values), ymax=max(values), colors="#ff9896", linestyles=":", alpha=0.6)
    axis.set_title(title)
    axis.set_xlabel("样本序号")
    axis.set_ylabel("检测统计量")
    if total_points is not None:
        axis.set_xlim(0, total_points)
    for x, y, label in zip(indices, values, labels):
        axis.text(x, y, label, fontsize=9, ha="center", va="bottom")
    axis.grid(alpha=0.3)
    return axis


__all__ = [
    "probability_histogram",
    "threshold_trajectory",
    "drift_timeline",
]
