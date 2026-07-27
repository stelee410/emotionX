"""Bradley-Terry：把成对比较还原成连续尺度。

    P(i 胜 j) = σ(θ_i − θ_j)

用 MM（minorization-maximization）迭代求 θ，再线性映射到目标值域。
平局记作双方各半场胜利，这是 BT 处理平局最常见的做法。

为什么值得：绝对数值标注的一致性历来很差（「这句话亲密度是 0.6 还是 0.7？」），
成对比较容易得多，每次 2–3 秒而不是 10–15 秒。n 条数据需要约 5n–8n 次比较，
总耗时反而更短，质量更高。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

# 正则化：给每个条目加一场对「虚拟平均对手」的平局，
# 否则从没输过的条目 θ 会跑到无穷大。
PRIOR_STRENGTH = 0.5
MAX_ITER = 500
TOL = 1e-8


def fit_bradley_terry(
    comparisons: Iterable[dict[str, Any]],
    item_ids: Iterable[int] | None = None,
) -> dict[int, float]:
    """返回 {item_id: θ}。θ 是对数尺度上的强度，可正可负。"""
    comps = [c for c in comparisons if c.get("winner") in {"left", "right", "tie"}]
    ids: set[int] = set(int(i) for i in (item_ids or []))
    for c in comps:
        ids.add(int(c["left_id"]))
        ids.add(int(c["right_id"]))
    if not ids:
        return {}

    index = {i: k for k, i in enumerate(sorted(ids))}
    n = len(index)
    wins = [PRIOR_STRENGTH] * n  # 每人先记 0.5 场对虚拟对手的胜
    pairs: dict[tuple[int, int], float] = {}

    for c in comps:
        li, ri = index[int(c["left_id"])], index[int(c["right_id"])]
        w = c["winner"]
        if w == "left":
            wins[li] += 1.0
        elif w == "right":
            wins[ri] += 1.0
        else:
            wins[li] += 0.5
            wins[ri] += 0.5
        key = (li, ri) if li < ri else (ri, li)
        pairs[key] = pairs.get(key, 0.0) + 1.0

    # MM 迭代
    p = [1.0] * n
    for _ in range(MAX_ITER):
        new = [0.0] * n
        denom = [PRIOR_STRENGTH / (p[k] + 1.0) for k in range(n)]  # 虚拟对手强度=1
        for (a, b), count in pairs.items():
            s = p[a] + p[b]
            denom[a] += count / s
            denom[b] += count / s
        max_delta = 0.0
        for k in range(n):
            value = wins[k] / denom[k] if denom[k] > 0 else p[k]
            max_delta = max(max_delta, abs(value - p[k]))
            new[k] = value
        # 归一化，避免整体漂移
        mean = sum(new) / n
        p = [max(1e-9, v / mean) for v in new]
        if max_delta < TOL:
            break

    return {item_id: math.log(p[k]) for item_id, k in index.items()}


def theta_to_scale(
    thetas: dict[int, float],
    lo: float = 0.0,
    hi: float = 1.0,
    anchors: dict[int, float] | None = None,
) -> dict[int, float]:
    """θ → 目标值域。

    有锚点（直接评分过的条目）时用最小二乘拟合 θ→值 的线性映射，
    这样比较得到的尺度和人工打分的绝对值对齐；没有锚点就按分位数铺开。
    """
    if not thetas:
        return {}
    usable = {k: v for k, v in (anchors or {}).items() if k in thetas}
    if len(usable) >= 2:
        xs = [thetas[k] for k in usable]
        ys = [usable[k] for k in usable]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        slope = (
            sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var if var else 0.0
        )
        intercept = my - slope * mx
        if slope != 0.0:
            return {
                k: max(lo, min(hi, slope * t + intercept)) for k, t in thetas.items()
            }

    values = sorted(thetas.values())
    span = values[-1] - values[0]
    if span <= 0:
        mid = (lo + hi) / 2
        return dict.fromkeys(thetas, mid)
    return {k: lo + (t - values[0]) / span * (hi - lo) for k, t in thetas.items()}


def agreement(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """标注者一致性：同一对被多人比较时，判断是否相同。

    成对比较用的是**一致率**而非 Kappa —— 比较任务只有 3 个选项且顺序有意义，
    Kappa 的期望一致率假设在这里不成立。
    """
    by_pair: dict[tuple[int, int], list[str]] = {}
    for c in comparisons:
        li, ri = int(c["left_id"]), int(c["right_id"])
        key = (li, ri) if li < ri else (ri, li)
        winner = c["winner"]
        if key != (li, ri) and winner in {"left", "right"}:
            winner = "right" if winner == "left" else "left"  # 归一化方向
        by_pair.setdefault(key, []).append(winner)

    multi = [v for v in by_pair.values() if len(v) >= 2]
    if not multi:
        return {"n_repeated_pairs": 0, "raw_agreement": None}
    agree = sum(1 for v in multi if len(set(v)) == 1)
    return {
        "n_repeated_pairs": len(multi),
        "raw_agreement": round(agree / len(multi), 4),
        "note": "低于 0.7 说明该维度的定义有歧义，先改标注指南再继续标",
    }


def build_scale(
    comparisons: list[dict[str, Any]],
    dimension: str,
    anchors: dict[int, float] | None = None,
    item_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    from .db import COMPARABLE

    if dimension not in COMPARABLE:
        raise ValueError(f"未知维度 {dimension!r}")
    lo, hi = (-1.0, 1.0) if dimension in {"affiliation_bid", "dominance_bid"} else (0.0, 1.0)
    thetas = fit_bradley_terry(comparisons, item_ids)
    scaled = theta_to_scale(thetas, lo, hi, anchors)
    counts: dict[int, int] = {}
    for c in comparisons:
        for k in ("left_id", "right_id"):
            counts[int(c[k])] = counts.get(int(c[k]), 0) + 1
    weak = [i for i, t in thetas.items() if counts.get(i, 0) < 3]
    return {
        "dimension": dimension,
        "range": [lo, hi],
        "n_items": len(thetas),
        "n_comparisons": len(comparisons),
        "values": {str(k): round(v, 4) for k, v in sorted(scaled.items())},
        "thetas": {str(k): round(v, 4) for k, v in sorted(thetas.items())},
        "anchored": bool(anchors and len(anchors) >= 2),
        "under_compared": sorted(weak)[:20],
        "warning": (
            f"{len(weak)} 个条目的比较次数少于 3，尺度估计不稳"
            if weak
            else None
        ),
        "agreement": agreement(comparisons),
    }
