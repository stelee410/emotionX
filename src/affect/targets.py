"""L1 的训练目标 —— 从「4 类策略标签」改为「UserMove 回归」。

为什么不能再用分类标签：策略取决于关系，而感知层看不到关系。
「我想要你」标成 positive，陌生人场景下 agent 会热情回应骚扰；标成 offensive，
情侣场景下会冷淡拒绝伴侣。**没有一个标签是对的** —— 问题不在标签集不够大，
而在于把关系相关的判断塞进了关系无关的层。

改成回归之后，L1 学的是**这句话本身的属性**：隐含多亲密、多亲近/敌意、
多支配/顺从、对方自身多痛苦。这些都与说话人是谁无关，关系条件化留给 L2。

评估指标随之改变：不用 macro-F1（没有类别了），用
  * MAE          —— 绝对误差
  * Spearman ρ   —— **更重要**。只要序关系对，绝对值可以靠 L2 的增益校准。
"""

from __future__ import annotations

from typing import Any

from .moves import UserMove

# 回归头的输出顺序。改动这里等于改动模型接口，需要重新导出 ONNX。
REGRESSION_TARGETS: tuple[str, ...] = (
    "affiliation_bid",  # [-1, 1]
    "dominance_bid",  # [-1, 1]
    "intimacy_bid",  # [ 0, 1]
    "distress_level",  # [ 0, 1]
    "intensity",  # [ 0, 1]
)

# 每个目标的值域，决定网络末端用 tanh 还是 sigmoid
TARGET_RANGES: dict[str, tuple[float, float]] = {
    "affiliation_bid": (-1.0, 1.0),
    "dominance_bid": (-1.0, 1.0),
    "intimacy_bid": (0.0, 1.0),
    "distress_level": (0.0, 1.0),
    "intensity": (0.0, 1.0),
}

# 二分类头：这句话是否指向 agent 本人
BINARY_TARGETS: tuple[str, ...] = ("directed_at_agent",)

# 各目标的损失权重。intimacy_bid 权重最高 —— 它是失配机制的输入，
# 错了会直接把「亲近」判成「越界」。
TARGET_WEIGHTS: dict[str, float] = {
    "affiliation_bid": 1.0,
    "dominance_bid": 0.7,
    "intimacy_bid": 1.4,
    "distress_level": 1.0,
    "intensity": 0.6,
}
DIRECTED_WEIGHT = 0.5


def move_to_targets(move: UserMove) -> dict[str, float]:
    return {name: float(getattr(move, name)) for name in REGRESSION_TARGETS}


def targets_to_move(
    values: dict[str, float] | list[float],
    directed_logit: float = 1.0,
    confidence: float = 0.5,
) -> UserMove:
    if isinstance(values, list):
        values = dict(zip(REGRESSION_TARGETS, values, strict=False))
    return UserMove(
        affiliation_bid=values.get("affiliation_bid", 0.0),
        dominance_bid=values.get("dominance_bid", 0.0),
        intimacy_bid=values.get("intimacy_bid", 0.0),
        distress_level=values.get("distress_level", 0.0),
        intensity=values.get("intensity", 0.0),
        directed_at_agent=directed_logit >= 0.0,
        confidence=confidence,
    )


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman 秩相关。样本少时比 Pearson 稳，且我们只关心序关系。"""
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def regression_metrics(
    truth: list[UserMove], pred: list[UserMove]
) -> dict[str, Any]:
    """§评估：MAE + Spearman。序关系比绝对值重要。"""
    if len(truth) != len(pred):
        raise ValueError("truth 与 pred 长度不一致")
    out: dict[str, Any] = {"n": len(truth), "per_target": {}}
    for name in REGRESSION_TARGETS:
        t = [float(getattr(x, name)) for x in truth]
        p = [float(getattr(x, name)) for x in pred]
        mae = sum(abs(a - b) for a, b in zip(t, p, strict=True)) / max(1, len(t))
        rho = spearman(t, p)
        out["per_target"][name] = {"mae": round(mae, 4), "spearman": round(rho, 4)}
    hits = sum(
        1 for a, b in zip(truth, pred, strict=True) if a.directed_at_agent == b.directed_at_agent
    )
    out["directed_accuracy"] = round(hits / max(1, len(truth)), 4)
    rhos = [
        m["spearman"] for m in out["per_target"].values() if m["spearman"] == m["spearman"]
    ]
    out["mean_spearman"] = round(sum(rhos) / len(rhos), 4) if rhos else None
    out["mean_mae"] = round(
        sum(m["mae"] for m in out["per_target"].values()) / len(REGRESSION_TARGETS), 4
    )
    return out
