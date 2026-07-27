"""6 通道情感状态，每通道独立的上升/下降时间常数。

与常见 VAD 模型的主要区别就在这里：**单一衰减系数无法同时表达
「惊了一下很快过去」和「被冒犯后很久才放松」。**

时间常数用**半衰期（轮）**表达而不是直接给 λ。理由很实际：
人对「被冒犯后大概几轮恢复正常」有直觉，对「λ 该取 0.9 还是 0.85」没有。
调参时改半衰期，λ 由 `lambda_from_half_life` 反解。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

ChannelName = Literal[
    "valence",  # 舒适 ← → 不适
    "arousal",  # 平静 ← → 激动
    "dominance",  # 退让 ← → 主导（语气的确定性）
    "concern",  # 对对方处境的关切
    "affiliation",  # 亲近意愿（想靠近）
    "threat",  # 戒备、防御、设界
]

CHANNEL_NAMES: tuple[str, ...] = get_args(ChannelName)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lambda_from_half_life(half_life_turns: float) -> float:
    """半衰期（轮）→ 每轮保留系数。half_life=3 表示 3 轮后偏离量减半。"""
    if half_life_turns <= 0:
        return 0.0
    return float(0.5 ** (1.0 / half_life_turns))


def half_life_from_lambda(lam: float) -> float:
    """反向换算，供 UI 展示。"""
    import math

    if lam <= 0:
        return 0.0
    if lam >= 1:
        return float("inf")
    return float(math.log(0.5) / math.log(lam))


@dataclass(frozen=True)
class ChannelSpec:
    """一个通道的静态定义。persona 可以覆盖 baseline / 半衰期 / 增益 / 边界。"""

    name: ChannelName
    lo: float
    hi: float
    baseline: float
    # 偏离量被**继续推大**时的半衰期（惯性：越大越黏）
    half_life_rise: float
    # 无刺激或反向刺激时回到 baseline 的半衰期（越大越难消退）
    half_life_fall: float
    gain: float = 1.0
    note: str = ""

    @property
    def lambda_rise(self) -> float:
        return lambda_from_half_life(self.half_life_rise)

    @property
    def lambda_fall(self) -> float:
        return lambda_from_half_life(self.half_life_fall)

    def clamp(self, value: float) -> float:
        return clamp(value, self.lo, self.hi)


# ---------------------------------------------------------------------------
# 默认通道定义
#
# 时间常数的取值依据是「可观察行为」而不是生理数据：
#   arousal      惊动一下 1~2 轮就过去           → 快升快降
#   valence      心情底色，被破坏后要好几轮       → 中升慢降
#   dominance    语气确定性，跟随互动节奏         → 中升中降
#   concern      对方一开口示弱就上来，退得较慢   → 快升中降
#   affiliation  亲近要慢慢建立，但建立后有余韵   → 中升很慢降
#   threat       被冒犯瞬间拉满，很久才放松       → 极快升 极慢降
# ---------------------------------------------------------------------------
DEFAULT_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("valence", -1.0, 1.0, 0.10, 3.0, 6.0, 1.0, "整体语气底色"),
    ChannelSpec("arousal", 0.0, 1.0, 0.30, 1.0, 1.5, 1.0, "句子长度与节奏"),
    ChannelSpec("dominance", 0.0, 1.0, 0.50, 2.0, 3.0, 1.0, "给结论 vs 试探性措辞"),
    ChannelSpec("concern", 0.0, 1.0, 0.25, 1.2, 3.0, 1.0, "先接住情绪 vs 直接给方案"),
    ChannelSpec(
        "affiliation", 0.0, 1.0, 0.25, 2.5, 8.0, 1.0, "亲近意愿：主动披露、延展话题"
    ),
    ChannelSpec(
        "threat",
        0.0,
        1.0,
        0.05,
        0.8,
        12.0,
        1.0,
        "戒备：设界、变冷、简短。快升慢降是刻意的——被冒犯后不该下一轮就热络",
    ),
)

CHANNELS: dict[str, ChannelSpec] = {c.name: c for c in DEFAULT_CHANNELS}


@dataclass(frozen=True)
class AffectState:
    """agent 的情感状态。不可变——更新走 `evolve()` 返回新对象，便于测试与回放。"""

    valence: float = 0.10
    arousal: float = 0.30
    dominance: float = 0.50
    concern: float = 0.25
    affiliation: float = 0.25
    threat: float = 0.05
    updated_at: float = field(default_factory=time.time)

    # ---- 访问 ----
    def __getitem__(self, name: str) -> float:
        if name not in CHANNEL_NAMES:
            raise KeyError(f"未知通道 {name!r}，可用：{CHANNEL_NAMES}")
        return float(getattr(self, name))

    def __iter__(self) -> Iterator[tuple[str, float]]:
        for name in CHANNEL_NAMES:
            yield name, float(getattr(self, name))

    def as_vector(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in CHANNEL_NAMES}

    def evolve(self, values: dict[str, float], now: float | None = None) -> AffectState:
        unknown = set(values) - set(CHANNEL_NAMES)
        if unknown:
            raise KeyError(f"未知通道 {sorted(unknown)}")
        return replace(self, **values, updated_at=time.time() if now is None else now)

    # ---- 离散化，供表达层与显示层查表 ----
    def buckets(self) -> dict[str, str]:
        return {name: bucket_of(name, self[name]) for name in CHANNEL_NAMES}

    def to_bucket(self) -> str:
        b = self.buckets()
        return "|".join(f"{n[:3]}:{b[n]}" for n in CHANNEL_NAMES)

    def dominant_channel(self) -> str:
        """偏离 baseline 最远的通道 —— 显示层用它决定主导表现。"""
        return max(
            CHANNEL_NAMES,
            key=lambda n: abs(self[n] - CHANNELS[n].baseline)
            / max(1e-6, CHANNELS[n].hi - CHANNELS[n].lo),
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.as_vector()
        d["updated_at"] = self.updated_at
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffectState:
        kwargs = {n: float(data[n]) for n in CHANNEL_NAMES if n in data}
        return cls(**kwargs, updated_at=float(data.get("updated_at", time.time())))

    @classmethod
    def from_baselines(
        cls, baselines: dict[str, float] | None = None, now: float | None = None
    ) -> AffectState:
        vals = {n: CHANNELS[n].baseline for n in CHANNEL_NAMES}
        if baselines:
            unknown = set(baselines) - set(CHANNEL_NAMES)
            if unknown:
                raise KeyError(f"未知通道 {sorted(unknown)}")
            vals.update({k: float(v) for k, v in baselines.items()})
        return cls(**vals, updated_at=time.time() if now is None else now)


# --- bucket 阈值。分档而非连续值送给下游，是为了让 prompt 稳定、可测试 ---
BUCKET_THRESHOLDS: dict[str, tuple[float, float]] = {
    # channel: (low_below, high_above)
    "valence": (-0.15, 0.30),
    "arousal": (0.30, 0.60),
    "dominance": (0.35, 0.65),
    "concern": (0.30, 0.60),
    "affiliation": (0.30, 0.60),
    "threat": (0.20, 0.45),  # 阈值刻意低：轻微戒备就该改变行为
}


def bucket_of(channel: str, value: float) -> str:
    lo, hi = BUCKET_THRESHOLDS[channel]
    if value > hi:
        return "high"
    if value < lo:
        return "low"
    return "medium"
