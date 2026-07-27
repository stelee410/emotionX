"""§2 核心数据结构。

这里刻意只用 dataclass + 标准库：L2/L3 是纯 Python，生产环境不装 torch。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

# 离散策略标签 —— 按 agent 应采取的应答策略划分，而非心理学分类学
StrategyLabel = Literal[
    "neutral",  # 常规应答
    "distress",  # 用户处于困扰/焦虑/悲伤 → 安抚优先
    "frustration",  # 用户不耐烦/受阻/重复失败 → 提速直给，减少寒暄
    "positive",  # 用户满意/愉悦 → 可延展、可轻松
]

STRATEGY_LABELS: tuple[str, ...] = get_args(StrategyLabel)
STRATEGY_TO_ID: dict[str, int] = {name: i for i, name in enumerate(STRATEGY_LABELS)}
ID_TO_STRATEGY: dict[int, str] = {i: name for name, i in STRATEGY_TO_ID.items()}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class UserAffect:
    """L1 输出：对用户当前状态的感知"""

    valence: float  # [-1, 1]  负面 ←→ 正面
    arousal: float  # [ 0, 1]  平静 ←→ 激动
    strategy: StrategyLabel
    intensity: float  # [ 0, 1]  强度，独立于类别
    confidence: float  # [ 0, 1]  模型置信度
    raw_logits: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.valence = clamp(float(self.valence), -1.0, 1.0)
        self.arousal = clamp(float(self.arousal), 0.0, 1.0)
        self.intensity = clamp(float(self.intensity), 0.0, 1.0)
        self.confidence = clamp(float(self.confidence), 0.0, 1.0)
        if self.strategy not in STRATEGY_LABELS:
            raise ValueError(f"unknown strategy label: {self.strategy!r}")

    @property
    def is_high_intensity(self) -> bool:
        """intensity 分档阈值，appraisal 规则表用它区分 high/low 行。"""
        return self.intensity >= HIGH_INTENSITY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HIGH_INTENSITY_THRESHOLD = 0.55


@dataclass
class ConversationEvent:
    """非情感来源的 appraisal 输入，由业务层填充"""

    task_succeeded: bool = False  # 任务完成
    task_failed: bool = False  # 明确失败
    user_repeated_query: bool = False  # 用户重复同一诉求（挫败的强信号）
    turn_count: int = 0
    latency_ms: int | None = None  # 本轮响应耗时（慢会加剧挫败）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# L2 状态向量的维度顺序，state_machine / store 共用
AFFECT_DIMS: tuple[str, ...] = ("valence", "arousal", "dominance", "concern")


@dataclass
class AgentAffect:
    """L2 输出：agent 自身的情感状态"""

    valence: float  # [-1, 1]
    arousal: float  # [ 0, 1]
    dominance: float  # [ 0, 1]  退让/顺从 ←→ 主导/自信
    concern: float  # [ 0, 1]  关切度（共情强度，非镜像）
    updated_at: float = field(default_factory=time.time)

    # ---- bucket 化（供 L3 查表）----
    # 阈值与 config/expression_templates.yaml 的注释保持一致
    CONCERN_HIGH = 0.6
    CONCERN_LOW = 0.3
    DOMINANCE_HIGH = 0.65
    DOMINANCE_LOW = 0.35
    AROUSAL_HIGH = 0.6
    AROUSAL_LOW = 0.3
    VALENCE_HIGH = 0.3
    VALENCE_LOW = -0.15

    @staticmethod
    def _bucket(value: float, low: float, high: float) -> str:
        if value > high:
            return "high"
        if value < low:
            return "low"
        return "medium"

    def buckets(self) -> dict[str, str]:
        return {
            "valence": self._bucket(self.valence, self.VALENCE_LOW, self.VALENCE_HIGH),
            "arousal": self._bucket(self.arousal, self.AROUSAL_LOW, self.AROUSAL_HIGH),
            "dominance": self._bucket(self.dominance, self.DOMINANCE_LOW, self.DOMINANCE_HIGH),
            "concern": self._bucket(self.concern, self.CONCERN_LOW, self.CONCERN_HIGH),
        }

    def to_bucket(self) -> str:
        """离散化，供 L3 查表 / 日志聚合。例：'c:high|d:low|a:high|v:medium'"""
        b = self.buckets()
        return f"c:{b['concern']}|d:{b['dominance']}|a:{b['arousal']}|v:{b['valence']}"

    def as_vector(self) -> dict[str, float]:
        return {d: float(getattr(self, d)) for d in AFFECT_DIMS}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentAffect:
        return cls(
            valence=float(data["valence"]),
            arousal=float(data["arousal"]),
            dominance=float(data["dominance"]),
            concern=float(data["concern"]),
            updated_at=float(data.get("updated_at", time.time())),
        )


@dataclass
class TurnTrace:
    """一轮的完整可解释性记录 —— §4.4 要求必须落日志。"""

    session_id: str
    persona: str
    turn_index: int
    user_affect: dict[str, Any]
    event: dict[str, Any]
    prev_state: dict[str, float]
    decayed_state: dict[str, float]
    delta: dict[str, float]
    matched_rules: list[str]
    next_state: dict[str, float]
    bucket: str
    idle_seconds: float
    idle_reset_applied: bool
    safety_bypass: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
