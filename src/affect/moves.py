"""用户的关系性动作 —— 感知层每轮的输出。

**这些量全部是关系无关的**：它们描述这句话本身携带了什么，而不是它在某段关系里
意味着什么。「我想要你」的亲密度就是 0.9，不管说话人是谁；它是亲近还是冒犯，
由关系层（`RelationalFrame`）判定。

这条分工是整个架构成立的关键：
  * 感知层可以保持小（~18M）且只需要关系无关的标注数据
  * 关系条件化的行为**不需要任何训练数据** —— 它是确定性逻辑，用反事实测试验证
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .channels import clamp

# 强度分档阈值：≥ 该值走 appraisal 表的 high 档
HIGH_INTENSITY = 0.55
# 置信度低于此值时，依赖感知判断的评价规则降权
LOW_CONFIDENCE = 0.45
LOW_CONFIDENCE_SCALE = 0.4


@dataclass
class UserMove:
    """L1 输出。人际环状模型的两个轴 + 亲密度 + 若干标量。"""

    # 敌意 ← → 亲近
    affiliation_bid: float = 0.0
    # 顺从 ← → 支配（要求、命令、质问为正；请求、示弱为负）
    dominance_bid: float = 0.0
    # 这句话隐含的亲密度。「你好」≈0.1，「宝贝」≈0.8，「我想要你」≈0.9
    intimacy_bid: float = 0.0
    # 指向 agent 本人，还是在说第三方/世界。
    # 「我讨厌他」和「我讨厌你」在评价上完全不同，前者不该让 agent 戒备。
    directed_at_agent: bool = True
    # 对方自身的痛苦程度（与敌意无关，用于共情而非防御）
    distress_level: float = 0.0
    intensity: float = 0.0
    confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.affiliation_bid = clamp(float(self.affiliation_bid), -1.0, 1.0)
        self.dominance_bid = clamp(float(self.dominance_bid), -1.0, 1.0)
        self.intimacy_bid = clamp(float(self.intimacy_bid), 0.0, 1.0)
        self.distress_level = clamp(float(self.distress_level), 0.0, 1.0)
        self.intensity = clamp(float(self.intensity), 0.0, 1.0)
        self.confidence = clamp(float(self.confidence), 0.0, 1.0)

    @property
    def is_high_intensity(self) -> bool:
        return self.intensity >= HIGH_INTENSITY

    @property
    def is_hostile(self) -> bool:
        """敌意：与亲密度无关的负向亲和，且指向 agent。"""
        return self.directed_at_agent and self.affiliation_bid < -0.2

    @property
    def confidence_scale(self) -> float:
        """低置信度时对「依赖感知判断」的规则降权。业务事实类规则不受影响。"""
        return LOW_CONFIDENCE_SCALE if self.confidence < LOW_CONFIDENCE else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserMove:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TurnContext:
    """非语言来源的评价输入，由业务层填充。"""

    task_succeeded: bool = False
    task_failed: bool = False
    user_repeated_query: bool = False
    turn_count: int = 0
    latency_ms: int | None = None
    # 用户是否在道歉/退让 —— threat 的**修复通路**，见 appraisal 引擎。
    # 没有这条，戒备一旦顶高就会锁死几十轮，用户无法挽回。
    user_repaired: bool = False
    # 外部记忆系统注入的补充信息（本项目不实现记忆系统本身）
    memory_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
