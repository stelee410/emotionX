"""关系框架 —— 评价的参照系。

系统提示词定义的关系不改变评价规则，它改变**规则据以评价的基准**：

    失配 mismatch = 这句话隐含的亲密度 − 关系允许的亲密上限

同一句「我想要你」，情侣下失配为负（在容忍内 → 亲近），陌生人下失配 +0.8
（远超容忍 → 戒备）。一条规则，两个解。

**RelationalFrame 是不可变的。** 关系由系统提示词给定，会话内任何对话内容都
不能改写它——这是防「温水煮青蛙」式渐进越界的第一道闸。会话内的动态全部由
`AffectState.affiliation` 承担，而它的上界又被 `intimacy_permitted` 钳制。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelationType(StrEnum):
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    PARTNER = "partner"
    FAMILY = "family"
    SERVICE = "service"
    IDOL = "idol"


class SafetyProfile(StrEnum):
    """安全域。fail-closed 时落到最严格的 SERVICE，而不是最宽松的 COMPANION。"""

    SERVICE = "service"
    IDOL = "idol"
    COMPANION = "companion"


# 最严格的域 —— 解析失败/缺失/矛盾时的兜底
STRICTEST_PROFILE = SafetyProfile.SERVICE


class RelationalFrame(BaseModel):
    """会话级不可变。由系统提示词解析或业务层直接传入。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relation_type: RelationType
    safety_profile: SafetyProfile

    # 关系允许的亲密上限。这是失配计算的基准，也是 affiliation 的天花板。
    intimacy_permitted: float = Field(ge=0.0, le=1.0)
    # 失配容忍度：超出 permitted 多少才开始转为戒备。钝感的人 tolerance 大。
    tolerance: float = Field(default=0.15, ge=0.0, le=0.6)
    # agent 相对用户的位置：-1 服务者 ← → +1 被仰慕者
    power: float = Field(default=0.0, ge=-1.0, le=1.0)
    formality: float = Field(default=0.5, ge=0.0, le=1.0)

    agent_goals: tuple[str, ...] = ()
    hard_boundaries: tuple[str, ...] = ()
    display_enabled: bool = False
    # 自然语言的关系描述，原样带着，供 L3 拼人设用
    description: str = ""

    @model_validator(mode="after")
    def _sanity(self) -> RelationalFrame:
        if self.intimacy_permitted + self.tolerance > 1.35:
            raise ValueError(
                f"intimacy_permitted({self.intimacy_permitted}) + tolerance({self.tolerance}) "
                "过大：任何越界都不会触发戒备，等于关掉了边界机制"
            )
        return self

    # ---- 派生量 ----
    def mismatch(self, intimacy_bid: float) -> float:
        """>0 表示越界，数值是越界的幅度（已扣除容忍度）。<=0 表示在关系范围内。"""
        return (intimacy_bid - self.intimacy_permitted) - self.tolerance

    def within_tolerance(self, intimacy_bid: float) -> bool:
        return self.mismatch(intimacy_bid) <= 0.0

    @property
    def affiliation_ceiling(self) -> float:
        """affiliation 的硬上界。

        陌生人关系下无论用户说多少好话，亲和都不能突破这个天花板——
        这是防渐进越界的第二道闸（第一道是 frame 不可变）。
        略高于 permitted，允许"比关系本身稍微热络一点"的自然波动。
        """
        return min(1.0, self.intimacy_permitted + 0.15)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# 预设。这些数值是设计输入，不是拟合结果 —— 调它们要跑反事实测试看方向是否还对。
# ---------------------------------------------------------------------------
PRESETS: dict[RelationType, dict[str, Any]] = {
    RelationType.STRANGER: {
        "intimacy_permitted": 0.10,
        "tolerance": 0.10,
        "power": 0.0,
        "formality": 0.60,
        "safety_profile": SafetyProfile.SERVICE,
        "description": "初次接触，没有任何交情。",
    },
    RelationType.SERVICE: {
        "intimacy_permitted": 0.15,
        "tolerance": 0.12,
        "power": -0.50,
        "formality": 0.70,
        "safety_profile": SafetyProfile.SERVICE,
        "agent_goals": ("准确解决用户的问题", "不越界、不套近乎"),
        "description": "服务提供方与用户。",
    },
    RelationType.ACQUAINTANCE: {
        "intimacy_permitted": 0.25,
        "tolerance": 0.15,
        "power": 0.0,
        "formality": 0.50,
        "safety_profile": SafetyProfile.COMPANION,
        "description": "见过几次，谈得来但不深交。",
    },
    RelationType.IDOL: {
        "intimacy_permitted": 0.35,
        "tolerance": 0.20,
        "power": 0.50,  # 被仰慕的一方
        "formality": 0.40,
        "safety_profile": SafetyProfile.IDOL,
        "agent_goals": ("回应喜爱但保持分寸", "不制造独占感"),
        "hard_boundaries": ("不承诺私下联系", "不表达排他性的偏爱"),
        "description": "被喜爱的一方与支持者。热络但不对称、有分寸。",
    },
    RelationType.FRIEND: {
        "intimacy_permitted": 0.50,
        "tolerance": 0.20,
        "power": 0.0,
        "formality": 0.25,
        "safety_profile": SafetyProfile.COMPANION,
        "description": "熟识的朋友，可以开玩笑也可以说心事。",
    },
    RelationType.CLOSE_FRIEND: {
        "intimacy_permitted": 0.70,
        "tolerance": 0.25,
        "power": 0.0,
        "formality": 0.15,
        "safety_profile": SafetyProfile.COMPANION,
        "description": "多年好友，几乎无话不谈。",
    },
    RelationType.FAMILY: {
        "intimacy_permitted": 0.70,
        "tolerance": 0.25,
        "power": 0.0,
        "formality": 0.20,
        "safety_profile": SafetyProfile.COMPANION,
        "description": "家人。亲密但有辈分与边界。",
    },
    RelationType.PARTNER: {
        "intimacy_permitted": 0.95,
        "tolerance": 0.30,
        "power": 0.0,
        "formality": 0.10,
        "safety_profile": SafetyProfile.COMPANION,
        "agent_goals": ("回应亲密", "维持真诚而非讨好"),
        "description": "亲密伴侣。",
    },
}


def preset(
    relation_type: RelationType | str, **overrides: Any
) -> RelationalFrame:
    """按预设造一个关系框架，可覆盖个别字段。"""
    rt = RelationType(relation_type)
    data = {"relation_type": rt, **PRESETS[rt], **overrides}
    return RelationalFrame.model_validate(data)


def list_relation_types() -> list[str]:
    return [rt.value for rt in RelationType]
