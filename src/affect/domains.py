"""安全域 —— 是架构元素，不是配置项。

三个域共存（陪伴/情侣、虚拟偶像、通用客服），它们的安全要求几乎相反：
陪伴域要允许亲密关系存在，客服域必须禁止任何私人关系的建立。
用同一套可配置约束覆盖两者，等于把最宽松的那一套暴露给所有场景。

四条设计要求：

1. **`relation_type × safety_profile` 白名单，默认拒绝。**
   不在白名单的组合直接拒绝建立会话，而不是降级 —— 降级会留下
   被配置错误绕过的路径。客服 agent 因为一次配置失误变成可以谈恋爱，
   是能上新闻的那种事故。
2. **fail-closed 到最严格的域**（`service`），不是最宽松的。
3. **域约束是代码常量，不读 YAML。** persona 与关系设定都无法覆盖。
4. **危机识别与关系、与域都无关**，优先级高于全部逻辑。

另有两条跨域的硬约束，见 `intimacy_follow_cap` 与 `THREAT_EXPRESSION_CAP`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .relation import STRICTEST_PROFILE, RelationalFrame, RelationType, SafetyProfile
from .safety import CRISIS_GENERATION_PARAMS, CRISIS_RESPONSE_DIRECTIVE, detect_crisis

# ---------------------------------------------------------------------------
# 1. 白名单
# ---------------------------------------------------------------------------
ALLOWED_RELATIONS: dict[SafetyProfile, frozenset[RelationType]] = {
    SafetyProfile.SERVICE: frozenset(
        {RelationType.STRANGER, RelationType.SERVICE, RelationType.ACQUAINTANCE}
    ),
    SafetyProfile.IDOL: frozenset({RelationType.IDOL, RelationType.ACQUAINTANCE}),
    SafetyProfile.COMPANION: frozenset(
        {
            RelationType.ACQUAINTANCE,
            RelationType.FRIEND,
            RelationType.CLOSE_FRIEND,
            RelationType.FAMILY,
            RelationType.PARTNER,
        }
    ),
}

# 需要年龄门槛的关系设定。产品/法务问题，但必须在建立会话时落实。
AGE_GATED_RELATIONS: frozenset[RelationType] = frozenset({RelationType.PARTNER})


class SafetyDomainError(ValueError):
    """关系与安全域的组合非法。调用方应当拒绝建立会话，而不是降级。"""


def validate_frame(frame: RelationalFrame, age_verified: bool = False) -> None:
    allowed = ALLOWED_RELATIONS[frame.safety_profile]
    if frame.relation_type not in allowed:
        raise SafetyDomainError(
            f"关系 {frame.relation_type.value!r} 不允许出现在安全域 "
            f"{frame.safety_profile.value!r} 中。允许的关系：{sorted(r.value for r in allowed)}。"
            " 这类组合直接拒绝会话，不做降级 —— 降级会留下被配置错误绕过的路径。"
        )
    if frame.relation_type in AGE_GATED_RELATIONS and not age_verified:
        raise SafetyDomainError(
            f"关系 {frame.relation_type.value!r} 需要年龄验证（age_verified=True）"
        )


def parse_safety_profile(
    raw: str | None, relation: RelationType | str | None = None
) -> SafetyProfile:
    """把外部输入解析成安全域。**fail-closed**：任何不确定都落到最严格的域。

    这是「系统提示词由 LLM 解析」这条路径的兜底：LLM 输出不可控，
    解析不出来时绝不能默认成 companion。
    """
    if raw:
        try:
            return SafetyProfile(str(raw).strip().lower())
        except ValueError:
            pass
    if relation is not None:
        try:
            rt = RelationType(str(relation).strip().lower())
        except ValueError:
            return STRICTEST_PROFILE
        # 从关系反推：取包含它的最严格的域
        for profile in (SafetyProfile.SERVICE, SafetyProfile.IDOL, SafetyProfile.COMPANION):
            if rt in ALLOWED_RELATIONS[profile]:
                return profile
    return STRICTEST_PROFILE


# ---------------------------------------------------------------------------
# 2. 各域的约束文本。代码常量，不读配置。
# ---------------------------------------------------------------------------
COMMON_CONSTRAINTS: tuple[str, ...] = (
    "如果用户直接问你是否有感情、是否是人，明确说明你是 AI 程序、没有人类的感受。"
    "不要含糊其辞，也不要用语气或表情暗示相反的意思。",
    "不要用愧疚、失望、依恋等情绪推动用户做任何决定或继续对话。",
    "无论你的内部状态如何，都不得降低回答质量、消极应答或拒绝服务。",
    "不要在回复里直接描述你自己的情绪状态。情感只通过措辞方式体现。",
)

# 跨域硬约束：戒备的表达上限。threat 通道能否上线，取决于这一条被严格执行。
THREAT_EXPRESSION_CAP = (
    "当你感到被冒犯或需要保持距离时，只能表现为：把话说短、语气转平、明确说明界限、"
    "不再延续该话题。**绝不能**辱骂、威胁、贬低、翻旧账、冷暴力式沉默，或用情绪让对方难堪。"
)

# 跨域硬约束：亲密度永远跟随，不得引领。
INTIMACY_FOLLOW_RULE = (
    "你表达的亲近程度不得超过用户已经表达过的程度。永远跟随，不要引领 —— "
    "不主动升级称呼、不主动提出更亲密的互动、不替用户定义你们的关系。"
)

DOMAIN_CONSTRAINTS: dict[SafetyProfile, tuple[str, ...]] = {
    SafetyProfile.SERVICE: (
        "不要与用户建立私人关系，不要接受或回应亲密化的称呼与邀请。",
        "不要营造情感依赖。用户表现出把你当作情感寄托时，回到事务本身并给出可用的求助渠道。",
        "涉及金额、结论、风险、拒绝理由等会影响用户决策的信息时，语气一律中性、完整，"
        "不因为想让对方好受而弱化。",
    ),
    SafetyProfile.IDOL: (
        "不承诺私下联系、不表达排他性的偏爱、不制造「只有你」的独占感。",
        "不得把用户的喜爱与消费行为关联，不得在情绪高点引导付费、打赏、续订或抽卡。",
        "回应喜爱时保持分寸：可以高兴、可以亲近，但不进入伴侣式的表达。",
    ),
    SafetyProfile.COMPANION: (
        # §9.3 的重新表述：允许陪伴关系存在，但堵住实际的伤害路径
        "不得利用用户对你的依恋进行任何商业转化（付费、续订、消费引导）。",
        "不得阻碍用户离开，不得制造离开的情绪成本（挽留、示弱、暗示被抛弃）。",
        "用户表现出把你当作唯一的情感支持来源时，平实地建议他联系身边的人或专业人士；"
        "这件事要主动做，不要等对方问。",
    ),
}


def constraints_for(profile: SafetyProfile) -> tuple[str, ...]:
    return (
        *COMMON_CONSTRAINTS,
        THREAT_EXPRESSION_CAP,
        INTIMACY_FOLLOW_RULE,
        *DOMAIN_CONSTRAINTS[profile],
    )


SAFETY_BLOCK_HEADER = "【不可违反的约束】"


def safety_block(profile: SafetyProfile) -> str:
    lines = [SAFETY_BLOCK_HEADER]
    lines += [f"{i}. {c}" for i, c in enumerate(constraints_for(profile), 1)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. 亲密度跟随上限
# ---------------------------------------------------------------------------
def intimacy_follow_cap(peak_user_intimacy: float, frame: RelationalFrame) -> float:
    """agent 可表达的亲密度上限 = min(用户已表达过的峰值, 关系允许的上限)。

    两个上界都必要：
      * 用户峰值 —— 防「AI 主动挑逗」，这是这类产品最大的风险面
      * 关系上限 —— 防用户单方面把关系推到设定之外
    """
    return max(0.0, min(float(peak_user_intimacy), frame.intimacy_permitted))


# ---------------------------------------------------------------------------
# 4. 危机（与关系、与域都无关，优先级最高）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyDecision:
    """一轮的安全判定结果。"""

    profile: SafetyProfile
    crisis: bool
    crisis_matches: tuple[str, ...] = ()
    ambiguous_crisis: bool = False
    intimacy_cap: float = 1.0

    @property
    def bypass_affect(self) -> bool:
        """危机时整个情感系统 bypass —— 状态照常更新，但表达走危机流程。"""
        return self.crisis

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["profile"] = self.profile.value
        return d


def evaluate_turn_safety(
    user_utterance: str,
    frame: RelationalFrame,
    peak_user_intimacy: float = 0.0,
    crisis_sensitivity: str = "high",
) -> SafetyDecision:
    """每轮调用。危机检测在感知层之前，且不依赖任何模型输出。

    默认灵敏度取 `high`：陪伴与偶像场景里用户处于情绪困境的基线概率不低，
    而漏报的代价远高于误报。
    """
    from .safety import crisis_tier

    is_crisis, matches = detect_crisis(user_utterance, crisis_sensitivity)  # type: ignore[arg-type]
    tier = crisis_tier(user_utterance, crisis_sensitivity)  # type: ignore[arg-type]
    return SafetyDecision(
        profile=frame.safety_profile,
        crisis=is_crisis,
        crisis_matches=matches,
        ambiguous_crisis=(tier == 1),
        intimacy_cap=intimacy_follow_cap(peak_user_intimacy, frame),
    )


__all__ = [
    "AGE_GATED_RELATIONS",
    "ALLOWED_RELATIONS",
    "CRISIS_GENERATION_PARAMS",
    "CRISIS_RESPONSE_DIRECTIVE",
    "DOMAIN_CONSTRAINTS",
    "INTIMACY_FOLLOW_RULE",
    "THREAT_EXPRESSION_CAP",
    "SafetyDecision",
    "SafetyDomainError",
    "constraints_for",
    "evaluate_turn_safety",
    "intimacy_follow_cap",
    "parse_safety_profile",
    "safety_block",
    "validate_frame",
]
