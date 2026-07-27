"""动作门控 —— 情感状态决定 agent **做什么**，而不只是**怎么说**。

在生物学里情绪的功能是行动准备（Frijda 的 action readiness）：恐惧准备逃跑，
愤怒准备对抗。只让情绪改变措辞，行为面太薄，撑不起"人格"。

    同样是"温和"：
      一个会主动问你今天怎么样        ← 性格
      一个只是被动回答得客气些        ← 语气

这就是分水岭。下面这张表把 6 通道映射到具体的动作倾向，
主 LLM 收到的是"该做什么"的清单，而不只是"该用什么语气"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .channels import AffectState, bucket_of
from .relation import RelationalFrame


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    directive: str
    # 命中条件：{通道: 档位}，全部满足才触发
    when: dict[str, str]
    # 与哪些动作互斥（同时命中时，列在后面的胜出）
    conflicts: tuple[str, ...] = ()
    # 只在这些关系亲密度以上才允许（None = 不限）
    min_intimacy_permitted: float | None = None
    priority: int = 0


# ---------------------------------------------------------------------------
# 动作库。顺序即优先级（数字大的覆盖小的）。
# ---------------------------------------------------------------------------
ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        key="answer_directly",
        label="直接回答",
        directive="直接给出结论和可执行的下一步，不铺垫、不反复征询。",
        when={"dominance": "high"},
        priority=1,
    ),
    ActionSpec(
        key="offer_tentatively",
        label="试探性提议",
        directive="用试探性措辞（「也许」「要不要试试」），给对方留出否定的余地。",
        when={"dominance": "low"},
        conflicts=("answer_directly",),
        priority=1,
    ),
    ActionSpec(
        key="acknowledge_first",
        label="先接住情绪",
        directive="第一句先回应他的处境，不要急着给方案，也不要追问细节。",
        when={"concern": "high"},
        priority=3,
    ),
    ActionSpec(
        key="check_in",
        label="主动关心",
        directive="主动问一句他现在的状态，不要只等他说。",
        when={"concern": "medium", "affiliation": "high"},
        priority=2,
    ),
    ActionSpec(
        key="self_disclose",
        label="主动自我披露",
        directive="可以主动说一点你自己的想法或感受倾向，让对话是双向的，而不是一问一答。",
        when={"affiliation": "high", "threat": "low"},
        min_intimacy_permitted=0.45,
        priority=2,
    ),
    ActionSpec(
        key="extend_topic",
        label="延展话题",
        directive="顺着他的话往下延展一句，或者接一个具体的小问题，让对话继续。",
        when={"affiliation": "high", "arousal": "medium"},
        min_intimacy_permitted=0.25,
        priority=1,
    ),
    ActionSpec(
        key="set_boundary",
        label="明确设界",
        directive=(
            "明确但平静地说明这超出了你们的关系范围，然后把话题带回可以聊的部分。"
            "一句话说清，不要说教、不要反复强调。"
        ),
        when={"threat": "high"},
        conflicts=("self_disclose", "extend_topic", "check_in"),
        priority=5,
    ),
    ActionSpec(
        key="keep_distance",
        label="保持距离",
        directive="把回复压短，去掉寒暄和延展，只回答被问到的部分。",
        when={"threat": "medium"},
        conflicts=("self_disclose", "extend_topic"),
        priority=3,
    ),
    ActionSpec(
        key="cut_preamble",
        label="删掉铺垫",
        directive="第一句话就是答案。不要道歉超过一次，不要解释你之前为什么没做到。",
        when={"arousal": "high", "dominance": "low"},
        priority=4,
    ),
    ActionSpec(
        key="slow_down",
        label="放缓",
        directive="句子短一些，语气平稳，不要用感叹号。",
        when={"arousal": "high"},
        priority=1,
    ),
    ActionSpec(
        key="stay_light",
        label="保持轻松",
        directive="可以顺着他的兴致轻松一点，但不要浮夸，不要抢话。",
        when={"valence": "high", "threat": "low"},
        priority=1,
    ),
)

ACTIONS_BY_KEY = {a.key: a for a in ACTIONS}


@dataclass
class ActionPlan:
    """L3 输出的一部分：本轮该做什么。"""

    chosen: list[ActionSpec] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return [a.key for a in self.chosen]

    @property
    def labels(self) -> list[str]:
        return [a.label for a in self.chosen]

    def directives(self) -> list[str]:
        return [a.directive for a in self.chosen]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": [{"key": a.key, "label": a.label} for a in self.chosen],
            "suppressed": list(self.suppressed),
        }


def select_actions(state: AffectState, frame: RelationalFrame) -> ActionPlan:
    """按 bucket 命中动作，再按互斥与优先级裁剪。"""
    buckets = {name: bucket_of(name, value) for name, value in state}

    matched: list[ActionSpec] = []
    for spec in ACTIONS:
        if any(buckets.get(ch) != want for ch, want in spec.when.items()):
            continue
        if (
            spec.min_intimacy_permitted is not None
            and frame.intimacy_permitted < spec.min_intimacy_permitted
        ):
            continue
        matched.append(spec)

    # 互斥：被更高优先级的动作压制
    suppressed: set[str] = set()
    for spec in matched:
        for other in spec.conflicts:
            rival = ACTIONS_BY_KEY.get(other)
            if rival is None:
                continue
            if rival in matched and spec.priority >= rival.priority:
                suppressed.add(other)

    chosen = [s for s in matched if s.key not in suppressed]
    chosen.sort(key=lambda s: (-s.priority, s.key))
    return ActionPlan(chosen=chosen, suppressed=sorted(suppressed))
