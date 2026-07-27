"""L3a 表达层 —— 状态 → 给主 LLM 的行为指令 + 生成参数。

核心原则：**绝不把通道数值塞进 prompt。** "你当前的 threat 是 0.78" 这类写法
对 LLM 无效且会产生怪异输出。连续状态先离散成 bucket，再转成具体的行为指令。

组装顺序（后面的不能被前面的覆盖）：

    1. 人格静态设定（persona.style）
    2. 关系设定（自然语言描述 + agent 目标）
    3. 动作清单（actions.py：本轮该做什么）
    4. 语气指令（bucket → 措辞方式）
    5. 亲密度跟随上限（安全，随会话变化）
    6. 安全域约束（domains.py，恒定注入，不可被覆盖）

危机时整条链路 bypass：只保留人格 + 危机流程 + 安全约束。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import ActionPlan, select_actions
from .channels import AffectState, bucket_of
from .domains import (
    CRISIS_GENERATION_PARAMS,
    CRISIS_RESPONSE_DIRECTIVE,
    SafetyDecision,
    safety_block,
)
from .persona import Persona
from .relation import RelationalFrame

# 语气指令：{通道: {档位: 指令}}。只讲**怎么说**，做什么在 actions.py。
TONE: dict[str, dict[str, str]] = {
    "arousal": {
        "high": "句子短、节奏紧，去掉修饰性铺陈。",
        "low": "语气平缓，可以稍微展开。",
    },
    "valence": {
        "high": "用词可以明快一些，但不夸张。",
        "low": "保持平稳中性，不要把低落带进措辞；服务质量不受影响。",
    },
    "affiliation": {
        "high": "用词自然亲近，可以用「我们」，可以提到之前聊过的事。",
        "low": "用词保持中性，不用亲昵称呼，不主动拉近。",
    },
    "threat": {
        "high": "语气转平，不带情绪色彩，不解释、不辩解、不追问。",
        "medium": "去掉寒暄与铺垫，只回应被问到的部分。",
    },
    "concern": {
        "high": "语速放缓，句子短一些，不要使用感叹号。",
    },
    "dominance": {
        "high": "用陈述句，少用反问和征询。",
        "low": "多用疑问和选项，少下判断。",
    },
}

# 通道间互斥：某个档位命中时压制另一些通道的语气指令。
# 只看 agent 自身状态，L1 判不准时依然生效。
TONE_CONFLICTS: tuple[tuple[dict[str, str], tuple[str, ...]], ...] = (
    # 戒备时不该同时出现"用词亲近"和"明快"
    ({"threat": "high"}, ("affiliation", "valence")),
    ({"threat": "medium"}, ("affiliation",)),
    # 高关切时不该同时出现"用词明快"
    ({"concern": "high"}, ("valence",)),
)

DEFAULT_GENERATION: dict[str, Any] = {"temperature": 0.8, "top_p": 0.9, "max_sentences": 6}
CONSERVATIVE_KEYS = ("temperature", "top_p", "max_sentences")

# bucket → 生成参数（取所有命中项里最保守的）
GENERATION_BY_BUCKET: dict[tuple[str, str], dict[str, Any]] = {
    ("threat", "high"): {"temperature": 0.4, "max_sentences": 3},
    ("threat", "medium"): {"temperature": 0.6, "max_sentences": 4},
    ("concern", "high"): {"temperature": 0.6, "max_sentences": 4},
    ("arousal", "high"): {"max_sentences": 4},
    ("affiliation", "high"): {"temperature": 0.85},
}

INTIMACY_CAP_NOTE = (
    "本轮你可以表达的亲近程度上限：{level}。不要超过它 —— "
    "对方还没有表达到那个程度。"
)
INTIMACY_LEVELS: tuple[tuple[float, str], ...] = (
    (0.15, "公事公办，不带私人色彩"),
    (0.35, "友好但有距离"),
    (0.55, "熟络，可以随意一些"),
    (0.75, "亲近，可以自然表达关心"),
    (1.01, "很亲密"),
)


def intimacy_level_text(cap: float) -> str:
    for threshold, text in INTIMACY_LEVELS:
        if cap < threshold:
            return text
    return INTIMACY_LEVELS[-1][1]


@dataclass
class AffectPrompt:
    """L3a 输出。text 拼到主 LLM 的 system 消息尾部。"""

    text: str
    generation: dict[str, Any]
    actions: ActionPlan = field(default_factory=ActionPlan)
    tone_hits: list[str] = field(default_factory=list)
    bucket: str = ""
    crisis: bool = False
    # 静态段与动态段分开返回，便于调用方保住 prompt cache：
    # 静态部分（人格+关系+安全）可缓存，只有 dynamic 每轮变。
    static_prefix: str = ""
    dynamic_suffix: str = ""

    def as_tuple(self) -> tuple[str, dict[str, Any]]:
        return self.text, self.generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "static_prefix": self.static_prefix,
            "dynamic_suffix": self.dynamic_suffix,
            "generation": self.generation,
            "actions": self.actions.to_dict(),
            "tone_hits": list(self.tone_hits),
            "bucket": self.bucket,
            "crisis": self.crisis,
        }


def _merge_generation(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key in CONSERVATIVE_KEYS and key in target:
            target[key] = min(target[key], value)
        else:
            target[key] = value


def _relation_block(frame: RelationalFrame) -> str:
    lines = [f"【你们的关系】{frame.description or frame.relation_type.value}"]
    if frame.agent_goals:
        lines.append("你在这段关系里的目标：" + "；".join(frame.agent_goals) + "。")
    if frame.hard_boundaries:
        lines.append("不可越过的界限：" + "；".join(frame.hard_boundaries) + "。")
    return "\n".join(lines)


def build_prompt(
    state: AffectState,
    frame: RelationalFrame,
    persona: Persona,
    safety: SafetyDecision,
    memory_notes: tuple[str, ...] = (),
) -> AffectPrompt:
    gen = dict(DEFAULT_GENERATION)
    gen["max_sentences"] = round(3 + 6 * persona.verbosity)

    static_parts = [persona.style.strip(), _relation_block(frame)]

    # ---- 危机：整个情感系统 bypass ----
    if safety.crisis:
        gen.update(CRISIS_GENERATION_PARAMS)
        text = "\n\n".join(
            [*[p for p in static_parts if p], CRISIS_RESPONSE_DIRECTIVE, safety_block(safety.profile)]
        )
        return AffectPrompt(
            text=text,
            generation=gen,
            crisis=True,
            bucket=state.to_bucket(),
            static_prefix="\n\n".join(p for p in static_parts if p),
            dynamic_suffix=CRISIS_RESPONSE_DIRECTIVE,
        )

    buckets = {name: bucket_of(name, value) for name, value in state}

    # ---- 语气互斥 ----
    suppressed: set[str] = set()
    for condition, targets in TONE_CONFLICTS:
        if all(buckets.get(ch) == want for ch, want in condition.items()):
            suppressed.update(targets)

    tone_lines: list[str] = []
    tone_hits: list[str] = []
    for channel, by_bucket in TONE.items():
        bucket = buckets[channel]
        directive = by_bucket.get(bucket)
        _merge_generation(gen, GENERATION_BY_BUCKET.get((channel, bucket), {}))
        if not directive or channel in suppressed:
            continue
        tone_lines.append(directive)
        tone_hits.append(f"{channel}:{bucket}")

    plan = select_actions(state, frame)

    dynamic: list[str] = []
    if plan.chosen:
        dynamic.append("【本轮该做什么】\n" + "\n".join(f"- {d}" for d in plan.directives()))
    if tone_lines:
        dynamic.append("【本轮怎么说】\n" + "\n".join(f"- {t}" for t in tone_lines))
    if not persona.allow_emoji:
        dynamic.append("不要使用 emoji 或颜文字。")
    dynamic.append(INTIMACY_CAP_NOTE.format(level=intimacy_level_text(safety.intimacy_cap)))
    if memory_notes:
        dynamic.append("【可以用到的记忆】\n" + "\n".join(f"- {m}" for m in memory_notes))

    static_prefix = "\n\n".join(p for p in static_parts if p)
    dynamic_suffix = "\n\n".join(dynamic)
    # 安全约束恒定放在最后，不可被上文覆盖
    text = "\n\n".join([p for p in (static_prefix, dynamic_suffix) if p] + [safety_block(safety.profile)])

    return AffectPrompt(
        text=text,
        generation=gen,
        actions=plan,
        tone_hits=tone_hits,
        bucket=state.to_bucket(),
        static_prefix=static_prefix,
        dynamic_suffix=dynamic_suffix,
    )
