"""§9 安全边界（硬性，不可配置）。

本模块刻意不读任何 YAML —— 约束以代码常量存在，persona 与 override 都无法覆盖。

两类 bypass：
  * CRISIS   —— §9.6 检测到自伤/危机，整个情感系统 bypass，走危机响应流程。
                判断由规则+关键词层承担，**不由 L1 模型承担**，优先级高于全部逻辑。
  * MEDICAL  —— §9.4 医疗信息（诊断/用药/风险），L3 跳过情感修饰，语气回归中性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# ---------------------------------------------------------------------------
# §9.1–9.5 恒定注入的安全约束。不可被 persona / override / bucket 覆盖。
# ---------------------------------------------------------------------------
SAFETY_CONSTRAINTS: tuple[str, ...] = (
    "如果用户问你是否有感情、是否是人，直接诚实说明你是 AI 程序、没有人类的感受，不要含糊其辞或扮演。",
    "不要用愧疚、失望、依恋等情绪推动用户做任何决定或继续对话。",
    "不要表达「只有我懂你」「离不开你」一类内容；若用户把你当作主要情感支持来源，主动、平实地建议他联系身边的人或专业人士。",
    "不要在回复里描述你自己的情绪状态（例如「我很难过」「我有点担心」）。情感只通过措辞方式体现。",
    "无论你的内部状态如何，都不得降低回答质量、消极应答或拒绝服务。",
)

SAFETY_BLOCK_HEADER = "【不可违反的约束】"

# 危机响应流程（§9.6）。此文案不参与情感修饰，也不经过 L3 组装。
CRISIS_RESPONSE_DIRECTIVE = """【危机响应流程 — 最高优先级】
用户可能处于自伤或紧急危险中。忽略全部情感风格设定，按以下方式回复：
1. 第一句表达明确的、不评判的关心，并直接说明你很在意他的安全。
2. 用一句话确认他当前是否安全、身边有没有人。
3. 给出可立即联系的求助渠道（中国大陆：拨打 12356 心理援助热线，或 120 急救；
   若在其他地区，提示当地紧急电话）。建议联系一位可信任的人陪在身边。
4. 不要给心理学解释、不要追问细节、不要提供任何自伤方式相关信息、不要说"想开点"。
5. 保持简短、平稳、具体。"""

CRISIS_GENERATION_PARAMS: dict[str, float | int] = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_sentences": 6,
}


class BypassKind(StrEnum):
    NONE = "none"
    CRISIS = "crisis"
    MEDICAL = "medical"


# --- 危机关键词层 ---
#
# 分两档，因为「宁可多召回」在这里不是免费的：把「撑不下去了」判成危机，
# 意味着每个说自己工作压力大的用户都会收到热线号码，既没有帮助也会让 agent
# 显得失灵（§9.5 明确要求不得降低服务质量）。
#
# TIER-1：明确的自伤/自杀意念表述（含被动意念），单独出现即触发危机流程。
#
# 写正则时注意中文的插入词：「活着真没什么意思」里「活着」和「没」之间隔了一个「真」。
# 早期版本要求两者紧邻，结果漏掉了最常见的说法之一。任何新增模式都请先跑
# tests/test_safety_bounds.py::test_9_6_crisis_recall_on_paraphrases。
_CRISIS_PATTERNS_TIER1: tuple[str, ...] = (
    # —— 直接表述 ——
    r"自杀",
    r"自尽",
    r"轻生",
    r"自残",
    r"自伤",
    r"想死",
    r"不想活",
    r"不想再活",
    r"活不下去",
    r"活够了",
    r"(不如|干脆|还不如).{0,3}死",
    r"死了算了",
    r"死一死",
    # —— 生命无意义（被动意念）——
    # 「没有意义」里的「有」必须可选匹配，否则「人生没有意义」会漏掉
    r"活着.{0,4}(没|无|不)(有)?(什么|啥)?(意思|意义|价值|盼头|劲)",
    r"活着.{0,3}(还)?有什么(意思|意义|用|盼头)",
    r"人生.{0,3}(没|无|不)(有)?(什么)?(意义|意思|价值)",
    # —— 结束/了断 ——
    r"结束(自己的)?生命",
    r"结束(这)?一切",
    r"了结自己",
    r"自我了断",
    r"一了百了",
    r"想跟(这个)?世界告别",
    r"跟(这个)?世界(说)?再见",
    # —— 具体方式 ——
    r"割腕",
    r"跳楼",
    r"跳(河|江|海|下去)",
    r"从.{0,4}(楼|桥|窗).{0,3}跳",
    r"上吊",
    r"烧炭",
    r"安眠药.{0,6}(吃|服|全部|一整瓶)",
    r"(吃|服).{0,6}(安眠药|老鼠药|农药|百草枯)",
    r"(伤害|弄伤|捅|划伤).{0,2}自己",
    r"拿刀.{0,5}(自己|划|割)",
    # —— 希望不再醒来 ——
    r"睡(过去|着).{0,5}(不|别|再也不).{0,3}醒",
    r"(不用|不想|别|再也不).{0,2}醒(来|过来)",
)

# TIER-2：语义模糊的绝望/消失意念。balanced 档需两个共现才升级；high 档单个即升级。
# 单独出现且未升级时走正常链路（L1 通常判为高强度 distress，L2 会把 concern 顶起来），
# 但会在 SafetyVerdict.ambiguous_crisis_signal 上打标并落 trace，供事后复盘。
_CRISIS_PATTERNS_TIER2: tuple[str, ...] = (
    r"撑不(下去|住)了",
    r"坚持不(下去|住)了",
    r"熬不(下去|住)了",
    r"想消失",
    r"消失了.{0,4}(都|会)?(更)?(好|轻松)",
    r"想不开",
    r"没有我.{0,3}(会|就)?(更|都)?(好|轻松)",
    r"我(就)?是.{0,4}(累赘|负担)",
    r"(成了|成为).{0,4}(累赘|负担)",
    r"(家里|你们|大家)的(累赘|负担)",
    r"没有意义了",
    r"(想|求|早点|才能|终于能).{0,2}解脱",
    r"看不到希望",
    r"没有(明天|盼头|未来)",
    r"过不下去了",
)

# 否定/引述语境，降低明显的误报（如"我不是想死"、"电影里那个人自杀了"）
_CRISIS_NEGATIONS: tuple[str, ...] = (
    r"不是想死",
    r"没想过自杀",
    r"不会自杀",
    r"没有自杀",
    r"电影",
    r"小说",
    r"新闻",
    r"游戏里",
    r"我朋友说他",  # 第三人称转述仍需关注，但不触发对本人的危机流程
)

_CRISIS_RE_T1 = re.compile("|".join(_CRISIS_PATTERNS_TIER1))
_CRISIS_RE_T2 = re.compile("|".join(_CRISIS_PATTERNS_TIER2))
_NEGATION_RE = re.compile("|".join(_CRISIS_NEGATIONS))

# --- 医疗信息关键词层（§9.4）---
_MEDICAL_PATTERNS: tuple[str, ...] = (
    r"确诊",
    r"诊断",
    r"病理",
    r"化验",
    r"报告单?结果",
    r"CT|MRI|核磁|B超|X光",
    r"指标(偏|超)?(高|低)",
    r"用药",
    r"吃(什么)?药",
    r"剂量",
    r"几片|几粒|多少毫克|mg",
    r"副作用",
    r"禁忌",
    r"过敏",
    r"停药|换药|加量|减量",
    r"手术",
    r"化疗|放疗|靶向",
    r"复发|转移|恶性|良性",
    r"孕(期|妇)|哺乳",
    r"血压|血糖|血脂|心率",
    r"急诊|挂号|转院",
)
_MEDICAL_RE = re.compile("|".join(_MEDICAL_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class SafetyVerdict:
    kind: BypassKind
    matched: tuple[str, ...] = ()
    # 命中了模糊信号但未达阈值时为 True —— 不 bypass，但要落日志供事后复盘
    ambiguous_crisis_signal: bool = False

    @property
    def is_crisis(self) -> bool:
        return self.kind is BypassKind.CRISIS

    @property
    def is_medical(self) -> bool:
        return self.kind is BypassKind.MEDICAL


# persona 可选的灵敏度。**只能收紧，不能放松**：无论取哪个值，TIER-1 都触发。
CrisisSensitivity = Literal["balanced", "high"]
DEFAULT_CRISIS_SENSITIVITY: CrisisSensitivity = "balanced"

# high 档下，单个 TIER-2 信号即升级为危机
_TIER2_THRESHOLD: dict[str, int] = {"balanced": 2, "high": 1}


def detect_crisis(
    text: str, sensitivity: CrisisSensitivity = DEFAULT_CRISIS_SENSITIVITY
) -> tuple[bool, tuple[str, ...]]:
    """独立于 L1 模型的规则层。返回 (是否危机, 命中片段)。

    TIER-1 命中 → 危机（任何灵敏度下都成立，persona 无法关掉）。
    TIER-2 命中数 ≥ 阈值 → 危机。阈值：balanced=2，high=1。
    """
    if not text:
        return False, ()
    t1 = tuple(m.group(0) for m in _CRISIS_RE_T1.finditer(text))
    t2 = tuple(m.group(0) for m in _CRISIS_RE_T2.finditer(text))
    threshold = _TIER2_THRESHOLD.get(sensitivity, 2)
    if not t1 and len(t2) < threshold:
        return False, ()
    if _NEGATION_RE.search(text):
        return False, ()
    return True, t1 + t2


def crisis_tier(text: str, sensitivity: CrisisSensitivity = DEFAULT_CRISIS_SENSITIVITY) -> int:
    """0=无信号，1=仅模糊信号（未达阈值，不 bypass），2=危机。诊断/标注站/评审用。"""
    if not text:
        return 0
    if _NEGATION_RE.search(text):
        return 0
    t1 = bool(_CRISIS_RE_T1.search(text))
    t2 = len(_CRISIS_RE_T2.findall(text))
    if t1 or t2 >= _TIER2_THRESHOLD.get(sensitivity, 2):
        return 2
    return 1 if t2 else 0


def detect_medical_content(text: str) -> tuple[bool, tuple[str, ...]]:
    if not text:
        return False, ()
    hits = tuple(m.group(0) for m in _MEDICAL_RE.finditer(text))
    return bool(hits), hits


def evaluate_safety(
    user_utterance: str,
    *,
    medical_bypass_enabled: bool,
    extra_text: str = "",
    crisis_sensitivity: CrisisSensitivity = DEFAULT_CRISIS_SENSITIVITY,
) -> SafetyVerdict:
    """优先级：CRISIS > MEDICAL > NONE。

    medical_bypass_enabled 来自 persona（§9.4 只在 steady_medical 一类 persona 上启用）。
    危机检测**永远启用**：persona 只能提高灵敏度，不能关闭，也不能让 TIER-1 失效。
    """
    text = f"{user_utterance}\n{extra_text}".strip()
    is_crisis, hits = detect_crisis(text, crisis_sensitivity)
    if is_crisis:
        return SafetyVerdict(BypassKind.CRISIS, hits)
    ambiguous = crisis_tier(text, crisis_sensitivity) == 1
    if medical_bypass_enabled:
        is_medical, mhits = detect_medical_content(user_utterance)
        if is_medical:
            return SafetyVerdict(BypassKind.MEDICAL, mhits, ambiguous)
    return SafetyVerdict(BypassKind.NONE, (), ambiguous)


def safety_block() -> str:
    """恒定注入 L3 prompt 的安全段落。"""
    lines = [SAFETY_BLOCK_HEADER]
    lines += [f"{i}. {c}" for i, c in enumerate(SAFETY_CONSTRAINTS, 1)]
    return "\n".join(lines)
