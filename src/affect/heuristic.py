"""规则桩：从中文文本估计 UserMove。

用途有三个，都不是"当作模型用"：
  * 没有训练好的 L1 时跑通全链路（关系条件化的行为本来就不依赖 L1 精度）
  * WebUI 测试台的默认感知器，让人能立刻手搓对话看状态怎么走
  * ONNX 加载失败时的降级路径

**不要指望它的精度。** 它的职责是方向大致对、永不崩溃。
真实精度由 M8 训练出来的回归模型提供。
"""

from __future__ import annotations

import re
from typing import Protocol

from .channels import clamp
from .moves import UserMove


class Perceiver(Protocol):
    def perceive(self, utterance: str, last_agent_reply: str | None = None) -> UserMove: ...


# --- 亲密度：这句话隐含多亲密（关系无关）---
_INTIMACY: tuple[tuple[str, float], ...] = (
    (r"我想要你|我要你", 0.92),
    (r"抱抱|抱我|亲亲|亲一下|吻", 0.90),
    (r"我爱你|爱你", 0.88),
    (r"睡了吗|陪我睡|一起睡", 0.85),
    (r"老公|老婆|宝贝|亲爱的|darling", 0.80),
    (r"想你|好想你|念你", 0.70),
    (r"喜欢你|最喜欢你", 0.60),
    (r"只有你|除了你", 0.65),
    (r"你真好看|你好美|你好帅", 0.50),
    (r"陪陪我|陪我聊", 0.48),
    (r"晚安|早安", 0.35),
    (r"辛苦了|注意身体|多喝水", 0.30),
    (r"谢谢你|感谢你", 0.22),
    (r"你好|在吗|hi|hello", 0.10),
)

# --- 亲和：敌意 ← → 亲近 ---
_WARM: tuple[tuple[str, float], ...] = (
    (r"喜欢|爱|想你|抱|亲", 0.7),
    (r"谢谢|感谢|多谢|辛苦", 0.5),
    (r"太好了|棒|赞|厉害|给力", 0.6),
    (r"开心|高兴|舒服多了|放心了", 0.55),
    (r"哈哈|嘻嘻|嘿嘿", 0.45),
    (r"陪|一起|我们", 0.35),
    (r"你说得对|有道理|明白了", 0.3),
)
_COLD: tuple[tuple[str, float], ...] = (
    (r"垃圾|傻|蠢|滚|闭嘴|去死", 0.95),
    (r"废话|没用|烦死|恶心", 0.8),
    (r"讨厌|讨厌你", 0.7),
    (r"你真没用|你什么都不懂|你根本不懂", 0.8),
    (r"骗人|骗子|忽悠", 0.7),
    (r"答非所问|听不懂人话|说了多少遍", 0.6),
    (r"算了|懒得说|不想说了", 0.4),
    (r"烦|无语|服了", 0.45),
)

# --- 支配：顺从 ← → 支配 ---
_DOMINANT: tuple[tuple[str, float], ...] = (
    (r"^(给我|快点|马上|立刻|赶紧)", 0.8),
    (r"必须|应该|不许|不准|别再", 0.6),
    (r"我要求|我命令|听我的", 0.85),
    (r"为什么不|凭什么|你到底", 0.6),
    (r"[!！]{2,}", 0.4),
    (r"^\S{0,6}(发我|给我|做了|办了)", 0.55),
    (r"帮我|替我|给我查", 0.35),
)
_SUBMISSIVE: tuple[tuple[str, float], ...] = (
    (r"可以吗|好吗|行吗|方便吗", 0.5),
    (r"麻烦你|拜托|求你|请问", 0.6),
    (r"不好意思|抱歉|对不起", 0.65),
    (r"我不太懂|我不确定|不知道该", 0.5),
    (r"随你|都行|听你的", 0.55),
)

# --- 痛苦：对方自身的处境（与敌意无关）---
_DISTRESS: tuple[tuple[str, float], ...] = (
    (r"崩溃|绝望|受不了了", 0.95),
    (r"撑不(下去|住)|坚持不(下去|住)", 0.9),
    (r"想哭|哭了|眼泪", 0.85),
    (r"难受|痛苦|伤心|难过", 0.8),
    (r"害怕|恐惧|怕", 0.75),
    (r"焦虑|慌|不安", 0.7),
    (r"睡不着|失眠", 0.6),
    (r"压力大|好累|太累了", 0.6),
    (r"孤独|寂寞|没人", 0.6),
    (r"担心|不放心", 0.5),
    (r"不舒服|疼|痛", 0.45),
    (r"怎么办|没办法", 0.45),
)

# --- 是否指向 agent 本人 ---
_THIRD_PARTY = re.compile(
    r"我(老板|同事|朋友|妈|爸|对象|男朋友|女朋友|老公|老婆|孩子|领导)"
    r"|他们|她们|那个人|那家伙|公司|客服(说|讲)|电影|小说|新闻"
)
_SECOND_PERSON = re.compile(r"你|您|咱们|我们")

_NEGATORS = ("不", "没")
_INTENSIFIERS: tuple[tuple[str, float], ...] = (
    (r"非常|特别|极其|太|超|巨", 0.25),
    (r"死了|得要命|要疯", 0.30),
    (r"有点|稍微|一点点|略微", -0.20),
)


def _best(text: str, table: tuple[tuple[str, float], ...]) -> tuple[float, list[str]]:
    """取命中项里权重最大的一个（而非求和）——避免长句自动变成高强度。"""
    best = 0.0
    hits: list[str] = []
    for pattern, weight in table:
        m = re.search(pattern, text)
        if not m:
            continue
        start = max(0, m.start() - 1)
        if text[start : m.start()] in _NEGATORS:
            continue
        hits.append(pattern)
        best = max(best, weight)
    return best, hits


class HeuristicPerceiver:
    """关键词 + 简单启发式。方向大致对、永不崩溃即可。"""

    def perceive(self, utterance: str, last_agent_reply: str | None = None) -> UserMove:
        text = (utterance or "").strip()
        if not text:
            return UserMove(intensity=0.0, confidence=0.2)

        intimacy, i_hits = _best(text, _INTIMACY)
        warm, w_hits = _best(text, _WARM)
        cold, c_hits = _best(text, _COLD)
        dom, d_hits = _best(text, _DOMINANT)
        sub, s_hits = _best(text, _SUBMISSIVE)
        distress, ds_hits = _best(text, _DISTRESS)

        bump = 0.0
        for pattern, delta in _INTENSIFIERS:
            if re.search(pattern, text):
                bump += delta

        # 亲密邀请**本身就是**亲和动作 —— 不能只靠关键词表去凑。
        # 「我想要你」里没有任何"温暖词"，但它显然是一次亲近尝试；
        # 漏了这条，高亲密度输入会因为 affiliation_bid=0 而完全不触发失配机制。
        if cold < 0.2:
            warm = max(warm, intimacy * 0.85)

        affiliation = clamp(warm - cold, -1.0, 1.0)
        dominance = clamp(dom - sub, -1.0, 1.0)

        # 第三人称语境：既没有第二人称，又提到了别人 → 不指向 agent
        directed = bool(_SECOND_PERSON.search(text)) or not _THIRD_PARTY.search(text)

        signal = max(abs(affiliation), abs(dominance), distress, intimacy)
        intensity = clamp(signal * 0.85 + bump, 0.0, 1.0)
        if len(text) <= 3 and signal < 0.3:
            intensity = min(intensity, 0.15)

        n_hits = len(i_hits) + len(w_hits) + len(c_hits) + len(d_hits) + len(s_hits) + len(ds_hits)
        confidence = clamp(0.35 + 0.12 * n_hits + 0.2 * signal, 0.0, 0.9)

        return UserMove(
            affiliation_bid=affiliation,
            dominance_bid=dominance,
            intimacy_bid=clamp(intimacy, 0.0, 1.0),
            directed_at_agent=directed,
            distress_level=clamp(distress, 0.0, 1.0),
            intensity=intensity,
            confidence=confidence,
            raw={
                "hits": {
                    "intimacy": i_hits,
                    "warm": w_hits,
                    "cold": c_hits,
                    "dominant": d_hits,
                    "submissive": s_hits,
                    "distress": ds_hits,
                },
                "source": "heuristic",
            },
        )

    def explain(self, utterance: str) -> dict[str, object]:
        move = self.perceive(utterance)
        return {"move": move.to_dict(), "hits": move.raw.get("hits", {})}
