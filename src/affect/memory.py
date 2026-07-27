"""外部记忆系统的适配口。

**本项目不实现记忆系统。** 记忆由外部系统提供，这里只定义接口和两条测试通路：

    ManualMemory   手动注入 —— 在 WebUI 里直接写几条记忆，看它怎么影响表达
    HttpMemory     调用外部 HTTP 服务

一条刻意的设计约束：**检索结果不回写情感状态**。

心境一致性检索（情绪偏置检索）+ 检索结果反过来更新情绪 = 正反馈。
在人身上这就是抑郁性反刍的机制：心情差 → 想起坏事 → 心情更差 → 想起更坏的事。
闭上这个环，系统会自发滑向单一情绪吸引子，而且滑得慢、要跑几十轮才显形、
上线后极难归因。

所以这里是**单向**的：情感状态可以影响检索（作为查询的一部分传出去），
但检索回来的内容只进 prompt，不进 appraisal。这是一个「特意不实现的生物学机制」。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .channels import AffectState
from .relation import RelationalFrame

# 单次注入 prompt 的记忆条数上限 —— 太多会淹没本轮的行为指令
MAX_NOTES = 5
DEFAULT_TIMEOUT = 2.0


@dataclass
class MemoryQuery:
    """发给外部记忆系统的查询。情感状态作为**检索偏置**传出去。"""

    session_id: str
    utterance: str
    relation_type: str
    # 当前情感状态，供外部系统做心境一致性检索
    affect: dict[str, float] = field(default_factory=dict)
    turn: int = 0
    limit: int = MAX_NOTES

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MemoryAdapter(Protocol):
    def recall(self, query: MemoryQuery) -> tuple[str, ...]: ...


class NullMemory:
    """默认：没有记忆系统。"""

    def recall(self, query: MemoryQuery) -> tuple[str, ...]:
        return ()


@dataclass
class ManualMemory:
    """手动注入。WebUI 的测试台用这个 —— 写几条记忆立刻看到表达变化。

    可选按关键词过滤：notes 里形如 "关键词|内容" 的条目只在 utterance
    含该关键词时召回，方便手搓「情境触发」的测试。
    """

    notes: list[str] = field(default_factory=list)

    def recall(self, query: MemoryQuery) -> tuple[str, ...]:
        out: list[str] = []
        text = query.utterance or ""
        for note in self.notes:
            if "|" in note:
                keyword, content = note.split("|", 1)
                if keyword.strip() and keyword.strip() not in text:
                    continue
                out.append(content.strip())
            else:
                out.append(note.strip())
        return tuple(n for n in out if n)[: query.limit]


@dataclass
class HttpMemory:
    """调用外部记忆服务。

    约定：POST {url}，请求体是 MemoryQuery 的 JSON，
    响应体为 {"notes": ["...", ...]} 或直接一个字符串数组。

    失败时返回空而不是抛异常 —— 记忆系统不可用不应当拖垮整条对话链路。
    """

    url: str
    timeout: float = DEFAULT_TIMEOUT
    headers: dict[str, str] = field(default_factory=dict)
    last_error: str | None = field(default=None, init=False)

    def recall(self, query: MemoryQuery) -> tuple[str, ...]:
        self.last_error = None
        payload = json.dumps(query.to_dict(), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ()
        notes = body.get("notes", body) if isinstance(body, dict) else body
        if not isinstance(notes, list):
            self.last_error = f"响应格式不对：期望 list，得到 {type(notes).__name__}"
            return ()
        return tuple(str(n).strip() for n in notes if str(n).strip())[: query.limit]


def build_query(
    session_id: str,
    utterance: str,
    state: AffectState,
    frame: RelationalFrame,
    turn: int = 0,
    limit: int = MAX_NOTES,
) -> MemoryQuery:
    return MemoryQuery(
        session_id=session_id,
        utterance=utterance,
        relation_type=frame.relation_type.value,
        affect={k: round(v, 3) for k, v in state.as_vector().items()},
        turn=turn,
        limit=limit,
    )
