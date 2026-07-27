"""§4.4 状态持久化：Redis + 内存两种 backend。

内存 backend 用于测试与单机 demo；生产用 Redis，key 按 session_id，TTL 24h。
另含 TraceLogger：把每轮的 S_t 落到 JSONL，供 §8.2 轨迹评审。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from .types import AgentAffect, TurnTrace

DEFAULT_TTL_SECONDS = 24 * 3600
KEY_PREFIX = "affect:state:"


class StateStore(Protocol):
    def get(self, session_id: str) -> AgentAffect | None: ...
    def set(self, session_id: str, state: AgentAffect) -> None: ...
    def delete(self, session_id: str) -> None: ...


class InMemoryStateStore:
    """线程安全的内存 backend，带 TTL 惰性过期。"""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl = ttl_seconds
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> AgentAffect | None:
        with self._lock:
            item = self._data.get(session_id)
            if item is None:
                return None
            written_at, payload = item
            if time.time() - written_at > self.ttl:
                del self._data[session_id]
                return None
            return AgentAffect.from_dict(payload)

    def set(self, session_id: str, state: AgentAffect) -> None:
        with self._lock:
            self._data[session_id] = (time.time(), state.to_dict())

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)


class RedisStateStore:
    """Redis backend。序列化为 JSON，字段即 AgentAffect。"""

    def __init__(
        self,
        url: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        client: Any | None = None,
    ) -> None:
        self.ttl = int(ttl_seconds)
        if client is not None:
            self._r = client
        else:
            import redis  # 延迟导入：内存 backend 用户无需装 redis-py

            self._r = redis.Redis.from_url(
                url or os.environ.get("AFFECT_REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}{session_id}"

    def get(self, session_id: str) -> AgentAffect | None:
        raw = self._r.get(self._key(session_id))
        if not raw:
            return None
        try:
            return AgentAffect.from_dict(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            # 脏数据不应打断服务：当作冷启动
            self._r.delete(self._key(session_id))
            return None

    def set(self, session_id: str, state: AgentAffect) -> None:
        self._r.setex(self._key(session_id), self.ttl, json.dumps(state.to_dict()))

    def delete(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


def make_store(backend: str = "memory", **kwargs: Any) -> StateStore:
    """backend: 'memory' | 'redis'（也可用环境变量 AFFECT_STORE_BACKEND）。"""
    backend = (backend or os.environ.get("AFFECT_STORE_BACKEND", "memory")).lower()
    if backend == "memory":
        return InMemoryStateStore(**kwargs)
    if backend == "redis":
        return RedisStateStore(**kwargs)
    raise ValueError(f"未知 store backend: {backend!r}")


class TraceLogger:
    """§4.4：必须记录状态轨迹。JSONL 追加写，一行一轮。"""

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        default = Path(__file__).resolve().parents[2] / "logs" / "affect_trace.jsonl"
        self.path = Path(path) if path else default
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, trace: TurnTrace) -> None:
        if not self.enabled:
            return
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("session_id") == session_id:
                    out.append(rec)
        return out
