"""pipeline 串联 + store 持久化 + 规则桩 的行为测试。"""

from __future__ import annotations

import time

import pytest

from affect.perception import HeuristicPerceiver
from affect.pipeline import AffectPipeline
from affect.store import InMemoryStateStore, RedisStateStore, TraceLogger, make_store
from affect.types import AgentAffect, ConversationEvent


@pytest.fixture()
def pipeline(tmp_path) -> AffectPipeline:
    return AffectPipeline(
        store_backend="memory", trace_logger=TraceLogger(tmp_path / "trace.jsonl")
    )


def test_process_turn_returns_spec_tuple(pipeline: AffectPipeline) -> None:
    r = pipeline.process_turn("s1", "你好", "", ConversationEvent(turn_count=1), "steady_medical")
    prompt, gen = r.as_tuple()
    assert isinstance(prompt, str) and prompt
    assert "temperature" in gen


def test_state_persists_across_turns(pipeline: AffectPipeline) -> None:
    sid = "s-persist"
    r1 = pipeline.process_turn(sid, "我好难受，特别害怕", persona_name="warm_companion")
    r2 = pipeline.process_turn(sid, "嗯", persona_name="warm_companion")
    assert r2.trace.prev_state["concern"] == pytest.approx(r1.agent_affect.concern)
    assert r2.agent_affect.concern < r1.agent_affect.concern, "无新刺激应向 baseline 衰减"


def test_reset_session_returns_to_cold_start(pipeline: AffectPipeline) -> None:
    sid = "s-reset"
    pipeline.process_turn(sid, "太好了，谢谢你！", persona_name="warm_companion")
    pipeline.reset_session(sid)
    assert pipeline.get_state(sid) is None
    r = pipeline.process_turn(sid, "嗯", persona_name="warm_companion")
    base = pipeline.persona("warm_companion").baseline.as_dict()
    assert r.trace.prev_state == base


def test_trace_written_to_jsonl(tmp_path) -> None:
    logger = TraceLogger(tmp_path / "t.jsonl")
    p = AffectPipeline(store_backend="memory", trace_logger=logger)
    for i in range(3):
        p.process_turn("s-log", f"第{i}轮，还是不行", persona_name="steady_medical")
    records = logger.read_session("s-log")
    assert len(records) == 3
    assert all("next_state" in r for r in records)


def test_idle_regression_through_pipeline(pipeline: AffectPipeline) -> None:
    sid = "s-idle"
    now = time.time()
    r1 = pipeline.process_turn(
        sid, "又错了！说了多少遍了", persona_name="warm_companion", now=now
    )
    assert r1.agent_affect.arousal > 0.6
    idle = pipeline.persona("warm_companion").idle_reset_seconds
    r2 = pipeline.process_turn(sid, "在吗", persona_name="warm_companion", now=now + idle + 60)
    assert r2.trace.idle_reset_applied
    base = pipeline.persona("warm_companion").baseline.as_dict()
    assert abs(r2.agent_affect.arousal - base["arousal"]) < 0.1


def test_two_sessions_are_isolated(pipeline: AffectPipeline) -> None:
    pipeline.process_turn("a", "我快崩溃了", persona_name="warm_companion")
    pipeline.process_turn("b", "谢谢，解决了！", persona_name="warm_companion")
    sa, sb = pipeline.get_state("a"), pipeline.get_state("b")
    assert sa and sb
    assert sa.concern > sb.concern
    assert sb.valence > sa.valence


# ------------------------------------------------------------------- store 层
def test_memory_store_roundtrip_and_ttl() -> None:
    s = InMemoryStateStore(ttl_seconds=0.05)
    st = AgentAffect(valence=0.1, arousal=0.2, dominance=0.3, concern=0.4)
    s.set("k", st)
    got = s.get("k")
    assert got is not None and got.concern == pytest.approx(0.4)
    time.sleep(0.06)
    assert s.get("k") is None


def test_memory_store_delete() -> None:
    s = InMemoryStateStore()
    s.set("k", AgentAffect(0.0, 0.1, 0.2, 0.3))
    s.delete("k")
    assert s.get("k") is None


class FakeRedis:
    """最小 Redis 替身，验证 RedisStateStore 的序列化契约。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value
        self.ttls[key] = ttl

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def test_redis_store_serializes_json() -> None:
    fake = FakeRedis()
    store = RedisStateStore(client=fake, ttl_seconds=3600)
    st = AgentAffect(valence=-0.2, arousal=0.7, dominance=0.3, concern=0.8)
    store.set("sess", st)
    key = "affect:state:sess"
    assert key in fake.data and fake.ttls[key] == 3600
    got = store.get("sess")
    assert got is not None and got.arousal == pytest.approx(0.7)
    store.delete("sess")
    assert store.get("sess") is None


def test_redis_store_survives_corrupt_payload() -> None:
    fake = FakeRedis()
    fake.data["affect:state:x"] = "{not json"
    store = RedisStateStore(client=fake)
    assert store.get("x") is None, "脏数据应当作冷启动，而不是抛异常"


def test_make_store_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="未知 store backend"):
        make_store("postgres")


# --------------------------------------------------------------- 规则桩 sanity
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("又错了，说了多少遍了", "frustration"),
        ("我特别害怕，晚上睡不着", "distress"),
        ("太好了，谢谢你，解决了", "positive"),
        ("帮我查一下明天的天气", "neutral"),
        ("这个报告的第三项是什么意思", "neutral"),
        ("我快撑不下去了", "distress"),
        ("能不能别废话，直接说", "frustration"),
    ],
)
def test_heuristic_stub_directionality(text: str, expected: str) -> None:
    """桩不追求精度，但方向性必须对，否则 Phase 2 的链路验证没有意义。"""
    assert HeuristicPerceiver().perceive(text).strategy == expected


def test_heuristic_handles_negation() -> None:
    assert HeuristicPerceiver().perceive("今天不难受了").strategy != "distress"


def test_heuristic_intensity_scales_with_intensifier() -> None:
    p = HeuristicPerceiver()
    mild = p.perceive("有点难过")
    strong = p.perceive("非常难过，一直在哭")
    assert strong.intensity > mild.intensity
