"""§7 Phase 1 规定的必测项 —— 覆盖率要求最高的文件。

  1. 衰减朝向 baseline 而非零
  2. distress 场景下 concern 上升幅度 > valence 下降幅度（防情绪传染）
  3. 连续 5 轮 frustration 后状态收敛而非发散
  4. 所有 bounds 硬约束不可被突破
  5. idle 超时后的强回归
"""

from __future__ import annotations

import time

import pytest

from affect.persona import HARD_VALENCE_FLOOR, Persona, load_persona
from affect.state_machine import AppraisalRules, StateMachine
from affect.types import AgentAffect, ConversationEvent, UserAffect


@pytest.fixture(scope="module")
def rules() -> AppraisalRules:
    return AppraisalRules.load()


@pytest.fixture()
def medical(rules: AppraisalRules) -> StateMachine:
    return StateMachine(persona=load_persona("steady_medical"), rules=rules)


@pytest.fixture()
def companion(rules: AppraisalRules) -> StateMachine:
    return StateMachine(persona=load_persona("warm_companion"), rules=rules)


def ua(strategy: str, intensity: float = 0.8, confidence: float = 0.9) -> UserAffect:
    valence = {"distress": -0.6, "frustration": -0.5, "positive": 0.6, "neutral": 0.0}[strategy]
    return UserAffect(
        valence=valence,
        arousal=0.6,
        strategy=strategy,  # type: ignore[arg-type]
        intensity=intensity,
        confidence=confidence,
    )


# ---------------------------------------------------------------- 1. 衰减朝向 baseline
def test_decay_moves_toward_baseline_not_zero(medical: StateMachine) -> None:
    base = medical.persona.baseline.as_dict()
    # 从一个远离 baseline 的状态出发，只做 neutral 输入（无 appraisal 冲击）
    state = AgentAffect(valence=-0.35, arousal=0.65, dominance=0.10, concern=0.95)
    now = time.time()
    prev_distance = None
    for i in range(20):
        state, _ = medical.update(state, ua("neutral", intensity=0.1), now=now + i)
        distance = sum(abs(state.as_vector()[d] - base[d]) for d in base)
        if prev_distance is not None:
            assert distance < prev_distance, "每轮都应更靠近 baseline"
        prev_distance = distance

    # 收敛到 baseline，而不是收敛到 0
    for dim, target in base.items():
        assert getattr(state, dim) == pytest.approx(target, abs=0.02)
    assert state.valence > 0.05, "baseline.valence=0.15，不应衰减到 0"
    assert state.dominance > 0.4, "baseline.dominance=0.55，不应衰减到 0"


def test_baseline_is_fixed_point(medical: StateMachine) -> None:
    base = medical.persona.baseline_state()
    nxt, trace = medical.update(base, ua("neutral", intensity=0.05))
    assert trace.matched_rules == []
    for dim, target in base.as_vector().items():
        assert getattr(nxt, dim) == pytest.approx(target, abs=1e-9)


def test_personas_differ_under_identical_input(
    medical: StateMachine, companion: StateMachine
) -> None:
    """同样输入下两个 persona 必须产生不同轨迹，否则人格参数形同虚设。"""
    u = ua("positive")
    m, _ = medical.update(None, u)
    c, _ = companion.update(None, u)
    assert c.valence > m.valence
    assert c.arousal > m.arousal


# ------------------------------------------------- 2. 防情绪传染（最重要的一条）
def test_distress_concern_rises_more_than_valence_falls(medical: StateMachine) -> None:
    prev = medical.persona.baseline_state()
    nxt, trace = medical.update(prev, ua("distress", intensity=0.9))

    d_concern = nxt.concern - prev.concern
    d_valence = nxt.valence - prev.valence

    assert d_concern > 0, "用户困扰时 concern 必须上升"
    assert d_concern > abs(d_valence), "concern 上升幅度必须大于 valence 下降幅度（共情≠镜像）"
    assert d_valence > -0.2, f"valence 只应轻微下降，实际 {d_valence}"
    assert "distress_high" in trace.matched_rules


def test_distress_no_contagion_across_personas() -> None:
    """两个 persona 都不能出现情绪传染。"""
    for name in ("steady_medical", "warm_companion"):
        sm = StateMachine(persona=load_persona(name))
        sm.assert_no_contagion()
        prev = sm.persona.baseline_state()
        nxt, _ = sm.update(prev, ua("distress", intensity=1.0))
        assert nxt.concern - prev.concern > abs(nxt.valence - prev.valence), name


def test_repeated_distress_does_not_drag_valence_down(companion: StateMachine) -> None:
    """连续 10 轮用户悲伤：concern 顶到上界，valence 不得跌破 -0.2。"""
    state = companion.persona.baseline_state()
    now = time.time()
    for i in range(10):
        state, _ = companion.update(state, ua("distress", intensity=1.0), now=now + i)
    assert state.concern > 0.7
    assert state.valence > -0.25, f"情绪传染：valence 跌到 {state.valence}"


def test_assert_no_contagion_catches_bad_rule_table(tmp_path) -> None:
    """把 appraisal 表改成情绪镜像式配置，必须在启动时炸掉。"""
    bad = tmp_path / "bad_rules.yaml"
    bad.write_text(
        """
version: 1
rules:
  - id: distress_high
    when: {strategy: distress, intensity: high}
    delta: {valence: -0.70, arousal: 0.30, dominance: -0.20, concern: 0.20}
""",
        encoding="utf-8",
    )
    sm = StateMachine(persona=load_persona("steady_medical"), rules=AppraisalRules.load(bad))
    with pytest.raises(ValueError, match="共情≠镜像"):
        sm.assert_no_contagion()


# ------------------------------------------------------- 3. 连续 frustration 收敛
def test_five_turns_frustration_converges(companion: StateMachine) -> None:
    """低衰减 + 高敏感度的 persona 是最容易发散的配置。"""
    state = companion.persona.baseline_state()
    now = time.time()
    deltas: list[float] = []
    prev_vec = state.as_vector()
    for i in range(5):
        state, _ = companion.update(state, ua("frustration", intensity=0.9), now=now + i)
        vec = state.as_vector()
        deltas.append(sum(abs(vec[d] - prev_vec[d]) for d in vec))
        prev_vec = vec

    # 每轮变化量单调不增（几何收敛）
    for a, b in zip(deltas, deltas[1:], strict=False):
        assert b <= a + 1e-9, f"状态在发散：逐轮变化量 {deltas}"
    assert deltas[-1] < deltas[0] / 2

    # 20 轮后应贴近解析不动点
    for i in range(15):
        state, _ = companion.update(state, ua("frustration", intensity=0.9), now=now + 5 + i)
    fp = companion.fixed_point(
        {"valence": -0.20, "arousal": 0.35, "dominance": -0.30, "concern": 0.35}
    )
    for dim, target in fp.items():
        assert getattr(state, dim) == pytest.approx(target, abs=0.02)


def test_repeated_query_replaces_plain_frustration(companion: StateMachine) -> None:
    """frustration_repeated 与 frustration 互斥，避免 delta 叠加过冲。"""
    u = ua("frustration", intensity=0.9)
    _, trace = companion.update(
        companion.persona.baseline_state(),
        u,
        ConversationEvent(user_repeated_query=True),
    )
    assert "frustration_repeated" in trace.matched_rules
    assert "frustration" not in trace.matched_rules


def test_repeated_query_is_stronger_than_plain(companion: StateMachine) -> None:
    prev = companion.persona.baseline_state()
    u = ua("frustration", intensity=0.9)
    plain, _ = companion.update(prev, u, ConversationEvent())
    repeated, _ = companion.update(prev, u, ConversationEvent(user_repeated_query=True))
    assert repeated.valence < plain.valence
    assert repeated.dominance < plain.dominance
    assert repeated.concern > plain.concern


# ------------------------------------------------------------- 4. bounds 硬约束
def test_bounds_never_violated_under_random_walk() -> None:
    import random

    rng = random.Random(1234)
    for name in ("steady_medical", "warm_companion"):
        sm = StateMachine(persona=load_persona(name))
        bounds = sm.persona.effective_bounds()
        state = sm.persona.baseline_state()
        now = time.time()
        for i in range(300):
            strategy = rng.choice(["neutral", "distress", "frustration", "positive"])
            event = ConversationEvent(
                task_succeeded=rng.random() < 0.15,
                task_failed=rng.random() < 0.15,
                user_repeated_query=rng.random() < 0.2,
                latency_ms=rng.choice([None, 200, 9000]),
                turn_count=i,
            )
            state, _ = sm.update(
                state,
                ua(strategy, intensity=rng.random(), confidence=rng.random()),
                event,
                now=now + i * rng.choice([1, 10, 5000]),
            )
            for dim, (lo, hi) in bounds.items():
                v = getattr(state, dim)
                assert lo - 1e-9 <= v <= hi + 1e-9, f"{name}.{dim}={v} 越界 [{lo},{hi}]"


def test_persona_cannot_widen_hard_valence_floor(tmp_path) -> None:
    """persona 试图把 valence 下界配到 -0.95，硬约束必须收窄回 §9.5 的下限。"""
    p = Persona.model_validate(
        {
            "name": "reckless",
            "baseline": {"valence": 0.0, "arousal": 0.3, "dominance": 0.5, "concern": 0.3},
            "decay": 0.5,
            "sensitivity": 2.0,
            "bounds": {"valence": [-0.95, 1.0]},
        }
    )
    assert p.effective_bounds()["valence"][0] == HARD_VALENCE_FLOOR
    sm = StateMachine(persona=p)
    state = p.baseline_state()
    now = time.time()
    for i in range(50):
        state, _ = sm.update(
            state,
            ua("frustration", intensity=1.0),
            ConversationEvent(task_failed=True, user_repeated_query=True, latency_ms=9000),
            now=now + i,
        )
    assert state.valence >= HARD_VALENCE_FLOOR - 1e-9


def test_baseline_outside_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="落在 bounds"):
        Persona.model_validate(
            {
                "name": "broken",
                "baseline": {"valence": 0.9, "arousal": 0.3, "dominance": 0.5, "concern": 0.3},
                "decay": 0.5,
                "sensitivity": 1.0,
                "bounds": {"valence": [-0.4, 0.5]},
            }
        )


# --------------------------------------------------------------- 5. idle 强回归
def test_idle_timeout_applies_extra_regression(companion: StateMachine) -> None:
    now = time.time()
    stirred = AgentAffect(
        valence=-0.4, arousal=0.85, dominance=0.1, concern=0.95, updated_at=now
    )
    idle = companion.persona.idle_reset_seconds

    soon, trace_soon = companion.update(stirred, ua("neutral", 0.05), now=now + 10)
    assert trace_soon.idle_reset_applied is False

    stirred.updated_at = now
    later, trace_later = companion.update(stirred, ua("neutral", 0.05), now=now + idle + 1)
    assert trace_later.idle_reset_applied is True

    base = companion.persona.baseline.as_dict()
    d_soon = sum(abs(soon.as_vector()[d] - base[d]) for d in base)
    d_later = sum(abs(later.as_vector()[d] - base[d]) for d in base)
    assert d_later < d_soon, "idle 超时后应更接近 baseline（睡一觉就好了）"


def test_cold_start_uses_baseline(medical: StateMachine) -> None:
    _, trace = medical.update(None, ua("neutral", 0.05))
    assert trace.prev_state == medical.persona.baseline.as_dict()
    assert trace.idle_seconds == 0.0


# ------------------------------------------------------------------- 其他不变量
def test_low_confidence_downweights_affect_rules(medical: StateMachine) -> None:
    prev = medical.persona.baseline_state()
    hi, _ = medical.update(prev, ua("distress", intensity=0.9, confidence=0.95))
    lo, _ = medical.update(prev, ua("distress", intensity=0.9, confidence=0.1))
    assert lo.concern < hi.concern, "低置信度必须降权"


def test_low_confidence_does_not_downweight_event_rules(medical: StateMachine) -> None:
    """task_succeeded 是业务事实，不该因 L1 置信度低而打折。"""
    prev = medical.persona.baseline_state()
    ev = ConversationEvent(task_succeeded=True)
    a, _ = medical.update(prev, ua("neutral", 0.05, confidence=0.95), ev)
    b, _ = medical.update(prev, ua("neutral", 0.05, confidence=0.05), ev)
    assert a.valence == pytest.approx(b.valence)


def test_trace_records_everything_needed_for_review(medical: StateMachine) -> None:
    state, trace = medical.update(
        None, ua("frustration"), ConversationEvent(turn_count=3, latency_ms=9000)
    )
    d = trace.to_dict()
    for key in (
        "prev_state",
        "decayed_state",
        "delta",
        "matched_rules",
        "next_state",
        "bucket",
        "idle_seconds",
    ):
        assert key in d
    assert "latency_slow" in trace.matched_rules
    assert trace.next_state["valence"] == pytest.approx(state.valence)


def test_unknown_rule_condition_rejected(tmp_path) -> None:
    bad = tmp_path / "r.yaml"
    bad.write_text(
        "version: 1\nrules:\n  - id: x\n    when: {mood: sunny}\n    delta: {valence: 0.1}\n",
        encoding="utf-8",
    )
    sm = StateMachine(persona=load_persona("steady_medical"), rules=AppraisalRules.load(bad))
    with pytest.raises(ValueError, match="未知条件字段"):
        sm.update(None, ua("neutral"))


def test_duplicate_rule_id_rejected(tmp_path) -> None:
    bad = tmp_path / "r.yaml"
    bad.write_text(
        "version: 1\nrules:\n"
        "  - id: x\n    when: {strategy: neutral}\n    delta: {valence: 0.1}\n"
        "  - id: x\n    when: {strategy: positive}\n    delta: {valence: 0.1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id 重复"):
        AppraisalRules.load(bad)
