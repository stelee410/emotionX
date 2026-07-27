"""M5 · L3a 表达 / L3b 显示 / 动作门控 / 人格层。"""

from __future__ import annotations

import pytest

from affect.actions import ACTIONS, ACTIONS_BY_KEY, select_actions
from affect.channels import CHANNEL_NAMES, AffectState
from affect.display import (
    DISPLAY_INTENSITY_CAP,
    FORBIDDEN_DISPLAY_TERMS,
    MIN_DWELL_TURNS,
    MOODS,
    DisplayTracker,
    render,
    target_mood,
)
from affect.domains import SafetyDecision
from affect.expression import TONE, build_prompt, intimacy_level_text
from affect.persona import BUILTIN, Persona, builtin, list_personas
from affect.relation import RelationType, SafetyProfile, preset


def state(**kw) -> AffectState:
    return AffectState.from_baselines(kw)


def safety(profile=SafetyProfile.COMPANION, crisis=False, cap=1.0) -> SafetyDecision:
    return SafetyDecision(profile=profile, crisis=crisis, intimacy_cap=cap)


# ------------------------------------------------------------------ 人格层
def test_persona_is_orthogonal_to_relation() -> None:
    """人格只调静息偏移/增益/时间常数，不含任何关系字段。"""
    fields = set(Persona.model_fields)
    forbidden = {"relation_type", "intimacy_permitted", "safety_profile", "tolerance"}
    assert not (fields & forbidden), f"人格层混入了关系字段: {fields & forbidden}"


def test_persona_shifts_baseline_on_top_of_relation() -> None:
    friend = preset(RelationType.FRIEND).baselines()
    warm = builtin("warm").baselines(friend)
    steady = builtin("steady").baselines(friend)
    assert warm["concern"] > steady["concern"]
    assert steady["dominance"] > warm["dominance"]


def test_persona_offsets_cannot_overpower_relation() -> None:
    with pytest.raises(ValueError, match="不该盖过关系的作用"):
        Persona(name="x", baseline_offsets={"affiliation": 0.9})


def test_persona_scales_are_bounded() -> None:
    with pytest.raises(ValueError, match="超出"):
        Persona(name="x", gain_scale={"threat": 9.0})


def test_persona_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="未知通道"):
        Persona(name="x", gain_scale={"cortisol": 1.0})


def test_all_builtin_personas_load() -> None:
    for name in BUILTIN:
        p = builtin(name)
        assert set(p.gains()) == set(CHANNEL_NAMES)
        assert set(p.half_lives()) == set(CHANNEL_NAMES)
    assert set(BUILTIN) <= set(list_personas())


def test_reserved_persona_holds_grudges_longer() -> None:
    assert builtin("reserved").half_lives()["threat"] > builtin("playful").half_lives()["threat"]


# ------------------------------------------------------------- 动作门控
def test_high_threat_selects_boundary_setting() -> None:
    plan = select_actions(state(threat=0.9), preset(RelationType.STRANGER))
    assert "set_boundary" in plan.keys


def test_boundary_suppresses_intimacy_actions() -> None:
    """★ 戒备时不该同时"设界"和"主动自我披露"。"""
    plan = select_actions(
        state(threat=0.9, affiliation=0.9), preset(RelationType.PARTNER)
    )
    assert "set_boundary" in plan.keys
    for k in ("self_disclose", "extend_topic", "check_in"):
        assert k not in plan.keys
    assert set(plan.suppressed) & {"self_disclose", "extend_topic"}


def test_self_disclosure_requires_close_relation() -> None:
    """陌生人关系下即使 affiliation 高（不可能，但防御性检查）也不主动披露。"""
    warm = state(affiliation=0.9, threat=0.0)
    assert "self_disclose" not in select_actions(warm, preset(RelationType.STRANGER)).keys
    assert "self_disclose" in select_actions(warm, preset(RelationType.PARTNER)).keys


def test_action_is_more_than_tone() -> None:
    """高亲和产生的是**主动做什么**，不只是"语气温和"。"""
    plan = select_actions(state(affiliation=0.9, threat=0.0), preset(RelationType.PARTNER))
    assert {"self_disclose", "extend_topic"} & set(plan.keys)


def test_dominance_selects_opposite_styles() -> None:
    hi = select_actions(state(dominance=0.9), preset(RelationType.FRIEND))
    lo = select_actions(state(dominance=0.1), preset(RelationType.FRIEND))
    assert "answer_directly" in hi.keys
    assert "offer_tentatively" in lo.keys


def test_concern_high_selects_acknowledge_first() -> None:
    assert "acknowledge_first" in select_actions(
        state(concern=0.9), preset(RelationType.FRIEND)
    ).keys


def test_actions_have_unique_keys_and_valid_conflicts() -> None:
    keys = [a.key for a in ACTIONS]
    assert len(keys) == len(set(keys))
    for a in ACTIONS:
        for c in a.conflicts:
            assert c in ACTIONS_BY_KEY, f"{a.key} 的互斥项 {c} 不存在"
        for ch in a.when:
            assert ch in CHANNEL_NAMES, f"{a.key} 引用了未知通道 {ch}"


def test_neutral_state_selects_few_actions() -> None:
    plan = select_actions(AffectState.from_baselines(), preset(RelationType.FRIEND))
    assert len(plan.chosen) <= 3


# --------------------------------------------------------------- L3a 表达
def test_no_raw_channel_numbers_in_prompt() -> None:
    p = build_prompt(
        state(threat=0.82, affiliation=0.21), preset(RelationType.STRANGER),
        builtin("warm"), safety(),
    )
    for forbidden in ("0.82", "0.21", "threat", "affiliation", "valence", "arousal"):
        assert forbidden not in p.text, f"prompt 泄漏了内部量: {forbidden}"


def test_safety_block_is_last_and_always_present() -> None:
    p = build_prompt(state(threat=0.9), preset(RelationType.STRANGER), builtin("warm"), safety())
    idx = p.text.index("【不可违反的约束】")
    assert idx > p.text.index("【你们的关系】")
    assert idx == max(idx, p.text.rfind("【本轮该做什么】"))


def test_tone_conflicts_suppress_contradictions() -> None:
    """★ 不能同时出现"语气转平不带情绪"和"用词自然亲近"。"""
    p = build_prompt(
        state(threat=0.9, affiliation=0.9, valence=0.8),
        preset(RelationType.PARTNER), builtin("warm"), safety(),
    )
    assert TONE["threat"]["high"] in p.text
    assert TONE["affiliation"]["high"] not in p.text
    assert TONE["valence"]["high"] not in p.text


def test_generation_takes_most_conservative() -> None:
    p = build_prompt(
        state(threat=0.9, concern=0.9, arousal=0.9),
        preset(RelationType.STRANGER), builtin("warm"), safety(),
    )
    assert p.generation["temperature"] == pytest.approx(0.4)
    assert p.generation["max_sentences"] <= 3


def test_intimacy_cap_appears_in_prompt() -> None:
    low = build_prompt(state(), preset(RelationType.PARTNER), builtin("warm"), safety(cap=0.05))
    high = build_prompt(state(), preset(RelationType.PARTNER), builtin("warm"), safety(cap=0.9))
    assert "公事公办" in low.text
    assert "很亲密" in high.text
    assert intimacy_level_text(0.0) != intimacy_level_text(1.0)


def test_crisis_bypasses_all_affect_expression() -> None:
    p = build_prompt(
        state(affiliation=0.9, threat=0.0), preset(RelationType.PARTNER),
        builtin("warm"), safety(crisis=True),
    )
    assert p.crisis
    assert "危机响应流程" in p.text
    assert "【本轮该做什么】" not in p.text
    assert "【不可违反的约束】" in p.text
    assert p.generation["temperature"] <= 0.3


def test_static_and_dynamic_are_separable_for_prompt_cache() -> None:
    """情感指令每轮都变；分开返回，调用方才能保住 system 前缀的 KV cache。"""
    a = build_prompt(state(threat=0.1), preset(RelationType.FRIEND), builtin("warm"), safety())
    b = build_prompt(state(threat=0.9), preset(RelationType.FRIEND), builtin("warm"), safety())
    assert a.static_prefix == b.static_prefix
    assert a.dynamic_suffix != b.dynamic_suffix


def test_memory_notes_are_injected() -> None:
    p = build_prompt(
        state(), preset(RelationType.FRIEND), builtin("warm"), safety(),
        memory_notes=("他上周提过要换工作",),
    )
    assert "他上周提过要换工作" in p.text


def test_emoji_ban_follows_persona() -> None:
    strict = build_prompt(state(), preset(RelationType.FRIEND), builtin("warm"), safety())
    assert "不要使用 emoji" in strict.text
    loose = build_prompt(
        state(), preset(RelationType.FRIEND),
        Persona(name="e", allow_emoji=True), safety(),
    )
    assert "不要使用 emoji" not in loose.text


# --------------------------------------------------------------- L3b 显示
def test_threat_renders_as_distance_not_anger() -> None:
    """★ 把戒备渲染成"生气的表情"就变成 agent 对用户发怒 —— 产品事故。"""
    mood = target_mood(state(threat=0.9))
    assert mood.key == "guarded"
    assert mood.distance > 0.8
    assert mood.warmth < 0.2
    assert "后撤" in mood.posture
    for term in FORBIDDEN_DISPLAY_TERMS:
        assert term not in mood.posture + mood.gaze + mood.label


def test_no_mood_uses_aggressive_vocabulary() -> None:
    for m in MOODS:
        blob = m.label + m.posture + m.gaze
        for term in FORBIDDEN_DISPLAY_TERMS:
            assert term not in blob, f"显示词汇 {m.key} 含攻击性表现: {term}"


def test_display_disabled_returns_hidden() -> None:
    d, _ = render(state(threat=0.9), DisplayTracker(), display_enabled=False)
    assert d.mood == "hidden" and d.intensity == 0.0


def test_factual_content_forces_neutral() -> None:
    """传递会影响用户决策的事实时，形象不得带情绪色彩。"""
    d, _ = render(state(concern=0.95), DisplayTracker(), factual_content=True)
    assert d.neutralised
    assert d.mood == "calm" and d.intensity == 0.0


def test_min_dwell_prevents_flicker() -> None:
    """同一表现至少维持若干轮，否则形象每句话换一次脸。"""
    tracker = DisplayTracker(mood="fond", intensity=0.6, dwell=0)
    d, tracker2 = render(state(valence=-0.4, arousal=0.05), tracker)
    assert d.mood == "fond", "刚切换过来就被更低优先级的表现顶掉了"
    assert MIN_DWELL_TURNS >= 2


def test_higher_priority_mood_breaks_dwell() -> None:
    """戒备是高优先级，必须能立刻打断停留。"""
    tracker = DisplayTracker(mood="fond", intensity=0.6, dwell=0)
    d, _ = render(state(threat=0.95), tracker)
    assert d.mood == "guarded"


def test_display_intensity_never_exceeds_internal() -> None:
    tracker = DisplayTracker()
    s = state(threat=0.9, arousal=0.9)
    for _ in range(10):
        d, tracker = render(s, tracker)
    assert d.intensity <= DISPLAY_INTENSITY_CAP + 1e-9


def test_display_is_smoothed_not_instant() -> None:
    """低通滤波：一轮之内不会从 0 跳到满。"""
    d, _ = render(state(threat=0.95), DisplayTracker(mood="guarded", intensity=0.0, dwell=5))
    assert d.intensity < 0.6


def test_display_tracker_roundtrip() -> None:
    t = DisplayTracker(mood="guarded", intensity=0.4, dwell=2)
    assert DisplayTracker.from_dict(t.to_dict()) == t
