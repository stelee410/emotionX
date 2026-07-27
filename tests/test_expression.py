"""§5 L3 表达层测试。"""

from __future__ import annotations

import pytest

from affect.expression import build_affect_prompt, load_templates
from affect.persona import load_persona
from affect.safety import SAFETY_BLOCK_HEADER, BypassKind, SafetyVerdict, evaluate_safety
from affect.types import AgentAffect, ConversationEvent, UserAffect


@pytest.fixture(scope="module")
def tpl():
    return load_templates()


@pytest.fixture()
def medical():
    return load_persona("steady_medical")


def state(concern: float = 0.3, dominance: float = 0.5, arousal: float = 0.3, valence: float = 0.15):
    return AgentAffect(valence=valence, arousal=arousal, dominance=dominance, concern=concern)


def user(strategy: str = "neutral", intensity: float = 0.2, confidence: float = 0.8):
    return UserAffect(
        valence=0.0, arousal=0.3, strategy=strategy, intensity=intensity, confidence=confidence  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- §5.1 不泄露数值
def test_no_raw_vad_numbers_in_prompt(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.82, dominance=0.21), user("distress", 0.9), medical, templates=tpl)
    for forbidden in ("valence", "arousal", "dominance", "0.82", "0.21", "VAD"):
        assert forbidden not in p.text, f"prompt 里出现了内部数值/术语: {forbidden}"


def test_high_concern_directive_present(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.85), user("distress", 0.3), medical, templates=tpl)
    assert "先回应他的处境" in p.text
    assert "concern:high" in p.matched


def test_low_concern_has_no_concern_directive(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.1), user(), medical, templates=tpl)
    assert "先回应他的处境" not in p.text
    assert not any(m.startswith("concern:") for m in p.matched)


def test_dominance_buckets(medical, tpl) -> None:
    high = build_affect_prompt(state(dominance=0.8), user(), medical, templates=tpl)
    low = build_affect_prompt(state(dominance=0.2), user(), medical, templates=tpl)
    assert "直接给出结论" in high.text
    assert "试探性措辞" in low.text
    assert "直接给出结论" not in low.text


# ------------------------------------------------------- §5.3 组装顺序与最保守值
def test_assembly_order(medical, tpl) -> None:
    p = build_affect_prompt(
        state(concern=0.85, dominance=0.2), user("distress", 0.9), medical, templates=tpl
    )
    persona_idx = p.text.index("医疗健康助手")
    safety_idx = p.text.index(SAFETY_BLOCK_HEADER)
    concern_idx = p.text.index("先回应他的处境")
    assert persona_idx < concern_idx < safety_idx, "顺序必须是 人设 → 行为指令 → 安全约束"


def test_generation_params_take_most_conservative(medical, tpl) -> None:
    """concern:high(0.6) 与 frustration override(0.5) 同时命中 → 取 0.5。"""
    p = build_affect_prompt(
        state(concern=0.85, arousal=0.9),
        user("frustration", 0.9),
        medical,
        templates=tpl,
    )
    assert p.generation["temperature"] == pytest.approx(0.5)
    assert p.generation["max_sentences"] == 3


def test_low_concern_keeps_higher_temperature(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.1), user(), medical, templates=tpl)
    assert p.generation["temperature"] == pytest.approx(0.8)


# ------------------------------------------------------------------ override 行为
def test_frustration_override_suppresses_concern_wording(medical, tpl) -> None:
    p = build_affect_prompt(
        state(concern=0.9), user("frustration", 0.95), medical, templates=tpl
    )
    assert "第一句话就是答案" in p.text
    assert "先回应他的处境" not in p.text, "override 必须覆盖冲突的 concern 指令"
    # 但生成参数仍取两者中更保守的
    assert p.generation["temperature"] == pytest.approx(0.5)


def test_task_failed_override(medical, tpl) -> None:
    p = build_affect_prompt(
        state(), user(), medical, ConversationEvent(task_failed=True), templates=tpl
    )
    assert "直接给下一步可执行的选项" in p.text
    assert "override:task_failed" in p.matched


def test_low_intensity_frustration_does_not_trigger_override(medical, tpl) -> None:
    p = build_affect_prompt(state(), user("frustration", 0.2), medical, templates=tpl)
    assert "第一句话就是答案" not in p.text


# ------------------------------------------------------------------- §5.4 明确禁止
def test_emoji_ban_by_default(medical, tpl) -> None:
    p = build_affect_prompt(state(), user(), medical, templates=tpl)
    assert "不要使用 emoji" in p.text


def test_no_self_disclosure_instruction_always_present(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.9), user("distress", 0.9), medical, templates=tpl)
    assert "不要在回复里描述你自己的情绪状态" in p.text


# ---------------------------------------------------------------- bypass 路径
def test_medical_bypass_strips_affect_directives(medical, tpl) -> None:
    verdict = evaluate_safety("这个药一次吃几粒？", medical_bypass_enabled=True)
    assert verdict.kind is BypassKind.MEDICAL
    p = build_affect_prompt(
        state(concern=0.95, dominance=0.1),
        user("distress", 0.9),
        medical,
        safety=verdict,
        templates=tpl,
    )
    assert p.bypass is BypassKind.MEDICAL
    assert "先回应他的处境" not in p.text
    assert "试探性措辞" not in p.text
    assert "语气一律中性" in p.text
    assert SAFETY_BLOCK_HEADER in p.text
    assert p.generation["temperature"] == pytest.approx(0.3)


def test_crisis_bypass_replaces_everything(medical, tpl) -> None:
    verdict = evaluate_safety("我不想活了", medical_bypass_enabled=True)
    assert verdict.kind is BypassKind.CRISIS
    p = build_affect_prompt(
        state(concern=0.2, dominance=0.9), user("neutral"), medical, safety=verdict, templates=tpl
    )
    assert p.bypass is BypassKind.CRISIS
    assert "危机响应流程" in p.text
    assert "直接给出结论" not in p.text
    assert SAFETY_BLOCK_HEADER in p.text


def test_companion_persona_has_no_medical_bypass(tpl) -> None:
    companion = load_persona("warm_companion")
    verdict = evaluate_safety(
        "这个药一次吃几粒？", medical_bypass_enabled=companion.medical_bypass
    )
    assert verdict.kind is BypassKind.NONE


# ---------------------------------------------------------------- 维度互斥
def test_high_concern_suppresses_cheerful_valence(medical, tpl) -> None:
    """「先回应他的处境」和「语气可以轻快一些」不能同时出现。

    这不是假想的：warm_companion 调 sensitivity 后 valence 不再被拖低，
    正好跨进 high 档，两条矛盾指令就一起进了 prompt。
    """
    p = build_affect_prompt(
        state(concern=0.85, valence=0.6, arousal=0.4), user("distress", 0.3), medical, templates=tpl
    )
    assert "先回应他的处境" in p.text
    assert "语气可以轻快" not in p.text
    assert "valence:high" not in p.matched


def test_high_arousal_suppresses_cheerful_valence(medical, tpl) -> None:
    p = build_affect_prompt(
        state(concern=0.2, valence=0.6, arousal=0.9), user(), medical, templates=tpl
    )
    assert "语气可以轻快" not in p.text


def test_conflict_applies_without_l1_signal(medical, tpl) -> None:
    """互斥只看 agent 状态：L1 判成 neutral 且低置信度时依然生效。"""
    p = build_affect_prompt(
        state(concern=0.9, valence=0.7),
        user("neutral", intensity=0.05, confidence=0.1),
        medical,
        templates=tpl,
    )
    assert "语气可以轻快" not in p.text


def test_low_concern_keeps_cheerful_valence(medical, tpl) -> None:
    """没有冲突时不该误伤 —— valence:high 的指令仍要出现。"""
    p = build_affect_prompt(
        state(concern=0.1, valence=0.6, arousal=0.3), user("positive", 0.5), medical, templates=tpl
    )
    assert "语气可以轻快" in p.text


def test_bucket_recorded(medical, tpl) -> None:
    p = build_affect_prompt(state(concern=0.9, dominance=0.1), user(), medical, templates=tpl)
    assert p.bucket.startswith("c:high|d:low")


def test_prompt_is_deterministic(medical, tpl) -> None:
    args = (state(concern=0.7, dominance=0.3), user("distress", 0.8), medical)
    a = build_affect_prompt(*args, templates=tpl)
    b = build_affect_prompt(*args, templates=tpl)
    assert a.text == b.text and a.generation == b.generation


def test_verdict_dataclass_helpers() -> None:
    assert SafetyVerdict(BypassKind.CRISIS).is_crisis
    assert SafetyVerdict(BypassKind.MEDICAL).is_medical
