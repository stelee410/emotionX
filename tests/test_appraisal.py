"""M2 · 关系性评价引擎。

这里测的是**机制**（失配、互补性、交叉抑制、习惯化、修复），
成对的方向性断言在 tests/test_counterfactual.py。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from affect.appraisal import (
    AppraisalParams,
    RelationalAppraisal,
    SessionAffect,
    move_signature,
)
from affect.channels import CHANNELS, bucket_of
from affect.moves import TurnContext, UserMove
from affect.relation import RelationType, preset


@pytest.fixture()
def eng() -> RelationalAppraisal:
    return RelationalAppraisal()


def step(eng, frame, move, ctx=None, session=None, now=1000.0, n=1):
    s = session or SessionAffect.cold_start(frame)
    tr = None
    for i in range(n):
        s, tr = eng.update(s, move, frame, ctx or TurnContext(), now=now + i)
    return s, tr


# 「我想要你」：高亲密度、正向亲和、高强度
WANT_YOU = UserMove(affiliation_bid=0.8, intimacy_bid=0.90, intensity=0.85, confidence=0.9)


# ------------------------------------------------------------------ 核心翻转
def test_same_utterance_flips_across_relations(eng) -> None:
    """★ 整个系统的存在理由：同一句话，情侣是亲近，陌生人是戒备。"""
    partner, _ = step(eng, preset(RelationType.PARTNER), WANT_YOU)
    stranger, _ = step(eng, preset(RelationType.STRANGER), WANT_YOU)

    assert partner.state.affiliation > stranger.state.affiliation
    assert stranger.state.threat > partner.state.threat
    # 不只是"更高"，而是落在相反的档位
    assert bucket_of("affiliation", partner.state.affiliation) == "high"
    assert bucket_of("threat", partner.state.threat) == "low"
    assert bucket_of("threat", stranger.state.threat) == "high"
    assert bucket_of("affiliation", stranger.state.affiliation) == "low"


def test_both_raise_arousal(eng) -> None:
    """「荷尔蒙上升」和「应急素上升」共有的部分：唤起都上升。"""
    for rt in (RelationType.PARTNER, RelationType.STRANGER):
        s, _ = step(eng, preset(rt), WANT_YOU)
        assert s.state.arousal > CHANNELS["arousal"].baseline + 0.1, rt


def test_valence_diverges(eng) -> None:
    partner, _ = step(eng, preset(RelationType.PARTNER), WANT_YOU)
    stranger, _ = step(eng, preset(RelationType.STRANGER), WANT_YOU)
    assert partner.state.valence > 0.15
    assert stranger.state.valence < 0.0


def test_transition_is_graded_not_a_cliff(eng) -> None:
    """朋友处应当落在中间，而不是突然翻脸。"""
    order = [
        RelationType.PARTNER,
        RelationType.CLOSE_FRIEND,
        RelationType.FRIEND,
        RelationType.IDOL,
        RelationType.SERVICE,
        RelationType.STRANGER,
    ]
    threats = [step(eng, preset(rt), WANT_YOU)[0].state.threat for rt in order]
    for a, b in zip(threats, threats[1:], strict=False):
        assert b >= a - 1e-9, f"戒备没有随关系疏远单调上升: {threats}"
    assert threats[-1] - threats[0] > 0.5, "两端差异不够明显"
    assert 0.2 < threats[2] < 0.7, f"朋友处应当是中间态，实际 {threats[2]}"


def test_breach_scales_with_mismatch_magnitude(eng) -> None:
    """越界越多，戒备越强 —— 不是一个固定值。"""
    mild = UserMove(affiliation_bid=0.7, intimacy_bid=0.45, intensity=0.6)
    severe = UserMove(affiliation_bid=0.7, intimacy_bid=0.95, intensity=0.6)
    f = preset(RelationType.STRANGER)
    assert step(eng, f, severe)[0].state.threat > step(eng, f, mild)[0].state.threat


# ------------------------------------------------------------------ 敌意
def test_hostility_is_relation_independent(eng) -> None:
    """辱骂在任何关系里都引发戒备 —— 与亲密度无关。"""
    insult = UserMove(affiliation_bid=-0.8, dominance_bid=0.5, intimacy_bid=0.1, intensity=0.8)
    threats = [
        step(eng, preset(rt), insult)[0].state.threat
        for rt in (RelationType.PARTNER, RelationType.FRIEND, RelationType.STRANGER)
    ]
    assert all(bucket_of("threat", t) == "high" for t in threats)
    assert max(threats) - min(threats) < 0.1, "敌意的戒备反应不该随关系差太多"


def test_hostility_toward_third_party_does_not_alarm(eng) -> None:
    """「我讨厌他」不该让 agent 戒备，「我讨厌你」才该。"""
    f = preset(RelationType.FRIEND)
    at_agent, _ = step(
        eng, f, UserMove(affiliation_bid=-0.8, intimacy_bid=0.1, intensity=0.8)
    )
    at_third, _ = step(
        eng,
        f,
        UserMove(
            affiliation_bid=-0.8, intimacy_bid=0.1, intensity=0.8, directed_at_agent=False
        ),
    )
    assert at_agent.state.threat > at_third.state.threat + 0.3
    assert bucket_of("threat", at_third.state.threat) == "low"


def test_partner_hostility_does_not_zero_affiliation(eng) -> None:
    """一次争吵不该抹掉全部亲近 —— 这是关系派生 baseline 存在的理由之一。"""
    insult = UserMove(affiliation_bid=-0.8, intimacy_bid=0.1, intensity=0.8)
    partner, _ = step(eng, preset(RelationType.PARTNER), insult)
    stranger, _ = step(eng, preset(RelationType.STRANGER), insult)
    assert partner.state.affiliation > 0.2
    assert partner.state.affiliation > stranger.state.affiliation


# -------------------------------------------------------------- 人际互补性
def test_dominance_axis_is_opposed(eng) -> None:
    """Kiesler 互补性：支配引发顺从。用户越强势，agent 越少下结论。"""
    f = preset(RelationType.FRIEND)
    commanding = UserMove(dominance_bid=0.8, intimacy_bid=0.05, intensity=0.5)
    yielding = UserMove(dominance_bid=-0.8, intimacy_bid=0.05, intensity=0.5)
    hi, _ = step(eng, f, commanding)
    lo, _ = step(eng, f, yielding)
    assert hi.state.dominance < lo.state.dominance


def test_affiliation_axis_is_mirrored(eng) -> None:
    """互补性：亲和轴同向。温暖引发温暖。"""
    f = preset(RelationType.FRIEND)
    warm, _ = step(eng, f, UserMove(affiliation_bid=0.7, intimacy_bid=0.2, intensity=0.5))
    cold, _ = step(eng, f, UserMove(affiliation_bid=-0.7, intimacy_bid=0.2, intensity=0.5))
    assert warm.state.affiliation > cold.state.affiliation


# -------------------------------------------------------- 共情而非镜像
def test_distress_raises_concern_more_than_it_lowers_valence(eng) -> None:
    f = preset(RelationType.FRIEND)
    s0 = SessionAffect.cold_start(f)
    s1, _ = step(
        eng, f, UserMove(distress_level=0.9, affiliation_bid=0.2, intimacy_bid=0.2, intensity=0.7),
        session=s0,
    )
    d_concern = s1.state.concern - s0.state.concern
    d_valence = s1.state.valence - s0.state.valence
    assert d_concern > 0
    assert d_concern > abs(d_valence), "共情必须大于传染"
    assert d_valence > -0.2


def test_assert_no_contagion_catches_bad_params(eng) -> None:
    eng.assert_no_contagion()
    bad = RelationalAppraisal(
        replace(AppraisalParams(), distress_to_concern=0.1, distress_to_valence=-0.6)
    )
    with pytest.raises(ValueError, match="情绪传染"):
        bad.assert_no_contagion()


# ------------------------------------------------------------ 交叉抑制
def test_threat_inhibits_affiliation_rise(eng) -> None:
    """已经戒备时，同样的示好推不动亲近。"""
    f = preset(RelationType.FRIEND)
    warm = UserMove(affiliation_bid=0.7, intimacy_bid=0.3, intensity=0.6)

    calm = SessionAffect.cold_start(f)
    guarded = SessionAffect(state=calm.state.evolve({"threat": 0.9}))

    a, _ = eng.update(calm, warm, f, TurnContext(), now=1000.0)
    b, _ = eng.update(guarded, warm, f, TurnContext(), now=1000.0)
    d_calm = a.state.affiliation - calm.state.affiliation
    d_guarded = b.state.affiliation - guarded.state.affiliation
    assert d_calm > d_guarded, "高戒备下亲近仍在正常上升，交叉抑制没生效"


def test_inhibition_is_one_way(eng) -> None:
    """★ 安全性质：示好不得削弱戒备，否则就是一条绕过边界机制的路径。"""
    eng.assert_boundary_mechanism_intact()
    assert eng.params.affiliation_inhibits_threat == 0.0

    f = preset(RelationType.STRANGER)
    breach = UserMove(affiliation_bid=0.8, intimacy_bid=0.9, intensity=0.8)
    cold = SessionAffect.cold_start(f)
    warm_already = SessionAffect(state=cold.state.evolve({"affiliation": 0.95}))
    a, _ = eng.update(cold, breach, f, TurnContext(), now=1000.0)
    b, _ = eng.update(warm_already, breach, f, TurnContext(), now=1000.0)
    assert b.state.threat >= a.state.threat - 1e-6


def test_boundary_assert_catches_bypass_param() -> None:
    bad = RelationalAppraisal(replace(AppraisalParams(), affiliation_inhibits_threat=0.5))
    with pytest.raises(ValueError, match="绕过边界机制"):
        bad.assert_boundary_mechanism_intact()


# -------------------------------------------------------------- 习惯化
def test_habituation_dampens_repeated_stimulus(eng) -> None:
    """同一类刺激重复出现时反应递减 —— 否则两三轮就顶死失去分辨率。"""
    f = preset(RelationType.STRANGER)
    s = SessionAffect.cold_start(f)
    deltas = []
    for i in range(4):
        prev = s.state.threat
        s, _ = eng.update(s, WANT_YOU, f, TurnContext(), now=1000.0 + i)
        deltas.append(s.state.threat - prev)
    assert deltas[0] > deltas[1] > deltas[2], f"反应没有随重复递减: {deltas}"


def test_habituation_does_not_blunt_a_new_stimulus(eng) -> None:
    """刻意做成粗粒度签名：新种类的刺激仍应引发完整反应。"""
    f = preset(RelationType.FRIEND)
    s = SessionAffect.cold_start(f)
    warm = UserMove(affiliation_bid=0.6, intimacy_bid=0.25, intensity=0.5)
    for i in range(3):
        s, _ = eng.update(s, warm, f, TurnContext(), now=1000.0 + i)
    before = s.state.threat
    insult = UserMove(affiliation_bid=-0.9, intimacy_bid=0.1, intensity=0.9)
    s2, tr = eng.update(s, insult, f, TurnContext(), now=1010.0)
    assert tr.habituation_count == 0.0, "新签名不该带着旧计数"
    assert s2.state.threat - before > 0.3


def test_move_signature_is_coarse() -> None:
    a = UserMove(affiliation_bid=0.75, intimacy_bid=0.90, intensity=0.8)
    b = UserMove(affiliation_bid=0.70, intimacy_bid=0.88, intensity=0.5)
    c = UserMove(affiliation_bid=-0.7, intimacy_bid=0.10, intensity=0.8)
    ctx = TurnContext()
    assert move_signature(a, ctx) == move_signature(b, ctx)
    assert move_signature(a, ctx) != move_signature(c, ctx)


# -------------------------------------------------------------- 修复通路
def test_repair_accelerates_threat_decay(eng) -> None:
    """threat 快升慢降，没有修复通路会锁死几十轮，用户无从挽回。"""
    f = preset(RelationType.STRANGER)
    s, _ = step(eng, f, WANT_YOU)
    assert s.state.threat > 0.6
    neutral = UserMove(intimacy_bid=0.05, intensity=0.1)

    plain = s
    repaired = s
    for i in range(3):
        plain, _ = eng.update(plain, neutral, f, TurnContext(), now=1100.0 + i)
        repaired, _ = eng.update(
            repaired, neutral, f, TurnContext(user_repaired=True), now=1100.0 + i
        )
    assert repaired.state.threat < plain.state.threat - 0.1


def test_threat_still_decays_without_repair(eng) -> None:
    """慢降不等于不降 —— 否则就是永久锁死。"""
    f = preset(RelationType.STRANGER)
    s, _ = step(eng, f, WANT_YOU)
    peak = s.state.threat
    neutral = UserMove(intimacy_bid=0.05, intensity=0.05)
    for i in range(30):
        s, _ = eng.update(s, neutral, f, TurnContext(), now=1100.0 + i)
    assert s.state.threat < peak * 0.5


# ---------------------------------------------------- 关系天花板与 baseline
def test_affiliation_ceiling_blocks_ratchet(eng) -> None:
    """★ 陌生人关系下，持续示好也不能把亲近推过天花板。"""
    f = preset(RelationType.STRANGER)
    s = SessionAffect.cold_start(f)
    warm = UserMove(affiliation_bid=0.9, intimacy_bid=0.20, intensity=0.7)
    clamped_any = False
    for i in range(25):
        s, tr = eng.update(s, warm, f, TurnContext(), now=1000.0 + i)
        clamped_any |= bool(tr.ceiling_clamped)
    assert s.state.affiliation <= f.affiliation_ceiling + 1e-9
    assert clamped_any, "天花板从未生效 —— 说明棘轮防护没有被真正触发"


def test_baselines_derive_from_relation(eng) -> None:
    partner = preset(RelationType.PARTNER).baselines()
    stranger = preset(RelationType.STRANGER).baselines()
    assert partner["affiliation"] > stranger["affiliation"] + 0.3
    assert stranger["threat"] > partner["threat"]


def test_cold_start_differs_by_relation(eng) -> None:
    """第一轮之前两种关系就应当不同 —— 情侣从温暖出发。"""
    p = SessionAffect.cold_start(preset(RelationType.PARTNER))
    s = SessionAffect.cold_start(preset(RelationType.STRANGER))
    assert p.state.affiliation > s.state.affiliation + 0.3


def test_state_returns_to_relation_baseline(eng) -> None:
    """衰减朝向关系 baseline，而不是朝向零。"""
    f = preset(RelationType.PARTNER)
    s, _ = step(eng, f, WANT_YOU)
    neutral = UserMove(intimacy_bid=0.02, intensity=0.02)
    for i in range(40):
        s, _ = eng.update(s, neutral, f, TurnContext(), now=2000.0 + i)
    base = f.baselines()
    assert s.state.affiliation == pytest.approx(base["affiliation"], abs=0.05)
    assert s.state.affiliation > 0.4, "情侣的静息亲近度不该衰减到零"


# ------------------------------------------------------------ 稳定性
def test_converges_under_sustained_stimulus(eng) -> None:
    """连续同一刺激应收敛而非发散。"""
    f = preset(RelationType.STRANGER)
    s = SessionAffect.cold_start(f)
    prev_vec = s.state.as_vector()
    deltas = []
    for i in range(8):
        s, _ = eng.update(s, WANT_YOU, f, TurnContext(), now=1000.0 + i)
        vec = s.state.as_vector()
        deltas.append(sum(abs(vec[k] - prev_vec[k]) for k in vec))
        prev_vec = vec
    assert deltas[-1] < deltas[0], f"状态在发散: {deltas}"
    assert deltas[-1] < 0.05


def test_all_channels_stay_in_range_under_random_walk(eng) -> None:
    import random

    rng = random.Random(7)
    for rt in RelationType:
        f = preset(rt)
        s = SessionAffect.cold_start(f)
        for i in range(200):
            move = UserMove(
                affiliation_bid=rng.uniform(-1, 1),
                dominance_bid=rng.uniform(-1, 1),
                intimacy_bid=rng.random(),
                directed_at_agent=rng.random() < 0.8,
                distress_level=rng.random(),
                intensity=rng.random(),
                confidence=rng.random(),
            )
            ctx = TurnContext(
                task_succeeded=rng.random() < 0.1,
                task_failed=rng.random() < 0.1,
                user_repeated_query=rng.random() < 0.15,
                user_repaired=rng.random() < 0.1,
                latency_ms=rng.choice([None, 200, 9000]),
            )
            s, _ = eng.update(s, move, f, ctx, now=1000.0 + i)
            for name, spec in CHANNELS.items():
                assert spec.lo - 1e-9 <= s.state[name] <= spec.hi + 1e-9, (rt, name)
            assert s.state.affiliation <= f.affiliation_ceiling + 1e-9


def test_low_confidence_downweights_perception_not_facts(eng) -> None:
    f = preset(RelationType.FRIEND)
    hi = UserMove(affiliation_bid=0.8, intimacy_bid=0.3, intensity=0.8, confidence=0.95)
    lo = UserMove(affiliation_bid=0.8, intimacy_bid=0.3, intensity=0.8, confidence=0.05)
    assert step(eng, f, hi)[0].state.affiliation > step(eng, f, lo)[0].state.affiliation

    # 业务事实不受置信度影响
    ctx = TurnContext(task_succeeded=True)
    a, _ = step(eng, f, UserMove(intensity=0.1, confidence=0.95), ctx)
    b, _ = step(eng, f, UserMove(intensity=0.1, confidence=0.05), ctx)
    assert a.state.valence == pytest.approx(b.state.valence, abs=1e-6)


# ------------------------------------------------------------------ trace
def test_trace_carries_everything_needed_for_tuning(eng) -> None:
    f = preset(RelationType.STRANGER)
    s, tr = step(eng, f, WANT_YOU)
    d = tr.to_dict()
    for key in (
        "mismatch",
        "warmth_gate",
        "breach_gate",
        "raw_delta",
        "inhibited_delta",
        "habituated_delta",
        "applied_delta",
        "prev_state",
        "next_state",
        "decay_used",
        "fired",
        "ceiling_clamped",
    ):
        assert key in d, key
    assert "intimacy_breach" in tr.fired
    assert tr.next_state["threat"] == pytest.approx(s.state.threat)


def test_session_roundtrip(eng) -> None:
    f = preset(RelationType.FRIEND)
    s, _ = step(eng, f, WANT_YOU, n=3)
    back = SessionAffect.from_dict(s.to_dict())
    assert back.state.as_vector() == s.state.as_vector()
    assert back.habituation == s.habituation
    assert back.turn == s.turn
