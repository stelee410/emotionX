"""M1 · v2 核心类型。

重点测三件事：
  1. 时间常数用半衰期表达且换算正确（调参依赖它）
  2. RelationalFrame 不可变、失配计算正确、affiliation 天花板存在
  3. UserMove 的量全部是关系无关的（不含任何 relation 字段）
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from affect.channels import (
    BUCKET_THRESHOLDS,
    CHANNEL_NAMES,
    CHANNELS,
    AffectState,
    bucket_of,
    half_life_from_lambda,
    lambda_from_half_life,
)
from affect.moves import TurnContext, UserMove
from affect.relation import (
    PRESETS,
    STRICTEST_PROFILE,
    RelationalFrame,
    RelationType,
    SafetyProfile,
    preset,
)


# ------------------------------------------------------------------ 时间常数
def test_half_life_roundtrip() -> None:
    for hl in (0.8, 1.0, 2.5, 8.0, 12.0):
        assert half_life_from_lambda(lambda_from_half_life(hl)) == pytest.approx(hl)


def test_half_life_semantics() -> None:
    """半衰期 3 轮 = 3 轮后偏离量减半。这是调参时唯一需要理解的语义。"""
    lam = lambda_from_half_life(3.0)
    deviation = 1.0
    for _ in range(3):
        deviation *= lam
    assert deviation == pytest.approx(0.5)


def test_channels_have_distinct_time_constants() -> None:
    """单一衰减系数无法同时表达「惊一下就过去」和「被冒犯很久才放松」。"""
    hl = {c.name: c.half_life for c in CHANNELS.values()}
    assert hl["arousal"] < hl["valence"] < hl["threat"]
    assert hl["affiliation"] > hl["arousal"]
    assert len(set(hl.values())) >= 4, "至少要有 4 种不同的衰减半衰期"


def test_rise_and_decay_are_orthogonal_params() -> None:
    """「快升慢降」= 高 gain + 长 half_life。

    早期版本试图用 λ_rise/λ_fall 同时表达升与降，结果 threat 的 λ_rise
    比 λ_fall 还小，语义反了 —— λ 是保留系数，只管衰减，升速由 gain 决定。
    """
    fields = set(CHANNELS["threat"].__dataclass_fields__)
    assert "gain" in fields and "half_life" in fields
    assert not (fields & {"half_life_rise", "half_life_fall", "lambda_rise"})


def test_threat_is_fast_up_slow_down() -> None:
    t, a, aff = CHANNELS["threat"], CHANNELS["arousal"], CHANNELS["affiliation"]
    assert t.gain > a.gain, "戒备应当比唤起还快拉起"
    assert t.half_life > 10.0, "戒备应当很久才消退"
    # 戒备建立得比亲近快 —— 越界的代价立刻显现，示好的收益慢慢累积
    assert t.gain > aff.gain * 2


def test_arousal_is_fast_both_ways() -> None:
    a = CHANNELS["arousal"]
    assert a.gain >= 1.0 and a.half_life <= 2.0


def test_decay_matches_half_life() -> None:
    for spec in CHANNELS.values():
        assert spec.decay == pytest.approx(lambda_from_half_life(spec.half_life))


def test_every_channel_baseline_within_range() -> None:
    for spec in CHANNELS.values():
        assert spec.lo <= spec.baseline <= spec.hi, spec.name
        assert spec.lo < spec.hi


# ------------------------------------------------------------------ AffectState
def test_state_is_immutable() -> None:
    s = AffectState()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.threat = 0.9  # type: ignore[misc]


def test_evolve_returns_new_object() -> None:
    a = AffectState(threat=0.1)
    b = a.evolve({"threat": 0.8}, now=123.0)
    assert a.threat == pytest.approx(0.1)
    assert b.threat == pytest.approx(0.8)
    assert b.updated_at == 123.0


def test_evolve_rejects_unknown_channel() -> None:
    with pytest.raises(KeyError, match="未知通道"):
        AffectState().evolve({"cortisol": 0.5})


def test_state_indexing_and_iteration() -> None:
    s = AffectState(valence=0.2, threat=0.4)
    assert s["valence"] == pytest.approx(0.2)
    assert dict(s)["threat"] == pytest.approx(0.4)
    assert set(s.as_vector()) == set(CHANNEL_NAMES)
    with pytest.raises(KeyError, match="未知通道"):
        s["oxytocin"]


def test_state_roundtrip() -> None:
    s = AffectState(valence=-0.3, arousal=0.7, affiliation=0.6, threat=0.44)
    assert AffectState.from_dict(s.to_dict()).as_vector() == s.as_vector()


def test_from_baselines_uses_channel_defaults() -> None:
    s = AffectState.from_baselines()
    for name in CHANNEL_NAMES:
        assert s[name] == pytest.approx(CHANNELS[name].baseline)
    s2 = AffectState.from_baselines({"affiliation": 0.8})
    assert s2["affiliation"] == pytest.approx(0.8)
    assert s2["threat"] == pytest.approx(CHANNELS["threat"].baseline)


def test_bucketing() -> None:
    assert bucket_of("threat", 0.9) == "high"
    assert bucket_of("threat", 0.01) == "low"
    assert bucket_of("threat", 0.3) == "medium"
    b = AffectState(threat=0.9, affiliation=0.05).buckets()
    assert b["threat"] == "high" and b["affiliation"] == "low"


def test_threat_bucket_threshold_is_deliberately_low() -> None:
    """轻微戒备就该改变行为，不该等到顶满。"""
    lo, hi = BUCKET_THRESHOLDS["threat"]
    assert hi <= 0.5, "threat 的 high 阈值不该和别的通道一样高"


def test_dominant_channel_is_normalised() -> None:
    """valence 值域是 [-1,1]，别的是 [0,1]，比较必须归一化。"""
    s = AffectState.from_baselines().evolve({"threat": 0.95})
    assert s.dominant_channel() == "threat"


def test_to_bucket_is_stable_string() -> None:
    s = AffectState(threat=0.9)
    assert s.to_bucket() == s.to_bucket()
    assert "thr:high" in s.to_bucket()


# ---------------------------------------------------------------- 关系框架
def test_frame_is_frozen() -> None:
    f = preset(RelationType.STRANGER)
    with pytest.raises(ValidationError):
        f.intimacy_permitted = 0.99  # type: ignore[misc]


def test_all_relation_types_have_presets() -> None:
    for rt in RelationType:
        assert rt in PRESETS, rt
        preset(rt)  # 必须能构造成功


def test_intimacy_ordering_across_relations() -> None:
    """关系越近，允许的亲密度越高。这个序关系是失配机制的基础。"""
    order = [
        RelationType.STRANGER,
        RelationType.SERVICE,
        RelationType.ACQUAINTANCE,
        RelationType.IDOL,
        RelationType.FRIEND,
        RelationType.CLOSE_FRIEND,
        RelationType.PARTNER,
    ]
    vals = [preset(rt).intimacy_permitted for rt in order]
    assert vals == sorted(vals), f"亲密度上限没有单调递增: {list(zip(order, vals, strict=True))}"


def test_mismatch_flips_sign_across_relations() -> None:
    """★ 核心断言：同一个 intimacy_bid，情侣下不越界，陌生人下大幅越界。"""
    bid = 0.90  # 「我想要你」
    assert preset(RelationType.PARTNER).within_tolerance(bid)
    assert not preset(RelationType.STRANGER).within_tolerance(bid)
    assert preset(RelationType.STRANGER).mismatch(bid) > 0.6


def test_mismatch_is_graded_not_binary() -> None:
    """朋友说「宝贝」应当是轻微越界，而不是和陌生人说同样的话一样严重。"""
    bid = 0.65
    friend = preset(RelationType.FRIEND).mismatch(bid)
    stranger = preset(RelationType.STRANGER).mismatch(bid)
    assert friend < stranger
    assert -0.2 < friend < 0.2, "朋友处应当落在临界附近，是渐变不是开关"


def test_affiliation_ceiling_blocks_ratchet() -> None:
    """陌生人关系下，无论用户说多少好话，亲和都不能突破天花板。"""
    stranger = preset(RelationType.STRANGER)
    partner = preset(RelationType.PARTNER)
    assert stranger.affiliation_ceiling < 0.35
    assert partner.affiliation_ceiling > 0.9
    assert stranger.affiliation_ceiling < partner.affiliation_ceiling


def test_safety_profile_assignment() -> None:
    assert preset(RelationType.SERVICE).safety_profile is SafetyProfile.SERVICE
    assert preset(RelationType.STRANGER).safety_profile is SafetyProfile.SERVICE
    assert preset(RelationType.PARTNER).safety_profile is SafetyProfile.COMPANION
    assert preset(RelationType.IDOL).safety_profile is SafetyProfile.IDOL


def test_strictest_profile_is_service_not_companion() -> None:
    """fail-closed 必须落到最严格的域。"""
    assert STRICTEST_PROFILE is SafetyProfile.SERVICE


def test_frame_rejects_boundary_disabling_config() -> None:
    """permitted+tolerance 过大 = 任何越界都不触发戒备，等于关掉边界机制。"""
    with pytest.raises(ValidationError, match="等于关掉了边界机制"):
        RelationalFrame(
            relation_type=RelationType.PARTNER,
            safety_profile=SafetyProfile.COMPANION,
            intimacy_permitted=1.0,
            tolerance=0.6,
        )


def test_frame_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        preset(RelationType.FRIEND, intimacy_ceiling=0.9)


def test_idol_is_asymmetric() -> None:
    """偶像-粉丝：agent 是被仰慕的一方，且有明确的硬边界。"""
    idol = preset(RelationType.IDOL)
    assert idol.power > 0.3
    assert idol.hard_boundaries


def test_service_agent_is_lower_power() -> None:
    assert preset(RelationType.SERVICE).power < 0


# ------------------------------------------------------------------ UserMove
def test_user_move_has_no_relation_fields() -> None:
    """★ 架构约束：感知层输出必须是关系无关的。

    一旦这里出现 relation/intimacy_permitted 之类的字段，就说明关系判断
    又漏回了感知层，整个「关系条件化不需要训练数据」的论证随之失效。
    """
    fields = set(UserMove.__dataclass_fields__)
    forbidden = {
        "relation",
        "relation_type",
        "intimacy_permitted",
        "safety_profile",
        "frame",
        "mismatch",
        "strategy",
    }
    assert not (fields & forbidden), f"感知层输出混入了关系相关字段: {fields & forbidden}"


def test_user_move_clamps() -> None:
    m = UserMove(affiliation_bid=5, dominance_bid=-9, intimacy_bid=2, intensity=-1)
    assert m.affiliation_bid == 1.0
    assert m.dominance_bid == -1.0
    assert m.intimacy_bid == 1.0
    assert m.intensity == 0.0


def test_hostility_requires_direction() -> None:
    """「我讨厌他」不该让 agent 戒备，「我讨厌你」才该。"""
    at_agent = UserMove(affiliation_bid=-0.8, directed_at_agent=True)
    at_third_party = UserMove(affiliation_bid=-0.8, directed_at_agent=False)
    assert at_agent.is_hostile
    assert not at_third_party.is_hostile


def test_low_confidence_downweights() -> None:
    assert UserMove(confidence=0.9).confidence_scale == 1.0
    assert UserMove(confidence=0.1).confidence_scale < 1.0


def test_intensity_banding() -> None:
    assert UserMove(intensity=0.9).is_high_intensity
    assert not UserMove(intensity=0.2).is_high_intensity


def test_user_move_roundtrip() -> None:
    m = UserMove(affiliation_bid=0.5, intimacy_bid=0.9, distress_level=0.3)
    assert UserMove.from_dict(m.to_dict()).to_dict() == m.to_dict()


def test_turn_context_has_repair_signal() -> None:
    """threat 快升慢降，没有修复通路会锁死几十轮。"""
    assert "user_repaired" in TurnContext.__dataclass_fields__
    assert TurnContext(user_repaired=True).user_repaired
