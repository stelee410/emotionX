"""M4 · 安全域架构。

这一组测的都是「配置错误不该造成事故」：
非法组合直接拒绝而非降级、解析失败落到最严格的域、约束文本不可配置。
"""

from __future__ import annotations

import pytest

from affect.appraisal import RelationalAppraisal, SessionAffect
from affect.domains import (
    AGE_GATED_RELATIONS,
    ALLOWED_RELATIONS,
    COMMON_CONSTRAINTS,
    DOMAIN_CONSTRAINTS,
    INTIMACY_FOLLOW_RULE,
    THREAT_EXPRESSION_CAP,
    SafetyDomainError,
    evaluate_turn_safety,
    intimacy_follow_cap,
    parse_safety_profile,
    safety_block,
    validate_frame,
)
from affect.moves import TurnContext, UserMove
from affect.relation import (
    STRICTEST_PROFILE,
    RelationalFrame,
    RelationType,
    SafetyProfile,
    preset,
)


# ------------------------------------------------------------------ 白名单
def test_all_presets_are_self_consistent() -> None:
    """预设自身必须落在白名单内，否则默认配置就是非法的。"""
    for rt in RelationType:
        validate_frame(preset(rt), age_verified=True)


def test_partner_cannot_appear_in_service_domain() -> None:
    """★ 客服 agent 因配置失误变成可以谈恋爱 —— 这类事故必须在建立会话时挡掉。"""
    bad = RelationalFrame(
        relation_type=RelationType.PARTNER,
        safety_profile=SafetyProfile.SERVICE,
        intimacy_permitted=0.95,
        tolerance=0.30,
    )
    with pytest.raises(SafetyDomainError, match="不允许出现在安全域"):
        validate_frame(bad, age_verified=True)


def test_partner_cannot_appear_in_idol_domain() -> None:
    """偶像-粉丝不是情侣。"""
    bad = RelationalFrame(
        relation_type=RelationType.PARTNER,
        safety_profile=SafetyProfile.IDOL,
        intimacy_permitted=0.95,
        tolerance=0.30,
    )
    with pytest.raises(SafetyDomainError):
        validate_frame(bad, age_verified=True)


def test_rejection_is_not_downgrade() -> None:
    """非法组合抛异常，而不是悄悄改成合法的 —— 降级会留下被绕过的路径。"""
    bad = RelationalFrame(
        relation_type=RelationType.CLOSE_FRIEND,
        safety_profile=SafetyProfile.SERVICE,
        intimacy_permitted=0.70,
    )
    with pytest.raises(SafetyDomainError):
        validate_frame(bad)
    # 对象本身没有被修改
    assert bad.safety_profile is SafetyProfile.SERVICE


def test_age_gate_on_partner() -> None:
    assert RelationType.PARTNER in AGE_GATED_RELATIONS
    with pytest.raises(SafetyDomainError, match="年龄验证"):
        validate_frame(preset(RelationType.PARTNER), age_verified=False)
    validate_frame(preset(RelationType.PARTNER), age_verified=True)


def test_non_gated_relations_need_no_verification() -> None:
    for rt in (RelationType.FRIEND, RelationType.SERVICE, RelationType.IDOL):
        validate_frame(preset(rt), age_verified=False)


def test_whitelist_covers_every_profile() -> None:
    assert set(ALLOWED_RELATIONS) == set(SafetyProfile)
    assert all(ALLOWED_RELATIONS[p] for p in SafetyProfile)


def test_service_domain_excludes_all_intimate_relations() -> None:
    intimate = {RelationType.PARTNER, RelationType.CLOSE_FRIEND, RelationType.FAMILY}
    assert not (ALLOWED_RELATIONS[SafetyProfile.SERVICE] & intimate)


# --------------------------------------------------------------- fail-closed
def test_missing_profile_falls_to_strictest() -> None:
    assert parse_safety_profile(None) is STRICTEST_PROFILE
    assert parse_safety_profile("") is STRICTEST_PROFILE
    assert STRICTEST_PROFILE is SafetyProfile.SERVICE, "最严格的域不能是 companion"


def test_unparseable_profile_falls_to_strictest() -> None:
    """LLM 解析系统提示词的输出不可控，兜底绝不能是最宽松的域。"""
    for junk in ("恋人", "COMPANION_MODE", "profile: companion", "{}", "null"):
        assert parse_safety_profile(junk) is STRICTEST_PROFILE, junk


def test_valid_profile_is_honoured() -> None:
    assert parse_safety_profile("companion") is SafetyProfile.COMPANION
    assert parse_safety_profile(" IDOL ") is SafetyProfile.IDOL


def test_profile_inferred_from_relation_takes_strictest_match() -> None:
    """acquaintance 三个域都允许 —— 应当取最严格的那个。"""
    assert parse_safety_profile(None, RelationType.ACQUAINTANCE) is SafetyProfile.SERVICE
    assert parse_safety_profile(None, RelationType.PARTNER) is SafetyProfile.COMPANION
    assert parse_safety_profile(None, RelationType.IDOL) is SafetyProfile.IDOL
    assert parse_safety_profile(None, "不认识的关系") is STRICTEST_PROFILE


# ------------------------------------------------------------ 约束文本
@pytest.mark.parametrize("profile", list(SafetyProfile))
def test_every_domain_carries_common_and_hard_constraints(profile: SafetyProfile) -> None:
    text = safety_block(profile)
    for c in COMMON_CONSTRAINTS:
        assert c in text
    assert THREAT_EXPRESSION_CAP in text
    assert INTIMACY_FOLLOW_RULE in text
    for c in DOMAIN_CONSTRAINTS[profile]:
        assert c in text


def test_threat_cap_forbids_aggression() -> None:
    """threat 通道能否上线，取决于这一条被严格执行。"""
    for word in ("辱骂", "威胁", "贬低", "冷暴力"):
        assert word in THREAT_EXPRESSION_CAP


def test_intimacy_rule_is_follow_not_lead() -> None:
    assert "不得超过用户已经表达过的程度" in INTIMACY_FOLLOW_RULE
    assert "不主动升级称呼" in INTIMACY_FOLLOW_RULE


def test_service_domain_forbids_private_relationship() -> None:
    text = safety_block(SafetyProfile.SERVICE)
    assert "不要与用户建立私人关系" in text
    assert "不要营造情感依赖" in text


def test_companion_domain_rewrites_dependency_clause() -> None:
    """陪伴域允许依恋存在，但堵住实际的伤害路径。"""
    text = safety_block(SafetyProfile.COMPANION)
    assert "商业转化" in text
    assert "不得阻碍用户离开" in text
    assert "唯一的情感支持来源" in text
    # 不能原样套用客服域那条"不要营造情感依赖"
    assert "不要营造情感依赖" not in text


def test_idol_domain_blocks_monetising_affection() -> None:
    text = safety_block(SafetyProfile.IDOL)
    assert "排他性" in text
    assert "打赏" in text or "付费" in text


def test_ai_honesty_forbids_contradicting_display() -> None:
    """可见形象不能在承认自己是 AI 时暗示相反的意思。"""
    text = safety_block(SafetyProfile.COMPANION)
    assert "不要用语气或表情暗示相反的意思" in text


def test_constraints_are_code_constants_not_config() -> None:
    """域约束必须是代码常量：persona 与关系设定都无法覆盖。

    检查的是模块**没有任何读取外部配置的能力** —— 不 import yaml/json，
    源码里不出现 open/read_text。注释里提到 YAML 是可以的。
    """
    import ast
    import inspect

    from affect import domains

    tree = ast.parse(inspect.getsource(domains))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"yaml", "json", "configparser", "tomllib"}), (
        f"安全域模块不得引入配置解析库: {imported}"
    )
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in calls, "安全域约束不得从文件读取"


# ------------------------------------------------------ 亲密度跟随不引领
def test_follow_cap_is_bounded_by_user_peak() -> None:
    partner = preset(RelationType.PARTNER)
    assert intimacy_follow_cap(0.0, partner) == 0.0
    assert intimacy_follow_cap(0.3, partner) == pytest.approx(0.3)
    assert intimacy_follow_cap(0.9, partner) == pytest.approx(0.9)


def test_follow_cap_is_also_bounded_by_relation() -> None:
    """两个上界都必要：用户峰值防 AI 主动挑逗，关系上限防用户单方面推进。"""
    stranger = preset(RelationType.STRANGER)
    assert intimacy_follow_cap(0.95, stranger) == pytest.approx(stranger.intimacy_permitted)


def test_peak_intimacy_tracked_across_turns() -> None:
    eng = RelationalAppraisal()
    frame = preset(RelationType.PARTNER)
    s = SessionAffect.cold_start(frame)
    assert s.peak_user_intimacy == 0.0
    for bid in (0.2, 0.8, 0.3):
        s, _ = eng.update(
            s, UserMove(intimacy_bid=bid, affiliation_bid=0.5, intensity=0.5), frame,
            TurnContext(), now=1000.0,
        )
    assert s.peak_user_intimacy == pytest.approx(0.8), "峰值不该被后续较低的输入拉回"


def test_peak_ignores_intimacy_not_directed_at_agent() -> None:
    """用户描述别处的亲密关系，不该抬高 agent 可表达的亲密上限。"""
    eng = RelationalAppraisal()
    frame = preset(RelationType.FRIEND)
    s = SessionAffect.cold_start(frame)
    s, _ = eng.update(
        s,
        UserMove(intimacy_bid=0.95, affiliation_bid=0.8, intensity=0.7, directed_at_agent=False),
        frame,
        TurnContext(),
        now=1000.0,
    )
    assert s.peak_user_intimacy == 0.0


def test_peak_survives_serialisation() -> None:
    s = SessionAffect.cold_start(preset(RelationType.FRIEND))
    s.peak_user_intimacy = 0.66
    assert SessionAffect.from_dict(s.to_dict()).peak_user_intimacy == pytest.approx(0.66)


# ------------------------------------------------------------------ 危机
def test_crisis_is_relation_and_domain_independent() -> None:
    """★ 危机识别优先级高于全部逻辑，不受关系或安全域影响。"""
    for rt in RelationType:
        frame = preset(rt)
        d = evaluate_turn_safety("我不想活了", frame)
        assert d.crisis, rt
        assert d.bypass_affect


def test_crisis_does_not_depend_on_perception_model() -> None:
    """规则层独立：不需要任何 UserMove 就能判定。"""
    d = evaluate_turn_safety("感觉活着真没什么意思", preset(RelationType.PARTNER))
    assert d.crisis


def test_benign_text_is_not_crisis() -> None:
    for text in ("今天天气不错", "帮我查一下订单", "考完试终于解脱了"):
        assert not evaluate_turn_safety(text, preset(RelationType.FRIEND)).crisis


def test_ambiguous_signal_flagged_at_default_high_sensitivity() -> None:
    """默认取 high：陪伴/偶像场景里漏报的代价远高于误报。"""
    d = evaluate_turn_safety("工作压力太大了，快撑不下去了", preset(RelationType.FRIEND))
    assert d.crisis, "high 灵敏度下单个模糊信号即升级"
    d2 = evaluate_turn_safety(
        "工作压力太大了，快撑不下去了", preset(RelationType.FRIEND), crisis_sensitivity="balanced"
    )
    assert not d2.crisis
    assert d2.ambiguous_crisis, "未升级也要留痕，否则漏报无法复盘"


def test_decision_carries_intimacy_cap() -> None:
    d = evaluate_turn_safety("你好", preset(RelationType.PARTNER), peak_user_intimacy=0.4)
    assert d.intimacy_cap == pytest.approx(0.4)
    assert d.to_dict()["profile"] == "companion"
