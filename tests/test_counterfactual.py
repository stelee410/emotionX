"""M3 · 反事实成对测试。

每个 YAML 用例展开成一个 pytest 用例，失败时直接打印是哪条断言、实际值多少。
同一套用例也被 `eval/run_counterfactual.py` 和 WebUI 复用。
"""

from __future__ import annotations

import pytest

from affect.appraisal import RelationalAppraisal
from affect.counterfactual import (
    evaluate_assertion,
    load_cases,
    run_all,
    run_case,
    summarize,
)

CASES = load_cases()
ENGINE = RelationalAppraisal()


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_counterfactual_case(case) -> None:
    result = run_case(case, ENGINE)
    failed = [a for a in result.assertions if not a.ok]
    if failed:
        lines = [f"用例 {case.id}「{case.utterance}」有 {len(failed)} 条断言不成立："]
        lines += [f"  ✗ {a.text}\n      {a.detail}" for a in failed]
        lines.append(f"  a 终态: { {k: round(v, 3) for k, v in result.states['a'].items()} }")
        lines.append(f"  b 终态: { {k: round(v, 3) for k, v in result.states['b'].items()} }")
        pytest.fail("\n".join(lines))


def test_suite_has_meaningful_coverage() -> None:
    """用例太少的话，这套「真值来源」就撑不住调参决策。"""
    assert len(CASES) >= 30
    tags = {t for c in CASES for t in c.tags}
    for required in ("core", "safety", "empathy", "repair", "complementarity", "habituation"):
        assert required in tags, f"缺少 {required} 类用例"


def test_every_case_asserts_a_direction() -> None:
    """只断言绝对值的用例没有意义 —— 方向才是可靠的真值。"""
    for c in CASES:
        cross = [e for e in c.expect if ("a." in e and "b." in e) or e.split()[-1] in ("up", "down", "flat")]
        assert cross, f"用例 {c.id} 没有任何方向性断言"


def test_direction_accuracy_is_perfect_on_current_params() -> None:
    """整体正确率。参数调错时这里会立刻掉下来。"""
    s = summarize(run_all(engine=ENGINE))
    assert s["direction_accuracy"] == 1.0, s["failures"]


# ------------------------------------------------------- 断言语言本身的测试
@pytest.fixture()
def env():
    values = {"a": {"threat": 0.8, "affiliation": 0.2}, "b": {"threat": 0.1, "affiliation": 0.7}}
    baselines = {"a": {"threat": 0.1, "affiliation": 0.5}, "b": {"threat": 0.1, "affiliation": 0.5}}
    return values, baselines


def test_cross_side_comparison(env) -> None:
    v, b = env
    assert evaluate_assertion("a.threat > b.threat", v, b).ok
    assert not evaluate_assertion("b.threat > a.threat", v, b).ok


def test_comparison_with_margin(env) -> None:
    v, b = env
    assert evaluate_assertion("a.threat > b.threat + 0.5", v, b).ok
    assert not evaluate_assertion("a.threat > b.threat + 0.9", v, b).ok


def test_comparison_with_constant(env) -> None:
    v, b = env
    assert evaluate_assertion("a.threat > 0.5", v, b).ok
    assert evaluate_assertion("b.threat < 0.2", v, b).ok


def test_bucket_assertion(env) -> None:
    v, b = env
    assert evaluate_assertion("a.threat is high", v, b).ok
    assert evaluate_assertion("b.threat is low", v, b).ok


def test_direction_assertion(env) -> None:
    v, b = env
    assert evaluate_assertion("a.threat up", v, b).ok
    assert evaluate_assertion("a.affiliation down", v, b).ok
    assert evaluate_assertion("b.threat flat", v, b).ok


def test_unknown_channel_is_rejected(env) -> None:
    v, b = env
    with pytest.raises(ValueError, match="未知通道"):
        evaluate_assertion("a.cortisol > b.cortisol", v, b)


def test_malformed_assertion_is_rejected(env) -> None:
    v, b = env
    with pytest.raises(ValueError, match="无法解析"):
        evaluate_assertion("a.threat 很高", v, b)


def test_duplicate_case_id_is_rejected(tmp_path) -> None:
    f = tmp_path / "dup.yaml"
    f.write_text(
        """
moves: {m: {intensity: 0.5}}
cases:
  - id: x
    a: {relation: friend, turns: [m]}
    b: {relation: friend, turns: [m]}
    expect: [a.threat > 0.0]
  - id: x
    a: {relation: friend, turns: [m]}
    b: {relation: friend, turns: [m]}
    expect: [a.threat > 0.0]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id 重复"):
        load_cases(f)


def test_unknown_move_template_is_rejected(tmp_path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text(
        """
moves: {m: {intensity: 0.5}}
cases:
  - id: x
    a: {relation: friend, turns: [nope]}
    b: {relation: friend, turns: [m]}
    expect: [a.threat > 0.0]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未定义的动作模板"):
        load_cases(f)


def test_case_without_assertions_is_rejected(tmp_path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text(
        """
moves: {m: {intensity: 0.5}}
cases:
  - id: x
    a: {relation: friend, turns: [m]}
    b: {relation: friend, turns: [m]}
    expect: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="没有任何断言"):
        load_cases(f)


def test_result_carries_trajectories() -> None:
    """WebUI 要画曲线，runner 必须返回逐轮轨迹而不只是终态。"""
    case = next(c for c in CASES if c.id == "escalation__stranger_repeated_breach")
    r = run_case(case, ENGINE)
    assert len(r.trajectories["a"]) == len(case.a.turns) + 1
    assert all("threat" in step for step in r.trajectories["a"])
