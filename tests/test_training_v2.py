"""M8 · L1 训练目标从分类改为 UserMove 回归。"""

from __future__ import annotations

import json

import pytest

from affect.moves import UserMove
from affect.targets import (
    BINARY_TARGETS,
    REGRESSION_TARGETS,
    TARGET_RANGES,
    TARGET_WEIGHTS,
    move_to_targets,
    regression_metrics,
    spearman,
    targets_to_move,
)


# ------------------------------------------------------------------ 目标定义
def test_no_classification_labels_remain() -> None:
    """★ 分类标签必须彻底消失。

    策略取决于关系，而感知层看不到关系 —— 留着任何一个策略标签，
    就意味着关系判断又漏回了感知层。
    """
    import affect

    assert not hasattr(affect, "StrategyLabel")
    assert not hasattr(affect, "STRATEGY_LABELS")
    for name in REGRESSION_TARGETS:
        assert name in UserMove.__dataclass_fields__


def test_targets_match_user_move_fields() -> None:
    for name in (*REGRESSION_TARGETS, *BINARY_TARGETS):
        assert name in UserMove.__dataclass_fields__, name


def test_target_ranges_cover_all_targets() -> None:
    assert set(TARGET_RANGES) == set(REGRESSION_TARGETS)
    assert set(TARGET_WEIGHTS) == set(REGRESSION_TARGETS)


def test_intimacy_has_highest_weight() -> None:
    """intimacy_bid 是失配机制的输入，错了会直接把「亲近」判成「越界」。"""
    assert TARGET_WEIGHTS["intimacy_bid"] == max(TARGET_WEIGHTS.values())


def test_signed_targets_are_marked() -> None:
    assert TARGET_RANGES["affiliation_bid"][0] < 0
    assert TARGET_RANGES["dominance_bid"][0] < 0
    assert TARGET_RANGES["intimacy_bid"][0] == 0.0


def test_move_target_roundtrip() -> None:
    m = UserMove(
        affiliation_bid=0.6, dominance_bid=-0.3, intimacy_bid=0.8,
        distress_level=0.2, intensity=0.7, directed_at_agent=False,
    )
    back = targets_to_move(move_to_targets(m), directed_logit=-1.0)
    for name in REGRESSION_TARGETS:
        assert getattr(back, name) == pytest.approx(getattr(m, name))
    assert back.directed_at_agent is False


def test_targets_accept_list_form() -> None:
    m = targets_to_move([0.1, 0.2, 0.3, 0.4, 0.5])
    assert m.affiliation_bid == pytest.approx(0.1)
    assert m.intensity == pytest.approx(0.5)


# ------------------------------------------------------------------ 指标
def test_spearman_perfect_and_inverse() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ignores_scale() -> None:
    """★ 序关系对就够了 —— 绝对值可以靠 L2 的增益校准。"""
    assert spearman([0.1, 0.2, 0.3], [0.5, 0.7, 0.9]) == pytest.approx(1.0)


def test_spearman_handles_ties_and_short_input() -> None:
    assert spearman([1, 1, 1], [1, 2, 3]) != spearman([1, 2, 3], [1, 2, 3])
    assert spearman([1, 2], [1, 2]) != spearman([1, 2], [1, 2])  # nan != nan


def test_regression_metrics_on_perfect_prediction() -> None:
    moves = [
        UserMove(affiliation_bid=0.5, intimacy_bid=0.3, intensity=0.4),
        UserMove(affiliation_bid=-0.5, intimacy_bid=0.9, intensity=0.8),
        UserMove(affiliation_bid=0.0, intimacy_bid=0.1, intensity=0.2),
    ]
    m = regression_metrics(moves, list(moves))
    assert m["mean_mae"] == 0.0
    assert m["directed_accuracy"] == 1.0
    assert m["per_target"]["intimacy_bid"]["spearman"] == pytest.approx(1.0)


def test_regression_metrics_detect_directed_errors() -> None:
    t = [UserMove(directed_at_agent=True), UserMove(directed_at_agent=False)]
    p = [UserMove(directed_at_agent=True), UserMove(directed_at_agent=True)]
    assert regression_metrics(t, p)["directed_accuracy"] == pytest.approx(0.5)


def test_regression_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        regression_metrics([UserMove()], [UserMove(), UserMove()])


# ------------------------------------------------- 数据层（不需要 torch）
def test_annotations_require_all_targets(tmp_path) -> None:
    from training.datasets.registry import load_annotations

    p = tmp_path / "a.jsonl"
    complete = {"utterance": "a", **{k: 0.1 for k in REGRESSION_TARGETS}}
    partial = {"utterance": "b", "affiliation_bid": 0.5}
    p.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in (complete, partial)),
        encoding="utf-8",
    )
    records = load_annotations(p)
    assert len(records) == 1, "缺任何一个目标的条目都应视为未标注"
    assert records[0].targets is not None


def test_golden_set_still_rejects_distilled(tmp_path) -> None:
    from training.datasets.registry import load_golden_set

    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps(
            {"utterance": "x", "source": "distilled", **{k: 0.1 for k in REGRESSION_TARGETS}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="绝不能包含开源数据或蒸馏数据"):
        load_golden_set(p)


def test_bootstrap_maps_to_regression_targets() -> None:
    from training.datasets.registry import (
        BOOTSTRAP_MOVE_MAP,
        AffectRecord,
        bootstrap_stage2_from_stage1,
    )

    src = [
        AffectRecord(text="t", dataset="ewect", native_label=lab)
        for lab in ("angry", "sad", "happy")
    ]
    out = bootstrap_stage2_from_stage1(src)
    assert len(out) == 3
    for r in out:
        assert r.dataset == "bootstrap", "bootstrap 数据必须可被识别"
        assert r.weight < 1.0, "弱标签必须降权"
        assert set(r.targets or {}) == set(REGRESSION_TARGETS)
    # angry 应当是敌意 + 支配；sad 应当是痛苦 + 顺从
    angry = next(r for r in out if r.native_label == "angry").targets or {}
    sad = next(r for r in out if r.native_label == "sad").targets or {}
    assert angry["affiliation_bid"] < -0.3
    assert angry["dominance_bid"] > 0.3
    assert sad["distress_level"] > 0.5
    assert sad["dominance_bid"] < 0
    assert set(BOOTSTRAP_MOVE_MAP) >= {"angry", "sad", "happy", "neutral"}


def test_bootstrap_targets_are_in_range() -> None:
    from training.datasets.registry import BOOTSTRAP_MOVE_MAP

    for label, targets in BOOTSTRAP_MOVE_MAP.items():
        assert set(targets) == set(REGRESSION_TARGETS), label
        for name, value in targets.items():
            lo, hi = TARGET_RANGES[name]
            assert lo <= value <= hi, (label, name, value)
