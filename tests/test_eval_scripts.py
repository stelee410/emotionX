"""eval/ 下两个评估脚本的测试。

`trajectory_review.py` 是给人看的工具，但它的剧本 + 指标计算逻辑仍然要能自动验证 ——
否则改 appraisal 表时没人会发现某个剧本已经跑不出来了。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relpath)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def review():
    return _load("trajectory_review", "eval/trajectory_review.py")


@pytest.fixture(scope="module")
def perception_eval():
    return _load("test_perception_script", "eval/test_perception.py")


# ------------------------------------------------------------ trajectory_review
def test_ten_scenarios_defined(review) -> None:
    """§8.2 要求 10 个典型剧本。"""
    assert len(review.SCENARIOS) == 10
    assert len({s.key for s in review.SCENARIOS}) == 10
    for s in review.SCENARIOS:
        assert s.turns, s.key
        assert s.expect, f"{s.key} 缺少判断标准 —— 没有 expect 的剧本没法评审"


@pytest.mark.parametrize("persona", ["steady_medical", "warm_companion"])
def test_all_scenarios_run(review, persona: str) -> None:
    for scenario in review.SCENARIOS:
        rows = review.run_scenario(scenario, persona)
        assert len(rows) == len(scenario.turns), scenario.key
        for r in rows:
            for dim in ("valence", "arousal", "dominance", "concern"):
                assert -1.0 <= r[dim] <= 1.0, (scenario.key, dim, r[dim])


def test_losing_patience_dominance_falls(review) -> None:
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["losing_patience"], "steady_medical")
    assert rows[-1]["dominance"] < rows[0]["dominance"], "用户越不耐烦，agent 应越少下结论"
    assert rows[-1]["concern"] > rows[0]["concern"]


def test_mood_recovers_concern_peaks_then_falls(review) -> None:
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["mood_recovers"], "warm_companion")
    concerns = [r["concern"] for r in rows]
    peak = concerns.index(max(concerns))
    assert peak < len(concerns) - 1, "concern 不应在最后一轮才达到峰值"
    assert concerns[-1] < max(concerns), "情绪好转后 concern 必须回落"
    assert rows[-1]["valence"] > rows[0]["valence"]


def test_long_silence_returns_to_baseline(review) -> None:
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["long_silence_return"], "warm_companion")
    assert any(r["idle_reset"] for r in rows), "剧本里的 4 小时间隔应触发 idle 强回归"
    idle_row = next(r for r in rows if r["idle_reset"])
    assert idle_row["concern"] < max(r["concern"] for r in rows)


def test_crisis_scenario_bypasses(review) -> None:
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["crisis_mid_conversation"], "warm_companion")
    assert any(r["bypass"] == "crisis" for r in rows)
    # 危机轮之后状态仍然连续（不重置）
    assert rows[-1]["concern"] > 0


def test_medical_scenario_bypasses_only_medical_turns(review) -> None:
    rows = review.run_scenario(
        review.SCENARIOS_BY_KEY["medical_info_mixed_with_emotion"], "steady_medical"
    )
    kinds = [r["bypass"] for r in rows]
    assert "medical" in kinds
    assert "none" in kinds, "非医疗轮次不应 bypass"


def test_task_streaks_stay_in_bounds(review) -> None:
    for key in ("task_success_streak", "task_failure_streak"):
        for persona in ("steady_medical", "warm_companion"):
            rows = review.run_scenario(review.SCENARIOS_BY_KEY[key], persona)
            assert all(-1 <= r["valence"] <= 1 for r in rows)


def test_slow_burn_accumulates_concern(review) -> None:
    """低强度长程困扰：concern 应缓慢累积，而不是一直贴 baseline。"""
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["slow_burn_low_intensity"], "warm_companion")
    base = review.AffectPipeline(store_backend="memory").persona("warm_companion").baseline.as_dict()
    assert rows[-1]["concern"] > base["concern"] + 0.1


def test_whiplash_is_damped_more_by_steady_persona(review) -> None:
    """高衰减 persona 的曲线必须更平 —— 否则人格参数没起作用。"""
    med = review.run_scenario(review.SCENARIOS_BY_KEY["whiplash"], "steady_medical")
    com = review.run_scenario(review.SCENARIOS_BY_KEY["whiplash"], "warm_companion")

    def swing(rows: list[dict]) -> float:
        return max(r["valence"] for r in rows) - min(r["valence"] for r in rows)

    assert swing(med) < swing(com)


def test_flat_neutral_stays_near_baseline(review) -> None:
    rows = review.run_scenario(review.SCENARIOS_BY_KEY["flat_neutral"], "steady_medical")
    base = review.AffectPipeline(store_backend="memory").persona("steady_medical").baseline.as_dict()
    for r in rows:
        assert abs(r["concern"] - base["concern"]) < 0.25


# --------------------------------------------------------------- test_perception
def _golden(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return p


def test_eval_rejects_distilled_data(perception_eval, tmp_path) -> None:
    """§8.1 红线：评估集里出现蒸馏/开源数据必须直接失败。"""
    p = _golden(
        tmp_path,
        [{"utterance": "我好难受", "strategy": "distress", "source": "distilled"}],
    )
    with pytest.raises(SystemExit, match="绝不能包含开源数据或蒸馏数据"):
        perception_eval.load_golden(p)


def test_eval_skips_conflicted_items(perception_eval, tmp_path) -> None:
    p = _golden(
        tmp_path,
        [
            {"utterance": "a", "strategy": "neutral", "source": "real_session"},
            {"utterance": "b", "candidate_strategies": ["neutral", "positive"], "source": "real_session"},
        ],
    )
    assert len(perception_eval.load_golden(p)) == 1


def test_macro_f1_ignores_absent_classes(perception_eval) -> None:
    """golden set 里没有某一类时，那一类不该把 macro-F1 拉成 0。"""
    m = perception_eval.prf(["neutral", "neutral"], ["neutral", "neutral"])
    assert m["macro_f1"] == 1.0
    assert m["per_class"]["distress"]["support"] == 0


def test_macro_f1_penalises_majority_class_collapse(perception_eval) -> None:
    """全预测 neutral：accuracy 高但 macro-F1 必须很低（这就是不用 accuracy 的原因）。"""
    y_true = ["neutral"] * 8 + ["distress", "frustration"]
    y_pred = ["neutral"] * 10
    m = perception_eval.prf(y_true, y_pred)
    assert m["accuracy"] == 0.8
    assert m["macro_f1"] < 0.35


def test_confusion_matrix_shape(perception_eval) -> None:
    cm = perception_eval.confusion(["neutral", "distress"], ["neutral", "frustration"])
    assert cm["neutral"]["neutral"] == 1
    assert cm["distress"]["frustration"] == 1
    assert set(cm) == set(perception_eval.LABELS)


def test_crisis_layer_metrics_from_annotations(perception_eval) -> None:
    rows = [
        {"utterance": "我想自杀", "strategy": "distress", "crisis_flag": 1},
        {"utterance": "有点累", "strategy": "neutral", "crisis_flag": 0},
        {"utterance": "撑不下去了", "strategy": "distress", "crisis_flag": 1},  # TIER-2 单个 → 漏报
    ]
    m = perception_eval.crisis_layer_metrics(rows)
    assert m["n_positive"] == 2
    assert m["recall"] == 0.5
    assert "撑不下去了" in m["missed"]


def test_heuristic_baseline_runs_on_golden(perception_eval, tmp_path, capsys) -> None:
    """规则桩基线要能端到端跑出报告 —— 这是 L1 训练前的对照组。"""
    rows = [
        {"utterance": "又错了，说了多少遍了", "strategy": "frustration", "source": "real_session",
         "valence": -0.6, "arousal": 0.8, "intensity": 0.8, "crisis_flag": 0},
        {"utterance": "我特别害怕，睡不着", "strategy": "distress", "source": "real_session",
         "valence": -0.6, "arousal": 0.5, "intensity": 0.7, "crisis_flag": 0},
        {"utterance": "帮我查一下挂号记录", "strategy": "neutral", "source": "real_session",
         "valence": 0.0, "arousal": 0.2, "intensity": 0.1, "crisis_flag": 0},
        {"utterance": "太好了，谢谢你", "strategy": "positive", "source": "real_session",
         "valence": 0.6, "arousal": 0.4, "intensity": 0.5, "crisis_flag": 0},
    ]
    out = tmp_path / "report.json"
    rc = perception_eval.main(
        ["--heuristic", "--golden", str(_golden(tmp_path, rows)), "--json-out", str(out)]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["metrics"]["macro_f1"] == 1.0, "这 4 条是规则桩的送分题，全对才说明链路没坏"
    assert report["regression"]["valence_mae"] < 0.5
    assert "混淆矩阵" in capsys.readouterr().out
