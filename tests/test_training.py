"""训练层测试：数据统一层、损失、类别权重、§8.1 红线。

不含真实训练（那在 logs/ 与 artifacts/ 里有实跑记录），只测那些出错会静默毁掉
训练结果的地方。
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from affect.types import STRATEGY_LABELS  # noqa: E402
from training.datasets.registry import (  # noqa: E402
    BOOTSTRAP_STRATEGY_MAP,
    VAD_PRIORS,
    AffectRecord,
    bootstrap_stage2_from_stage1,
    label_distribution,
    load_annotations,
    load_golden_set,
)
from training.model import compute_class_weights, multitask_loss  # noqa: E402


def _rec(**kw) -> AffectRecord:
    base = {"text": "[USER] 测试", "dataset": "t", "native_label": "neutral"}
    return AffectRecord(**{**base, **kw})


# --------------------------------------------------------------- §8.1 红线
def test_golden_set_rejects_non_real_sessions(tmp_path) -> None:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        json.dumps(
            {"utterance": "我好难受", "strategy": "distress", "source": "distilled"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="绝不能包含开源数据或蒸馏数据"):
        load_golden_set(p)


def test_golden_set_accepts_real_sessions(tmp_path) -> None:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        json.dumps(
            {
                "utterance": "我好难受",
                "last_agent_reply": "怎么了",
                "strategy": "distress",
                "source": "real_session",
                "valence": -0.6,
                "arousal": 0.4,
                "intensity": 0.7,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_golden_set(p)
    assert len(records) == 1
    r = records[0]
    assert r.strategy == "distress"
    assert r.vad_is_human is True
    assert r.text == "[USER] 我好难受 [SEP] [AGENT] 怎么了"


def test_annotations_skip_conflicted_items(tmp_path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"utterance": "a", "strategy": "neutral"}, ensure_ascii=False),
                json.dumps(
                    {"utterance": "b", "candidate_strategies": ["neutral", "positive"]},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert len(load_annotations(p)) == 1


def test_bootstrap_is_marked_and_downweighted() -> None:
    src = [_rec(native_label="angry"), _rec(native_label="sad"), _rec(native_label="happy")]
    out = bootstrap_stage2_from_stage1(src)
    assert [r.strategy for r in out] == ["frustration", "distress", "positive"]
    assert all(r.dataset == "bootstrap" for r in out), "bootstrap 数据必须可被识别出来"
    assert all(r.weight < 1.0 for r in out), "弱标签必须降权"
    assert all(r.vad_is_human is False for r in out)


def test_bootstrap_map_covers_all_priors() -> None:
    """新增 VAD 先验时别忘了同步 bootstrap 映射，否则那些样本会被静默丢掉。"""
    missing = sorted(set(VAD_PRIORS) - set(BOOTSTRAP_STRATEGY_MAP))
    assert not missing, f"这些标签缺 bootstrap 映射: {missing}"


def test_bootstrap_map_targets_are_valid_strategies() -> None:
    assert set(BOOTSTRAP_STRATEGY_MAP.values()) <= set(STRATEGY_LABELS)


def test_vad_priors_within_range() -> None:
    for label, (v, a, i) in VAD_PRIORS.items():
        assert -1.0 <= v <= 1.0, label
        assert 0.0 <= a <= 1.0, label
        assert 0.0 <= i <= 1.0, label


def test_label_distribution() -> None:
    recs = [_rec(native_label="a"), _rec(native_label="a"), _rec(native_label="b")]
    assert label_distribution(recs) == {"a": 2, "b": 1}


# --------------------------------------------------------------- 损失函数
def _loss_inputs(n: int = 4, c: int = 4):
    return {
        "logits": torch.zeros(n, c, requires_grad=True),
        "vad_pred": torch.zeros(n, 2),
        "intensity_pred": torch.zeros(n, 1),
        "labels": torch.zeros(n, dtype=torch.long),
        "vad_target": torch.zeros(n, 2),
        "vad_mask": torch.ones(n),
        "intensity_target": torch.zeros(n),
        "intensity_mask": torch.ones(n),
    }


def test_loss_weights_match_spec() -> None:
    """L = L_strategy + 0.5*L_vad + 0.3*L_intensity（§3.2）。"""
    kw = _loss_inputs()
    kw["vad_pred"] = torch.full((4, 2), 1.0)  # MSE = 1
    kw["intensity_pred"] = torch.full((4, 1), 1.0)  # MSE = 1
    out = multitask_loss(**kw)
    expected = float(out.cls) + 0.5 * 1.0 + 0.3 * 1.0
    assert float(out.total.detach()) == pytest.approx(expected, abs=1e-5)


def test_vad_mask_zero_removes_supervision() -> None:
    kw = _loss_inputs()
    kw["vad_pred"] = torch.full((4, 2), 5.0)
    kw["vad_mask"] = torch.zeros(4)
    out = multitask_loss(**kw)
    assert float(out.vad) == pytest.approx(0.0)


def test_partial_vad_mask_averages_only_supervised() -> None:
    kw = _loss_inputs()
    kw["vad_pred"] = torch.tensor([[1.0, 1.0], [9.0, 9.0], [1.0, 1.0], [9.0, 9.0]])
    kw["vad_mask"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = multitask_loss(**kw)
    assert float(out.vad) == pytest.approx(1.0, abs=1e-5)


def test_prior_vad_weight_reduces_contribution() -> None:
    """先验来的 VAD（mask=0.3）贡献应显著小于人工标注（mask=1.0）。"""
    strong = _loss_inputs()
    strong["vad_pred"] = torch.full((4, 2), 1.0)
    weak = dict(strong)
    weak["vad_mask"] = torch.full((4,), 0.3)
    # mask 同时作为分子权重和分母，单独看 vad 项一样；关键是它相对 cls 的梯度更小
    a = multitask_loss(**strong)
    b = multitask_loss(**weak)
    assert float(a.vad) == pytest.approx(float(b.vad), abs=1e-5)


def test_kd_loss_active_only_on_masked_samples() -> None:
    kw = _loss_inputs()
    kw["teacher_logits"] = torch.tensor(
        [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    kw["kd_mask"] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = multitask_loss(**kw, kd_temperature=2.0, kd_alpha=0.5)
    assert out.kd is not None and float(out.kd) > 0


def test_kd_zero_when_student_matches_teacher() -> None:
    kw = _loss_inputs()
    kw["logits"] = torch.tensor([[3.0, 0.0, 0.0, 0.0]], requires_grad=True)
    for key in ("vad_pred", "intensity_pred", "vad_target", "intensity_target", "vad_mask", "intensity_mask", "labels"):
        kw[key] = kw[key][:1]
    kw["teacher_logits"] = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    kw["kd_mask"] = torch.tensor([1.0])
    out = multitask_loss(**kw)
    assert float(out.kd) == pytest.approx(0.0, abs=1e-6)


def test_no_kd_without_teacher() -> None:
    out = multitask_loss(**_loss_inputs())
    assert out.kd is None


def test_sample_weight_scales_cls_loss() -> None:
    a = multitask_loss(**_loss_inputs())
    kw = _loss_inputs()
    kw["sample_weight"] = torch.full((4,), 0.5)
    b = multitask_loss(**kw)
    assert float(b.cls) == pytest.approx(float(a.cls) * 0.5, abs=1e-6)


def test_loss_is_differentiable() -> None:
    kw = _loss_inputs()
    out = multitask_loss(**kw)
    out.total.backward()
    assert kw["logits"].grad is not None


# --------------------------------------------------------------- 类别权重
def test_class_weights_favour_rare_classes() -> None:
    """neutral 占 80% 时，稀有类必须拿到更大权重（§8.1 macro-F1 导向）。"""
    labels = [0] * 800 + [1] * 100 + [2] * 60 + [3] * 40
    w = compute_class_weights(labels, 4, mode="inv_sqrt")
    assert w[0] < w[1] < w[2] < w[3]
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)


def test_class_weights_none_is_uniform() -> None:
    w = compute_class_weights([0, 0, 1], 4, mode="none")
    assert torch.allclose(w, torch.ones(4))


def test_class_weights_handles_empty_class() -> None:
    w = compute_class_weights([0, 0, 0], 4, mode="inv_sqrt")
    assert torch.isfinite(w).all()
    assert w[3] > w[0]


def test_class_weights_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知 class weight 模式"):
        compute_class_weights([0], 2, mode="magic")


# --------------------------------------------------------------- 模型结构
def test_model_output_ranges() -> None:
    """valence∈[-1,1] 用 tanh，arousal/intensity∈[0,1] 用 sigmoid —— 值域约束在网络内。"""
    from training.model import AffectEncoder

    model = AffectEncoder(strategy_labels=list(STRATEGY_LABELS))
    ids = torch.randint(100, 1000, (3, 16))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits, vad, intensity = model(ids, mask, torch.zeros_like(ids))
    assert logits.shape == (3, 4)
    assert vad.shape == (3, 2) and intensity.shape == (3, 1)
    assert (vad[:, 0] >= -1).all() and (vad[:, 0] <= 1).all()
    assert (vad[:, 1] >= 0).all() and (vad[:, 1] <= 1).all()
    assert (intensity >= 0).all() and (intensity <= 1).all()


def test_attach_strategy_head_discards_stage1_heads() -> None:
    """§3.3 阶段二：丢弃阶段一的分类头。"""
    from training.model import AffectEncoder, HeadSpec

    model = AffectEncoder(aux_heads=[HeadSpec("ewect", ["a", "b", "c"])])
    assert "ewect" in model.aux_heads
    model.attach_strategy_head(list(STRATEGY_LABELS))
    assert len(model.aux_heads) == 0
    assert model.head_strategy is not None
    assert model.head_strategy.out_features == 4


def test_mean_pooling_ignores_padding() -> None:
    """padding 必须不参与池化，否则不同长度的同一句话会得到不同表征。"""
    from training.model import AffectEncoder

    model = AffectEncoder(strategy_labels=list(STRATEGY_LABELS)).eval()
    ids_short = torch.tensor([[101, 500, 600, 102]])
    mask_short = torch.ones_like(ids_short)
    ids_pad = torch.tensor([[101, 500, 600, 102, 0, 0, 0, 0]])
    mask_pad = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    with torch.no_grad():
        a = model.pooled(ids_short, mask_short, torch.zeros_like(ids_short))
        b = model.pooled(ids_pad, mask_pad, torch.zeros_like(ids_pad))
    assert torch.allclose(a, b, atol=1e-4)
