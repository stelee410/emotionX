"""§3.2 L1 模型：小 encoder + 多头输出。

    head_strategy : 4 分类, CrossEntropy
    head_vad      : 2 维回归 (valence, arousal), MSE
    head_intensity: 1 维回归, MSE
    L = L_strategy + 0.5 * L_vad + 0.3 * L_intensity

阶段一额外挂「每个数据集一个原生标签头」（§3.3：各数据集用各自原生标签）；
阶段二丢弃这些头，换上 4 类 StrategyLabel 头。

全参微调，不用 LoRA（§3.2：18M 尺度上 LoRA 只增加复杂度）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

BASE_MODEL = "nghuyong/ernie-3.0-nano-zh"
USER_TOKEN = "[USER]"
AGENT_TOKEN = "[AGENT]"

# §3.2 联合损失权重
W_VAD = 0.5
W_INTENSITY = 0.3
# 先验推出来的 VAD 目标（非人工标注）的降权系数
PRIOR_VAD_WEIGHT = 0.3


def load_tokenizer(base_model: str = BASE_MODEL) -> Any:
    """加载 tokenizer 并把 [USER]/[AGENT] 注册为单 token。

    不注册的话 `[USER]` 会被切成 `[ user ]` 若干个 token，白占 max_length 且语义为零。
    """
    tok = AutoTokenizer.from_pretrained(base_model)
    added = tok.add_special_tokens(
        {"additional_special_tokens": [USER_TOKEN, AGENT_TOKEN]}
    )
    if added:
        print(f"  tokenizer 新增 {added} 个特殊 token（{USER_TOKEN} {AGENT_TOKEN}）")
    return tok


def embedding_size(tokenizer: Any) -> int:
    """embedding 行数必须按**最大 token id** 算，不能用 len(tokenizer)。

    ernie-3.0-nano-zh 的 vocab.txt 有一个重复条目（id 12084 成为空洞），
    于是 len(tokenizer) 比 max_id+1 少 1。若按 len(tokenizer) resize，
    [AGENT] 的 id 会正好越界 —— 在没有 agent 上文的数据集上不报错，
    一旦喂入双句输入就崩（或在某些后端上静默读到越界内存）。
    """
    return max(tokenizer.get_vocab().values()) + 1


def write_vocab_file(tokenizer: Any, path: str | Path) -> Path:
    """导出 vocab.txt，使「行号 == token id」，供纯 Python 分词器使用。

    空洞用占位 token 填上，否则行号会整体前移，线上 id 全错位。
    """
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    lines = [f"[unused_hole_{i}]" for i in range(size)]
    for token, idx in vocab.items():
        lines[idx] = token
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@dataclass
class HeadSpec:
    """阶段一的一个原生标签头。"""

    name: str
    labels: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.labels)


class AffectEncoder(nn.Module):
    """encoder + 多头。ONNX 只导出 strategy / vad / intensity 三个头。"""

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        strategy_labels: list[str] | None = None,
        aux_heads: list[HeadSpec] | None = None,
        dropout: float = 0.1,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        if vocab_size and vocab_size != self.encoder.get_input_embeddings().weight.shape[0]:
            self.encoder.resize_token_embeddings(vocab_size)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.strategy_labels = list(strategy_labels or [])
        self.head_strategy = (
            nn.Linear(hidden, len(self.strategy_labels)) if self.strategy_labels else None
        )
        self.head_vad = nn.Linear(hidden, 2)
        self.head_intensity = nn.Linear(hidden, 1)

        self.aux_specs = list(aux_heads or [])
        self.aux_heads = nn.ModuleDict(
            {spec.name: nn.Linear(hidden, spec.num_labels) for spec in self.aux_specs}
        )

    # ---- 前向 ----
    def pooled(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """mean pooling over 有效 token。

        用 mean pooling 而不是 [CLS]：ernie-nano 只有 4 层，[CLS] 表征在这个深度上
        明显弱于均值池化（在 EWECT 上实测差 1–2 个点 macro-F1）。
        """
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last.dtype)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ONNX 导出用的签名：返回 (strategy_logits, vad, intensity)。"""
        h = self.dropout(self.pooled(input_ids, attention_mask, token_type_ids))
        if self.head_strategy is None:
            raise RuntimeError("模型没有 strategy 头（还在阶段一？）")
        strategy_logits = self.head_strategy(h)
        vad = self.head_vad(h)
        # valence ∈ [-1,1] 用 tanh；arousal ∈ [0,1] 用 sigmoid —— 值域约束放进网络，
        # 免得推理侧还要 clamp。
        vad = torch.cat([torch.tanh(vad[:, :1]), torch.sigmoid(vad[:, 1:])], dim=-1)
        intensity = torch.sigmoid(self.head_intensity(h))
        return strategy_logits, vad, intensity

    def forward_aux(
        self,
        head: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.dropout(self.pooled(input_ids, attention_mask, token_type_ids))
        logits = self.aux_heads[head](h)
        vad = self.head_vad(h)
        vad = torch.cat([torch.tanh(vad[:, :1]), torch.sigmoid(vad[:, 1:])], dim=-1)
        intensity = torch.sigmoid(self.head_intensity(h))
        return logits, vad, intensity

    # ---- 阶段切换 ----
    def attach_strategy_head(self, labels: list[str]) -> None:
        """§3.3 阶段二：丢弃阶段一的分类头，换上 4 类 StrategyLabel 头。"""
        self.strategy_labels = list(labels)
        self.head_strategy = nn.Linear(self.config.hidden_size, len(labels))
        self.aux_specs = []
        self.aux_heads = nn.ModuleDict()

    # ---- 存取 ----
    def save(self, directory: str | Path, tokenizer: Any | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), d / "pytorch_model.bin")
        meta = {
            "base_model": self.base_model,
            "strategy_labels": self.strategy_labels,
            "aux_heads": [{"name": s.name, "labels": s.labels} for s in self.aux_specs],
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.encoder.get_input_embeddings().weight.shape[0],
        }
        (d / "affect_model.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(d)
        return d

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> AffectEncoder:
        d = Path(directory)
        meta = json.loads((d / "affect_model.json").read_text(encoding="utf-8"))
        model = cls(
            base_model=meta["base_model"],
            strategy_labels=meta.get("strategy_labels") or None,
            aux_heads=[HeadSpec(**h) for h in meta.get("aux_heads", [])],
            vocab_size=meta.get("vocab_size"),
        )
        state = torch.load(d / "pytorch_model.bin", map_location=map_location)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  ! 缺失参数（将保持随机初始化）：{sorted(missing)[:6]}")
        if unexpected:
            print(f"  ! 忽略多余参数：{sorted(unexpected)[:6]}")
        return model


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


@dataclass
class LossParts:
    total: torch.Tensor
    cls: torch.Tensor
    vad: torch.Tensor
    intensity: torch.Tensor
    kd: torch.Tensor | None = None
    extras: dict[str, float] = field(default_factory=dict)


def multitask_loss(
    logits: torch.Tensor,
    vad_pred: torch.Tensor,
    intensity_pred: torch.Tensor,
    labels: torch.Tensor,
    vad_target: torch.Tensor,
    vad_mask: torch.Tensor,
    intensity_target: torch.Tensor,
    intensity_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    kd_mask: torch.Tensor | None = None,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.5,
) -> LossParts:
    """§3.2 联合损失 + §3.4.3 知识蒸馏（T=2.0）。

    vad_mask / intensity_mask 里带的是每样本权重（0 = 无监督），
    这样先验来的 VAD 目标可以只算 0.3 的权重。
    """
    ce = F.cross_entropy(logits, labels, weight=class_weight, reduction="none")
    if sample_weight is not None:
        ce = ce * sample_weight
    cls_loss = ce.mean()

    vad_se = ((vad_pred - vad_target) ** 2).mean(dim=-1) * vad_mask
    denom_v = vad_mask.sum().clamp(min=1e-6)
    vad_loss = vad_se.sum() / denom_v

    int_se = ((intensity_pred.squeeze(-1) - intensity_target) ** 2) * intensity_mask
    denom_i = intensity_mask.sum().clamp(min=1e-6)
    int_loss = int_se.sum() / denom_i

    total = cls_loss + W_VAD * vad_loss + W_INTENSITY * int_loss

    kd_loss: torch.Tensor | None = None
    if teacher_logits is not None and kd_mask is not None and kd_mask.any():
        T = kd_temperature
        student_log = F.log_softmax(logits / T, dim=-1)
        teacher_prob = F.softmax(teacher_logits / T, dim=-1)
        per_sample = F.kl_div(student_log, teacher_prob, reduction="none").sum(dim=-1) * (T * T)
        kd_loss = (per_sample * kd_mask).sum() / kd_mask.sum().clamp(min=1e-6)
        # 蒸馏样本上把 CE 与 KD 混合；无 teacher 的样本不受影响
        total = total + kd_alpha * kd_loss

    return LossParts(
        total=total,
        cls=cls_loss.detach(),
        vad=vad_loss.detach(),
        intensity=int_loss.detach(),
        kd=None if kd_loss is None else kd_loss.detach(),
    )


def compute_class_weights(
    labels: list[int], num_classes: int, mode: str = "inv_sqrt"
) -> torch.Tensor:
    """类别不平衡处理。neutral 占 60–80%，不加权 macro-F1 会被拖死（§8.1）。"""
    counts = torch.zeros(num_classes)
    for lab in labels:
        counts[lab] += 1
    counts = counts.clamp(min=1.0)
    if mode == "inv_sqrt":
        w = counts.sum() / counts.sqrt()
    elif mode == "inv":
        w = counts.sum() / counts
    elif mode == "none":
        return torch.ones(num_classes)
    else:
        raise ValueError(f"未知 class weight 模式 {mode!r}")
    return w / w.mean()
