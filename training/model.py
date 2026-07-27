"""§3.2 L1 模型：小 encoder + 多头输出。

    head_move     : 5 维回归（affiliation/dominance/intimacy/distress/intensity）
    head_directed : 1 维二分类（这句话是否指向 agent 本人）

    分类标签在 v2 里被废弃了：策略取决于关系，而感知层看不到关系，
    所以没有任何一个标签是对的。改成回归后 L1 学的是句子本身的属性。

阶段一额外挂「每个数据集一个原生标签头」（§3.3：各数据集用各自原生标签）；
阶段二丢弃这些头，换上 4 类 StrategyLabel 头。

全参微调，不用 LoRA（§3.2：18M 尺度上 LoRA 只增加复杂度）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _P
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from affect.targets import (  # noqa: E402
    DIRECTED_WEIGHT,
    REGRESSION_TARGETS,
    TARGET_RANGES,
    TARGET_WEIGHTS,
)

BASE_MODEL = "nghuyong/ernie-3.0-nano-zh"
USER_TOKEN = "[USER]"
AGENT_TOKEN = "[AGENT]"

N_TARGETS = len(REGRESSION_TARGETS)
# 值域为 [-1,1] 的目标用 tanh，[0,1] 的用 sigmoid —— 约束放进网络，推理侧不用再 clamp
TANH_MASK: tuple[bool, ...] = tuple(TARGET_RANGES[t][0] < 0 for t in REGRESSION_TARGETS)
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
    """encoder + 多头。ONNX 只导出 move 与 directed 两个头。"""

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        move_head: bool = True,
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

        self.has_move_head = bool(move_head)
        self.head_move = nn.Linear(hidden, N_TARGETS) if move_head else None
        self.head_directed = nn.Linear(hidden, 1) if move_head else None

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ONNX 导出签名：返回 (move[5], directed_logit[1])。"""
        h = self.dropout(self.pooled(input_ids, attention_mask, token_type_ids))
        if self.head_move is None or self.head_directed is None:
            raise RuntimeError("模型没有 move 头（还在阶段一？）")
        raw = self.head_move(h)
        cols = [
            torch.tanh(raw[:, i : i + 1]) if use_tanh else torch.sigmoid(raw[:, i : i + 1])
            for i, use_tanh in enumerate(TANH_MASK)
        ]
        return torch.cat(cols, dim=-1), self.head_directed(h)

    def forward_aux(
        self,
        head: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """阶段一：某个数据集的原生标签分类头。"""
        h = self.dropout(self.pooled(input_ids, attention_mask, token_type_ids))
        return self.aux_heads[head](h)

    # ---- 阶段切换 ----
    def attach_move_head(self) -> None:
        """阶段二：丢弃阶段一的原生标签分类头，换上 UserMove 回归头。"""
        hidden = self.config.hidden_size
        self.has_move_head = True
        self.head_move = nn.Linear(hidden, N_TARGETS)
        self.head_directed = nn.Linear(hidden, 1)
        self.aux_specs = []
        self.aux_heads = nn.ModuleDict()

    # ---- 存取 ----
    def save(self, directory: str | Path, tokenizer: Any | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), d / "pytorch_model.bin")
        meta = {
            "base_model": self.base_model,
            "move_head": self.has_move_head,
            "targets": list(REGRESSION_TARGETS),
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
            move_head=bool(meta.get("move_head", True)),
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
    move: torch.Tensor
    directed: torch.Tensor
    kd: torch.Tensor | None = None


def move_loss(
    pred: torch.Tensor,
    directed_logit: torch.Tensor,
    target: torch.Tensor,
    directed_target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    teacher: torch.Tensor | None = None,
    kd_mask: torch.Tensor | None = None,
    kd_alpha: float = 0.5,
) -> LossParts:
    """回归损失。每个目标有自己的权重 —— intimacy_bid 最高，因为它是
    失配机制的输入，错了会直接把「亲近」判成「越界」。"""
    weights = torch.tensor(
        [TARGET_WEIGHTS[t] for t in REGRESSION_TARGETS], device=pred.device, dtype=pred.dtype
    )
    # Huber：标注里总有几条离谱的，L2 会被它们带偏
    err = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.2)
    err = err * weights
    if target_mask is not None:
        err = err * target_mask
    per_sample = err.mean(dim=-1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    move = per_sample.mean()

    directed = torch.nn.functional.binary_cross_entropy_with_logits(
        directed_logit.squeeze(-1), directed_target
    )

    total = move + DIRECTED_WEIGHT * directed

    kd = None
    if teacher is not None and kd_mask is not None and kd_mask.any():
        per = (torch.nn.functional.smooth_l1_loss(pred, teacher, reduction="none", beta=0.2)
               * weights).mean(dim=-1)
        kd = (per * kd_mask).sum() / kd_mask.sum().clamp(min=1e-6)
        total = total + kd_alpha * kd

    return LossParts(total=total, move=move.detach(), directed=directed.detach(),
                     kd=None if kd is None else kd.detach())


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
