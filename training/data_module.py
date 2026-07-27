"""AffectRecord → torch batch。stage1（分类）与 stage2（回归）共用。"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from affect.targets import REGRESSION_TARGETS  # noqa: E402
from training.datasets.registry import AffectRecord  # noqa: E402

MAX_LENGTH = 128
N_TARGETS = len(REGRESSION_TARGETS)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    # stage1：原生标签；stage2：为 0
    labels: torch.Tensor
    # stage2：回归目标
    targets: torch.Tensor
    target_mask: torch.Tensor
    directed: torch.Tensor
    sample_weight: torch.Tensor
    teacher: torch.Tensor | None
    kd_mask: torch.Tensor | None

    def to(self, device: str) -> Batch:
        return Batch(
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


class AffectDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AffectRecord],
        tokenizer: Any,
        label_to_id: dict[str, int] | None = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id or {}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        enc = self.tokenizer(
            r.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
        )
        has_targets = r.targets is not None
        targets = [float((r.targets or {}).get(t, 0.0)) for t in REGRESSION_TARGETS]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids") or [0] * self.max_length,
            "label": self.label_to_id.get(str(r.native_label), 0),
            "targets": targets,
            "target_mask": [1.0 if has_targets else 0.0] * N_TARGETS,
            "directed": 1.0 if (r.directed_at_agent is not False) else 0.0,
            "sample_weight": float(r.weight),
            "teacher": r.teacher_logits,
        }


def collate(batch: list[dict[str, Any]]) -> Batch:
    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([b[key] for b in batch], dtype=dtype)

    rows = [b["teacher"] for b in batch]
    has_teacher = any(t is not None and len(t) == N_TARGETS for t in rows)
    teacher = kd_mask = None
    if has_teacher:
        teacher = torch.tensor(
            [t if (t is not None and len(t) == N_TARGETS) else [0.0] * N_TARGETS for t in rows],
            dtype=torch.float,
        )
        kd_mask = torch.tensor(
            [1.0 if (t is not None and len(t) == N_TARGETS) else 0.0 for t in rows],
            dtype=torch.float,
        )

    return Batch(
        input_ids=stack("input_ids", torch.long),
        attention_mask=stack("attention_mask", torch.long),
        token_type_ids=stack("token_type_ids", torch.long),
        labels=stack("label", torch.long),
        targets=stack("targets", torch.float),
        target_mask=stack("target_mask", torch.float),
        directed=stack("directed", torch.float),
        sample_weight=stack("sample_weight", torch.float),
        teacher=teacher,
        kd_mask=kd_mask,
    )


def make_loader(
    records: Sequence[AffectRecord],
    tokenizer: Any,
    label_to_id: dict[str, int] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0,
) -> DataLoader:
    ds = AffectDataset(records, tokenizer, label_to_id, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )


def stratified_split(
    records: Sequence[AffectRecord],
    dev_ratio: float = 0.1,
    seed: int = 42,
    label_field: str = "native_label",
) -> tuple[list[AffectRecord], list[AffectRecord]]:
    """按标签分层切 dev，保证小类别在 dev 里也有样本。"""
    buckets: dict[str, list[AffectRecord]] = {}
    for r in records:
        buckets.setdefault(str(getattr(r, label_field)), []).append(r)
    rng = random.Random(seed)
    train: list[AffectRecord] = []
    dev: list[AffectRecord] = []
    for _, items in sorted(buckets.items()):
        items = list(items)
        rng.shuffle(items)
        k = max(1, int(len(items) * dev_ratio)) if len(items) > 1 else 0
        dev.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def label_vocab(records: Iterable[AffectRecord], field: str = "native_label") -> dict[str, int]:
    labels = sorted({str(getattr(r, field)) for r in records})
    return {lab: i for i, lab in enumerate(labels)}
