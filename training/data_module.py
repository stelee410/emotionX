"""AffectRecord → torch batch。stage1/stage2 共用。"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from training.datasets.registry import AffectRecord
from training.model import PRIOR_VAD_WEIGHT

MAX_LENGTH = 128


@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor
    vad_target: torch.Tensor
    vad_mask: torch.Tensor
    intensity_target: torch.Tensor
    intensity_mask: torch.Tensor
    sample_weight: torch.Tensor
    teacher_logits: torch.Tensor | None
    kd_mask: torch.Tensor | None

    def to(self, device: str) -> Encoded:
        def mv(x: Any) -> Any:
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Encoded(
            **{k: mv(v) for k, v in self.__dict__.items()}  # type: ignore[arg-type]
        )


class AffectDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AffectRecord],
        tokenizer: Any,
        label_to_id: dict[str, int],
        label_field: str = "native_label",
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.label_field = label_field
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
        label_name = str(getattr(r, self.label_field) or "")
        vad_weight = 1.0 if r.vad_is_human else PRIOR_VAD_WEIGHT
        has_vad = r.valence is not None and r.arousal is not None
        has_int = r.intensity is not None
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids") or [0] * self.max_length,
            "label": self.label_to_id[label_name],
            "valence": float(r.valence or 0.0),
            "arousal": float(r.arousal or 0.0),
            "vad_mask": vad_weight if has_vad else 0.0,
            "intensity": float(r.intensity or 0.0),
            "intensity_mask": vad_weight if has_int else 0.0,
            "sample_weight": float(r.weight),
            "teacher_logits": r.teacher_logits,
        }


def collate(batch: list[dict[str, Any]], num_classes: int) -> Encoded:
    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([b[key] for b in batch], dtype=dtype)

    teacher_rows = [b["teacher_logits"] for b in batch]
    has_teacher = any(t is not None and len(t) == num_classes for t in teacher_rows)
    teacher_logits = None
    kd_mask = None
    if has_teacher:
        teacher_logits = torch.tensor(
            [t if (t is not None and len(t) == num_classes) else [0.0] * num_classes for t in teacher_rows],
            dtype=torch.float,
        )
        kd_mask = torch.tensor(
            [1.0 if (t is not None and len(t) == num_classes) else 0.0 for t in teacher_rows],
            dtype=torch.float,
        )

    return Encoded(
        input_ids=stack("input_ids", torch.long),
        attention_mask=stack("attention_mask", torch.long),
        token_type_ids=stack("token_type_ids", torch.long),
        labels=stack("label", torch.long),
        vad_target=torch.stack(
            [
                torch.tensor([b["valence"], b["arousal"]], dtype=torch.float)
                for b in batch
            ]
        ),
        vad_mask=stack("vad_mask", torch.float),
        intensity_target=stack("intensity", torch.float),
        intensity_mask=stack("intensity_mask", torch.float),
        sample_weight=stack("sample_weight", torch.float),
        teacher_logits=teacher_logits,
        kd_mask=kd_mask,
    )


def make_loader(
    records: Sequence[AffectRecord],
    tokenizer: Any,
    label_to_id: dict[str, int],
    batch_size: int = 64,
    shuffle: bool = True,
    label_field: str = "native_label",
    max_length: int = MAX_LENGTH,
    num_workers: int = 0,
) -> DataLoader:
    ds = AffectDataset(records, tokenizer, label_to_id, label_field, max_length)
    n = len(label_to_id)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, n),
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
