"""§3.3 阶段一 · 通用情感感知预训练。

    python training/stage1_pretrain.py --epochs 3 --datasets ewect simplifyweibo

目标：让 encoder 学会「中文里什么样的表达携带情绪」这一通用能力。
每个数据集用**自己的原生标签**接一个独立分类头；VAD/intensity 头是共享的，
监督信号来自标签→VAD 先验（权重压到 0.3，见 model.PRIOR_VAD_WEIGHT）。

⚠️ 不要把阶段一和阶段二的数据混在一起一次训完（§3.3 结尾），效果会明显更差。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from training.data_module import label_vocab, make_loader, stratified_split  # noqa: E402
from training.datasets.registry import STAGE1_DEFAULT, load_stage1  # noqa: E402
from training.model import (  # noqa: E402
    BASE_MODEL,
    AffectEncoder,
    HeadSpec,
    compute_class_weights,
    embedding_size,
    load_tokenizer,
    write_vocab_file,
)

DEFAULT_OUT = ROOT / "artifacts" / "l1_stage1"


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p != c)
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
            continue
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


@torch.no_grad()
def evaluate_head(
    model: AffectEncoder, head: str, loader: Any, device: str, num_classes: int
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        b = batch.to(device)
        logits = model.forward_aux(head, b.input_ids, b.attention_mask, b.token_type_ids)
        y_true.extend(b.labels.tolist())
        y_pred.extend(logits.argmax(dim=-1).tolist())
    acc = (
        sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p) / len(y_true)
        if y_true
        else 0.0
    )
    return {
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "accuracy": acc,
        "n": len(y_true),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段一 · 通用情感感知预训练")
    ap.add_argument("--datasets", nargs="+", default=list(STAGE1_DEFAULT))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-per-dataset", type=int, default=None, help="每个数据集截断条数（调试用）")
    ap.add_argument("--class-weight", default="inv_sqrt", choices=["inv_sqrt", "inv", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device={device}  base={args.base_model}")

    print("加载数据集：")
    by_dataset = load_stage1(args.datasets)
    if args.max_per_dataset:
        by_dataset = {k: v[: args.max_per_dataset] for k, v in by_dataset.items()}

    tokenizer = load_tokenizer(args.base_model)

    # 每个数据集一个原生标签头
    specs: list[HeadSpec] = []
    vocabs: dict[str, dict[str, int]] = {}
    splits: dict[str, tuple[list, list]] = {}
    for name, records in by_dataset.items():
        vocab = label_vocab(records)
        vocabs[name] = vocab
        specs.append(HeadSpec(name=name, labels=list(vocab)))
        splits[name] = stratified_split(records, dev_ratio=0.08, seed=args.seed)

    model = AffectEncoder(
        base_model=args.base_model,
        move_head=False,
        aux_heads=specs,
        vocab_size=embedding_size(tokenizer),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M")

    train_loaders = {}
    dev_loaders = {}
    class_weights = {}
    for name, (train, dev) in splits.items():
        train_loaders[name] = make_loader(
            train,
            tokenizer,
            vocabs[name],
            batch_size=args.batch_size,
            shuffle=True,
            max_length=args.max_length,
        )
        dev_loaders[name] = make_loader(
            dev,
            tokenizer,
            vocabs[name],
            batch_size=args.batch_size * 2,
            shuffle=False,
            max_length=args.max_length,
        )
        cw = compute_class_weights(
            [vocabs[name][r.native_label] for r in train],
            len(vocabs[name]),
            mode=args.class_weight,
        ).to(device)
        class_weights[name] = cw
        print(f"  {name}: train={len(train)} dev={len(dev)} 类别={len(vocabs[name])}")

    # 多数据集交错训练：按各自 loader 长度加权抽样，避免大数据集把小的淹没
    steps_per_epoch = sum(len(dl) for dl in train_loaders.values())
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        iters = {name: iter(dl) for name, dl in train_loaders.items()}
        remaining = {name: len(dl) for name, dl in train_loaders.items()}
        running = {"total": 0.0, "n": 0}
        while any(v > 0 for v in remaining.values()):
            pool = [name for name, v in remaining.items() if v > 0]
            # 按剩余步数加权，让所有数据集在 epoch 内均匀铺开
            name = rng.choices(pool, weights=[remaining[n] for n in pool], k=1)[0]
            batch = next(iters[name])
            remaining[name] -= 1

            b = batch.to(device)
            logits = model.forward_aux(name, b.input_ids, b.attention_mask, b.token_type_ids)
            loss_value = torch.nn.functional.cross_entropy(
                logits, b.labels, weight=class_weights[name]
            )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running["total"] += float(loss_value.detach())
            running["n"] += 1
            global_step += 1
            if global_step % args.log_every == 0:
                n = running["n"]
                print(
                    f"  e{epoch} step {global_step}/{total_steps} "
                    f"loss={running['total'] / n:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)"
                )
                running = {"total": 0.0, "n": 0}

        epoch_metrics = {"epoch": epoch}
        for name, dl in dev_loaders.items():
            m = evaluate_head(model, name, dl, device, len(vocabs[name]))
            epoch_metrics[name] = m
            print(
                f"  [dev] {name}: macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} (n={m['n']})"
            )
        history.append(epoch_metrics)
        model.train()

    model.save(out_dir, tokenizer=tokenizer)
    write_vocab_file(tokenizer, out_dir / "vocab.txt")
    (out_dir / "stage1_history.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "history": history,
                "label_vocabs": vocabs,
                "elapsed_seconds": round(time.time() - t0, 1),
                "n_params": n_params,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n阶段一模型 → {out_dir}  （{time.time() - t0:.0f}s）")
    print("下一步：training/stage2_finetune.py --stage1 " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
