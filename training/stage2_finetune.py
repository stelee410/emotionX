"""§3.3 阶段二 · 策略标签微调（+ §3.4.3 知识蒸馏）。

    # 有人工标注了（正常路径）
    python training/stage2_finetune.py --stage1 artifacts/l1_stage1 \
        --annotations data/exports/stage2_train.jsonl

    # 还没标注，先用开源标签弱映射把链路跑通（bootstrap，**不可上线**）
    python training/stage2_finetune.py --stage1 artifacts/l1_stage1 --bootstrap 6000

做法：丢弃阶段一的分类头，换上 4 类 StrategyLabel 头，lr=1e-5，2–3 epoch。

⚠️ dev 集永远从 `--annotations` 里切；bootstrap 数据只进训练集，
   否则「学生像不像老师」会被当成「模型准不准」（§8.1）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from affect.types import STRATEGY_LABELS  # noqa: E402
from training.data_module import make_loader, stratified_split  # noqa: E402
from training.datasets.registry import (  # noqa: E402
    AffectRecord,
    bootstrap_stage2_from_stage1,
    label_distribution,
    load_annotations,
    load_distilled,
    load_stage1,
)
from training.model import (  # noqa: E402
    AffectEncoder,
    compute_class_weights,
    embedding_size,
    load_tokenizer,
    multitask_loss,
    write_vocab_file,
)
from training.stage1_pretrain import macro_f1, pick_device, set_seed  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "l1_stage2"
LABELS = list(STRATEGY_LABELS)
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}


@torch.no_grad()
def evaluate(model: AffectEncoder, loader: Any, device: str) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    vad_err = 0.0
    vad_n = 0.0
    int_err = 0.0
    int_n = 0.0
    for batch in loader:
        b = batch.to(device)
        logits, vad, intensity = model(b.input_ids, b.attention_mask, b.token_type_ids)
        y_true.extend(b.labels.tolist())
        y_pred.extend(logits.argmax(dim=-1).tolist())
        vad_err += ((vad - b.vad_target).abs().mean(dim=-1) * b.vad_mask).sum().item()
        vad_n += b.vad_mask.sum().item()
        int_err += (
            ((intensity.squeeze(-1) - b.intensity_target).abs() * b.intensity_mask).sum().item()
        )
        int_n += b.intensity_mask.sum().item()

    per_class: dict[str, dict[str, float]] = {}
    for c, name in enumerate(LABELS):
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return {
        "macro_f1": round(macro_f1(y_true, y_pred, len(LABELS)), 4),
        "accuracy": round(
            sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p) / max(1, len(y_true)), 4
        ),
        "vad_mae": round(vad_err / vad_n, 4) if vad_n else None,
        "intensity_mae": round(int_err / int_n, 4) if int_n else None,
        "per_class": per_class,
        "n": len(y_true),
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[AffectRecord], list[AffectRecord], dict[str, Any]]:
    """返回 (train, dev, 数据来源说明)。"""
    human: list[AffectRecord] = []
    if args.annotations:
        human = load_annotations(args.annotations)
        print(f"人工标注 {len(human)} 条: {label_distribution(human, 'strategy')}")

    distilled: list[AffectRecord] = []
    if args.distilled:
        distilled = load_distilled(args.distilled)
        n_kd = sum(1 for r in distilled if r.teacher_logits)
        print(f"蒸馏数据 {len(distilled)} 条（含 soft logits {n_kd} 条）")

    boot: list[AffectRecord] = []
    if args.bootstrap:
        pools = load_stage1(args.bootstrap_datasets)
        flat = [r for records in pools.values() for r in records]
        boot = bootstrap_stage2_from_stage1(flat, limit=args.bootstrap)
        print(f"bootstrap 弱标签 {len(boot)} 条: {label_distribution(boot, 'strategy')}")

    if not (human or distilled or boot):
        raise SystemExit(
            "没有任何阶段二训练数据。\n"
            "  正常路径：用标注站标 300–1000 条真实会话，导出后 --annotations 传进来。\n"
            "  临时验证链路：--bootstrap 6000"
        )

    # dev 只从人工标注里切（§8.1）；没有人工标注时退化成从 bootstrap 切，
    # 并在指标里显式标注「不可作为验收依据」。
    dev_source = "human"
    if human:
        train_h, dev = stratified_split(human, dev_ratio=args.dev_ratio, seed=args.seed, label_field="strategy")
    else:
        dev_source = "bootstrap(不可作为验收依据)"
        train_h, dev = stratified_split(boot, dev_ratio=args.dev_ratio, seed=args.seed, label_field="strategy")
        boot = train_h
        train_h = []

    train = [*train_h, *distilled, *boot]
    info = {
        "n_human": len(human),
        "n_distilled": len(distilled),
        "n_bootstrap": len(boot),
        "dev_source": dev_source,
        "n_train": len(train),
        "n_dev": len(dev),
        "train_distribution": label_distribution(train, "strategy"),
        "dev_distribution": label_distribution(dev, "strategy"),
    }
    return train, dev, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段二 · 策略标签微调")
    ap.add_argument("--stage1", default=str(ROOT / "artifacts" / "l1_stage1"))
    ap.add_argument("--annotations", default=None, help="标注站导出的 JSONL")
    ap.add_argument("--distilled", default=None, help="教师蒸馏 JSONL（含 teacher_logits）")
    ap.add_argument("--bootstrap", type=int, default=0, help="用开源标签弱映射 N 条（仅验证链路）")
    ap.add_argument("--bootstrap-datasets", nargs="+", default=["ewect"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="§3.3 指定 1e-5")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--class-weight", default="inv_sqrt", choices=["inv_sqrt", "inv", "none"])
    ap.add_argument("--kd-temperature", type=float, default=2.0, help="§3.4.3 指定 T=2.0")
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--from-scratch", action="store_true", help="不加载阶段一权重（用于对照实验）")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device={device}")

    train, dev, info = build_dataset(args)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["dev_source"] != "human":
        print(
            "\n!! dev 集来自 bootstrap 弱标签。这里的 macro-F1 只说明链路通了，\n"
            "!! **不能**当作 §8.1 的验收指标。真实验收必须跑 eval/test_perception.py。\n"
        )

    tokenizer = load_tokenizer()
    stage1_dir = Path(args.stage1)
    if args.from_scratch or not (stage1_dir / "affect_model.json").exists():
        if not args.from_scratch:
            print(f"! 未找到阶段一权重 {stage1_dir}，退化为直接微调基座（效果会明显更差，见 §3.3）")
        model = AffectEncoder(strategy_labels=LABELS, vocab_size=embedding_size(tokenizer))
    else:
        print(f"加载阶段一权重: {stage1_dir}")
        model = AffectEncoder.load(stage1_dir)
        # §3.3：丢弃阶段一的分类头，换上本系统的 4 类 StrategyLabel 头
        model.attach_strategy_head(LABELS)
    model = model.to(device)

    train_loader = make_loader(
        train,
        tokenizer,
        LABEL_TO_ID,
        batch_size=args.batch_size,
        shuffle=True,
        label_field="strategy",
        max_length=args.max_length,
    )
    dev_loader = make_loader(
        dev,
        tokenizer,
        LABEL_TO_ID,
        batch_size=args.batch_size * 2,
        shuffle=False,
        label_field="strategy",
        max_length=args.max_length,
    )
    class_weight = compute_class_weights(
        [LABEL_TO_ID[r.strategy] for r in train if r.strategy], len(LABELS), mode=args.class_weight
    ).to(device)
    print(f"class_weight={[round(float(x), 3) for x in class_weight]}")

    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    t0 = time.time()
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "vad": 0.0, "int": 0.0, "kd": 0.0, "n": 0}
        for batch in train_loader:
            b = batch.to(device)
            logits, vad, intensity = model(b.input_ids, b.attention_mask, b.token_type_ids)
            loss = multitask_loss(
                logits=logits,
                vad_pred=vad,
                intensity_pred=intensity,
                labels=b.labels,
                vad_target=b.vad_target,
                vad_mask=b.vad_mask,
                intensity_target=b.intensity_target,
                intensity_mask=b.intensity_mask,
                sample_weight=b.sample_weight,
                class_weight=class_weight,
                teacher_logits=b.teacher_logits,
                kd_mask=b.kd_mask,
                kd_temperature=args.kd_temperature,
                kd_alpha=args.kd_alpha,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running["total"] += float(loss.total.detach())
            running["cls"] += float(loss.cls)
            running["vad"] += float(loss.vad)
            running["int"] += float(loss.intensity)
            running["kd"] += float(loss.kd) if loss.kd is not None else 0.0
            running["n"] += 1
            step += 1
            if step % args.log_every == 0:
                n = running["n"]
                print(
                    f"  e{epoch} step {step}/{total_steps} loss={running['total'] / n:.4f} "
                    f"cls={running['cls'] / n:.4f} vad={running['vad'] / n:.4f} "
                    f"kd={running['kd'] / n:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running = {k: (0.0 if k != "n" else 0) for k in running}

        metrics = evaluate(model, dev_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"  [dev] e{epoch} macro-F1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"vad_mae={metrics['vad_mae']} int_mae={metrics['intensity_mae']}"
        )
        for name, m in metrics["per_class"].items():
            print(f"        {name:<12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            model.save(out_dir, tokenizer=tokenizer)
            write_vocab_file(tokenizer, out_dir / "vocab.txt")
            print(f"        ↑ 保存最优 (epoch {epoch})")
        model.train()

    (out_dir / "stage2_history.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "data": info,
                "history": history,
                "best": {"epoch": best_epoch, "macro_f1": best_f1},
                "labels": LABELS,
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n阶段二模型 → {out_dir}（best epoch {best_epoch}, dev macro-F1 {best_f1:.4f}）")
    print(f"下一步：python training/export_onnx.py --model-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
