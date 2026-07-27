"""阶段二 · UserMove 回归微调。

    # 有人工标注（正常路径）
    python training/stage2_finetune.py --stage1 artifacts/l1_stage1 \
        --annotations data/exports/stage2_train.jsonl

    # 还没标注，先用开源标签弱映射把链路跑通（**不可上线**）
    python training/stage2_finetune.py --stage1 artifacts/l1_stage1 --bootstrap 6000

分类标签在 v2 里被废弃：策略取决于关系，而感知层看不到关系，
所以没有任何一个标签是对的。这里学的是句子本身的属性
（亲和/支配/亲密/痛苦/强度 + 是否指向 agent），关系条件化留给 L2。

指标是 MAE 与 **Spearman ρ** —— 序关系比绝对值重要，绝对值可由 L2 的增益校准。
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

from affect.targets import REGRESSION_TARGETS, spearman  # noqa: E402
from training.data_module import make_loader, stratified_split  # noqa: E402
from training.datasets.registry import (  # noqa: E402
    AffectRecord,
    bootstrap_stage2_from_stage1,
    load_annotations,
    load_distilled,
    load_stage1,
)
from training.model import (  # noqa: E402
    AffectEncoder,
    embedding_size,
    load_tokenizer,
    move_loss,
    write_vocab_file,
)
from training.stage1_pretrain import pick_device, set_seed  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "l1_stage2"


@torch.no_grad()
def evaluate(model: AffectEncoder, loader: Any, device: str) -> dict[str, Any]:
    model.eval()
    truth: list[list[float]] = []
    pred: list[list[float]] = []
    d_true: list[float] = []
    d_pred: list[float] = []
    for batch in loader:
        b = batch.to(device)
        move, directed = model(b.input_ids, b.attention_mask, b.token_type_ids)
        truth.extend(b.targets.tolist())
        pred.extend(move.tolist())
        d_true.extend(b.directed.tolist())
        d_pred.extend((directed.squeeze(-1) >= 0).float().tolist())

    per_target: dict[str, dict[str, float]] = {}
    for i, name in enumerate(REGRESSION_TARGETS):
        t = [row[i] for row in truth]
        p = [row[i] for row in pred]
        mae = sum(abs(a - b) for a, b in zip(t, p, strict=True)) / max(1, len(t))
        rho = spearman(t, p)
        per_target[name] = {
            "mae": round(mae, 4),
            "spearman": round(rho, 4) if rho == rho else None,
        }
    rhos = [m["spearman"] for m in per_target.values() if m["spearman"] is not None]
    return {
        "n": len(truth),
        "per_target": per_target,
        "mean_mae": round(sum(m["mae"] for m in per_target.values()) / len(per_target), 4),
        "mean_spearman": round(sum(rhos) / len(rhos), 4) if rhos else None,
        "directed_accuracy": round(
            sum(1 for a, b in zip(d_true, d_pred, strict=True) if a == b) / max(1, len(d_true)), 4
        ),
    }


def build_dataset(args: argparse.Namespace) -> tuple[list, list, dict[str, Any]]:
    human: list[AffectRecord] = []
    if args.annotations:
        human = load_annotations(args.annotations)
        print(f"人工标注 {len(human)} 条")

    distilled: list[AffectRecord] = []
    if args.distilled:
        distilled = load_distilled(args.distilled)
        print(f"蒸馏数据 {len(distilled)} 条")

    boot: list[AffectRecord] = []
    if args.bootstrap:
        pools = load_stage1(args.bootstrap_datasets)
        flat = [r for records in pools.values() for r in records]
        boot = bootstrap_stage2_from_stage1(flat, limit=args.bootstrap)
        print(f"bootstrap 弱标签 {len(boot)} 条")

    if not (human or distilled or boot):
        raise SystemExit(
            "没有阶段二训练数据。\n"
            "  正常路径：标注站标 500–1000 条真实会话 → --annotations\n"
            "  临时验证：--bootstrap 6000"
        )

    # dev 只从人工标注里切；没有人工标注时退化并显式标注不可作为验收依据
    if human:
        train_h, dev = stratified_split(human, dev_ratio=args.dev_ratio, seed=args.seed)
        dev_source = "human"
    else:
        train_h, dev = stratified_split(boot, dev_ratio=args.dev_ratio, seed=args.seed)
        boot, train_h, dev_source = train_h, [], "bootstrap(不可作为验收依据)"

    train = [*train_h, *distilled, *boot]
    return train, dev, {
        "n_human": len(human),
        "n_distilled": len(distilled),
        "n_bootstrap": len(boot),
        "dev_source": dev_source,
        "n_train": len(train),
        "n_dev": len(dev),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段二 · UserMove 回归微调")
    ap.add_argument("--stage1", default=str(ROOT / "artifacts" / "l1_stage1"))
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--distilled", default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--bootstrap-datasets", nargs="+", default=["ewect"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device={device}")

    train, dev, info = build_dataset(args)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["dev_source"] != "human":
        print(
            "\n!! dev 集来自 bootstrap 弱标签。这里的指标只说明链路通了，\n"
            "!! **不能**当作验收依据。真实验收要跑 eval/test_perception.py。\n"
        )

    tokenizer = load_tokenizer()
    stage1_dir = Path(args.stage1)
    if args.from_scratch or not (stage1_dir / "affect_model.json").exists():
        if not args.from_scratch:
            print(f"! 未找到阶段一权重 {stage1_dir}，退化为直接微调基座（效果会明显更差）")
        model = AffectEncoder(move_head=True, vocab_size=embedding_size(tokenizer))
    else:
        print(f"加载阶段一权重: {stage1_dir}")
        model = AffectEncoder.load(stage1_dir)
        model.attach_move_head()  # 丢弃阶段一的原生标签头
    model = model.to(device)

    train_loader = make_loader(
        train, tokenizer, batch_size=args.batch_size, shuffle=True, max_length=args.max_length
    )
    dev_loader = make_loader(
        dev, tokenizer, batch_size=args.batch_size * 2, shuffle=False, max_length=args.max_length
    )

    total_steps = max(1, len(train_loader) * args.epochs)
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
    best = -1.0
    best_epoch = 0
    t0 = time.time()
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "move": 0.0, "dir": 0.0, "n": 0}
        for batch in train_loader:
            b = batch.to(device)
            move, directed = model(b.input_ids, b.attention_mask, b.token_type_ids)
            loss = move_loss(
                pred=move,
                directed_logit=directed,
                target=b.targets,
                directed_target=b.directed,
                sample_weight=b.sample_weight,
                target_mask=b.target_mask,
                teacher=b.teacher,
                kd_mask=b.kd_mask,
                kd_alpha=args.kd_alpha,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running["total"] += float(loss.total.detach())
            running["move"] += float(loss.move)
            running["dir"] += float(loss.directed)
            running["n"] += 1
            step += 1
            if step % args.log_every == 0:
                n = running["n"]
                print(
                    f"  e{epoch} step {step}/{total_steps} loss={running['total'] / n:.4f} "
                    f"move={running['move'] / n:.4f} dir={running['dir'] / n:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running = {"total": 0.0, "move": 0.0, "dir": 0.0, "n": 0}

        metrics = evaluate(model, dev_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"  [dev] e{epoch} mean_MAE={metrics['mean_mae']} "
            f"mean_Spearman={metrics['mean_spearman']} "
            f"directed_acc={metrics['directed_accuracy']}"
        )
        for name, m in metrics["per_target"].items():
            print(f"        {name:<16} MAE={m['mae']:.4f}  ρ={m['spearman']}")

        score = metrics["mean_spearman"] if metrics["mean_spearman"] is not None else -metrics["mean_mae"]
        if score > best:
            best, best_epoch = score, epoch
            model.save(out_dir, tokenizer=tokenizer)
            write_vocab_file(tokenizer, out_dir / "vocab.txt")
            print(f"        ↑ 保存最优 (epoch {epoch})")

    (out_dir / "stage2_history.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "data": info,
                "history": history,
                "best": {"epoch": best_epoch, "score": best},
                "targets": list(REGRESSION_TARGETS),
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n阶段二模型 → {out_dir}（best epoch {best_epoch}）")
    print(f"下一步：python training/export_onnx.py --model-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
