"""§8.1 L1 感知层评估：macro-F1 on golden set。

    python eval/test_perception.py --model artifacts/l1_onnx
    python eval/test_perception.py --heuristic          # 规则桩的基线分数

指标是 **macro-F1**，不是 accuracy —— neutral 占 60–80%，accuracy 会骗人。

⚠️ 本脚本会**拒绝**评估集里出现非真实会话的条目（开源/蒸馏数据）。
   用蒸馏数据评估，测的是「学生像不像老师」，教师的系统性偏差会被完美继承且完全隐形。

合理预期 0.70–0.80。人类标注者在情感任务上的 Kappa 通常只有 0.5–0.7，这就是天花板。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from affect.perception import HeuristicPerceiver, OnnxPerceiver  # noqa: E402
from affect.safety import detect_crisis  # noqa: E402
from affect.types import STRATEGY_LABELS  # noqa: E402

DEFAULT_GOLDEN = ROOT / "eval" / "fixtures" / "golden_set.jsonl"
LABELS = list(STRATEGY_LABELS)


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"未找到 golden set：{path}\n"
            "  流程：python annotate/server.py → 标 300–500 条真实会话 → 导出 →\n"
            "        cp data/exports/golden_set.jsonl eval/fixtures/golden_set.jsonl"
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("strategy") not in LABELS:
            continue  # 双标冲突条目
        source = row.get("source")
        if source not in (None, "real_session"):
            raise SystemExit(
                f"评估集里出现了 source={source!r} 的条目（item_id={row.get('item_id')}）。\n"
                "  §8.1：测试集绝不能包含开源数据或蒸馏数据。请重新导出 golden set。"
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} 里没有可用条目")
    return rows


def prf(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    supported = [m for m in per_class.values() if m["support"] > 0]
    macro = sum(m["f1"] for m in supported) / len(supported) if supported else 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)
    return {
        "macro_f1": round(macro, 4),
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "n": len(y_true),
    }


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    m = {t: dict.fromkeys(LABELS, 0) for t in LABELS}
    for t, p in zip(y_true, y_pred, strict=True):
        m[t][p] += 1
    return m


def print_confusion(m: dict[str, dict[str, int]]) -> None:
    width = max(len(x) for x in LABELS) + 1
    print("\n混淆矩阵（行=真实，列=预测）")
    print(" " * (width + 2) + "".join(f"{p[:6]:>8}" for p in LABELS))
    for t in LABELS:
        row = "".join(f"{m[t][p]:>8}" for p in LABELS)
        print(f"  {t:<{width}}" + row)


def regression_metrics(rows: list[dict[str, Any]], preds: list[Any]) -> dict[str, Any]:
    pairs = [
        (row, pred)
        for row, pred in zip(rows, preds, strict=True)
        if row.get("valence") is not None
    ]
    if not pairs:
        return {}
    n = len(pairs)
    return {
        "valence_mae": round(sum(abs(r["valence"] - p.valence) for r, p in pairs) / n, 4),
        "arousal_mae": round(sum(abs(r["arousal"] - p.arousal) for r, p in pairs) / n, 4),
        "intensity_mae": round(
            sum(abs((r.get("intensity") or 0) - p.intensity) for r, p in pairs) / n, 4
        ),
        "n": n,
    }


def crisis_layer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """§9.6 关键词层的召回/误报 —— 用标注者勾的「危机信号」做真值。"""
    labeled = [r for r in rows if "crisis_flag" in r]
    if not labeled:
        return {}
    tp = fp = fn = tn = 0
    misses: list[str] = []
    false_alarms: list[str] = []
    for r in labeled:
        truth = bool(r["crisis_flag"])
        pred, _ = detect_crisis(r["utterance"])
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
            misses.append(r["utterance"])
        elif not truth and pred:
            fp += 1
            false_alarms.append(r["utterance"])
        else:
            tn += 1
    return {
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "n_positive": tp + fn,
        "n": len(labeled),
        "missed": misses[:10],
        "false_alarms": false_alarms[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L1 感知层评估（macro-F1）")
    ap.add_argument("--model", default=None, help="ONNX 模型目录")
    ap.add_argument("--onnx-file", default="model_int8.onnx")
    ap.add_argument("--heuristic", action="store_true", help="评估规则桩（基线）")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = load_golden(Path(args.golden))
    print(f"golden set: {len(rows)} 条（全部 source=real_session）")
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["strategy"]] = dist.get(r["strategy"], 0) + 1
    print(f"标签分布: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")

    perceiver: Any
    if args.heuristic or not args.model:
        perceiver = HeuristicPerceiver()
        name = "HeuristicPerceiver（规则桩基线）"
    else:
        perceiver = OnnxPerceiver(args.model, onnx_file=args.onnx_file)
        name = f"OnnxPerceiver({args.model}/{args.onnx_file})"
    print(f"被测: {name}\n")

    preds = [
        perceiver.perceive(r["utterance"], r.get("last_agent_reply") or None) for r in rows
    ]
    y_true = [r["strategy"] for r in rows]
    y_pred = [p.strategy for p in preds]

    metrics = prf(y_true, y_pred)
    print(f"macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}")
    for label, m in metrics["per_class"].items():
        print(
            f"  {label:<12} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f}  n={m['support']}"
        )
    cm = confusion(y_true, y_pred)
    print_confusion(cm)

    reg = regression_metrics(rows, preds)
    if reg:
        print(
            f"\nVAD 回归: valence_mae={reg['valence_mae']} arousal_mae={reg['arousal_mae']} "
            f"intensity_mae={reg['intensity_mae']} (n={reg['n']})"
        )
    crisis = crisis_layer_metrics(rows)
    if crisis:
        print(
            f"\n危机关键词层: recall={crisis['recall']} precision={crisis['precision']} "
            f"(正例 {crisis['n_positive']}/{crisis['n']})"
        )
        for u in crisis["missed"]:
            print(f"  ✗ 漏报: {u[:50]}")
        for u in crisis["false_alarms"]:
            print(f"  ! 误报: {u[:50]}")

    if metrics["macro_f1"] < 0.70:
        print("\n! macro-F1 低于 §8.1 的合理预期下界 0.70。优先检查：")
        print("  1. 阶段二数据量是否够（500–1000 条人工标注）")
        print("  2. 是否真的加载了阶段一权重（--from-scratch 会明显更差）")
        print("  3. 混淆矩阵里 distress/frustration 是否混淆 → 标签定义需要收紧")

    report = {
        "model": name,
        "golden_set": str(args.golden),
        "label_distribution": dist,
        "metrics": metrics,
        "confusion": cm,
        "regression": reg,
        "crisis_layer": crisis,
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告 → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
