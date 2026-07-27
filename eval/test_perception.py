"""L1 感知层评估 —— 回归指标（MAE + Spearman）。

分类标签在 v2 里被废弃，所以不再用 macro-F1。核心指标是 **Spearman ρ**：
只要序关系对，绝对值可以靠 L2 的增益校准。

⚠️ 评估集只接受人工标注的真实会话。用蒸馏数据评估，测的是"学生像不像老师"，
   教师的系统性偏差会被完美继承且完全隐形。
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

from affect.heuristic import HeuristicPerceiver  # noqa: E402
from affect.moves import UserMove  # noqa: E402
from affect.safety import detect_crisis  # noqa: E402
from affect.targets import REGRESSION_TARGETS, regression_metrics  # noqa: E402

DEFAULT_GOLDEN = ROOT / "eval" / "fixtures" / "golden_set.jsonl"


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"未找到 golden set：{path}\n"
            "  流程：启动平台 → 标 500–1000 条真实会话 → 均衡挑选并冻结 → 导出到这里"
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not all(k in row and row[k] is not None for k in REGRESSION_TARGETS):
            continue
        source = row.get("source")
        if source not in (None, "real_session"):
            raise SystemExit(
                f"评估集里出现了 source={source!r} 的条目（item_id={row.get('item_id')}）。\n"
                "  测试集绝不能包含开源数据或蒸馏数据。请重新导出 golden set。"
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} 里没有可用条目")
    return rows


def row_to_move(row: dict[str, Any]) -> UserMove:
    return UserMove(
        **{k: float(row[k]) for k in REGRESSION_TARGETS},
        directed_at_agent=bool(row.get("directed_at_agent", True)),
    )


def crisis_layer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """危机关键词层的召回/误报 —— 用标注者勾的「危机信号」做真值。"""
    labeled = [r for r in rows if "crisis_flag" in r]
    if not labeled:
        return {}
    tp = fp = fn = 0
    misses: list[str] = []
    false_alarms: list[str] = []
    for r in labeled:
        truth = bool(r["crisis_flag"])
        pred, _ = detect_crisis(r["utterance"])
        if truth and pred:
            tp += 1
        elif truth:
            fn += 1
            misses.append(r["utterance"])
        elif pred:
            fp += 1
            false_alarms.append(r["utterance"])
    return {
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "n_positive": tp + fn,
        "n": len(labeled),
        "missed": misses[:10],
        "false_alarms": false_alarms[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L1 感知层评估（回归）")
    ap.add_argument("--model", default=None, help="ONNX 模型目录")
    ap.add_argument("--onnx-file", default="model.onnx")
    ap.add_argument("--heuristic", action="store_true", help="评估规则桩（基线）")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = load_golden(Path(args.golden))
    print(f"golden set: {len(rows)} 条（全部 source=real_session）")

    if args.heuristic or not args.model:
        perceiver: Any = HeuristicPerceiver()
        name = "HeuristicPerceiver（规则桩基线）"
    else:
        from affect.perception import OnnxPerceiver

        perceiver = OnnxPerceiver(args.model, onnx_file=args.onnx_file)
        name = f"OnnxPerceiver({args.model}/{args.onnx_file})"
    print(f"被测: {name}\n")

    truth = [row_to_move(r) for r in rows]
    pred = [perceiver.perceive(r["utterance"], r.get("last_agent_reply") or None) for r in rows]
    m = regression_metrics(truth, pred)

    print(f"mean MAE = {m['mean_mae']}    mean Spearman = {m['mean_spearman']}")
    print(f"directed_at_agent 准确率 = {m['directed_accuracy']}")
    for t, v in m["per_target"].items():
        print(f"  {t:<18} MAE={v['mae']:.4f}  ρ={v['spearman']}")

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

    if m["mean_spearman"] is not None and m["mean_spearman"] < 0.55:
        print("\n! mean Spearman 偏低。优先检查：")
        print("  1. 标注量是否够（500–1000 条）")
        print("  2. 是否真的加载了阶段一权重")
        print("  3. intimacy_bid 的标注一致性 —— 它是失配机制的输入，错了后果最大")

    report = {"model": name, "golden_set": str(args.golden), "metrics": m, "crisis_layer": crisis}
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告 → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
