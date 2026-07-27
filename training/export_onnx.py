"""§3.5 导出 ONNX + int8 动态量化。

    python training/export_onnx.py --model-dir artifacts/l1_stage2 --out artifacts/l1_onnx

产出目录（`OnnxPerceiver` 直接吃这个目录）：
    model.onnx        fp32
    model_int8.onnx   int8 动态量化
    vocab.txt         行号 == token id，供纯 Python 分词器
    l1_meta.json      标签、max_length、导出信息

⚠️ §3.5：**量化和延迟基准都必须在目标 Linux 服务器上做**，不得用开发机（M5）的数据。
本脚本在 Mac 上跑出的 int8 模型仅用于验证导出链路是否正确。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from affect.targets import REGRESSION_TARGETS  # noqa: E402
from affect.text_format import MAX_LENGTH, build_l1_input  # noqa: E402
from affect.tokenization import WordPieceTokenizer  # noqa: E402
from training.model import AffectEncoder, load_tokenizer, write_vocab_file  # noqa: E402

SANITY_TEXTS = [
    ("你好，帮我查一下挂号记录", ""),
    ("又错了，我说了多少遍了", "请再试一次"),
    ("我特别害怕，晚上一直睡不着", "报告已经收到了"),
    ("太好了，谢谢你，解决了", "问题已经修复"),
]


def export(
    model_dir: str | Path,
    out_dir: str | Path,
    opset: int = 17,
    quantize: bool = True,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AffectEncoder.load(model_dir, map_location="cpu").eval()
    if not model.has_move_head:
        raise SystemExit(f"{model_dir} 里的模型没有 move 头 —— 先跑 stage2_finetune.py")

    # vocab.txt：优先复用训练时导出的那份（保证 id 完全一致）
    src_vocab = model_dir / "vocab.txt"
    if src_vocab.exists():
        shutil.copy(src_vocab, out / "vocab.txt")
    else:
        write_vocab_file(load_tokenizer(model.base_model), out / "vocab.txt")

    dummy = torch.ones(2, max_length, dtype=torch.long)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        (dummy, torch.ones_like(dummy), torch.zeros_like(dummy)),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["move", "directed"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "move": {0: "batch"},
            "directed": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    result: dict[str, Any] = {
        "fp32_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "targets": list(REGRESSION_TARGETS),
    }

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out / "model_int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        result["int8_mb"] = round(int8_path.stat().st_size / 1e6, 2)

    meta = {
        "targets": list(REGRESSION_TARGETS),
        "max_length": max_length,
        "base_model": model.base_model,
        "hidden_size": model.config.hidden_size,
        "input_format": "[USER] {utterance} [SEP] [AGENT] {last_agent_reply}",
        "outputs": ["move(5)", "directed(1)"],
        "exported_from": str(model_dir),
        "note": "int8 量化与延迟基准必须在目标 Linux 服务器上重做（spec §3.5）",
    }
    (out / "l1_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def verify(out_dir: str | Path, model_dir: str | Path, tolerance: float = 2e-3) -> dict[str, Any]:
    """ONNX 输出必须与 torch 输出一致，且纯 Python 分词器与训练侧一致。"""
    out = Path(out_dir)
    torch_model = AffectEncoder.load(Path(model_dir), map_location="cpu").eval()
    tok_py = WordPieceTokenizer.from_pretrained_dir(out)
    meta = json.loads((out / "l1_meta.json").read_text(encoding="utf-8"))
    max_length = int(meta["max_length"])

    texts = [build_l1_input(u, a) for u, a in SANITY_TEXTS]
    enc = tok_py.encode_batch(texts, max_length=max_length)

    with torch.no_grad():
        t_move, t_dir = torch_model(
            torch.tensor(enc["input_ids"]),
            torch.tensor(enc["attention_mask"]),
            torch.tensor(enc["token_type_ids"]),
        )

    import onnxruntime as ort

    report: dict[str, Any] = {}
    for name in ("model.onnx", "model_int8.onnx"):
        path = out / name
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {k: np.asarray(v, dtype=np.int64) for k, v in enc.items()}
        o_move, o_dir = sess.run(None, feed)
        max_diff = float(np.abs(o_move - t_move.numpy()).max())
        sign_agree = int(((o_dir >= 0) == (t_dir.numpy() >= 0)).sum())
        report[name] = {
            "max_move_diff": round(max_diff, 5),
            "directed_agreement": f"{sign_agree}/{len(texts)}",
        }
        if name == "model.onnx" and max_diff > tolerance:
            raise SystemExit(f"fp32 ONNX 与 torch 输出不一致：max_diff={max_diff}")
    return report


def benchmark(
    out_dir: str | Path, onnx_file: str = "model_int8.onnx", n: int = 200, threads: int = 1
) -> dict[str, Any]:
    from affect.perception import OnnxPerceiver

    p = OnnxPerceiver(out_dir, onnx_file=onnx_file, num_threads=threads)
    utterance = "我按你说的做了，但是还是提交不了，这个问题到底怎么解决"
    reply = "请确认一下发票格式是否为 PDF"
    p.perceive(utterance, reply)  # 预热

    times: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        p.perceive(utterance, reply)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "onnx_file": onnx_file,
        "threads": threads,
        "n": n,
        "mean_ms": round(statistics.fmean(times), 3),
        "p50_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
        "p99_ms": round(times[int(len(times) * 0.99)], 3),
        "note": "开发机数据，仅供参考。§3.5 要求在生产硬件上重测。",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX + int8 量化")
    ap.add_argument("--model-dir", default=str(ROOT / "artifacts" / "l1_stage2"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "l1_onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--bench", type=int, default=200, help="基准迭代次数，0 = 跳过")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    info = export(
        args.model_dir, args.out, opset=args.opset, quantize=not args.no_quantize
    )
    print(f"导出完成 → {args.out}")
    print(f"  fp32 {info['fp32_mb']} MB" + (f" / int8 {info.get('int8_mb')} MB" if "int8_mb" in info else ""))

    report: dict[str, Any] = {"export": info}
    if not args.no_verify:
        report["verify"] = verify(args.out, args.model_dir)
        print("一致性校验：")
        for name, r in report["verify"].items():
            print(f"  {name}: {r}")

    if args.bench:
        for onnx_file in ("model.onnx", "model_int8.onnx"):
            if (Path(args.out) / onnx_file).exists():
                b = benchmark(args.out, onnx_file, n=args.bench, threads=args.threads)
                report.setdefault("benchmark", []).append(b)
                print(
                    f"  {onnx_file} 单线程延迟: p50={b['p50_ms']}ms p95={b['p95_ms']}ms "
                    f"p99={b['p99_ms']}ms"
                )
        print("  ⚠️ 以上是开发机数据。§3.5：延迟基准必须在目标 Linux 服务器上测。")

    (Path(args.out) / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
