"""§3.4 本地教师蒸馏：MLX-LM LoRA 教师训练 + 批量标注（Apple Silicon 本机）。

三个子命令：

    # 1. 用人工标注的 300–500 条，生成 LoRA 训练数据
    python training/distill_labeler.py prepare --annotations data/exports/stage2_train.jsonl

    # 2. LoRA 微调教师（走 mlx_lm.lora，内部不联网调云 API）
    python training/distill_labeler.py train --model mlx-community/Qwen3-4B-Instruct-4bit

    # 3. 批量标注真实会话，同时保存 soft logits（T=2.0 蒸馏用）
    python training/distill_labeler.py label --pool data/raw/pool.jsonl \
        --adapter artifacts/teacher_lora --out data/distilled/labeled.jsonl

⚠️ §3.4.2 **医疗类会话数据必须走本地标注，不得调用云 API。** 本脚本只用 MLX 本地推理，
   不含任何网络请求（模型权重需事先 `huggingface-cli download` 到本地）。

soft logits 的取法：让教师只输出 A/B/C/D 之一，然后**单次前向**读取该位置上这 4 个
token 的 logits。比让模型生成再解析更稳，也真的拿到了分布而不是 one-hot。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from affect.types import STRATEGY_LABELS  # noqa: E402

LABELS = list(STRATEGY_LABELS)
CHOICES = ["A", "B", "C", "D"]
LABEL_TO_CHOICE = dict(zip(LABELS, CHOICES, strict=True))
CHOICE_TO_LABEL = {c: lab for lab, c in LABEL_TO_CHOICE.items()}

DEFAULT_TEACHER = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_ADAPTER = ROOT / "artifacts" / "teacher_lora"
DEFAULT_LORA_DATA = ROOT / "data" / "teacher_lora"

SYSTEM_PROMPT = """你是对话应答策略标注器。读用户这一轮的话，判断 agent 应该采取哪种应答策略。

A neutral      直接回答就好。事务性提问、确认、简短应答。
B distress     用户在表达痛苦/焦虑/无助，需要先接住情绪再谈事情。
C frustration  用户对交互过程不满（绕圈子、答非所问、反复失败），要的是效率不是共情。
D positive     用户满意/愉悦/道谢并且值得 agent 顺着轻松一下。

判断标准是「如果我是客服，下一句该先做什么」，不是「用户是什么情绪」。
只输出一个字母，不要解释。"""

USER_TEMPLATE = """{context}用户这一轮：{utterance}

答案（只输出 A/B/C/D 一个字母）："""


def _format_user(utterance: str, last_agent_reply: str = "") -> str:
    context = f"上一轮 agent 说：{last_agent_reply}\n" if last_agent_reply else ""
    return USER_TEMPLATE.format(context=context, utterance=utterance)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# prepare：人工标注 → mlx_lm.lora 的训练数据
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> int:
    rows = [r for r in _read_jsonl(args.annotations) if r.get("strategy") in LABELS]
    if len(rows) < 100:
        print(
            f"! 只有 {len(rows)} 条标注。§3.4 建议 300–500 条再训教师，"
            "否则教师的偏差会被学生完美继承。"
        )
    random.Random(args.seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_ratio))
    splits = {"valid": rows[:n_valid], "train": rows[n_valid:]}

    out_dir = Path(args.out)
    for split, items in splits.items():
        formatted = [
            {
                "prompt": (
                    f"{SYSTEM_PROMPT}\n\n"
                    + _format_user(r["utterance"], r.get("last_agent_reply") or "")
                ),
                "completion": LABEL_TO_CHOICE[r["strategy"]],
            }
            for r in items
        ]
        _write_jsonl(out_dir / f"{split}.jsonl", formatted)
        print(f"  {split}: {len(formatted)} 条 → {out_dir / f'{split}.jsonl'}")
    # mlx_lm.lora 需要 test.jsonl 存在
    _write_jsonl(out_dir / "test.jsonl", [])
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["strategy"]] = dist.get(r["strategy"], 0) + 1
    print(f"标签分布: {dist}")
    print(f"\n下一步：python training/distill_labeler.py train --data {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# train：调 mlx_lm.lora
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    if not (data_dir / "train.jsonl").exists():
        raise SystemExit(f"{data_dir}/train.jsonl 不存在 —— 先跑 prepare")
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(args.adapter),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--learning-rate",
        str(args.lr),
    ]
    print("执行: " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "\n! mlx_lm.lora 执行失败。检查：\n"
            "  1. 是否在 Apple Silicon 上（pip install '.[mac]'）\n"
            "  2. 教师模型是否已下载到本地 HF 缓存\n"
            "  3. 该版本 mlx-lm 的参数名是否变化（试 python -m mlx_lm lora --help）"
        )
        return result.returncode
    print(f"\n教师 adapter → {args.adapter}")
    print(
        "下一步：python training/distill_labeler.py label "
        f"--pool data/raw/pool.jsonl --adapter {args.adapter}"
    )
    return 0


# ---------------------------------------------------------------------------
# label：批量标注 + soft logits
# ---------------------------------------------------------------------------
class MLXTeacher:
    """封装 MLX 本地推理。单次前向取 4 个候选 token 的 logits。"""

    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - 仅 Apple Silicon
            raise SystemExit(
                "需要 mlx-lm：pip install '.[mac]'（仅 Apple Silicon）"
            ) from exc
        self.mx = mx
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        # 候选字母对应的 token id。用 encode 而不是硬编码，避免不同 tokenizer 错位。
        self.choice_ids: list[int] = []
        for c in CHOICES:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if not ids:
                raise SystemExit(f"tokenizer 无法编码候选 {c!r}")
            self.choice_ids.append(ids[0])
        if len(set(self.choice_ids)) != len(CHOICES):
            raise SystemExit(f"候选字母的 token id 有冲突: {self.choice_ids}")

    def _prompt_ids(self, utterance: str, last_agent_reply: str) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user(utterance, last_agent_reply)},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return list(
                self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
            )
        text = f"{SYSTEM_PROMPT}\n\n{_format_user(utterance, last_agent_reply)}"
        return list(self.tokenizer.encode(text))

    def classify(self, utterance: str, last_agent_reply: str = "") -> dict[str, Any]:
        ids = self._prompt_ids(utterance, last_agent_reply)
        logits = self.model(self.mx.array([ids]))[0, -1, :]
        raw = [float(logits[i]) for i in self.choice_ids]
        m = max(raw)
        exps = [math.exp(x - m) for x in raw]
        s = sum(exps)
        probs = [e / s for e in exps]
        best = max(range(len(probs)), key=lambda k: probs[k])
        return {
            "strategy": CHOICE_TO_LABEL[CHOICES[best]],
            "teacher_logits": [round(x, 4) for x in raw],
            "probs": [round(p, 4) for p in probs],
            "confidence": round(probs[best], 4),
        }


def cmd_label(args: argparse.Namespace) -> int:
    pool = _read_jsonl(args.pool)
    if args.limit:
        pool = pool[: args.limit]
    print(f"待标注 {len(pool)} 条；教师={args.model} adapter={args.adapter}")
    if not Path(args.adapter).exists():
        print(f"! adapter {args.adapter} 不存在，将用未微调的基座模型标注（质量会差很多）")
        adapter = None
    else:
        adapter = str(args.adapter)

    teacher = MLXTeacher(args.model, adapter)
    out_rows: list[dict[str, Any]] = []
    low_conf = 0
    for i, row in enumerate(pool, 1):
        utterance = str(row.get("utterance") or row.get("text") or "").strip()
        if not utterance:
            continue
        reply = str(row.get("last_agent_reply") or "").strip()
        r = teacher.classify(utterance, reply)
        if r["confidence"] < args.min_confidence:
            low_conf += 1
            if args.drop_low_confidence:
                continue
        out_rows.append(
            {
                "utterance": utterance,
                "last_agent_reply": reply,
                "session_id": row.get("session_id", ""),
                "turn_index": row.get("turn_index", 0),
                "strategy": r["strategy"],
                "teacher_logits": r["teacher_logits"],
                "confidence": r["confidence"],
                # 低置信度样本降权，避免教师的犹豫被学生放大
                "weight": round(min(1.0, r["confidence"] / 0.8), 3),
                "teacher": f"{args.model}+{Path(args.adapter).name if adapter else 'base'}",
                "source": "distilled",
            }
        )
        if i % 200 == 0:
            print(f"  {i}/{len(pool)}")

    path = _write_jsonl(args.out, out_rows)
    dist: dict[str, int] = {}
    for r in out_rows:
        dist[r["strategy"]] = dist.get(r["strategy"], 0) + 1
    print(f"\n标注完成 {len(out_rows)} 条 → {path}")
    print(f"分布: {dist}；低置信度({args.min_confidence}) {low_conf} 条")
    print(
        "\n下一步：python training/stage2_finetune.py --stage1 artifacts/l1_stage1 \\\n"
        f"          --annotations data/exports/stage2_train.jsonl --distilled {path}"
    )
    print(
        "提醒 §8.1：蒸馏数据只能进训练集。golden set 必须是人工标注的真实会话。"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本地教师蒸馏（MLX，Apple Silicon）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="人工标注 → LoRA 训练数据")
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", default=str(DEFAULT_LORA_DATA))
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="LoRA 微调教师")
    t.add_argument("--model", default=DEFAULT_TEACHER)
    t.add_argument("--data", default=str(DEFAULT_LORA_DATA))
    t.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    t.add_argument("--iters", type=int, default=600)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--num-layers", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-5)
    t.set_defaults(func=cmd_train)

    lb = sub.add_parser("label", help="批量标注 + soft logits")
    lb.add_argument("--pool", required=True, help="待标注 JSONL（真实会话）")
    lb.add_argument("--model", default=DEFAULT_TEACHER)
    lb.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    lb.add_argument("--out", default=str(ROOT / "data" / "distilled" / "labeled.jsonl"))
    lb.add_argument("--limit", type=int, default=None)
    lb.add_argument("--min-confidence", type=float, default=0.5)
    lb.add_argument("--drop-low-confidence", action="store_true")
    lb.set_defaults(func=cmd_label)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
