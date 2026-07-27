"""§3 L1 感知层：推理封装。

两个实现，同一个协议：

* `HeuristicPerceiver` —— Phase 2 的规则桩（关键词 + 简单启发式）。零依赖，
  用来在没有模型时跑通全链路；也是 ONNX 加载失败时的降级路径。
* `OnnxPerceiver`     —— 生产路径。ONNX Runtime + 纯 Python WordPiece，不装 torch。

延迟预算 ≤10ms（CPU）。**基准必须在生产硬件上测**，不得用开发机数据（§3.5）。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

from .text_format import MAX_LENGTH, build_l1_input
from .tokenization import WordPieceTokenizer
from .types import ID_TO_STRATEGY, STRATEGY_LABELS, StrategyLabel, UserAffect, clamp


class Perceiver(Protocol):
    def perceive(self, user_utterance: str, last_agent_reply: str | None = None) -> UserAffect: ...


# ---------------------------------------------------------------------------
# 规则桩（Phase 2）
# ---------------------------------------------------------------------------

_FRUSTRATION_CUES = (
    (r"又(错|失败|不行|不对)", 0.9),
    (r"还是(不行|不对|没用|一样)", 0.9),
    (r"说了(多少|几)遍", 1.0),
    (r"重复", 0.5),
    (r"到底", 0.7),
    (r"能不能", 0.5),
    (r"听不懂(我说的)?话", 1.0),
    (r"废话", 0.9),
    (r"没用", 0.7),
    (r"算了", 0.6),
    (r"垃圾|傻|蠢", 1.0),
    (r"快点|赶紧|催", 0.6),
    (r"[?？]{2,}|[!！]{2,}", 0.6),
    (r"烦", 0.8),
    (r"搞什么|干什么呢", 0.8),
    (r"我不是问(这个|这)", 0.9),
)

_DISTRESS_CUES = (
    (r"难受|不舒服", 0.7),
    (r"疼|痛", 0.6),
    (r"害怕|怕", 0.8),
    (r"担心|焦虑|慌", 0.8),
    (r"睡不着|失眠", 0.6),
    (r"想哭|哭了", 0.9),
    (r"撑不(住|下去)", 1.0),
    (r"崩溃", 1.0),
    (r"无助|没办法了", 0.9),
    (r"绝望", 1.0),
    (r"压力(大|很大)", 0.7),
    (r"难过|伤心|痛苦", 0.9),
    (r"孤独|一个人", 0.5),
    (r"严重吗|会不会是", 0.6),
    (r"确诊|癌|恶性", 0.8),
    (r"怎么办", 0.6),
    (r"救救我|帮帮我", 0.9),
)

_POSITIVE_CUES = (
    (r"谢谢|感谢|多谢", 0.7),
    (r"太好了|棒|赞|给力", 0.9),
    (r"有用|解决了|好了", 0.8),
    (r"明白了|懂了|清楚了", 0.6),
    (r"开心|高兴|舒服多了", 0.9),
    (r"辛苦(你|了)", 0.5),
    (r"哈哈", 0.7),
    (r"喜欢", 0.6),
    (r"放心了|安心", 0.7),
)

# 注意：不要把「别」当否定词 —— 「特别难受」会被误判成否定。
_NEGATORS = ("不", "没")

_INTENSIFIERS = ((r"非常|特别|极其|太", 0.25), (r"有点|稍微|一点", -0.2), (r"死了|得要命", 0.3))


def _score(text: str, cues: tuple[tuple[str, float], ...]) -> tuple[float, list[str]]:
    total = 0.0
    hits: list[str] = []
    for pattern, weight in cues:
        m = re.search(pattern, text)
        if not m:
            continue
        # 简单否定处理："不难受" 不算 distress
        start = max(0, m.start() - 1)
        if text[start : m.start()] in _NEGATORS:
            continue
        total += weight
        hits.append(pattern)
    return total, hits


class HeuristicPerceiver:
    """关键词 + 简单启发式。**不要指望它的精度** —— 它的职责是让链路可跑通。"""

    def perceive(self, user_utterance: str, last_agent_reply: str | None = None) -> UserAffect:
        text = (user_utterance or "").strip()
        f_score, f_hits = _score(text, _FRUSTRATION_CUES)
        d_score, d_hits = _score(text, _DISTRESS_CUES)
        p_score, p_hits = _score(text, _POSITIVE_CUES)

        bump = 0.0
        for pattern, delta in _INTENSIFIERS:
            if re.search(pattern, text):
                bump += delta

        scores: dict[str, float] = {
            "frustration": f_score,
            "distress": d_score,
            "positive": p_score,
        }
        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]

        if best_score < 0.5:
            strategy: StrategyLabel = "neutral"
            intensity = clamp(0.15 + bump, 0.0, 1.0)
            valence, arousal = 0.0, 0.2
            confidence = 0.5
        else:
            strategy = best  # type: ignore[assignment]
            intensity = clamp(min(1.0, best_score / 2.0) + bump, 0.05, 1.0)
            runner_up = sorted(scores.values(), reverse=True)[1]
            margin = best_score - runner_up
            confidence = clamp(0.45 + 0.25 * min(best_score, 2.0) / 2.0 + 0.2 * min(margin, 1.0), 0.0, 0.95)
            if strategy == "frustration":
                valence, arousal = -0.45 * intensity - 0.15, clamp(0.45 + 0.4 * intensity, 0, 1)
            elif strategy == "distress":
                valence, arousal = -0.5 * intensity - 0.1, clamp(0.3 + 0.4 * intensity, 0, 1)
            else:
                valence, arousal = 0.4 * intensity + 0.2, clamp(0.3 + 0.3 * intensity, 0, 1)

        # 长句 + 全是标点的宣泄，额外提一点 arousal
        if len(text) > 60 and strategy != "neutral":
            arousal = clamp(arousal + 0.05, 0.0, 1.0)

        return UserAffect(
            valence=valence,
            arousal=arousal,
            strategy=strategy,
            intensity=intensity,
            confidence=confidence,
            raw_logits=[],
        )

    def explain(self, user_utterance: str) -> dict[str, Any]:
        f_score, f_hits = _score(user_utterance, _FRUSTRATION_CUES)
        d_score, d_hits = _score(user_utterance, _DISTRESS_CUES)
        p_score, p_hits = _score(user_utterance, _POSITIVE_CUES)
        return {
            "frustration": {"score": f_score, "hits": f_hits},
            "distress": {"score": d_score, "hits": d_hits},
            "positive": {"score": p_score, "hits": p_hits},
        }


# ---------------------------------------------------------------------------
# ONNX 生产路径
# ---------------------------------------------------------------------------


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


class OnnxPerceiver:
    """ONNX Runtime 推理。模型目录需含 `model.onnx` + `vocab.txt` + `l1_meta.json`。"""

    def __init__(
        self,
        model_dir: str | Path,
        onnx_file: str = "model.onnx",
        num_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort  # 延迟导入，便于纯 L2/L3 使用者不触发

        self.model_dir = Path(model_dir)
        onnx_path = self.model_dir / onnx_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"未找到 ONNX 模型: {onnx_path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

        meta_path = self.model_dir / "l1_meta.json"
        self.meta: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )
        self.labels: list[str] = self.meta.get("strategy_labels") or list(STRATEGY_LABELS)
        self.max_length: int = int(self.meta.get("max_length", MAX_LENGTH))
        self.tokenizer = WordPieceTokenizer.from_pretrained_dir(self.model_dir)

    def _run(self, texts: list[str]) -> list[dict[str, Any]]:
        import numpy as np

        enc = self.tokenizer.encode_batch(texts, max_length=self.max_length)
        feed = {
            name: np.asarray(enc[name], dtype=np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in self.input_names
        }
        outputs = self.session.run(None, feed)
        named = dict(zip(self.output_names, outputs, strict=False))
        logits = named.get("strategy_logits", outputs[0])
        vad = named.get("vad", outputs[1] if len(outputs) > 1 else None)
        intensity = named.get("intensity", outputs[2] if len(outputs) > 2 else None)

        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            row_logits = [float(x) for x in logits[i]]
            probs = _softmax(row_logits)
            idx = max(range(len(probs)), key=lambda k: probs[k])
            v, a = (
                (float(vad[i][0]), float(vad[i][1])) if vad is not None else (0.0, 0.3)
            )
            inten = (
                float(intensity[i][0] if hasattr(intensity[i], "__len__") else intensity[i])
                if intensity is not None
                else abs(v)
            )
            results.append(
                {
                    "label": self.labels[idx] if idx < len(self.labels) else ID_TO_STRATEGY[idx],
                    "probs": probs,
                    "logits": row_logits,
                    "valence": v,
                    "arousal": a,
                    "intensity": inten,
                }
            )
        return results

    def perceive(self, user_utterance: str, last_agent_reply: str | None = None) -> UserAffect:
        return self.perceive_batch([(user_utterance, last_agent_reply)])[0]

    def perceive_batch(
        self, pairs: list[tuple[str, str | None]]
    ) -> list[UserAffect]:
        texts = [build_l1_input(u, a) for u, a in pairs]
        out: list[UserAffect] = []
        for r in self._run(texts):
            out.append(
                UserAffect(
                    valence=r["valence"],
                    arousal=r["arousal"],
                    strategy=r["label"],
                    intensity=r["intensity"],
                    confidence=max(r["probs"]),
                    raw_logits=r["logits"],
                )
            )
        return out


def load_perceiver(
    model_dir: str | Path | None = None, fallback_to_heuristic: bool = True
) -> Perceiver:
    """有模型走 ONNX，没有则回退规则桩（Phase 2 → Phase 3 的平滑切换点）。"""
    if model_dir:
        try:
            return OnnxPerceiver(model_dir)
        except Exception:
            if not fallback_to_heuristic:
                raise
    return HeuristicPerceiver()
