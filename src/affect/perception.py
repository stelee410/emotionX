"""L1 感知层的 ONNX 推理封装。

输出 `UserMove`（关系无关）。延迟预算 ≤10ms @ CPU 单核。
**基准必须在生产硬件上测**，不得用开发机数据。

生产环境不装 torch 也不装 transformers：分词走纯 Python WordPiece。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .heuristic import HeuristicPerceiver, Perceiver
from .moves import UserMove
from .targets import REGRESSION_TARGETS, targets_to_move
from .text_format import MAX_LENGTH, build_l1_input
from .tokenization import WordPieceTokenizer


class OnnxPerceiver:
    """模型目录需含 model.onnx + vocab.txt + l1_meta.json。"""

    def __init__(
        self,
        model_dir: str | Path,
        onnx_file: str = "model.onnx",
        num_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

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
        self.targets: list[str] = self.meta.get("targets") or list(REGRESSION_TARGETS)
        self.max_length: int = int(self.meta.get("max_length", MAX_LENGTH))
        self.tokenizer = WordPieceTokenizer.from_pretrained_dir(self.model_dir)

    def perceive(self, utterance: str, last_agent_reply: str | None = None) -> UserMove:
        return self.perceive_batch([(utterance, last_agent_reply)])[0]

    def perceive_batch(self, pairs: list[tuple[str, str | None]]) -> list[UserMove]:
        import numpy as np

        texts = [build_l1_input(u, a) for u, a in pairs]
        enc = self.tokenizer.encode_batch(texts, max_length=self.max_length)
        feed = {
            name: np.asarray(enc[name], dtype=np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in self.input_names
        }
        outputs = self.session.run(None, feed)
        named = dict(zip(self.output_names, outputs, strict=False))
        move_out = named.get("move", outputs[0])
        directed_out = named.get("directed", outputs[1] if len(outputs) > 1 else None)

        results: list[UserMove] = []
        for i in range(len(texts)):
            values = dict(zip(self.targets, (float(x) for x in move_out[i]), strict=False))
            logit = float(directed_out[i][0]) if directed_out is not None else 1.0
            # 回归模型没有天然的置信度；用「离中性有多远」当代理，
            # 越接近全零的输出越可能是模型没看懂。
            spread = max(abs(v) for v in values.values()) if values else 0.0
            results.append(targets_to_move(values, logit, confidence=min(0.95, 0.35 + spread)))
        return results


def load_perceiver(
    model_dir: str | Path | None = None, fallback_to_heuristic: bool = True
) -> Perceiver:
    """有模型走 ONNX，没有则回退规则桩。"""
    if model_dir:
        try:
            return OnnxPerceiver(model_dir)
        except Exception:
            if not fallback_to_heuristic:
                raise
    return HeuristicPerceiver()
