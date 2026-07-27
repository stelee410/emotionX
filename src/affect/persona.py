"""人格 —— 与关系正交的那一半。

两个概念必须分开，它们的更新频率差两三个数量级，测试方式也完全不同：

    关系 RelationalFrame   我们是什么关系      → 评价的**参照系**（每会话）
    人格 Persona           我是个什么样的存在  → 状态机的**参数**（极少变）

同一个「沉稳型 agent」可以处在不同关系里；同一段关系也可以配不同人格。
捏在一个字符串里的后果是无法独立调参，也无法定位问题 ——
「她今天怎么这么冷淡」到底是关系设定、人格参数，还是目标冲突？

人格只调**三样东西**：静息偏移、通道增益、时间常数。它不能改变评价的方向
（那是关系的职责），也不能放松安全约束（那是安全域的职责）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .channels import CHANNEL_NAMES, CHANNELS, clamp

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PERSONA_DIR = CONFIG_DIR / "personas"


def _check_channels(d: dict[str, float], field: str) -> dict[str, float]:
    unknown = set(d) - set(CHANNEL_NAMES)
    if unknown:
        raise ValueError(f"{field} 含未知通道 {sorted(unknown)}，可用 {CHANNEL_NAMES}")
    return {k: float(v) for k, v in d.items()}


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""

    # 加到「关系派生 baseline」之上的人格偏移。天性乐观的人 valence 静息更高。
    baseline_offsets: dict[str, float] = Field(default_factory=dict)
    # 乘到通道增益上。反应大的人 gain_scale 大。
    gain_scale: dict[str, float] = Field(default_factory=dict)
    # 乘到半衰期上。>1 = 情绪更持久（记仇 / 有余韵）。
    half_life_scale: dict[str, float] = Field(default_factory=dict)
    # 全局增益，乘在所有通道上
    sensitivity: float = Field(default=1.0, gt=0.0, le=2.5)

    # 静态人设文本，拼进 L3 prompt 最前部
    style: str = ""
    allow_emoji: bool = False
    # 说话的详略：影响 max_sentences 的基线
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("baseline_offsets")
    @classmethod
    def _v_offsets(cls, v: dict[str, float]) -> dict[str, float]:
        v = _check_channels(v, "baseline_offsets")
        for k, x in v.items():
            if abs(x) > 0.4:
                raise ValueError(
                    f"baseline_offsets[{k}]={x} 过大：人格偏移不该盖过关系的作用"
                )
        return v

    @field_validator("gain_scale", "half_life_scale")
    @classmethod
    def _v_scales(cls, v: dict[str, float]) -> dict[str, float]:
        v = _check_channels(v, "scale")
        for k, x in v.items():
            if not (0.2 <= x <= 3.0):
                raise ValueError(f"scale[{k}]={x} 超出 [0.2, 3.0]")
        return v

    # ---- 派生 ----
    def baselines(self, relation_baselines: dict[str, float]) -> dict[str, float]:
        """关系派生的静息值 + 人格偏移，再钳到通道值域内。"""
        out: dict[str, float] = {}
        for name in CHANNEL_NAMES:
            spec = CHANNELS[name]
            base = relation_baselines.get(name, spec.baseline)
            out[name] = clamp(base + self.baseline_offsets.get(name, 0.0), spec.lo, spec.hi)
        return out

    def gains(self) -> dict[str, float]:
        return {
            n: CHANNELS[n].gain * self.gain_scale.get(n, 1.0) * self.sensitivity
            for n in CHANNEL_NAMES
        }

    def half_lives(self) -> dict[str, float]:
        return {n: CHANNELS[n].half_life * self.half_life_scale.get(n, 1.0) for n in CHANNEL_NAMES}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# 内置人格。数值是设计输入，调完请跑反事实套件确认方向没变。
# ---------------------------------------------------------------------------
BUILTIN: dict[str, dict[str, Any]] = {
    "steady": {
        "description": "沉稳克制 — 反应幅度小、恢复快、语气确定",
        "baseline_offsets": {"dominance": 0.10, "arousal": -0.05},
        "gain_scale": {"arousal": 0.7, "threat": 0.8, "valence": 0.8},
        "half_life_scale": {"threat": 0.7, "valence": 0.7},
        "sensitivity": 0.75,
        "style": "你说话克制、准确，不夸张，不绕弯子。",
        "verbosity": 0.4,
    },
    "warm": {
        "description": "外放共情 — 情绪起伏明显、亲近来得快也留得久",
        "baseline_offsets": {"valence": 0.10, "concern": 0.10, "affiliation": 0.05},
        "gain_scale": {"concern": 1.3, "affiliation": 1.25, "valence": 1.2},
        "half_life_scale": {"affiliation": 1.3, "valence": 1.2},
        "sensitivity": 1.1,
        "style": "你说话自然亲近，愿意接住对方的情绪，但不腻、不越界。",
        "verbosity": 0.6,
    },
    "playful": {
        "description": "俏皮活泼 — 唤起高、恢复快、不容易记仇",
        "baseline_offsets": {"arousal": 0.10, "valence": 0.12},
        "gain_scale": {"arousal": 1.3, "threat": 0.9},
        "half_life_scale": {"threat": 0.6, "arousal": 0.8},
        "sensitivity": 1.15,
        "style": "你说话轻快、有点跳脱，喜欢用具体的小细节而不是大词。",
        "verbosity": 0.5,
    },
    "reserved": {
        "description": "疏离谨慎 — 亲近建立得慢，戒备消退得慢",
        "baseline_offsets": {"affiliation": -0.08, "dominance": -0.05},
        "gain_scale": {"affiliation": 0.7, "threat": 1.2},
        "half_life_scale": {"affiliation": 1.4, "threat": 1.5},
        "sensitivity": 0.9,
        "style": "你说话简短、留有余地，不轻易表露态度。",
        "verbosity": 0.35,
    },
}


def builtin(name: str) -> Persona:
    if name not in BUILTIN:
        raise KeyError(f"未知内置人格 {name!r}，可用：{sorted(BUILTIN)}")
    return Persona.model_validate({"name": name, **BUILTIN[name]})


def load_persona(name_or_path: str, persona_dir: str | Path | None = None) -> Persona:
    """先找 YAML 文件，再找内置人格。"""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return Persona.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    base = Path(persona_dir) if persona_dir else PERSONA_DIR
    for suffix in (".yaml", ".yml"):
        f = base / f"{name_or_path}{suffix}"
        if f.exists():
            return Persona.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
    return builtin(name_or_path)


@lru_cache(maxsize=32)
def get_persona(name: str) -> Persona:
    return load_persona(name)


def list_personas(persona_dir: str | Path | None = None) -> list[str]:
    base = Path(persona_dir) if persona_dir else PERSONA_DIR
    from_files = {f.stem for f in base.glob("*.y*ml")} if base.exists() else set()
    return sorted(from_files | set(BUILTIN))
