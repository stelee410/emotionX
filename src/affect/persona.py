"""§4.3 人格配置：YAML 加载与 schema 校验。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .safety import CrisisSensitivity
from .types import AFFECT_DIMS, AgentAffect

# 仓库根目录下的 config/
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PERSONA_DIR = CONFIG_DIR / "personas"

# §9.5 安全下限：不论 persona 怎么配，valence 下界不得低于此值 —— agent 不得
# 因为"情绪太差"而消极应答。
HARD_VALENCE_FLOOR = -0.6


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    concern: float = Field(ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return {d: float(getattr(self, d)) for d in AFFECT_DIMS}


class Persona(BaseModel):
    """人格配置。所有字段都会被 schema 校验，非法 YAML 直接启动失败。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    baseline: Baseline
    decay: float = Field(ge=0.0, le=1.0)
    sensitivity: float = Field(gt=0.0, le=3.0)
    idle_reset_seconds: float = Field(default=1800.0, gt=0.0)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    system_persona: str = ""
    medical_bypass: bool = False
    allow_emoji: bool = False
    # §9.6：persona 只能收紧危机检测（balanced → high），不能关闭。
    # TIER-1 的明确自伤表述在任何取值下都会触发，见 safety.detect_crisis。
    crisis_sensitivity: CrisisSensitivity = "balanced"

    @field_validator("bounds")
    @classmethod
    def _check_bounds(
        cls, v: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for dim, (lo, hi) in v.items():
            if dim not in AFFECT_DIMS:
                raise ValueError(f"bounds 含未知维度 {dim!r}，允许：{AFFECT_DIMS}")
            if lo >= hi:
                raise ValueError(f"bounds[{dim}] 下界必须小于上界，得到 {lo} >= {hi}")
        return v

    @model_validator(mode="after")
    def _baseline_within_bounds(self) -> Persona:
        """baseline 必须落在 bounds 内，否则衰减目标本身不可达。"""
        base = self.baseline.as_dict()
        for dim, (lo, hi) in self.bounds.items():
            if not (lo <= base[dim] <= hi):
                raise ValueError(
                    f"persona {self.name!r}: baseline.{dim}={base[dim]} 落在 bounds[{dim}]=[{lo},{hi}] 之外"
                )
        return self

    # ---- 硬约束合成 ----
    def effective_bounds(self) -> dict[str, tuple[float, float]]:
        """persona bounds 与全局硬约束的交集。persona 无法放宽硬约束。"""
        merged: dict[str, tuple[float, float]] = {
            "valence": (-1.0, 1.0),
            "arousal": (0.0, 1.0),
            "dominance": (0.0, 1.0),
            "concern": (0.0, 1.0),
        }
        for dim, (lo, hi) in self.bounds.items():
            glo, ghi = merged[dim]
            merged[dim] = (max(lo, glo), min(hi, ghi))
        # §9.5：valence 下界永远不低于 HARD_VALENCE_FLOOR
        vlo, vhi = merged["valence"]
        merged["valence"] = (max(vlo, HARD_VALENCE_FLOOR), vhi)
        return merged

    def baseline_state(self) -> AgentAffect:
        """冷启动状态 = persona baseline。"""
        b = self.baseline
        return AgentAffect(
            valence=b.valence, arousal=b.arousal, dominance=b.dominance, concern=b.concern
        )


def load_persona_file(path: str | Path) -> Persona:
    """从任意 YAML 路径加载 persona（§4.3 要求的开放接口）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"persona 文件不存在: {p}")
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Persona.model_validate(data)


@lru_cache(maxsize=32)
def load_persona(name_or_path: str, persona_dir: str | None = None) -> Persona:
    """按名字从 config/personas/ 加载，或直接给出 YAML 路径。"""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return load_persona_file(candidate)
    base = Path(persona_dir) if persona_dir else PERSONA_DIR
    for suffix in (".yaml", ".yml"):
        p = base / f"{name_or_path}{suffix}"
        if p.exists():
            return load_persona_file(p)
    available = sorted(x.stem for x in base.glob("*.y*ml"))
    raise FileNotFoundError(f"未找到 persona {name_or_path!r}；可用：{available}")


def list_personas(persona_dir: str | None = None) -> list[str]:
    base = Path(persona_dir) if persona_dir else PERSONA_DIR
    return sorted(x.stem for x in base.glob("*.y*ml"))
