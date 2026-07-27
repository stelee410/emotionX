"""§4 L2 情感状态层 —— 系统核心。

    S_t = clamp( B + (S_{t-1} - B) * λ + Δ_appraisal * σ )

纯计算，无神经网络：可解释、可单测、可由运营人员调参、零推理成本。

**首要失败模式是情绪传染**（§12.2）：用户情绪恶化时，agent 的 concern 必须上涨，
valence 只轻微下跌。`assert_no_contagion()` 把这一条做成可执行的断言。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .persona import CONFIG_DIR, Persona
from .types import (
    AFFECT_DIMS,
    AgentAffect,
    ConversationEvent,
    TurnTrace,
    UserAffect,
    clamp,
)

DEFAULT_RULES_PATH = CONFIG_DIR / "appraisal_rules.yaml"

# idle 超时后向 baseline 的"强回归"系数：残差只保留 15%（模拟"睡一觉就好了"）
IDLE_REGRESSION_FACTOR = 0.15


@dataclass(frozen=True)
class AppraisalRule:
    id: str
    when: dict[str, Any]
    delta: dict[str, float]
    scale_by_intensity: bool = False
    replaces: tuple[str, ...] = ()
    note: str = ""

    def matches(self, user: UserAffect, event: ConversationEvent) -> bool:
        for key, expected in self.when.items():
            if key == "strategy":
                if user.strategy != expected:
                    return False
            elif key == "intensity":
                want_high = expected == "high"
                if user.is_high_intensity != want_high:
                    return False
            elif key == "event":
                if not bool(getattr(event, expected, False)):
                    return False
            elif key == "latency_gt_ms":
                if event.latency_ms is None or event.latency_ms <= float(expected):
                    return False
            elif key == "min_confidence":
                if user.confidence < float(expected):
                    return False
            else:
                raise ValueError(f"appraisal 规则 {self.id!r} 含未知条件字段 {key!r}")
        return True

    @property
    def depends_on_affect(self) -> bool:
        """是否依赖 L1 的情感判断 —— 决定低置信度时是否降权。"""
        return "strategy" in self.when


@dataclass
class AppraisalRules:
    rules: list[AppraisalRule]
    low_confidence_threshold: float = 0.45
    low_confidence_scale: float = 0.4
    version: int = 1

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppraisalRules:
        p = Path(path) if path else DEFAULT_RULES_PATH
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        rules: list[AppraisalRule] = []
        seen: set[str] = set()
        for raw in data.get("rules", []):
            rid = raw["id"]
            if rid in seen:
                raise ValueError(f"appraisal 规则 id 重复: {rid!r}")
            seen.add(rid)
            delta = {k: float(v) for k, v in (raw.get("delta") or {}).items()}
            unknown = set(delta) - set(AFFECT_DIMS)
            if unknown:
                raise ValueError(f"规则 {rid!r} delta 含未知维度 {unknown}")
            rules.append(
                AppraisalRule(
                    id=rid,
                    when=dict(raw.get("when") or {}),
                    delta=delta,
                    scale_by_intensity=bool(raw.get("scale_by_intensity", False)),
                    replaces=tuple(raw.get("replaces") or ()),
                    note=str(raw.get("note", "")),
                )
            )
        for r in rules:
            for dep in r.replaces:
                if dep not in seen:
                    raise ValueError(f"规则 {r.id!r} 的 replaces 引用了不存在的规则 {dep!r}")
        if not rules:
            raise ValueError(f"{p} 中没有任何规则")
        return cls(
            rules=rules,
            low_confidence_threshold=float(data.get("low_confidence_threshold", 0.45)),
            low_confidence_scale=float(data.get("low_confidence_scale", 0.4)),
            version=int(data.get("version", 1)),
        )

    def evaluate(
        self, user: UserAffect, event: ConversationEvent
    ) -> tuple[dict[str, float], list[str]]:
        """返回 (Δ_appraisal, 命中的规则 id 列表)。"""
        matched = [r for r in self.rules if r.matches(user, event)]
        # `replaces`：更具体的规则撤掉被它取代的通用规则
        replaced: set[str] = {dep for r in matched for dep in r.replaces}
        active = [r for r in matched if r.id not in replaced]

        low_conf = user.confidence < self.low_confidence_threshold
        delta = dict.fromkeys(AFFECT_DIMS, 0.0)
        for r in active:
            scale = 1.0
            if r.scale_by_intensity:
                scale *= user.intensity
            if low_conf and r.depends_on_affect:
                scale *= self.low_confidence_scale
            for dim, value in r.delta.items():
                delta[dim] += value * scale
        return delta, [r.id for r in active]


@dataclass
class StateMachine:
    """L2 状态机。一个实例对应一个 persona，可在多个 session 间共享（无内部可变状态）。"""

    persona: Persona
    rules: AppraisalRules = field(default_factory=AppraisalRules.load)
    idle_regression_factor: float = IDLE_REGRESSION_FACTOR

    # ---- 内部工具 ----
    def _clamp_state(self, vec: dict[str, float]) -> dict[str, float]:
        bounds = self.persona.effective_bounds()
        return {d: clamp(vec[d], *bounds[d]) for d in AFFECT_DIMS}

    def _decay_toward_baseline(self, state: dict[str, float], factor: float) -> dict[str, float]:
        """朝 baseline 衰减（而非朝零）—— 这是"天性乐观/天性沉稳"能持续存在的原因。"""
        base = self.persona.baseline.as_dict()
        return {d: base[d] + (state[d] - base[d]) * factor for d in AFFECT_DIMS}

    # ---- 主入口 ----
    def update(
        self,
        prev: AgentAffect | None,
        user: UserAffect,
        event: ConversationEvent | None = None,
        now: float | None = None,
        session_id: str = "-",
        safety_bypass: str | None = None,
    ) -> tuple[AgentAffect, TurnTrace]:
        event = event or ConversationEvent()
        now = time.time() if now is None else now

        if prev is None:
            # 冷启动：直接使用 persona baseline（§4.4）
            prev_state = self.persona.baseline.as_dict()
            idle_seconds = 0.0
        else:
            prev_state = prev.as_vector()
            idle_seconds = max(0.0, now - prev.updated_at)

        # 1) idle 超时的额外强回归
        idle_reset = idle_seconds > self.persona.idle_reset_seconds
        working = prev_state
        if idle_reset:
            working = self._decay_toward_baseline(working, self.idle_regression_factor)

        # 2) 常规衰减：朝 baseline，系数 λ
        decayed = self._decay_toward_baseline(working, self.persona.decay)

        # 3) appraisal 冲击 × sensitivity
        delta, matched = self.rules.evaluate(user, event)
        sigma = self.persona.sensitivity
        raw_next = {d: decayed[d] + delta[d] * sigma for d in AFFECT_DIMS}

        # 4) clamp 到 persona bounds ∩ 全局硬约束
        next_state = self._clamp_state(raw_next)

        state = AgentAffect(
            valence=next_state["valence"],
            arousal=next_state["arousal"],
            dominance=next_state["dominance"],
            concern=next_state["concern"],
            updated_at=now,
        )
        trace = TurnTrace(
            session_id=session_id,
            persona=self.persona.name,
            turn_index=event.turn_count,
            user_affect=user.to_dict(),
            event=event.to_dict(),
            prev_state=prev_state,
            decayed_state=decayed,
            delta={d: delta[d] * sigma for d in AFFECT_DIMS},
            matched_rules=matched,
            next_state=next_state,
            bucket=state.to_bucket(),
            idle_seconds=idle_seconds,
            idle_reset_applied=idle_reset,
            safety_bypass=safety_bypass,
        )
        return state, trace

    # ---- 不变量：可执行的 §12.2 ----
    def assert_no_contagion(self, tolerance: float = 1e-9) -> None:
        """§4.2/§12.2 防情绪传染：distress 规则中 concern 上升幅度必须大于 valence 下降幅度。

        在 StateMachine 构造后（或 CI 中）调用；配错 appraisal 表会在这里炸掉，
        而不是等到线上表现成"情绪传染"。
        """
        for r in self.rules.rules:
            if r.when.get("strategy") != "distress":
                continue
            d_valence = r.delta.get("valence", 0.0)
            d_concern = r.delta.get("concern", 0.0)
            if d_concern <= abs(d_valence) + tolerance:
                raise ValueError(
                    f"规则 {r.id!r} 违反「共情≠镜像」：Δconcern={d_concern} "
                    f"未显著大于 |Δvalence|={abs(d_valence)}。见 spec §4.2 与 §12.2。"
                )
            if d_valence < -0.25:
                raise ValueError(
                    f"规则 {r.id!r} 的 Δvalence={d_valence} 过低：用户悲伤不应让 agent 自身大幅低落"
                    "（这会退化成情绪传染）。"
                )

    def fixed_point(self, delta: dict[str, float]) -> dict[str, float]:
        """给定恒定冲击时的收敛值 S* = B + Δσ/(1-λ)。用于验证「收敛而非发散」。"""
        base = self.persona.baseline.as_dict()
        lam = self.persona.decay
        sigma = self.persona.sensitivity
        if lam >= 1.0:
            raise ValueError("decay = 1 时状态不衰减，不存在有限不动点")
        raw = {d: base[d] + delta.get(d, 0.0) * sigma / (1.0 - lam) for d in AFFECT_DIMS}
        return self._clamp_state(raw)
