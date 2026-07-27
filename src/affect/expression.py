"""§5 L3 表达层：状态 → 自然语言行为指令 + 生成参数。

核心原则（§5.1）：**绝不把 VAD 数值直接塞进 prompt。** 连续状态先离散成 bucket，
每个 bucket 对应一段具体的行为指令。

组装顺序（§5.3）：
  1. persona 基础人设（静态）
  2. concern bucket directive
  3. dominance bucket directive
  4. overrides（若命中，覆盖上述冲突项）
  5. 安全约束（§9，恒定注入，不可被覆盖）

生成参数取所有命中 bucket 中的**最保守值**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .persona import CONFIG_DIR, Persona
from .safety import (
    CRISIS_GENERATION_PARAMS,
    CRISIS_RESPONSE_DIRECTIVE,
    BypassKind,
    SafetyVerdict,
    safety_block,
)
from .types import AgentAffect, ConversationEvent, UserAffect

DEFAULT_TEMPLATES_PATH = CONFIG_DIR / "expression_templates.yaml"

# §5.3：这些参数「越小越保守」，多 bucket 命中时取 min
CONSERVATIVE_MIN_KEYS = ("temperature", "top_p", "max_sentences")

# bucket directive 的拼装顺序（concern → dominance → arousal → valence）
DIMENSION_ORDER = ("concern", "dominance", "arousal", "valence")


@dataclass
class ExpressionTemplates:
    concern: dict[str, dict[str, Any]] = field(default_factory=dict)
    dominance: dict[str, dict[str, Any]] = field(default_factory=dict)
    arousal: dict[str, dict[str, Any]] = field(default_factory=dict)
    valence: dict[str, dict[str, Any]] = field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    medical_bypass: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ExpressionTemplates:
        p = Path(path) if path else DEFAULT_TEMPLATES_PATH
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        return cls(
            concern=data.get("concern") or {},
            dominance=data.get("dominance") or {},
            arousal=data.get("arousal") or {},
            valence=data.get("valence") or {},
            overrides=data.get("overrides") or {},
            conflicts=data.get("conflicts") or [],
            medical_bypass=data.get("medical_bypass") or {},
            defaults=data.get("defaults") or {},
        )

    def dimension(self, name: str) -> dict[str, dict[str, Any]]:
        return getattr(self, name)


@lru_cache(maxsize=8)
def load_templates(path: str | None = None) -> ExpressionTemplates:
    return ExpressionTemplates.load(path)


@dataclass
class AffectPrompt:
    """L3 输出。text 直接注入主 LLM 的 system 消息尾部（§10 待决项 #4 默认值）。"""

    text: str
    generation: dict[str, Any]
    bypass: BypassKind = BypassKind.NONE
    matched: list[str] = field(default_factory=list)
    bucket: str = ""

    def as_tuple(self) -> tuple[str, dict[str, Any]]:
        return self.text, self.generation


def _merge_generation(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """取最保守值：CONSERVATIVE_MIN_KEYS 取 min，其余后写覆盖。"""
    for key, value in (incoming or {}).items():
        if key in CONSERVATIVE_MIN_KEYS and key in target:
            target[key] = min(target[key], value)
        else:
            target[key] = value


def _hit_overrides(user: UserAffect, event: ConversationEvent) -> list[str]:
    """决定命中哪些 override（顺序即优先级，后者可再压制前者的维度）。"""
    hits: list[str] = []
    if user.strategy == "frustration" and user.is_high_intensity:
        hits.append("user_frustration_high")
    if user.strategy == "distress" and user.is_high_intensity:
        hits.append("user_distress_high")
    if event.task_failed:
        hits.append("task_failed")
    return hits


def build_affect_prompt(
    state: AgentAffect,
    user: UserAffect,
    persona: Persona,
    event: ConversationEvent | None = None,
    safety: SafetyVerdict | None = None,
    templates: ExpressionTemplates | None = None,
) -> AffectPrompt:
    event = event or ConversationEvent()
    tpl = templates or load_templates()
    gen: dict[str, Any] = dict(tpl.defaults)

    # ---- §9.6 危机：整个情感系统 bypass ----
    if safety is not None and safety.kind is BypassKind.CRISIS:
        text = "\n\n".join(
            [persona.system_persona.strip(), CRISIS_RESPONSE_DIRECTIVE, safety_block()]
        ).strip()
        gen.update(CRISIS_GENERATION_PARAMS)
        return AffectPrompt(
            text=text, generation=gen, bypass=BypassKind.CRISIS, matched=["crisis"]
        )

    # ---- §9.4 医疗信息：跳过情感修饰，只留人设 + 中性指令 + 安全约束 ----
    if safety is not None and safety.kind is BypassKind.MEDICAL:
        med = tpl.medical_bypass
        _merge_generation(gen, med.get("generation", {}))
        text = "\n\n".join(
            [
                persona.system_persona.strip(),
                str(med.get("directive", "")).strip(),
                safety_block(),
            ]
        ).strip()
        return AffectPrompt(
            text=text,
            generation=gen,
            bypass=BypassKind.MEDICAL,
            matched=["medical_bypass"],
            bucket=state.to_bucket(),
        )

    # ---- 常规路径 ----
    buckets = state.buckets()
    override_ids = _hit_overrides(user, event)
    suppressed: set[str] = set()
    for oid in override_ids:
        suppressed.update(tpl.overrides.get(oid, {}).get("suppress", []) or [])
    # 维度互斥：只看 agent 状态，L1 判断不准时也生效
    for rule in tpl.conflicts:
        when = rule.get("when") or {}
        if all(buckets.get(dim) == want for dim, want in when.items()):
            suppressed.update(rule.get("suppress") or [])

    directives: list[str] = []
    matched: list[str] = []

    # 2–3. bucket directive（concern → dominance → arousal → valence）
    for dim in DIMENSION_ORDER:
        cfg = tpl.dimension(dim).get(buckets[dim], {})
        # 生成参数即使被 suppress 也要参与"取最保守值"——被压制的是措辞，不是安全性
        _merge_generation(gen, cfg.get("generation", {}))
        if dim in suppressed:
            continue
        directive = str(cfg.get("directive", "")).strip()
        if directive:
            directives.append(directive)
            matched.append(f"{dim}:{buckets[dim]}")

    # 4. overrides（覆盖冲突项）
    for oid in override_ids:
        cfg = tpl.overrides.get(oid, {})
        _merge_generation(gen, cfg.get("generation", {}))
        directive = str(cfg.get("directive", "")).strip()
        if directive:
            directives.append(directive)
            matched.append(f"override:{oid}")

    if not persona.allow_emoji:
        directives.append("不要使用 emoji 或颜文字。")

    parts = [persona.system_persona.strip()]
    if directives:
        parts.append("【本轮表达方式】\n" + "\n".join(f"- {d}" for d in directives))
    # 5. 安全约束，恒定注入且放在最后（不可被上文覆盖）
    parts.append(safety_block())

    return AffectPrompt(
        text="\n\n".join(p for p in parts if p).strip(),
        generation=gen,
        bypass=BypassKind.NONE,
        matched=matched,
        bucket=state.to_bucket(),
    )
