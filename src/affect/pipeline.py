"""三层串联的对外接口（§7 Phase 2）。

    L1 感知 → L2 状态更新 → L3 表达指令

安全层在 L1 之前跑：§9.6 危机识别优先级高于本系统全部逻辑，且不由 L1 承担。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .expression import AffectPrompt, ExpressionTemplates, build_affect_prompt, load_templates
from .perception import HeuristicPerceiver, Perceiver, load_perceiver
from .persona import Persona, load_persona
from .safety import BypassKind, evaluate_safety
from .state_machine import AppraisalRules, StateMachine
from .store import StateStore, TraceLogger, make_store
from .types import AgentAffect, ConversationEvent, TurnTrace, UserAffect


@dataclass
class TurnResult:
    affect_prompt: str
    generation_params: dict[str, Any]
    user_affect: UserAffect
    agent_affect: AgentAffect
    trace: TurnTrace
    bypass: BypassKind = BypassKind.NONE
    matched_expression: list[str] = field(default_factory=list)

    def as_tuple(self) -> tuple[str, dict[str, Any]]:
        """§7 Phase 2 指定的返回形态：(affect_prompt, generation_params)。"""
        return self.affect_prompt, self.generation_params


class AffectPipeline:
    """一个进程内长期存活的对象。无 per-session 可变状态（状态在 store 里）。"""

    def __init__(
        self,
        perceiver: Perceiver | None = None,
        store: StateStore | None = None,
        rules: AppraisalRules | None = None,
        templates: ExpressionTemplates | None = None,
        trace_logger: TraceLogger | None = None,
        model_dir: str | Path | None = None,
        store_backend: str = "memory",
        check_invariants: bool = True,
    ) -> None:
        self.perceiver = perceiver or load_perceiver(model_dir)
        self.store = store or make_store(store_backend)
        self.rules = rules or AppraisalRules.load()
        self.templates = templates or load_templates()
        self.tracer = trace_logger or TraceLogger()
        self._machines: dict[str, StateMachine] = {}
        self._check_invariants = check_invariants

    # ---- persona / 状态机缓存 ----
    def machine(self, persona_name: str) -> StateMachine:
        if persona_name not in self._machines:
            persona = load_persona(persona_name)
            sm = StateMachine(persona=persona, rules=self.rules)
            if self._check_invariants:
                # 启动即校验「共情≠镜像」，配错规则表在这里就炸（§12.2）
                sm.assert_no_contagion()
            self._machines[persona_name] = sm
        return self._machines[persona_name]

    def persona(self, persona_name: str) -> Persona:
        return self.machine(persona_name).persona

    # ---- 主入口 ----
    def process_turn(
        self,
        session_id: str,
        user_utterance: str,
        last_agent_reply: str = "",
        event: ConversationEvent | None = None,
        persona_name: str = "steady_medical",
        now: float | None = None,
    ) -> TurnResult:
        event = event or ConversationEvent()
        now = time.time() if now is None else now
        sm = self.machine(persona_name)
        persona = sm.persona

        # 0) 安全层（优先级最高，独立于 L1）
        verdict = evaluate_safety(
            user_utterance,
            medical_bypass_enabled=persona.medical_bypass,
            extra_text="",
            crisis_sensitivity=persona.crisis_sensitivity,
        )

        # 1) L1 感知
        user = self.perceiver.perceive(user_utterance, last_agent_reply or None)

        # 2) L2 状态更新
        #    危机场景下仍然更新状态（会话是连续的），但 L3 走 bypass。
        prev = self.store.get(session_id)
        state, trace = sm.update(
            prev=prev,
            user=user,
            event=event,
            now=now,
            session_id=session_id,
            safety_bypass=(
                None if verdict.kind is BypassKind.NONE else verdict.kind.value
            )
            or ("ambiguous_crisis" if verdict.ambiguous_crisis_signal else None),
        )
        self.store.set(session_id, state)
        self.tracer.log(trace)

        # 3) L3 表达
        prompt: AffectPrompt = build_affect_prompt(
            state=state,
            user=user,
            persona=persona,
            event=event,
            safety=verdict,
            templates=self.templates,
        )

        return TurnResult(
            affect_prompt=prompt.text,
            generation_params=prompt.generation,
            user_affect=user,
            agent_affect=state,
            trace=trace,
            bypass=prompt.bypass,
            matched_expression=prompt.matched,
        )

    # ---- 运维接口 ----
    def get_state(self, session_id: str) -> AgentAffect | None:
        return self.store.get(session_id)

    def reset_session(self, session_id: str) -> None:
        self.store.delete(session_id)


_default_pipeline: AffectPipeline | None = None


def get_pipeline(**kwargs: Any) -> AffectPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = AffectPipeline(**kwargs)
    return _default_pipeline


def process_turn(
    session_id: str,
    user_utterance: str,
    last_agent_reply: str,
    event: ConversationEvent,
    persona_name: str,
) -> tuple[str, dict[str, Any]]:
    """§7 Phase 2 规定的函数级接口。内部复用进程内单例 pipeline。"""
    result = get_pipeline().process_turn(
        session_id=session_id,
        user_utterance=user_utterance,
        last_agent_reply=last_agent_reply,
        event=event,
        persona_name=persona_name,
    )
    return result.as_tuple()


__all__ = [
    "AffectPipeline",
    "HeuristicPerceiver",
    "TurnResult",
    "get_pipeline",
    "process_turn",
]
