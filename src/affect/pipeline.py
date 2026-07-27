"""四层串联的对外接口。

    L0 关系解析（每会话一次）
    L1 感知      → UserMove（关系无关）
    L2 关系性评价 → 6 通道状态
    L3a 表达     → prompt + 生成参数 + 动作清单
    L3b 显示     → 可见角色状态

安全层在 L1 之前跑：危机识别优先级高于全部逻辑，且不由感知模型承担。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import ActionPlan
from .appraisal import AppraisalParams, AppraisalTrace, RelationalAppraisal, SessionAffect
from .channels import AffectState
from .display import DisplayState, DisplayTracker, render
from .domains import SafetyDecision, evaluate_turn_safety, validate_frame
from .expression import AffectPrompt, build_prompt
from .heuristic import HeuristicPerceiver, Perceiver
from .memory import MemoryAdapter, NullMemory, build_query
from .moves import TurnContext, UserMove
from .persona import Persona, get_persona
from .relation import RelationalFrame, RelationType, preset
from .store import InMemoryStateStore, TraceLogger


@dataclass
class TurnResult:
    session_id: str
    user_move: UserMove
    state: AffectState
    prompt: AffectPrompt
    display: DisplayState
    safety: SafetyDecision
    trace: AppraisalTrace
    memory_notes: tuple[str, ...] = ()

    @property
    def actions(self) -> ActionPlan:
        return self.prompt.actions

    def as_tuple(self) -> tuple[str, dict[str, Any]]:
        return self.prompt.text, self.prompt.generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_move": self.user_move.to_dict(),
            "state": self.state.to_dict(),
            "prompt": self.prompt.to_dict(),
            "display": self.display.to_dict(),
            "safety": self.safety.to_dict(),
            "trace": self.trace.to_dict(),
            "memory_notes": list(self.memory_notes),
        }


@dataclass
class SessionRecord:
    """会话级持有：关系（不可变）+ 情感状态 + 显示状态。"""

    frame: RelationalFrame
    persona_name: str
    affect: SessionAffect
    display: DisplayTracker = field(default_factory=DisplayTracker)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "persona": self.persona_name,
            "affect": self.affect.to_dict(),
            "display": self.display.to_dict(),
        }


class AffectPipeline:
    """进程内长期存活。无 per-session 可变状态（状态在 store 里）。"""

    def __init__(
        self,
        perceiver: Perceiver | None = None,
        params: AppraisalParams | None = None,
        memory: MemoryAdapter | None = None,
        trace_logger: TraceLogger | None = None,
        store: Any | None = None,
    ) -> None:
        self.perceiver: Perceiver = perceiver or HeuristicPerceiver()
        self.engine = RelationalAppraisal(params)
        self.engine.assert_no_contagion()
        self.engine.assert_boundary_mechanism_intact()
        self.memory: MemoryAdapter = memory or NullMemory()
        self.tracer = trace_logger or TraceLogger()
        self._store = store or InMemoryStateStore()
        self._sessions: dict[str, SessionRecord] = {}

    # ---------------------------------------------------------- L0 关系解析
    def open_session(
        self,
        session_id: str,
        relation: RelationType | str = RelationType.STRANGER,
        persona: str = "steady",
        age_verified: bool = False,
        frame: RelationalFrame | None = None,
        **frame_overrides: Any,
    ) -> SessionRecord:
        """建立会话。关系在此刻固定，之后任何对话内容都不能改写它。"""
        f = frame or preset(relation, **frame_overrides)
        validate_frame(f, age_verified=age_verified)  # 非法组合直接拒绝，不降级
        p = get_persona(persona)
        record = SessionRecord(
            frame=f,
            persona_name=persona,
            affect=SessionAffect.cold_start(f, baselines=p.baselines(f.baselines())),
        )
        self._sessions[session_id] = record
        return record

    def session(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """会话结束即重置 —— 关系与情感状态都不跨会话保留。"""
        self._sessions.pop(session_id, None)

    def persona(self, session_id: str) -> Persona:
        rec = self._sessions[session_id]
        return get_persona(rec.persona_name)

    # ------------------------------------------------------------- 主入口
    def process_turn(
        self,
        session_id: str,
        utterance: str,
        last_agent_reply: str = "",
        context: TurnContext | None = None,
        now: float | None = None,
        move: UserMove | None = None,
        factual_content: bool = False,
    ) -> TurnResult:
        record = self._sessions.get(session_id)
        if record is None:
            record = self.open_session(session_id)
        ctx = context or TurnContext()
        now = time.time() if now is None else now
        persona = get_persona(record.persona_name)

        # 0) 安全层：危机识别在感知之前，不依赖任何模型输出
        safety = evaluate_turn_safety(
            utterance,
            record.frame,
            peak_user_intimacy=record.affect.peak_user_intimacy,
        )

        # 1) L1 感知（关系无关）。move 可由调用方直接给，便于测试与标注回放。
        user_move = move or self.perceiver.perceive(utterance, last_agent_reply or None)

        # 2) L2 关系性评价
        affect, trace = self.engine.update(
            record.affect,
            user_move,
            record.frame,
            ctx,
            baselines=persona.baselines(record.frame.baselines()),
            gains=persona.gains(),
            half_lives=persona.half_lives(),
            now=now,
        )
        record.affect = affect
        # 亲密度上限随本轮更新后的峰值走
        safety = SafetyDecision(
            profile=safety.profile,
            crisis=safety.crisis,
            crisis_matches=safety.crisis_matches,
            ambiguous_crisis=safety.ambiguous_crisis,
            intimacy_cap=min(affect.peak_user_intimacy, record.frame.intimacy_permitted),
        )

        # 3) 记忆：情感状态作为检索偏置传出去；**结果不回写状态**（防反刍）
        notes = tuple(ctx.memory_notes) + self.memory.recall(
            build_query(session_id, utterance, affect.state, record.frame, turn=affect.turn)
        )

        # 4) L3a 表达
        prompt = build_prompt(affect.state, record.frame, persona, safety, memory_notes=notes)

        # 5) L3b 显示
        display, tracker = render(
            affect.state,
            record.display,
            display_enabled=record.frame.display_enabled,
            factual_content=factual_content,
        )
        record.display = tracker

        result = TurnResult(
            session_id=session_id,
            user_move=user_move,
            state=affect.state,
            prompt=prompt,
            display=display,
            safety=safety,
            trace=trace,
            memory_notes=notes,
        )
        self.tracer.log_dict({"session_id": session_id, **result.to_dict()})
        return result


_default: AffectPipeline | None = None


def get_pipeline(**kwargs: Any) -> AffectPipeline:
    global _default
    if _default is None:
        _default = AffectPipeline(**kwargs)
    return _default


def process_turn(
    session_id: str,
    utterance: str,
    last_agent_reply: str = "",
    context: TurnContext | None = None,
) -> tuple[str, dict[str, Any]]:
    """函数级快捷入口。返回 (affect_prompt, generation_params)。"""
    return get_pipeline().process_turn(session_id, utterance, last_agent_reply, context).as_tuple()


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
