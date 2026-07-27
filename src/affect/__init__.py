"""emotionX —— 关系条件化的情感反应引擎。

    L0 关系解析 → L1 感知（可训练）→ L2 关系性评价（核心，纯计算）→ L3a 表达 / L3b 显示

同一句输入在不同关系设定下产生方向相反的情感反应：关系不改变评价规则，
它改变规则据以评价的**参照系**。
"""

from .actions import ActionPlan, select_actions
from .appraisal import (
    AppraisalParams,
    AppraisalTrace,
    RelationalAppraisal,
    SessionAffect,
)
from .channels import CHANNEL_NAMES, CHANNELS, AffectState, bucket_of
from .display import DisplayState, DisplayTracker, render
from .domains import (
    SafetyDecision,
    SafetyDomainError,
    evaluate_turn_safety,
    safety_block,
    validate_frame,
)
from .expression import AffectPrompt, build_prompt
from .heuristic import HeuristicPerceiver, Perceiver
from .memory import HttpMemory, ManualMemory, MemoryQuery, NullMemory
from .moves import TurnContext, UserMove
from .persona import Persona, get_persona, list_personas
from .pipeline import AffectPipeline, SessionRecord, TurnResult, process_turn
from .relation import (
    RelationalFrame,
    RelationType,
    SafetyProfile,
    list_relation_types,
    preset,
)

__version__ = "0.2.0"

__all__ = [
    "CHANNELS",
    "CHANNEL_NAMES",
    "ActionPlan",
    "AffectPipeline",
    "AffectPrompt",
    "AffectState",
    "AppraisalParams",
    "AppraisalTrace",
    "DisplayState",
    "DisplayTracker",
    "HeuristicPerceiver",
    "HttpMemory",
    "ManualMemory",
    "MemoryQuery",
    "NullMemory",
    "Perceiver",
    "Persona",
    "RelationType",
    "RelationalAppraisal",
    "RelationalFrame",
    "SafetyDecision",
    "SafetyDomainError",
    "SafetyProfile",
    "SessionAffect",
    "SessionRecord",
    "TurnContext",
    "TurnResult",
    "UserMove",
    "bucket_of",
    "build_prompt",
    "evaluate_turn_safety",
    "get_persona",
    "list_personas",
    "list_relation_types",
    "preset",
    "process_turn",
    "render",
    "safety_block",
    "select_actions",
    "validate_frame",
]
