"""Affective Response Simulation System.

L1 感知（可训练） → L2 状态机（系统核心，纯计算） → L3 表达（prompt 组装）
"""

from .expression import build_affect_prompt
from .perception import HeuristicPerceiver, OnnxPerceiver, load_perceiver
from .persona import Persona, list_personas, load_persona
from .pipeline import AffectPipeline, TurnResult, process_turn
from .safety import BypassKind, evaluate_safety
from .state_machine import AppraisalRules, StateMachine
from .store import InMemoryStateStore, RedisStateStore, TraceLogger, make_store
from .types import AgentAffect, ConversationEvent, StrategyLabel, TurnTrace, UserAffect

__version__ = "0.1.0"

__all__ = [
    "AffectPipeline",
    "AgentAffect",
    "AppraisalRules",
    "BypassKind",
    "ConversationEvent",
    "HeuristicPerceiver",
    "InMemoryStateStore",
    "OnnxPerceiver",
    "Persona",
    "RedisStateStore",
    "StateMachine",
    "StrategyLabel",
    "TraceLogger",
    "TurnResult",
    "TurnTrace",
    "UserAffect",
    "build_affect_prompt",
    "evaluate_safety",
    "list_personas",
    "load_perceiver",
    "load_persona",
    "make_store",
    "process_turn",
]
