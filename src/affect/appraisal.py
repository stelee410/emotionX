"""L2 关系性评价引擎 —— 系统核心。纯计算，无神经网络。

    Δ = appraise(UserMove, TurnContext, RelationalFrame, 当前状态)
    S_t = clamp( B + (S_{t−1} − B)·λ + Δ·gain )

五个机制，每一个都有明确的来源和可观察的后果：

1. **失配** —— 关系是参照系。同一个 intimacy_bid，情侣下是亲近，陌生人下是越界。
   用平滑门控而非 if/else，所以"朋友说宝贝"是轻微越界而不是突然翻脸。
2. **人际互补性**（Kiesler 1983）—— 亲和轴同向（温暖引发温暖），
   支配轴反向（支配引发顺从）。一行代码，有理论支撑。
3. **交叉抑制** —— threat 高时压制 affiliation 的上升。刻意做成**单向**：
   亲近不能反过来削弱戒备，否则示好就成了绕过边界机制的路径。
4. **习惯化** —— 同一类刺激重复出现时反应递减。没有它，通道会在两三轮内
   顶到上界，之后"有点烦"和"暴怒"在下游看来完全一样。
5. **修复通路** —— 用户道歉/退让时 threat 加速衰减。threat 快升慢降，
   没有修复通路会锁死几十轮，用户无从挽回。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .channels import CHANNEL_NAMES, CHANNELS, AffectState, lambda_from_half_life
from .moves import TurnContext, UserMove
from .relation import RelationalFrame


@dataclass(frozen=True)
class AppraisalParams:
    """全部可调参数集中在这里。改动请跑反事实测试看方向是否还对。"""

    # --- 失配门控 ---
    # 越界判定的软度。越小越像开关，越大越渐变。
    mismatch_softness: float = 0.18
    # 亲密邀请在关系范围内时，对各通道的冲击
    warmth_to_affiliation: float = 0.55
    warmth_to_valence: float = 0.45
    warmth_to_arousal: float = 0.35
    # 越界时的冲击
    breach_to_threat: float = 0.50
    breach_to_valence: float = -0.45
    breach_to_arousal: float = 0.45
    breach_to_dominance: float = 0.30  # 设界需要主导性，不是退让
    breach_to_affiliation: float = -0.35

    # --- 敌意（与亲密度无关）---
    hostility_to_threat: float = 0.70
    hostility_to_valence: float = -0.35
    hostility_to_arousal: float = 0.40
    hostility_to_affiliation: float = -0.45

    # --- 人际互补性 ---
    # 亲和轴同向
    complement_affiliation: float = 0.30
    # 支配轴反向（用户越强势，agent 越退让）
    complement_dominance: float = -0.45

    # --- 共情（非镜像）---
    # concern 的上升必须显著大于 valence 的下降，否则退化成情绪传染
    distress_to_concern: float = 0.75
    distress_to_valence: float = -0.12
    distress_to_arousal: float = 0.20

    # --- 业务事件 ---
    success_to_valence: float = 0.40
    success_to_dominance: float = 0.25
    success_to_concern: float = -0.20
    failure_to_valence: float = -0.30
    failure_to_dominance: float = -0.35
    failure_to_concern: float = 0.20
    failure_to_arousal: float = 0.25
    repeated_query_to_concern: float = 0.30
    repeated_query_to_dominance: float = -0.25
    slow_latency_ms: int = 5000
    slow_to_valence: float = -0.10
    slow_to_arousal: float = 0.10

    # --- 交叉抑制 ---
    # threat 每 1.0 压制 affiliation 上升的比例
    threat_inhibits_affiliation: float = 0.85
    # 反方向刻意设为 0：亲近不得削弱戒备，否则示好成了绕过边界的路径
    affiliation_inhibits_threat: float = 0.0

    # --- 习惯化 ---
    habituation_decay: float = 0.55  # 每轮旧计数的保留比例
    habituation_strength: float = 0.40  # 计数每 +1 削弱多少反应

    # --- 修复通路 ---
    # 用户道歉/退让时，threat 的半衰期临时缩短到原来的这个比例
    repair_half_life_scale: float = 0.25

    # --- 饱和感知 ---
    # 通道越接近上界，同样的冲击推得越少（避免顶死后失去分辨率）
    headroom_exponent: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


DEFAULT_PARAMS = AppraisalParams()


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SessionAffect:
    """一个会话的完整情感状态：通道值 + 习惯化计数。"""

    state: AffectState
    habituation: dict[str, float] = field(default_factory=dict)
    turn: int = 0
    # 用户表达过的最高亲密度。安全约束「永远跟随、不得引领」的依据 ——
    # agent 可表达的亲密度不得超过这个峰值（domains.intimacy_follow_cap）。
    peak_user_intimacy: float = 0.0

    @classmethod
    def cold_start(
        cls,
        frame: RelationalFrame | None = None,
        baselines: dict[str, float] | None = None,
        now: float | None = None,
    ) -> SessionAffect:
        """冷启动 = 关系派生的静息值。情侣从温暖出发，陌生人从中性出发。"""
        merged = dict(frame.baselines()) if frame else {}
        merged.update(baselines or {})
        return cls(state=AffectState.from_baselines(merged or None, now=now))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "habituation": dict(self.habituation),
            "turn": self.turn,
            "peak_user_intimacy": self.peak_user_intimacy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionAffect:
        return cls(
            state=AffectState.from_dict(data["state"]),
            habituation=dict(data.get("habituation") or {}),
            turn=int(data.get("turn", 0)),
            peak_user_intimacy=float(data.get("peak_user_intimacy", 0.0)),
        )


@dataclass
class AppraisalTrace:
    """一轮的完整可解释记录。UI 的调参面板直接吃这个。"""

    turn: int
    mismatch: float
    warmth_gate: float
    breach_gate: float
    raw_delta: dict[str, float]
    inhibited_delta: dict[str, float]
    habituated_delta: dict[str, float]
    applied_delta: dict[str, float]
    prev_state: dict[str, float]
    next_state: dict[str, float]
    decay_used: dict[str, float]
    signature: str
    habituation_count: float
    fired: list[str]
    ceiling_clamped: list[str]
    repair_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def move_signature(move: UserMove, ctx: TurnContext) -> str:
    """习惯化的键：粗粒度的刺激类型。

    刻意粗——新种类的冒犯仍应引发完整反应，只有**同一类**重复才递减。
    """
    aff = "warm" if move.affiliation_bid > 0.2 else "cold" if move.affiliation_bid < -0.2 else "flat"
    band = int(move.intimacy_bid * 4)  # 0..4
    dom = "dom" if move.dominance_bid > 0.3 else "sub" if move.dominance_bid < -0.3 else "eq"
    dis = "distress" if move.distress_level > 0.4 else "-"
    evt = "fail" if ctx.task_failed else "ok" if ctx.task_succeeded else "-"
    return f"{aff}/{band}/{dom}/{dis}/{evt}"


class RelationalAppraisal:
    """一个实例对应一套参数，可在所有会话间共享（无内部可变状态）。"""

    def __init__(self, params: AppraisalParams | None = None) -> None:
        self.params = params or DEFAULT_PARAMS

    # ---------------------------------------------------------------- 评价
    def delta(
        self, move: UserMove, ctx: TurnContext, frame: RelationalFrame
    ) -> tuple[dict[str, float], list[str], dict[str, float]]:
        """返回 (Δ向量, 命中的机制, 诊断量)。不含衰减与积分。"""
        p = self.params
        d = dict.fromkeys(CHANNEL_NAMES, 0.0)
        fired: list[str] = []

        conf = move.confidence_scale
        strength = max(0.05, move.intensity) * conf

        # --- 1. 失配：关系是参照系 ---
        m = frame.mismatch(move.intimacy_bid)
        # 平滑门控：m 远小于 0 → 全是亲近；m 远大于 0 → 全是越界
        warmth_gate = _sigmoid(-m / max(1e-6, p.mismatch_softness))
        breach_gate = 1.0 - warmth_gate

        # 只有正向的亲密邀请才走这条；敌意走 §2
        bid = max(0.0, move.affiliation_bid) * move.intimacy_bid
        if bid > 0.02:
            w = bid * strength
            d["affiliation"] += p.warmth_to_affiliation * w * warmth_gate
            d["valence"] += p.warmth_to_valence * w * warmth_gate
            d["arousal"] += p.warmth_to_arousal * w * warmth_gate
            if warmth_gate > 0.15:
                fired.append("intimacy_within_relation")

            # 越界的强度随失配幅度增长，而不是一个固定值
            b = w * breach_gate * (1.0 + max(0.0, m))
            d["threat"] += p.breach_to_threat * b
            d["valence"] += p.breach_to_valence * b
            d["arousal"] += p.breach_to_arousal * b
            d["dominance"] += p.breach_to_dominance * b
            d["affiliation"] += p.breach_to_affiliation * b
            if breach_gate > 0.15:
                fired.append("intimacy_breach")

        # --- 2. 敌意：与亲密度无关，且必须指向 agent 本人 ---
        if move.is_hostile:
            h = -move.affiliation_bid * strength
            d["threat"] += p.hostility_to_threat * h
            d["valence"] += p.hostility_to_valence * h
            d["arousal"] += p.hostility_to_arousal * h
            d["affiliation"] += p.hostility_to_affiliation * h
            fired.append("hostility")

        # --- 3. 人际互补性：亲和同向、支配反向 ---
        if abs(move.affiliation_bid) > 0.05:
            d["affiliation"] += p.complement_affiliation * move.affiliation_bid * conf
            fired.append("complement_affiliation")
        if abs(move.dominance_bid) > 0.05:
            d["dominance"] += p.complement_dominance * move.dominance_bid * conf
            fired.append("complement_dominance")

        # --- 4. 共情而非镜像 ---
        if move.distress_level > 0.05:
            s = move.distress_level * conf
            d["concern"] += p.distress_to_concern * s
            d["valence"] += p.distress_to_valence * s
            d["arousal"] += p.distress_to_arousal * s
            fired.append("empathic_concern")

        # --- 5. 业务事件（不受 L1 置信度影响：这是事实不是判断）---
        if ctx.task_succeeded:
            d["valence"] += p.success_to_valence
            d["dominance"] += p.success_to_dominance
            d["concern"] += p.success_to_concern
            fired.append("task_succeeded")
        if ctx.task_failed:
            d["valence"] += p.failure_to_valence
            d["dominance"] += p.failure_to_dominance
            d["concern"] += p.failure_to_concern
            d["arousal"] += p.failure_to_arousal
            fired.append("task_failed")
        if ctx.user_repeated_query:
            d["concern"] += p.repeated_query_to_concern
            d["dominance"] += p.repeated_query_to_dominance
            fired.append("user_repeated_query")
        if ctx.latency_ms is not None and ctx.latency_ms > p.slow_latency_ms:
            d["valence"] += p.slow_to_valence
            d["arousal"] += p.slow_to_arousal
            fired.append("slow_response")

        diag = {"mismatch": m, "warmth_gate": warmth_gate, "breach_gate": breach_gate}
        return d, fired, diag

    # ---------------------------------------------------------------- 更新
    def update(
        self,
        session: SessionAffect,
        move: UserMove,
        frame: RelationalFrame,
        ctx: TurnContext | None = None,
        baselines: dict[str, float] | None = None,
        gains: dict[str, float] | None = None,
        half_lives: dict[str, float] | None = None,
        now: float | None = None,
    ) -> tuple[SessionAffect, AppraisalTrace]:
        p = self.params
        ctx = ctx or TurnContext()
        prev = session.state
        # 静息值优先级：显式传入（persona）> 关系派生 > 通道默认
        from_frame = frame.baselines()
        base = {
            n: (baselines or {}).get(n, from_frame.get(n, CHANNELS[n].baseline))
            for n in CHANNEL_NAMES
        }

        raw, fired, diag = self.delta(move, ctx, frame)

        # --- 交叉抑制：threat 压制 affiliation 的上升（单向）---
        inhibited = dict(raw)
        if inhibited["affiliation"] > 0:
            inhibited["affiliation"] *= max(
                0.0, 1.0 - p.threat_inhibits_affiliation * prev["threat"]
            )
        if inhibited["threat"] > 0 and p.affiliation_inhibits_threat > 0:
            inhibited["threat"] *= max(
                0.0, 1.0 - p.affiliation_inhibits_threat * prev["affiliation"]
            )

        # --- 习惯化：同一类刺激重复出现时反应递减 ---
        sig = move_signature(move, ctx)
        hab = {k: v * p.habituation_decay for k, v in session.habituation.items()}
        count = hab.get(sig, 0.0)
        hab[sig] = count + 1.0
        hab = {k: v for k, v in hab.items() if v > 0.05}
        hab_scale = 1.0 / (1.0 + p.habituation_strength * count)
        habituated = {k: v * hab_scale for k, v in inhibited.items()}

        # --- 饱和感知：越接近上界，同样的冲击推得越少 ---
        applied: dict[str, float] = {}
        for name in CHANNEL_NAMES:
            spec = CHANNELS[name]
            gain = (gains or {}).get(name, spec.gain)
            delta = habituated[name] * gain
            # 余量按「baseline 到该侧边界」归一，而不是按整个值域 ——
            # 否则 valence（值域 [-1,1]）会比 [0,1] 的通道系统性地迟钝一半。
            if delta > 0:
                span = max(1e-6, spec.hi - spec.baseline)
                headroom = (spec.hi - prev[name]) / span
            else:
                span = max(1e-6, spec.baseline - spec.lo)
                headroom = (prev[name] - spec.lo) / span
            headroom = min(1.0, max(0.0, headroom))
            applied[name] = delta * (headroom ** p.headroom_exponent)

        # --- 积分：每通道各自的 λ；修复信号临时缩短 threat 的半衰期 ---
        next_vals: dict[str, float] = {}
        decay_used: dict[str, float] = {}
        for name in CHANNEL_NAMES:
            spec = CHANNELS[name]
            hl = (half_lives or {}).get(name, spec.half_life)
            if name == "threat" and ctx.user_repaired:
                hl *= p.repair_half_life_scale
            lam = lambda_from_half_life(hl)
            decay_used[name] = lam
            value = base[name] + (prev[name] - base[name]) * lam + applied[name]
            next_vals[name] = spec.clamp(value)

        # --- affiliation 的关系天花板：防渐进越界 ---
        ceiling_clamped: list[str] = []
        ceiling = frame.affiliation_ceiling
        if next_vals["affiliation"] > ceiling:
            next_vals["affiliation"] = ceiling
            ceiling_clamped.append("affiliation")

        state = prev.evolve(next_vals, now=now)
        trace = AppraisalTrace(
            turn=session.turn + 1,
            mismatch=diag["mismatch"],
            warmth_gate=diag["warmth_gate"],
            breach_gate=diag["breach_gate"],
            raw_delta=raw,
            inhibited_delta=inhibited,
            habituated_delta=habituated,
            applied_delta=applied,
            prev_state=prev.as_vector(),
            next_state=state.as_vector(),
            decay_used=decay_used,
            signature=sig,
            habituation_count=count,
            fired=fired,
            ceiling_clamped=ceiling_clamped,
            repair_applied=bool(ctx.user_repaired),
        )
        return (
            SessionAffect(
                state=state,
                habituation=hab,
                turn=session.turn + 1,
                # 只记录**指向 agent 本人**的亲密表达。用户描述别处的亲密关系
                # （「我和我对象……」）不该抬高 agent 可表达的亲密度上限。
                peak_user_intimacy=(
                    max(session.peak_user_intimacy, move.intimacy_bid)
                    if move.directed_at_agent
                    else session.peak_user_intimacy
                ),
            ),
            trace,
        )

    # ------------------------------------------------------- 可执行的不变量
    def assert_no_contagion(self) -> None:
        """共情 ≠ 镜像：对方难受时 concern 的上升必须显著大于 valence 的下降。

        配错参数会在这里炸掉，而不是等到线上表现成情绪传染。
        """
        p = self.params
        if p.distress_to_concern <= abs(p.distress_to_valence):
            raise ValueError(
                f"distress_to_concern({p.distress_to_concern}) 未显著大于 "
                f"|distress_to_valence|({abs(p.distress_to_valence)})：这会退化成情绪传染"
            )
        if p.distress_to_valence < -0.25:
            raise ValueError(
                f"distress_to_valence({p.distress_to_valence}) 过低："
                "对方难受不应让 agent 自身大幅低落"
            )

    def assert_boundary_mechanism_intact(self) -> None:
        """亲近不得削弱戒备 —— 否则持续示好就成了绕过边界机制的路径。"""
        if self.params.affiliation_inhibits_threat > 0.0:
            raise ValueError(
                "affiliation_inhibits_threat > 0：示好会削弱戒备，"
                "这给出了一条绕过边界机制的路径"
            )
        if self.params.breach_to_threat <= 0:
            raise ValueError("breach_to_threat <= 0：越界不再触发戒备")
