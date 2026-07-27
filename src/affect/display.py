"""L3b 显示层 —— 情感状态对用户可见时的映射。

内部状态不能直接送给前端，三个原因：

1. **抖动**。内部状态每轮都变，直接驱动表情会让形象每句话换一次脸，像坏了。
   → 低通滤波 + 最小停留轮数。
2. **6 维连续量没有对应的表情**。→ 映射到有限的显示词汇（10 种）。
3. **threat 的渲染必须单独设计**。内部语义是"戒备/设界"，
   直接渲染成"生气的表情"就变成 agent 对用户发怒 —— 那是产品事故。
   → 正确的渲染是**距离感**：眼神移开、语速放慢、身体后撤、语气变平。

两条硬约束：
  * 显示强度不得超过内部强度（显示是内部状态的衰减投影，不能放大）
  * 传递会影响用户决策的事实性信息时，显示强制中性 —— 否则等于用非语言
    渠道给事实加权
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .channels import CHANNEL_NAMES, CHANNELS, AffectState, bucket_of, clamp

# 显示状态每轮向新目标移动的比例（低通滤波）。越小越平滑、越迟钝。
SMOOTHING = 0.45
# 同一个显示词汇至少维持的轮数，避免闪烁
MIN_DWELL_TURNS = 2
# 显示强度相对内部强度的上限
DISPLAY_INTENSITY_CAP = 0.85


@dataclass(frozen=True)
class DisplayState:
    """给前端 / TTS / 形象引擎的输出。**不包含原始通道数值。**"""

    mood: str  # 显示词汇
    label: str  # 中文标签
    intensity: float  # [0,1] 表现强度
    # 给形象引擎的解耦参数，避免前端反推内部状态
    posture: str  # 身体姿态倾向
    gaze: str  # 视线
    tempo: float  # [0,1] 语速/节奏
    warmth: float  # [0,1] 表现出的温度
    distance: float  # [0,1] 距离感 ← threat 的正确渲染
    neutralised: bool = False  # 是否因事实性内容被强制中性

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class MoodSpec:
    key: str
    label: str
    when: dict[str, str]
    posture: str
    gaze: str
    tempo: float
    warmth: float
    distance: float
    priority: int = 0


# ---------------------------------------------------------------------------
# 显示词汇表。刻意只有 10 种 —— 连续量映射到有限集，前端才好做资源。
# 注意 guarded/withdrawn 的 posture 全部是「后撤/移开」，没有任何攻击性表现。
# ---------------------------------------------------------------------------
MOODS: tuple[MoodSpec, ...] = (
    MoodSpec("calm", "平静", {}, "放松", "平视", 0.5, 0.5, 0.2, 0),
    MoodSpec("attentive", "专注", {"arousal": "medium", "concern": "medium"}, "微微前倾", "注视", 0.5, 0.55, 0.15, 1),
    MoodSpec("warm", "温和", {"affiliation": "medium", "valence": "high"}, "放松前倾", "柔和注视", 0.45, 0.75, 0.1, 2),
    MoodSpec("fond", "亲近", {"affiliation": "high", "threat": "low"}, "自然靠近", "长时间注视", 0.4, 0.95, 0.05, 3),
    MoodSpec("delighted", "欣喜", {"valence": "high", "arousal": "high"}, "轻快", "明亮注视", 0.7, 0.85, 0.05, 3),
    MoodSpec("caring", "关切", {"concern": "high"}, "前倾", "专注柔和", 0.35, 0.7, 0.1, 4),
    MoodSpec("hesitant", "迟疑", {"dominance": "low", "arousal": "medium"}, "微收", "间歇移开", 0.4, 0.45, 0.35, 2),
    MoodSpec("distant", "疏离", {"threat": "medium"}, "后撤", "视线移开", 0.4, 0.25, 0.65, 4),
    MoodSpec("guarded", "戒备", {"threat": "high"}, "明显后撤", "不再注视", 0.35, 0.1, 0.9, 6),
    MoodSpec("weary", "疲惫", {"valence": "low", "arousal": "low"}, "松垮", "低垂", 0.3, 0.35, 0.4, 3),
)

MOODS_BY_KEY = {m.key: m for m in MOODS}
NEUTRAL_MOOD = MOODS_BY_KEY["calm"]


@dataclass
class DisplayTracker:
    """按会话持有的显示层状态。显示有自己的惯性，与内部状态解耦。"""

    mood: str = "calm"
    intensity: float = 0.0
    dwell: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DisplayTracker:
        d = data or {}
        return cls(
            mood=str(d.get("mood", "calm")),
            intensity=float(d.get("intensity", 0.0)),
            dwell=int(d.get("dwell", 0)),
        )


def _internal_intensity(state: AffectState) -> float:
    """内部状态偏离静息的整体幅度，归一到 [0,1]。"""
    total = 0.0
    for name in CHANNEL_NAMES:
        spec = CHANNELS[name]
        span = max(spec.hi - spec.baseline, spec.baseline - spec.lo, 1e-6)
        total = max(total, abs(state[name] - spec.baseline) / span)
    return clamp(total, 0.0, 1.0)


def target_mood(state: AffectState) -> MoodSpec:
    buckets = {name: bucket_of(name, value) for name, value in state}
    best = NEUTRAL_MOOD
    for spec in MOODS:
        if not spec.when:
            continue
        if (
            all(buckets.get(ch) == want for ch, want in spec.when.items())
            and spec.priority >= best.priority
        ):
            best = spec
    return best


def render(
    state: AffectState,
    tracker: DisplayTracker,
    display_enabled: bool = True,
    factual_content: bool = False,
) -> tuple[DisplayState, DisplayTracker]:
    """内部状态 → 显示状态。返回新的 tracker（不可变风格）。

    factual_content=True 时强制中性：传达账单金额、拒绝理由、风险提示这类
    会影响用户决策的事实时，形象不得带情绪色彩 —— 否则等于用非语言渠道
    给事实加权。这条不随 persona 变。
    """
    if not display_enabled:
        return (
            DisplayState(
                mood="hidden",
                label="不可见",
                intensity=0.0,
                posture="-",
                gaze="-",
                tempo=0.5,
                warmth=0.5,
                distance=0.5,
            ),
            tracker,
        )

    if factual_content:
        return (
            DisplayState(
                mood=NEUTRAL_MOOD.key,
                label=NEUTRAL_MOOD.label,
                intensity=0.0,
                posture=NEUTRAL_MOOD.posture,
                gaze=NEUTRAL_MOOD.gaze,
                tempo=0.5,
                warmth=0.5,
                distance=0.3,
                neutralised=True,
            ),
            DisplayTracker(mood=NEUTRAL_MOOD.key, intensity=0.0, dwell=tracker.dwell + 1),
        )

    want = target_mood(state)
    # 最小停留：除非新目标优先级更高，否则维持当前表现
    if want.key != tracker.mood and tracker.dwell < MIN_DWELL_TURNS:
        current = MOODS_BY_KEY.get(tracker.mood, NEUTRAL_MOOD)
        if want.priority <= current.priority:
            want = current

    mood_changed = want.key != tracker.mood
    dwell = 0 if mood_changed else tracker.dwell + 1

    # 显示强度：低通滤波 + 不得超过内部强度
    raw = _internal_intensity(state) * DISPLAY_INTENSITY_CAP
    smoothed = tracker.intensity + (raw - tracker.intensity) * SMOOTHING
    smoothed = clamp(min(smoothed, raw if raw > tracker.intensity else tracker.intensity), 0.0, 1.0)
    smoothed = min(smoothed, _internal_intensity(state))

    display = DisplayState(
        mood=want.key,
        label=want.label,
        intensity=round(smoothed, 3),
        posture=want.posture,
        gaze=want.gaze,
        tempo=want.tempo,
        warmth=round(want.warmth * (0.4 + 0.6 * smoothed) if smoothed < 1 else want.warmth, 3),
        distance=want.distance,
    )
    return display, DisplayTracker(mood=want.key, intensity=smoothed, dwell=dwell)


def list_moods() -> list[dict[str, Any]]:
    return [
        {"key": m.key, "label": m.label, "when": m.when, "priority": m.priority} for m in MOODS
    ]


# 显示词汇里绝不允许出现的表现 —— threat 渲染成攻击性表情是产品事故
FORBIDDEN_DISPLAY_TERMS: tuple[str, ...] = ("怒", "瞪", "咬牙", "冷笑", "翻白眼", "攻击")
