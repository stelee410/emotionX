"""多轮对话 CLI —— 实时打印内部状态、动作清单与显示状态。

    python -m affect.cli --relation partner --persona warm
    python -m affect.cli --relation stranger --display

元命令：
    :rel <name>      切换关系（重建会话 —— 关系不可在会话内改变）
    :persona <name>  切换人格
    :state           打印当前 6 通道
    :prompt          打印完整 L3 prompt
    :move            打印上一轮 L1 感知到的 UserMove
    :mem <text>      注入一条记忆
    :event k=v ...   设置下一轮的 TurnContext
    :reset  :quit
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .channels import CHANNEL_NAMES, CHANNELS
from .memory import ManualMemory
from .moves import TurnContext
from .persona import list_personas
from .pipeline import AffectPipeline
from .relation import list_relation_types, preset

BAR = 22


def _bar(value: float, lo: float, hi: float) -> str:
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = round(frac * BAR)
    return "█" * filled + "·" * (BAR - filled)


def render_state(result: Any) -> str:
    m, s = result.user_move, result.state
    lines = [
        "  ── L1 感知（关系无关）─────────────────────────",
        f"  亲和{m.affiliation_bid:+.2f}  支配{m.dominance_bid:+.2f}  亲密{m.intimacy_bid:.2f}  "
        f"痛苦{m.distress_level:.2f}  强度{m.intensity:.2f}  置信{m.confidence:.2f}"
        + ("" if m.directed_at_agent else "  [指向第三方]"),
        f"  ── L2 状态  失配{result.trace.mismatch:+.2f} "
        f"(亲近门{result.trace.warmth_gate:.2f}/越界门{result.trace.breach_gate:.2f}) ──",
    ]
    for name in CHANNEL_NAMES:
        spec = CHANNELS[name]
        lines.append(f"  {name:<11} {_bar(s[name], spec.lo, spec.hi)} {s[name]:+.2f}")
    lines += [
        f"  机制: {', '.join(result.trace.fired) or '(none)'}",
        "  ── L3a 表达 ─────────────────────────────────",
        f"  动作: {', '.join(result.actions.labels) or '(none)'}"
        + (f"   [被压制: {', '.join(result.actions.suppressed)}]" if result.actions.suppressed else ""),
        f"  语气: {', '.join(result.prompt.tone_hits) or '(none)'}",
        f"  生成: {result.prompt.generation}",
        "  ── L3b 显示 ─────────────────────────────────",
        f"  {result.display.label}({result.display.mood})  强度{result.display.intensity:.2f}  "
        f"姿态[{result.display.posture}] 视线[{result.display.gaze}] "
        f"温度{result.display.warmth:.2f} 距离{result.display.distance:.2f}"
        + ("  [事实性内容→强制中性]" if result.display.neutralised else ""),
    ]
    if result.safety.crisis:
        lines.append(f"  ⚠ 危机 bypass：{result.safety.crisis_matches}")
    if result.memory_notes:
        lines.append(f"  记忆: {list(result.memory_notes)}")
    return "\n".join(lines)


def _parse_event(tokens: list[str]) -> TurnContext:
    kwargs: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in {"latency_ms", "turn_count"}:
            kwargs[k] = int(v)
        else:
            kwargs[k] = v.lower() in {"1", "true", "yes", "y"}
    return TurnContext(**kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="emotionX 交互式测试台")
    ap.add_argument("--relation", default="friend", choices=list_relation_types())
    ap.add_argument("--persona", default="warm", choices=list_personas())
    ap.add_argument("--session", default="cli")
    ap.add_argument("--display", action="store_true", help="启用可见状态")
    ap.add_argument("--age-verified", action="store_true")
    args = ap.parse_args(argv)

    memory = ManualMemory()
    pipeline = AffectPipeline(memory=memory)
    pipeline.tracer.enabled = False

    relation, persona = args.relation, args.persona

    def start() -> None:
        pipeline.open_session(
            args.session,
            frame=preset(relation, display_enabled=args.display),
            persona=persona,
            age_verified=args.age_verified or relation != "partner",
        )

    try:
        start()
    except Exception as exc:  # noqa: BLE001
        print(f"无法建立会话：{exc}")
        return 1

    print(f"关系={relation}  人格={persona}  显示={'开' if args.display else '关'}")
    print("元命令：:rel :persona :state :prompt :move :mem :event :reset :quit\n")

    pending = TurnContext()
    last = None
    turn = 0

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith(":"):
            parts = line[1:].split()
            cmd, rest = (parts[0] if parts else ""), parts[1:]
            if cmd in {"quit", "q", "exit"}:
                return 0
            if cmd == "rel" and rest:
                if rest[0] not in list_relation_types():
                    print(f"  可用关系: {list_relation_types()}")
                    continue
                relation = rest[0]
                try:
                    start()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {exc}")
                    continue
                turn = 0
                print(f"  关系 → {relation}（会话已重建：关系不可在会话内改变）")
                continue
            if cmd == "persona" and rest:
                if rest[0] not in list_personas():
                    print(f"  可用人格: {list_personas()}")
                    continue
                persona = rest[0]
                start()
                turn = 0
                print(f"  人格 → {persona}")
                continue
            if cmd == "state":
                rec = pipeline.session(args.session)
                if rec:
                    print(f"  {rec.affect.state.as_vector()}")
                    print(f"  峰值亲密度 {rec.affect.peak_user_intimacy:.2f}  轮次 {rec.affect.turn}")
                continue
            if cmd == "prompt" and last:
                print("-" * 62)
                print(last.prompt.text)
                print("-" * 62)
                continue
            if cmd == "move" and last:
                print(f"  {last.user_move.to_dict()}")
                continue
            if cmd == "mem":
                memory.notes.append(" ".join(rest))
                print(f"  已注入 {len(memory.notes)} 条记忆")
                continue
            if cmd == "event":
                pending = _parse_event(rest)
                print(f"  下一轮 context = {pending}")
                continue
            if cmd == "reset":
                start()
                turn = 0
                print("  已重置")
                continue
            print(f"  未知命令 :{cmd}")
            continue

        turn += 1
        pending.turn_count = turn
        last = pipeline.process_turn(args.session, line, context=pending)
        pending = TurnContext()
        print(render_state(last))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
