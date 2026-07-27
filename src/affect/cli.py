"""多轮对话 CLI demo：实时打印 agent 内部状态（§7 Phase 2）。

    python -m affect.cli --persona warm_companion
    python -m affect.cli --persona steady_medical --model-dir artifacts/l1_onnx

对话中可用的元命令：
    :state              打印当前状态
    :event k=v ...      设置下一轮的 ConversationEvent（如 :event task_failed=1）
    :prompt             打印上一轮完整的 L3 prompt
    :persona <name>     切换 persona（会重置状态）
    :reset              重置本 session 状态
    :quit
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .persona import list_personas
from .pipeline import AffectPipeline
from .types import ConversationEvent

BAR_WIDTH = 24


def _bar(value: float, lo: float = 0.0, hi: float = 1.0) -> str:
    frac = (value - lo) / (hi - lo)
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    filled = round(frac * BAR_WIDTH)
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def render_state(result: Any) -> str:
    s = result.agent_affect
    u = result.user_affect
    lines = [
        "  ── L1 用户感知 ─────────────────────────────────",
        f"  strategy={u.strategy:<12} intensity={u.intensity:.2f}  confidence={u.confidence:.2f}",
        f"  user valence={u.valence:+.2f}  arousal={u.arousal:.2f}",
        "  ── L2 agent 状态 ───────────────────────────────",
        f"  valence   {_bar(s.valence, -1, 1)} {s.valence:+.2f}",
        f"  arousal   {_bar(s.arousal)} {s.arousal:.2f}",
        f"  dominance {_bar(s.dominance)} {s.dominance:.2f}",
        f"  concern   {_bar(s.concern)} {s.concern:.2f}",
        f"  rules: {', '.join(result.trace.matched_rules) or '(none)'}",
        f"  bucket: {result.agent_affect.to_bucket()}",
        "  ── L3 表达 ─────────────────────────────────────",
        f"  bypass={result.bypass.value}  hits={', '.join(result.matched_expression) or '(none)'}",
        f"  gen={result.generation_params}",
    ]
    return "\n".join(lines)


def _parse_event(tokens: list[str]) -> ConversationEvent:
    kwargs: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "latency_ms" or k == "turn_count":
            kwargs[k] = int(v)
        else:
            kwargs[k] = v.lower() in {"1", "true", "yes", "y"}
    return ConversationEvent(**kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="affect system 多轮 demo")
    ap.add_argument("--persona", default="steady_medical", choices=list_personas())
    ap.add_argument("--session", default="demo-session")
    ap.add_argument("--model-dir", default=None, help="L1 ONNX 模型目录；省略则用规则桩")
    ap.add_argument("--no-trace-log", action="store_true")
    args = ap.parse_args(argv)

    pipeline = AffectPipeline(model_dir=args.model_dir)
    if args.no_trace_log:
        pipeline.tracer.enabled = False

    persona_name = args.persona
    kind = "ONNX" if args.model_dir else "规则桩 (HeuristicPerceiver)"
    print(f"persona={persona_name}  L1={kind}  session={args.session}")
    print("输入 :quit 退出，:state 查看状态，:event task_failed=1 设置事件\n")

    last_reply = ""
    pending_event = ConversationEvent()
    turn = 0
    last_result = None

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
            if cmd == "state":
                st = pipeline.get_state(args.session)
                print(f"  {st}")
                continue
            if cmd == "event":
                pending_event = _parse_event(rest)
                print(f"  下一轮 event = {pending_event}")
                continue
            if cmd == "prompt":
                if last_result:
                    print("-" * 60)
                    print(last_result.affect_prompt)
                    print("-" * 60)
                continue
            if cmd == "persona":
                if rest and rest[0] in list_personas():
                    persona_name = rest[0]
                    pipeline.reset_session(args.session)
                    turn = 0
                    print(f"  切换到 {persona_name}，状态已重置")
                else:
                    print(f"  可用 persona: {list_personas()}")
                continue
            if cmd == "reset":
                pipeline.reset_session(args.session)
                turn = 0
                print("  状态已重置")
                continue
            print(f"  未知命令 :{cmd}")
            continue

        turn += 1
        pending_event.turn_count = turn
        result = pipeline.process_turn(
            session_id=args.session,
            user_utterance=line,
            last_agent_reply=last_reply,
            event=pending_event,
            persona_name=persona_name,
        )
        last_result = result
        pending_event = ConversationEvent()
        print(render_state(result))
        print("  （:prompt 查看完整 L3 指令）\n")
        # demo 没有接主 LLM；把上一轮用户输入当作 last_agent_reply 的占位显然不对，
        # 因此这里留空，真实接入时由业务层回填 agent 的实际回复。
        last_reply = ""

    return 0


if __name__ == "__main__":
    sys.exit(main())
