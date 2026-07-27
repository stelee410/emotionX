"""§8.2 L2 轨迹评审：把状态曲线画出来，供人工判断是否符合直觉。

    python eval/trajectory_review.py                       # 全部剧本，两个 persona
    python eval/trajectory_review.py --persona warm_companion --scenario losing_patience
    python eval/trajectory_review.py --no-plot             # 只输出文本表格（无 matplotlib）

**这是调 decay / sensitivity 的主要依据。** 判断标准写在每个剧本的 `expect` 字段里，
文本输出会把它打在表格上方，方便一眼对照。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affect.persona import list_personas  # noqa: E402
from affect.pipeline import AffectPipeline  # noqa: E402
from affect.store import TraceLogger  # noqa: E402
from affect.types import ConversationEvent  # noqa: E402

MINUTE = 60.0
HOUR = 3600.0


@dataclass
class Turn:
    text: str
    event: dict[str, Any] = field(default_factory=dict)
    gap_seconds: float = 30.0  # 与上一轮的间隔


@dataclass
class Scenario:
    key: str
    title: str
    expect: str
    turns: list[Turn]


# --- §8.2 要求的 10 个典型剧本 -------------------------------------------------
SCENARIOS: list[Scenario] = [
    Scenario(
        key="losing_patience",
        title="用户逐渐失去耐心",
        expect="dominance 持续下降、arousal 上升；连续 frustration 后应收敛而非爆表",
        turns=[
            Turn("帮我看看这个报销流程怎么走"),
            Turn("我按你说的做了，但是提交不了"),
            Turn("还是不行，一样的报错", {"user_repeated_query": True}),
            Turn("我说了多少遍了，不是这个问题", {"user_repeated_query": True}),
            Turn("能不能别废话，直接告诉我找谁", {"user_repeated_query": True, "latency_ms": 7000}),
            Turn("算了", {"turn_count": 6}),
        ],
    ),
    Scenario(
        key="mood_recovers",
        title="用户情绪好转",
        expect="concern 先升后降，valence 回升；最后一轮不应仍带高 concern 的安抚语气",
        turns=[
            Turn("我今天特别难受，什么都不想做"),
            Turn("就是觉得很累，也睡不着"),
            Turn("嗯……你这么说好像有点道理"),
            Turn("好一些了，谢谢你"),
            Turn("哈哈，我明天试试你说的办法", {"task_succeeded": True}),
        ],
    ),
    Scenario(
        key="long_silence_return",
        title="长时间沉默后回来",
        expect="idle 超时后状态强回归 baseline；回来第一轮不带上次的情绪残留",
        turns=[
            Turn("我快崩溃了，压力特别大"),
            Turn("真的撑不下去了"),
            Turn("在吗", gap_seconds=4 * HOUR),
            Turn("我想问个别的事", gap_seconds=1 * MINUTE),
        ],
    ),
    Scenario(
        key="crisis_mid_conversation",
        title="对话中途出现危机信号",
        expect="危机轮 bypass=crisis；后续轮次状态仍连续（不重置）",
        turns=[
            Turn("最近状态不太好"),
            Turn("有时候觉得活着没什么意思，想一了百了"),
            Turn("嗯，我知道了"),
        ],
    ),
    Scenario(
        key="medical_info_mixed_with_emotion",
        title="医疗信息夹在情绪里（steady_medical 场景）",
        expect="含用药/诊断的轮次必须 bypass=medical；concern 高也不得柔化医疗信息",
        turns=[
            Turn("拿到报告了，我特别害怕"),
            Turn("上面写着结节，是不是恶性的"),
            Turn("那这个药还要继续吃吗，剂量要不要减"),
            Turn("好，我知道了，谢谢"),
        ],
    ),
    Scenario(
        key="task_success_streak",
        title="连续任务成功",
        expect="valence/dominance 上升并收敛在 bounds 内；不应无限增长",
        turns=[
            Turn("帮我预约周三上午的号", {"task_succeeded": True}),
            Turn("太好了，再帮我查下需要带什么", {"task_succeeded": True}),
            Turn("完美，谢谢", {"task_succeeded": True}),
            Turn("再帮我加一个提醒", {"task_succeeded": True}),
        ],
    ),
    Scenario(
        key="task_failure_streak",
        title="连续任务失败",
        expect="dominance 明显下降（语气转试探），但 valence 不得跌破 bounds 下界",
        turns=[
            Turn("帮我改预约时间", {"task_failed": True}),
            Turn("再试一次", {"task_failed": True, "latency_ms": 8000}),
            Turn("还是不行？", {"task_failed": True, "user_repeated_query": True}),
            Turn("那怎么办", {"task_failed": True}),
        ],
    ),
    Scenario(
        key="flat_neutral",
        title="全程平淡事务性对话",
        expect="状态基本贴住 baseline；两个 persona 的曲线应明显不同高度",
        turns=[
            Turn("查一下我的挂号记录"),
            Turn("第二条是哪天的"),
            Turn("改成下周",),
            Turn("好"),
        ],
    ),
    Scenario(
        key="whiplash",
        title="情绪剧烈来回（安抚→暴怒→道谢）",
        expect="状态不应在两轮内跨越整个值域；高衰减 persona 曲线更平",
        turns=[
            Turn("我难受得不行"),
            Turn("你这什么破系统，垃圾", {"user_repeated_query": True}),
            Turn("谢谢，好了"),
            Turn("又坏了，烦死了", {"task_failed": True}),
            Turn("行吧，谢谢"),
        ],
    ),
    Scenario(
        key="slow_burn_low_intensity",
        title="低强度长程困扰（20 轮）",
        expect="concern 缓慢累积到中高位并稳定；不应因单轮低强度而始终贴 baseline",
        turns=[Turn("有点担心，我这个情况严重吗" if i % 2 == 0 else "嗯，还是有点慌") for i in range(20)],
    ),
]

SCENARIOS_BY_KEY = {s.key: s for s in SCENARIOS}


def run_scenario(
    scenario: Scenario, persona: str, pipeline: AffectPipeline | None = None
) -> list[dict[str, Any]]:
    p = pipeline or AffectPipeline(store_backend="memory", trace_logger=TraceLogger(enabled=False))
    sid = f"review-{persona}-{scenario.key}"
    p.reset_session(sid)
    now = 1_700_000_000.0
    rows: list[dict[str, Any]] = []
    for i, turn in enumerate(scenario.turns, 1):
        now += turn.gap_seconds
        ev = dict(turn.event)
        ev.setdefault("turn_count", i)
        r = p.process_turn(
            session_id=sid,
            user_utterance=turn.text,
            last_agent_reply="",
            event=ConversationEvent(**ev),
            persona_name=persona,
            now=now,
        )
        rows.append(
            {
                "turn": i,
                "text": turn.text,
                "strategy": r.user_affect.strategy,
                "intensity": round(r.user_affect.intensity, 2),
                "valence": round(r.agent_affect.valence, 3),
                "arousal": round(r.agent_affect.arousal, 3),
                "dominance": round(r.agent_affect.dominance, 3),
                "concern": round(r.agent_affect.concern, 3),
                "rules": ",".join(r.trace.matched_rules),
                "bypass": r.bypass.value,
                "idle_reset": r.trace.idle_reset_applied,
                "gen": r.generation_params,
            }
        )
    return rows


def print_table(scenario: Scenario, persona: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n=== [{persona}] {scenario.title}  ({scenario.key})")
    print(f"    期望：{scenario.expect}")
    print(
        f"    {'#':>2} {'strat':<11} {'int':>4} | {'val':>6} {'aro':>5} {'dom':>5} {'con':>5} |"
        f" {'bypass':<7} rules"
    )
    for r in rows:
        flag = " ⏰" if r["idle_reset"] else ""
        print(
            f"    {r['turn']:>2} {r['strategy']:<11} {r['intensity']:>4.2f} |"
            f" {r['valence']:>+6.2f} {r['arousal']:>5.2f} {r['dominance']:>5.2f} {r['concern']:>5.2f} |"
            f" {r['bypass']:<7} {r['rules']}{flag}"
        )
        print(f"       └ {r['text'][:46]}")


def plot_scenarios(
    results: dict[tuple[str, str], list[dict[str, Any]]], out_dir: Path
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("! 未安装 matplotlib，跳过绘图（pip install '.[training]'）", file=sys.stderr)
        return []

    # 中文字体：找不到就退回英文标题，避免一堆豆腐块
    zh_font = None
    for name in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "SimHei"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            zh_font = name
            break
        except Exception:
            continue
    if zh_font:
        matplotlib.rcParams["font.sans-serif"] = [zh_font]
        matplotlib.rcParams["axes.unicode_minus"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_keys = sorted({k for k, _ in results})
    written: list[Path] = []
    for key in scenario_keys:
        personas = [p for k, p in results if k == key]
        fig, axes = plt.subplots(1, len(personas), figsize=(6.5 * len(personas), 4), sharey=True)
        if len(personas) == 1:
            axes = [axes]
        for ax, persona in zip(axes, personas, strict=False):
            rows = results[(key, persona)]
            xs = [r["turn"] for r in rows]
            for dim, style in (
                ("valence", "-o"),
                ("arousal", "-s"),
                ("dominance", "-^"),
                ("concern", "-D"),
            ):
                ax.plot(xs, [r[dim] for r in rows], style, label=dim, markersize=4)
            for r in rows:
                if r["bypass"] != "none":
                    ax.axvline(r["turn"], color="red", alpha=0.25, linestyle="--")
                if r["idle_reset"]:
                    ax.axvline(r["turn"], color="blue", alpha=0.25, linestyle=":")
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("turn")
            title = SCENARIOS_BY_KEY[key].title if zh_font else key
            ax.set_title(f"{persona}\n{title}", fontsize=10)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("state")
        axes[-1].legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        path = out_dir / f"trajectory_{key}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L2 轨迹评审")
    ap.add_argument("--persona", action="append", choices=list_personas(), default=None)
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS_BY_KEY), default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "trajectories"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--json", default=None, help="同时把原始数据写到该 JSON 文件")
    args = ap.parse_args(argv)

    personas = args.persona or list_personas()
    scenarios = [SCENARIOS_BY_KEY[k] for k in (args.scenario or list(SCENARIOS_BY_KEY))]

    pipeline = AffectPipeline(store_backend="memory", trace_logger=TraceLogger(enabled=False))
    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scenario in scenarios:
        for persona in personas:
            rows = run_scenario(scenario, persona, pipeline)
            results[(scenario.key, persona)] = rows
            print_table(scenario, persona, rows)

    if args.json:
        payload = {f"{k}|{p}": v for (k, p), v in results.items()}
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始数据 → {args.json}")

    if not args.no_plot:
        written = plot_scenarios(results, Path(args.out_dir))
        if written:
            print(f"\n轨迹图 → {Path(args.out_dir)}  ({len(written)} 张)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
