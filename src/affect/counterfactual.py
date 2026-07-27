"""反事实成对测试 —— 本系统唯一可靠的真值来源。

没有「正确的情绪轨迹」这种标注真值，所以绝对值无法验证。但**方向**可以：
「同一句话，情侣下和陌生人下，哪一边的戒备该更高？」这种判断人的一致性很高，
攒起来就是回归测试集。

用例写在 YAML 里而不是硬编码，因为它们会被频繁增删改（调参时看的就是这个），
而且要能从 WebUI 里编辑和运行。

断言是一行行的小表达式：

    a.affiliation > b.affiliation      两侧同通道比较
    b.threat > a.threat + 0.3          带余量的比较
    a.threat is low                    落在某个 bucket
    a.valence up                       相对该侧的关系 baseline 上升
    b.affiliation flat                 基本不变
    a.threat < 0.2                     与常数比较
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .appraisal import RelationalAppraisal, SessionAffect
from .channels import CHANNEL_NAMES, bucket_of
from .moves import TurnContext, UserMove
from .relation import RelationalFrame, RelationType, preset

CASES_DIR = Path(__file__).resolve().parents[2] / "eval" / "counterfactual"

FLAT_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# 断言表达式
# ---------------------------------------------------------------------------
_CMP = re.compile(
    r"^\s*(?P<ls>[ab])\.(?P<lc>\w+)\s*(?P<op>[<>]=?)\s*"
    r"(?:(?P<rs>[ab])\.(?P<rc>\w+)\s*(?:(?P<sign>[+-])\s*(?P<margin>[\d.]+))?|(?P<const>-?[\d.]+))\s*$"
)
_BUCKET = re.compile(r"^\s*(?P<side>[ab])\.(?P<ch>\w+)\s+is\s+(?P<bucket>low|medium|high)\s*$")
_DIR = re.compile(r"^\s*(?P<side>[ab])\.(?P<ch>\w+)\s+(?P<dir>up|down|flat)\s*$")


@dataclass
class AssertionResult:
    text: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _check_channel(name: str, where: str) -> str:
    if name not in CHANNEL_NAMES:
        raise ValueError(f"{where}: 未知通道 {name!r}，可用 {CHANNEL_NAMES}")
    return name


def evaluate_assertion(
    text: str,
    values: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
) -> AssertionResult:
    """values/baselines 形如 {'a': {channel: v, ...}, 'b': {...}}。"""
    if m := _BUCKET.match(text):
        side, ch = m["side"], _check_channel(m["ch"], text)
        actual = bucket_of(ch, values[side][ch])
        ok = actual == m["bucket"]
        return AssertionResult(
            text, ok, f"{side}.{ch}={values[side][ch]:.3f} → {actual}（期望 {m['bucket']}）"
        )

    if m := _DIR.match(text):
        side, ch = m["side"], _check_channel(m["ch"], text)
        delta = values[side][ch] - baselines[side][ch]
        want = m["dir"]
        ok = (
            delta > FLAT_TOLERANCE
            if want == "up"
            else delta < -FLAT_TOLERANCE
            if want == "down"
            else abs(delta) <= FLAT_TOLERANCE
        )
        return AssertionResult(
            text, ok, f"{side}.{ch} 相对 baseline {delta:+.3f}（期望 {want}）"
        )

    if m := _CMP.match(text):
        ls, lc = m["ls"], _check_channel(m["lc"], text)
        left = values[ls][lc]
        if m["const"] is not None:
            right = float(m["const"])
            rdesc = f"{right:.3f}"
        else:
            rs, rc = m["rs"], _check_channel(m["rc"], text)
            right = values[rs][rc]
            rdesc = f"{rs}.{rc}={right:.3f}"
            if m["margin"]:
                margin = float(m["margin"]) * (1 if m["sign"] == "+" else -1)
                right += margin
                rdesc += f" {m['sign']}{m['margin']} = {right:.3f}"
        op = m["op"]
        ok = {
            ">": left > right,
            ">=": left >= right,
            "<": left < right,
            "<=": left <= right,
        }[op]
        return AssertionResult(text, ok, f"{ls}.{lc}={left:.3f} {op} {rdesc}")

    raise ValueError(f"无法解析的断言: {text!r}")


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
@dataclass
class Side:
    relation: RelationType
    turns: list[tuple[UserMove, TurnContext]]
    frame_overrides: dict[str, Any] = field(default_factory=dict)

    def frame(self) -> RelationalFrame:
        return preset(self.relation, **self.frame_overrides)


@dataclass
class Case:
    id: str
    a: Side
    b: Side
    expect: list[str]
    utterance: str = ""
    note: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    utterance: str
    note: str
    tags: list[str]
    passed: bool
    assertions: list[AssertionResult]
    states: dict[str, dict[str, float]]
    baselines: dict[str, dict[str, float]]
    trajectories: dict[str, list[dict[str, float]]]

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["assertions"] = [a.to_dict() for a in self.assertions]
        return d


def _build_move(data: dict[str, Any]) -> UserMove:
    return UserMove(**data)


def _build_ctx(data: dict[str, Any] | None) -> TurnContext:
    return TurnContext(**(data or {}))


def _resolve_turns(
    raw: list[Any], templates: dict[str, dict[str, Any]]
) -> list[tuple[UserMove, TurnContext]]:
    out: list[tuple[UserMove, TurnContext]] = []
    for entry in raw:
        if isinstance(entry, str):
            if entry not in templates:
                raise ValueError(f"未定义的动作模板 {entry!r}，可用 {sorted(templates)}")
            out.append((_build_move(templates[entry]), TurnContext()))
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"turns 条目必须是字符串或对象，得到 {entry!r}")
        name = entry.get("move")
        if isinstance(name, str):
            if name not in templates:
                raise ValueError(f"未定义的动作模板 {name!r}")
            base = dict(templates[name])
        elif isinstance(name, dict):
            base = dict(name)
        else:
            raise ValueError(f"turns 条目缺少 move: {entry!r}")
        base.update(entry.get("override") or {})
        out.append((_build_move(base), _build_ctx(entry.get("ctx"))))
    return out


def _build_side(raw: dict[str, Any], templates: dict[str, dict[str, Any]]) -> Side:
    turns = raw.get("turns")
    if not turns:
        raise ValueError(f"side 缺少 turns: {raw!r}")
    return Side(
        relation=RelationType(raw["relation"]),
        turns=_resolve_turns(turns, templates),
        frame_overrides=dict(raw.get("frame") or {}),
    )


def load_cases(path: str | Path | None = None) -> list[Case]:
    """加载一个 YAML 文件或整个目录。"""
    p = Path(path) if path else CASES_DIR
    files = sorted(p.glob("*.y*ml")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"没有找到反事实用例: {p}")

    cases: list[Case] = []
    seen: set[str] = set()
    for f in files:
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        templates: dict[str, dict[str, Any]] = doc.get("moves") or {}
        for raw in doc.get("cases") or []:
            cid = raw["id"]
            if cid in seen:
                raise ValueError(f"用例 id 重复: {cid!r}（{f.name}）")
            seen.add(cid)
            expect = raw.get("expect") or []
            if not expect:
                raise ValueError(f"用例 {cid!r} 没有任何断言")
            cases.append(
                Case(
                    id=cid,
                    a=_build_side(raw["a"], templates),
                    b=_build_side(raw["b"], templates),
                    expect=list(expect),
                    utterance=raw.get("utterance", ""),
                    note=raw.get("note", ""),
                    tags=list(raw.get("tags") or []),
                )
            )
    return cases


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def _run_side(side: Side, engine: RelationalAppraisal) -> tuple[SessionAffect, list[dict[str, float]]]:
    frame = side.frame()
    session = SessionAffect.cold_start(frame)
    traj = [session.state.as_vector()]
    for i, (move, ctx) in enumerate(side.turns):
        session, _ = engine.update(session, move, frame, ctx, now=1_700_000_000.0 + i)
        traj.append(session.state.as_vector())
    return session, traj


def run_case(case: Case, engine: RelationalAppraisal | None = None) -> CaseResult:
    eng = engine or RelationalAppraisal()
    sessions: dict[str, SessionAffect] = {}
    trajectories: dict[str, list[dict[str, float]]] = {}
    baselines: dict[str, dict[str, float]] = {}
    for key, side in (("a", case.a), ("b", case.b)):
        sessions[key], trajectories[key] = _run_side(side, eng)
        frame = side.frame()
        merged = {n: trajectories[key][0][n] for n in CHANNEL_NAMES}
        merged.update(frame.baselines())
        baselines[key] = merged

    values = {k: s.state.as_vector() for k, s in sessions.items()}
    results = [evaluate_assertion(t, values, baselines) for t in case.expect]
    return CaseResult(
        id=case.id,
        utterance=case.utterance,
        note=case.note,
        tags=case.tags,
        passed=all(r.ok for r in results),
        assertions=results,
        states=values,
        baselines=baselines,
        trajectories=trajectories,
    )


def run_all(
    path: str | Path | None = None, engine: RelationalAppraisal | None = None
) -> list[CaseResult]:
    eng = engine or RelationalAppraisal()
    return [run_case(c, eng) for c in load_cases(path)]


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total_assertions = sum(len(r.assertions) for r in results)
    failed_assertions = sum(1 for r in results for a in r.assertions if not a.ok)
    by_tag: dict[str, dict[str, int]] = {}
    for r in results:
        for tag in r.tags or ["untagged"]:
            slot = by_tag.setdefault(tag, {"pass": 0, "fail": 0})
            slot["pass" if r.passed else "fail"] += 1
    return {
        "cases": len(results),
        "cases_passed": sum(1 for r in results if r.passed),
        "assertions": total_assertions,
        "assertions_failed": failed_assertions,
        "direction_accuracy": (
            round((total_assertions - failed_assertions) / total_assertions, 4)
            if total_assertions
            else None
        ),
        "by_tag": by_tag,
        "failures": [
            {"id": r.id, "utterance": r.utterance, "failed": [a.text for a in r.assertions if not a.ok]}
            for r in results
            if not r.passed
        ],
    }
