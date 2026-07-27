"""emotionX 本地集成平台。

    python studio/server.py                       → http://127.0.0.1:8080
    python studio/server.py --import data/raw/seed_pool.jsonl

五个面板：
    对话测试台   关系/人格实时切换、6 通道曲线、动作与显示状态、记忆注入
    参数调校     改 appraisal 参数立刻看反事实套件的方向正确率有没有掉
    反事实       用例列表、逐条断言、失败详情
    标注         成对比较（主）+ 直接评分（锚点）、Bradley-Terry 还原尺度
    训练         触发 stage1/stage2/export，实时看日志与指标

只监听 127.0.0.1：真实会话数据不离开本机。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as _exc:  # pragma: no cover - 启动期的可读报错
    import sys as _sys

    _sys.exit(
        f"缺少依赖 {_exc.name!r}。平台需要 studio extra，且要用项目的虚拟环境启动：\n\n"
        '    uv pip install --python .venv/bin/python -e ".[dev,studio]"\n'
        "    .venv/bin/python studio/server.py\n\n"
        f"（当前解释器：{_sys.executable}）"
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from affect.actions import ACTIONS  # noqa: E402
from affect.appraisal import DEFAULT_PARAMS  # noqa: E402
from affect.channels import BUCKET_THRESHOLDS, CHANNEL_NAMES, CHANNELS  # noqa: E402
from affect.counterfactual import load_cases, run_case, summarize  # noqa: E402
from affect.display import list_moods  # noqa: E402
from affect.domains import SafetyDomainError  # noqa: E402
from affect.heuristic import HeuristicPerceiver  # noqa: E402
from affect.memory import HttpMemory, ManualMemory  # noqa: E402
from affect.moves import TurnContext, UserMove  # noqa: E402
from affect.persona import BUILTIN  # noqa: E402
from affect.pipeline import AffectPipeline  # noqa: E402
from affect.relation import PRESETS, RelationType, preset  # noqa: E402
from affect.safety import crisis_tier  # noqa: E402
from affect.targets import REGRESSION_TARGETS  # noqa: E402
from studio.db import COMPARABLE, PlatformDB  # noqa: E402
from studio.scale import build_scale  # noqa: E402

STATIC = Path(__file__).parent / "static"
EXPORTS = ROOT / "data" / "exports"

app = FastAPI(title="emotionX 平台", docs_url="/api/docs")

_db = PlatformDB()

SEED_POOL = ROOT / "data" / "raw" / "seed_pool.jsonl"


def _autoload_seed(db_obj: PlatformDB) -> None:
    """首次启动时把种子样例灌进去 —— 空库的标注面板没法用，也看不出该怎么用。"""
    if db_obj.stats()["items"] or not SEED_POOL.exists():
        return
    records = [
        json.loads(line)
        for line in SEED_POOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    r = db_obj.import_records(records, source="seed")
    print(f"首次启动：已载入 {r['added']} 条种子样例（source=seed，不会进评估集）")


_autoload_seed(_db)
_memory = ManualMemory()
_pipeline = AffectPipeline(memory=_memory)
_pipeline.tracer.enabled = False
_params = DEFAULT_PARAMS
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def db() -> PlatformDB:
    return _db


# =========================================================== 静态页面
@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """前端启动时拉一次：通道、关系、人格、动作、显示词汇。"""
    return {
        "channels": [
            {
                "name": n,
                "lo": CHANNELS[n].lo,
                "hi": CHANNELS[n].hi,
                "baseline": CHANNELS[n].baseline,
                "gain": CHANNELS[n].gain,
                "half_life": CHANNELS[n].half_life,
                "note": CHANNELS[n].note,
                "thresholds": BUCKET_THRESHOLDS[n],
            }
            for n in CHANNEL_NAMES
        ],
        "relations": [
            {
                "key": rt.value,
                "intimacy_permitted": PRESETS[rt]["intimacy_permitted"],
                "tolerance": PRESETS[rt]["tolerance"],
                "safety_profile": str(PRESETS[rt]["safety_profile"]),
                "description": PRESETS[rt].get("description", ""),
            }
            for rt in RelationType
        ],
        "personas": [{"key": k, "description": v.get("description", "")} for k, v in BUILTIN.items()],
        "actions": [{"key": a.key, "label": a.label, "when": a.when} for a in ACTIONS],
        "moods": list_moods(),
        "targets": list(REGRESSION_TARGETS),
        "comparable": list(COMPARABLE),
        "params": _params.to_dict(),
    }


# =========================================================== 对话测试台
class OpenPayload(BaseModel):
    session_id: str = "ui"
    relation: str = "friend"
    persona: str = "warm"
    display: bool = True
    age_verified: bool = True


class TurnPayload(BaseModel):
    session_id: str = "ui"
    utterance: str
    last_agent_reply: str = ""
    factual_content: bool = False
    context: dict[str, Any] = Field(default_factory=dict)
    # 手动覆盖 L1 输出，用于绕开感知桩直接测 L2/L3
    move_override: dict[str, float] | None = None


@app.post("/api/session/open")
def open_session(p: OpenPayload) -> dict[str, Any]:
    try:
        rec = _pipeline.open_session(
            p.session_id,
            frame=preset(p.relation, display_enabled=p.display),
            persona=p.persona,
            age_verified=p.age_verified,
        )
    except SafetyDomainError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, "session": rec.to_dict()}


@app.post("/api/session/turn")
def turn(p: TurnPayload) -> dict[str, Any]:
    if _pipeline.session(p.session_id) is None:
        raise HTTPException(409, "会话不存在，先调用 /api/session/open")
    move = None
    if p.move_override:
        try:
            move = UserMove(**p.move_override)
        except TypeError as exc:
            raise HTTPException(422, f"move_override 字段不对: {exc}") from exc
    try:
        ctx = TurnContext(**p.context)
    except TypeError as exc:
        raise HTTPException(422, f"context 字段不对: {exc}") from exc
    result = _pipeline.process_turn(
        p.session_id,
        p.utterance,
        p.last_agent_reply,
        context=ctx,
        move=move,
        factual_content=p.factual_content,
    )
    rec = _pipeline.session(p.session_id)
    out = result.to_dict()
    out["peak_user_intimacy"] = rec.affect.peak_user_intimacy if rec else 0.0
    out["crisis_tier"] = crisis_tier(p.utterance)
    return out


@app.post("/api/session/reset")
def reset_session(session_id: str = "ui") -> dict[str, Any]:
    rec = _pipeline.session(session_id)
    if rec is None:
        return {"ok": True}
    _pipeline.open_session(
        session_id, frame=rec.frame, persona=rec.persona_name, age_verified=True
    )
    return {"ok": True}


class MemoryPayload(BaseModel):
    notes: list[str] = Field(default_factory=list)
    http_url: str | None = None


@app.post("/api/memory")
def set_memory(p: MemoryPayload) -> dict[str, Any]:
    """记忆由外部系统提供。这里两条测试通路：手动注入 / 调外部 HTTP 服务。"""
    global _memory
    if p.http_url:
        _pipeline.memory = HttpMemory(url=p.http_url)
        return {"mode": "http", "url": p.http_url}
    _memory = ManualMemory(notes=list(p.notes))
    _pipeline.memory = _memory
    return {"mode": "manual", "count": len(p.notes)}


@app.get("/api/memory")
def get_memory() -> dict[str, Any]:
    mem = _pipeline.memory
    if isinstance(mem, HttpMemory):
        return {"mode": "http", "url": mem.url, "last_error": mem.last_error}
    notes = getattr(mem, "notes", [])
    return {"mode": "manual", "notes": list(notes)}


@app.post("/api/perceive")
def perceive(utterance: str) -> dict[str, Any]:
    """单独看感知桩的输出，便于判断问题出在 L1 还是 L2。"""
    return HeuristicPerceiver().explain(utterance)


# =========================================================== 参数调校
class ParamsPayload(BaseModel):
    params: dict[str, float]


@app.post("/api/params")
def set_params(p: ParamsPayload) -> dict[str, Any]:
    """改参数并立刻跑反事实套件 —— 方向正确率掉了就说明改坏了。"""
    global _params
    unknown = set(p.params) - set(DEFAULT_PARAMS.to_dict())
    if unknown:
        raise HTTPException(422, f"未知参数: {sorted(unknown)}")
    try:
        candidate = replace(_params, **p.params)
    except TypeError as exc:
        raise HTTPException(422, str(exc)) from exc

    from affect.appraisal import RelationalAppraisal

    engine = RelationalAppraisal(candidate)
    try:
        engine.assert_no_contagion()
        engine.assert_boundary_mechanism_intact()
    except ValueError as exc:
        raise HTTPException(422, f"参数违反硬约束：{exc}") from exc

    summary = summarize([run_case(c, engine) for c in load_cases()])
    _params = candidate
    _pipeline.engine = engine
    return {"ok": True, "params": _params.to_dict(), "counterfactual": summary}


@app.post("/api/params/reset")
def reset_params() -> dict[str, Any]:
    global _params
    from affect.appraisal import RelationalAppraisal

    _params = DEFAULT_PARAMS
    _pipeline.engine = RelationalAppraisal(_params)
    return {"ok": True, "params": _params.to_dict()}


# =========================================================== 反事实
@app.get("/api/counterfactual")
def counterfactual(tag: str | None = None) -> dict[str, Any]:
    from affect.appraisal import RelationalAppraisal

    engine = RelationalAppraisal(_params)
    cases = load_cases()
    if tag:
        cases = [c for c in cases if tag in c.tags]
    results = [run_case(c, engine) for c in cases]
    return {
        "summary": summarize(results),
        "results": [r.to_dict() for r in results],
    }


@app.get("/api/counterfactual/source")
def counterfactual_source() -> dict[str, Any]:
    from affect.counterfactual import CASES_DIR

    return {
        "files": [
            {"name": f.name, "content": f.read_text(encoding="utf-8")}
            for f in sorted(CASES_DIR.glob("*.y*ml"))
        ]
    }


# =========================================================== 标注
class ImportPayload(BaseModel):
    content: str
    format: str = "jsonl"
    source: str = "real_session"


@app.post("/api/annotate/import")
def import_items(p: ImportPayload) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if p.format == "lines":
        records = [{"utterance": line} for line in p.content.splitlines() if line.strip()]
    else:
        for line in p.content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    result = db().import_records(records, source=p.source)
    result["stats"] = db().stats()
    return result


@app.get("/api/annotate/pair")
def next_pair(dimension: str, annotator: str) -> dict[str, Any]:
    if dimension not in COMPARABLE:
        raise HTTPException(422, f"维度必须是 {list(COMPARABLE)} 之一")
    pair = db().next_pair(dimension, annotator)
    if pair is None:
        return {"pair": None, "stats": db().stats()}
    left, right = pair
    return {"pair": {"left": left, "right": right}, "stats": db().stats()}


class ComparePayload(BaseModel):
    dimension: str
    left_id: int
    right_id: int
    winner: str
    annotator: str
    elapsed_ms: int | None = None


@app.post("/api/annotate/compare")
def save_comparison(p: ComparePayload) -> dict[str, Any]:
    try:
        db().save_comparison(
            p.dimension, p.left_id, p.right_id, p.winner, p.annotator, p.elapsed_ms
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, "stats": db().stats()}


@app.post("/api/annotate/undo")
def undo_comparison(annotator: str) -> dict[str, Any]:
    cid = db().undo_comparison(annotator)
    if cid is None:
        raise HTTPException(404, "没有可撤销的比较")
    return {"ok": True, "id": cid, "stats": db().stats()}


@app.get("/api/annotate/rate")
def next_rating(annotator: str) -> dict[str, Any]:
    item = db().next_unrated(annotator)
    if item is None:
        return {"item": None, "stats": db().stats()}
    item["crisis_hint_tier"] = crisis_tier(item["utterance"])
    item["suggestion"] = {
        k: round(float(getattr(HeuristicPerceiver().perceive(item["utterance"]), k)), 2)
        for k in REGRESSION_TARGETS
    }
    return {"item": item, "stats": db().stats()}


class RatingPayload(BaseModel):
    item_id: int
    annotator: str
    values: dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


@app.post("/api/annotate/rate")
def save_rating(p: RatingPayload) -> dict[str, Any]:
    if db().get_item(p.item_id) is None:
        raise HTTPException(404, "item 不存在")
    try:
        db().save_rating(p.item_id, p.annotator, p.values, p.skipped)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, "stats": db().stats()}


@app.get("/api/annotate/scale")
def get_scale(dimension: str) -> dict[str, Any]:
    """Bradley-Terry 还原尺度。有直接评分的条目作为锚点对齐绝对值。"""
    if dimension not in COMPARABLE:
        raise HTTPException(422, f"维度必须是 {list(COMPARABLE)} 之一")
    comps = db().comparisons(dimension)
    anchors = {
        int(r["item_id"]): float(r[dimension])
        for r in db().ratings()
        if r.get(dimension) is not None
    }
    item_ids = [int(i["id"]) for i in db().items(limit=5000)]
    return build_scale(comps, dimension, anchors=anchors, item_ids=item_ids)


@app.post("/api/annotate/golden")
def select_golden(n: int = 200, seed: int = 42, force: bool = False) -> dict[str, Any]:
    try:
        return db().select_golden(n=n, seed=seed, force=force)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/annotate/export")
def export_annotations(split: str | None = None, filename: str | None = None) -> dict[str, Any]:
    """导出训练/评估数据。比较得到的尺度会覆盖直接评分（比较的一致性更高）。"""
    scales = {d: get_scale(d).get("values", {}) for d in COMPARABLE}
    rows: list[dict[str, Any]] = []
    for r in db().ratings():
        if split and r["split"] != split:
            continue
        item_id = str(r["item_id"])
        record: dict[str, Any] = {
            "item_id": r["item_id"],
            "utterance": r["utterance"],
            "last_agent_reply": r["last_agent_reply"],
            "source": r["source"],
            "split": r["split"],
            "directed_at_agent": bool(r["directed_at_agent"]),
            "crisis_flag": int(r["crisis_flag"]),
            "annotator": r["annotator"],
        }
        for target in REGRESSION_TARGETS:
            value = scales.get(target, {}).get(item_id)
            if value is None:
                value = r.get(target)
            if value is None:
                break
            record[target] = round(float(value), 4)
        else:
            record["scale_source"] = {
                t: ("comparison" if scales.get(t, {}).get(item_id) is not None else "rating")
                for t in REGRESSION_TARGETS
            }
            rows.append(record)

    EXPORTS.mkdir(parents=True, exist_ok=True)
    path = EXPORTS / (filename or f"annotations_{split or 'all'}.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"written": len(rows), "path": str(path)}


@app.get("/api/annotate/guideline", response_class=PlainTextResponse)
def guideline() -> str:
    p = ROOT / "docs" / "label_guideline.md"
    return p.read_text(encoding="utf-8") if p.exists() else "（缺少标注指南）"


# =========================================================== 数据集
DATASET_INFO: dict[str, dict[str, Any]] = {
    "ewect": {
        "label": "SMP2020-EWECT",
        "detail": "微博 6 类情感，训练集 27,766 条。HuggingFace 直接下载。",
        "auto": True,
        "recommended": True,
    },
    "simplifyweibo": {
        "label": "simplifyweibo_4_moods",
        "detail": "微博 4 类，下采样 4 万条。标签噪声较大 —— 实测加进来会让 EWECT 掉 0.7 个点。",
        "auto": True,
        "recommended": False,
    },
    "cped": {
        "label": "CPED",
        "detail": "对话数据，13 类细粒度情感。域最接近，但需手动申请：github.com/scutcyr/CPED",
        "auto": False,
        "recommended": True,
    },
    "m3ed": {
        "label": "M3ED",
        "detail": "对话数据 24,449 句 7 类。只用文本模态。需手动获取：github.com/AIM3-RUC/RUCM3ED",
        "auto": False,
        "recommended": True,
    },
    "ocemotion": {
        "label": "OCEMOTION",
        "detail": "约 3.5 万条 7 类。天池赛题，需手动下载后放到 data/raw/ocemotion/train.csv",
        "auto": False,
        "recommended": False,
    },
}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    from training.datasets.registry import RAW_DIR

    out = []
    for key, info in DATASET_INFO.items():
        d = RAW_DIR / key
        present = d.exists() and any(d.iterdir())
        out.append(
            {
                "key": key,
                **info,
                "present": present,
                "path": str(d),
                "status": "已就绪" if present else ("首次运行自动下载" if info["auto"] else "需手动获取"),
                "usable": present or info["auto"],
            }
        )
    exports = sorted(str(f.name) for f in EXPORTS.glob("*.jsonl")) if EXPORTS.exists() else []
    models = (
        sorted(d.name for d in (ROOT / "artifacts").iterdir() if d.is_dir())
        if (ROOT / "artifacts").exists()
        else []
    )
    return {"datasets": out, "annotation_exports": exports, "artifacts": models}


# =========================================================== 训练
def _build_command(kind: str, cfg: dict[str, Any]) -> list[str]:
    """把面板上的配置翻译成命令行。

    刻意不接受自由文本参数 —— 让人背命令行参数是这个面板存在的反面。
    """
    py = sys.executable
    if kind == "stage1":
        ds = cfg.get("datasets") or ["ewect"]
        cmd = [py, "training/stage1_pretrain.py", "--datasets", *ds,
               "--epochs", str(cfg.get("epochs", 3)),
               "--batch-size", str(cfg.get("batch_size", 64)),
               "--lr", str(cfg.get("lr", 5e-5))]
        if cfg.get("max_per_dataset"):
            cmd += ["--max-per-dataset", str(cfg["max_per_dataset"])]
        if cfg.get("out"):
            cmd += ["--out", str(cfg["out"])]
        return cmd

    if kind == "stage2":
        cmd = [py, "training/stage2_finetune.py",
               "--stage1", str(cfg.get("stage1", "artifacts/l1_stage1")),
               "--epochs", str(cfg.get("epochs", 3)),
               "--lr", str(cfg.get("lr", 1e-5))]
        source = cfg.get("source", "bootstrap")
        if source == "annotations":
            path = cfg.get("annotations")
            if not path:
                raise ValueError("选择了「人工标注」但没有指定文件 —— 先在标注面板导出")
            cmd += ["--annotations", str(EXPORTS / path if not str(path).startswith("/") else path)]
        else:
            cmd += ["--bootstrap", str(cfg.get("bootstrap", 6000))]
        if cfg.get("distilled"):
            cmd += ["--distilled", str(cfg["distilled"])]
        return cmd

    if kind == "export":
        cmd = [py, "training/export_onnx.py",
               "--model-dir", str(cfg.get("model_dir", "artifacts/l1_stage2")),
               "--out", str(cfg.get("out", "artifacts/l1_onnx")),
               "--bench", str(cfg.get("bench", 200))]
        if not cfg.get("quantize", True):
            cmd.append("--no-quantize")
        return cmd

    if kind == "eval":
        model = cfg.get("model")
        if model and model != "heuristic":
            return [py, "eval/test_perception.py", "--model", str(ROOT / "artifacts" / model)]
        return [py, "eval/test_perception.py", "--heuristic"]

    if kind == "counterfactual":
        cmd = [py, "eval/run_counterfactual.py"]
        if cfg.get("verbose"):
            cmd.append("-v")
        if cfg.get("tag"):
            cmd += ["--tag", str(cfg["tag"])]
        return cmd

    raise ValueError(f"未知任务 {kind}")


JOB_KINDS = ("stage1", "stage2", "export", "eval", "counterfactual")


def _run_job(job_id: str, cmd: list[str]) -> None:
    env_path = ROOT
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=env_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with _jobs_lock:
            _jobs[job_id]["pid"] = proc.pid
        assert proc.stdout is not None
        for line in proc.stdout:
            with _jobs_lock:
                _jobs[job_id]["log"].append(line.rstrip())
                _jobs[job_id]["log"] = _jobs[job_id]["log"][-800:]
        code = proc.wait()
        with _jobs_lock:
            _jobs[job_id]["status"] = "done" if code == 0 else "failed"
            _jobs[job_id]["exit_code"] = code
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["log"].append(f"[平台] 启动失败: {exc}")
    finally:
        with _jobs_lock:
            _jobs[job_id]["finished_at"] = time.time()


class JobPayload(BaseModel):
    kind: str
    # 结构化配置，服务端翻译成命令行。刻意不接受自由文本参数 ——
    # 让人背命令行参数是这个面板存在的反面。
    config: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/train/preview")
def preview_job(p: JobPayload) -> dict[str, Any]:
    """把面板配置翻译成命令行给人看 —— 不执行。

    这样既能确认参数没配错，也让人知道该怎么在终端复现同一次训练。
    """
    if p.kind not in JOB_KINDS:
        raise HTTPException(422, f"未知任务 {p.kind}")
    try:
        cmd = _build_command(p.kind, p.config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"command": " ".join(str(c) for c in cmd)}


@app.post("/api/train/start")
def start_job(p: JobPayload) -> dict[str, Any]:
    if p.kind not in JOB_KINDS:
        raise HTTPException(422, f"未知任务 {p.kind}，可用 {sorted(JOB_KINDS)}")
    try:
        cmd = _build_command(p.kind, p.config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with _jobs_lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if running:
            raise HTTPException(409, f"已有任务在跑：{running[0]['kind']}")
        job_id = f"{p.kind}-{int(time.time())}"
        _jobs[job_id] = {
            "id": job_id,
            "kind": p.kind,
            "cmd": cmd,
            "status": "running",
            "log": [],
            "started_at": time.time(),
            "finished_at": None,
            "exit_code": None,
            "pid": None,
        }
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/train/jobs")
def list_jobs() -> dict[str, Any]:
    with _jobs_lock:
        return {
            "jobs": [
                {k: v for k, v in j.items() if k != "log"}
                for j in sorted(_jobs.values(), key=lambda x: -x["started_at"])
            ]
        }


@app.get("/api/train/log")
def job_log(job_id: str, offset: int = 0) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return {
            "status": job["status"],
            "exit_code": job["exit_code"],
            "lines": job["log"][offset:],
            "next_offset": len(job["log"]),
        }


@app.get("/api/train/artifacts")
def artifacts() -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    base = ROOT / "artifacts"
    if base.exists():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            info: dict[str, Any] = {"name": d.name, "files": sorted(f.name for f in d.iterdir())}
            for hist in ("stage1_history.json", "stage2_history.json"):
                f = d / hist
                if f.exists():
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        info["history"] = data.get("history", [])[-3:]
                        info["best"] = data.get("best")
                    except ValueError:
                        pass
            out.append(info)
    return {"artifacts": out}


# =========================================================== 入口
def main(argv: list[str] | None = None) -> int:
    global _db
    ap = argparse.ArgumentParser(description="emotionX 集成平台")
    ap.add_argument("--host", default="127.0.0.1", help="默认只监听本机：真实会话数据不出本机")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--db", default=None)
    ap.add_argument("--import", dest="import_path", default=None)
    ap.add_argument("--source", default="real_session")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args(argv)

    if args.db:
        _db = PlatformDB(args.db)
        _autoload_seed(_db)

    if args.import_path:
        p = Path(args.import_path)
        records = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"导入 {p}: {_db.import_records(records, source=args.source)}")

    if args.no_serve:
        print(json.dumps(_db.stats(), ensure_ascii=False, indent=2))
        return 0

    import uvicorn

    s = _db.stats()
    print(f"待标注 {s['remaining']} / 共 {s['items']} 条；比较 {s['comparisons_total']} 次")
    print(f"→ http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
