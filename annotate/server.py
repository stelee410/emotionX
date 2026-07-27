"""本地人工标注站点（§3.4 / §8.1）。

    python annotate/server.py                    # → http://127.0.0.1:8077
    python annotate/server.py --import data/raw/seed_pool.jsonl
    python annotate/server.py --port 9000 --db data/annotations.db

只监听 127.0.0.1：医疗类会话数据不得离开本机（§3.4）。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from affect.perception import HeuristicPerceiver  # noqa: E402
from affect.safety import crisis_tier  # noqa: E402
from annotate.db import AnnotationDB  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
GUIDELINE = ROOT / "config" / "label_guideline.md"
EXPORT_DIR = ROOT / "data" / "exports"

app = FastAPI(title="Affect 标注站", docs_url="/api/docs")
_db: AnnotationDB = AnnotationDB()
_suggester = HeuristicPerceiver()


def db() -> AnnotationDB:
    return _db


# --------------------------------------------------------------------- 请求体
class AnnotatePayload(BaseModel):
    item_id: int
    annotator: str = Field(min_length=1, max_length=40)
    strategy: str | None = None
    valence: float | None = None
    arousal: float | None = None
    intensity: float | None = None
    crisis_flag: bool = False
    skipped: bool = False
    note: str = ""
    suggestion_shown: bool = False
    elapsed_ms: int | None = None


class ImportPayload(BaseModel):
    """粘贴导入：JSONL 或 CSV 文本。"""

    content: str
    format: str = "jsonl"  # jsonl | csv | lines
    source: str = "real_session"
    golden_ratio: float = 0.0
    double_annotate_ratio: float = 0.2


# --------------------------------------------------------------------- 页面
@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/guideline", response_class=HTMLResponse)
def guideline_page() -> Any:
    return FileResponse(STATIC_DIR / "guideline.html")


@app.get("/api/guideline")
def guideline_md() -> dict[str, str]:
    text = GUIDELINE.read_text(encoding="utf-8") if GUIDELINE.exists() else "（缺少指南文件）"
    return {"markdown": text}


# --------------------------------------------------------------------- 取题
@app.get("/api/next")
def next_item(annotator: str, with_suggestion: bool = False) -> dict[str, Any]:
    item = db().next_item(annotator)
    if item is None:
        return {"item": None, "stats": db().stats()}
    payload: dict[str, Any] = {
        "id": item["id"],
        "utterance": item["utterance"],
        "last_agent_reply": item["last_agent_reply"],
        "source": item["source"],
        "split": item["split"],
        "session_id": item["session_id"],
        "turn_index": item["turn_index"],
        "meta": json.loads(item["meta"] or "{}"),
        # 危机信号提示：这是规则层的输出，不是模型预测，不构成标签锚定
        "crisis_hint_tier": crisis_tier(item["utterance"]),
    }
    if with_suggestion:
        s = _suggester.perceive(item["utterance"], item["last_agent_reply"] or None)
        payload["suggestion"] = {
            "strategy": s.strategy,
            "valence": round(s.valence, 2),
            "arousal": round(s.arousal, 2),
            "intensity": round(s.intensity, 2),
        }
    return {"item": payload, "stats": db().stats()}


@app.get("/api/item/{item_id}")
def get_item(item_id: int) -> dict[str, Any]:
    item = db().get_item(item_id)
    if item is None:
        raise HTTPException(404, "item 不存在")
    return item


# --------------------------------------------------------------------- 写标注
@app.post("/api/annotate")
def annotate(payload: AnnotatePayload) -> dict[str, Any]:
    if db().get_item(payload.item_id) is None:
        raise HTTPException(404, f"item {payload.item_id} 不存在")
    try:
        db().save_annotation(
            item_id=payload.item_id,
            annotator=payload.annotator.strip(),
            strategy=payload.strategy,
            valence=payload.valence,
            arousal=payload.arousal,
            intensity=payload.intensity,
            crisis_flag=payload.crisis_flag,
            skipped=payload.skipped,
            note=payload.note,
            suggestion_shown=payload.suggestion_shown,
            elapsed_ms=payload.elapsed_ms,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, "stats": db().stats()}


@app.post("/api/undo")
def undo(annotator: str) -> dict[str, Any]:
    item_id = db().undo_last(annotator)
    if item_id is None:
        raise HTTPException(404, "没有可撤销的标注")
    return {"ok": True, "item_id": item_id, "stats": db().stats()}


# --------------------------------------------------------------------- 统计/导出
@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return db().stats()


@app.get("/api/skipped")
def skipped() -> dict[str, Any]:
    return {"items": db().skipped_with_notes()}


@app.post("/api/golden/select")
def select_golden(per_class: int = 60, seed: int = 42, force: bool = False) -> dict[str, Any]:
    """标注完成后按标签均衡挑 golden set 并冻结（见 db.select_golden 的说明）。"""
    try:
        return db().select_golden(per_class=per_class, seed=seed, force=force)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/export")
def export(split: str | None = None, filename: str | None = None) -> dict[str, Any]:
    rows = db().export_rows(split=split)
    clean = [r for r in rows if not r["disagreement"]]
    conflicts = [r for r in rows if r["disagreement"]]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = filename or f"annotations_{split or 'all'}.jsonl"
    path = EXPORT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    conflict_path = None
    if conflicts:
        conflict_path = EXPORT_DIR / name.replace(".jsonl", "_conflicts.jsonl")
        with conflict_path.open("w", encoding="utf-8") as f:
            for r in conflicts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "written": len(clean),
        "path": str(path),
        "conflicts": len(conflicts),
        "conflicts_path": str(conflict_path) if conflict_path else None,
    }


# --------------------------------------------------------------------- 导入
def _iter_records(content: str, fmt: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if fmt == "jsonl":
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    elif fmt == "csv":
        for row in csv.DictReader(io.StringIO(content)):
            records.append(dict(row))
    elif fmt == "lines":
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append({"utterance": line})
    else:
        raise ValueError(f"未知格式 {fmt}")
    return records


def _stable_ratio(key: str) -> float:
    """跨进程稳定的 [0,1) 哈希。"""
    import hashlib

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def import_records(
    records: list[dict[str, Any]],
    source: str = "real_session",
    golden_ratio: float = 0.0,
    double_annotate_ratio: float = 0.2,
    database: AnnotationDB | None = None,
) -> dict[str, Any]:
    """golden_ratio > 0 时按稳定哈希把一部分条目划入 golden split。

    §8.1：golden set 只允许来自真实会话。source 不是 real_session 时强制 golden_ratio=0。
    """
    d = database or db()
    if source != "real_session" and golden_ratio > 0:
        golden_ratio = 0.0
    added = 0
    duplicates = 0
    for rec in records:
        utterance = str(
            rec.get("utterance") or rec.get("text") or rec.get("content") or ""
        ).strip()
        if not utterance:
            continue
        reply = str(rec.get("last_agent_reply") or rec.get("agent_reply") or "").strip()
        # 用 md5 而不是内置 hash()：后者对 str 加了随机盐，跨进程不稳定，
        # 会导致同一条数据重复导入时落到不同 split。
        split = "golden" if _stable_ratio(utterance) < golden_ratio else "train"
        # 双标抽样用另一个哈希切面，避免与 golden 划分相关
        dbl = _stable_ratio(utterance + "#dbl") < double_annotate_ratio
        item_id = d.add_item(
            utterance=utterance,
            last_agent_reply=reply,
            source=source,
            split=split,
            session_id=str(rec.get("session_id") or ""),
            turn_index=int(rec.get("turn_index") or 0),
            double_annotate=dbl,
            meta={
                k: v
                for k, v in rec.items()
                if k
                not in {
                    "utterance",
                    "text",
                    "content",
                    "last_agent_reply",
                    "agent_reply",
                    "session_id",
                    "turn_index",
                }
            },
        )
        if item_id is None:
            duplicates += 1
        else:
            added += 1
    return {"added": added, "duplicates": duplicates, "total": len(records)}


@app.post("/api/import")
def import_endpoint(payload: ImportPayload) -> dict[str, Any]:
    try:
        records = _iter_records(payload.content, payload.format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = import_records(
        records,
        source=payload.source,
        golden_ratio=payload.golden_ratio,
        double_annotate_ratio=payload.double_annotate_ratio,
    )
    result["stats"] = db().stats()
    return result


# --------------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    global _db
    ap = argparse.ArgumentParser(description="本地标注站")
    ap.add_argument("--host", default="127.0.0.1", help="默认只监听本机（医疗数据不得外泄）")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--db", default=None)
    ap.add_argument("--import", dest="import_path", default=None, help="启动前导入 JSONL/CSV")
    ap.add_argument("--source", default="real_session")
    ap.add_argument("--golden-ratio", type=float, default=0.0)
    ap.add_argument("--double-annotate-ratio", type=float, default=0.2)
    ap.add_argument("--no-serve", action="store_true", help="只导入，不启动服务")
    args = ap.parse_args(argv)

    if args.db:
        _db = AnnotationDB(args.db)

    if args.import_path:
        p = Path(args.import_path)
        fmt = "csv" if p.suffix.lower() == ".csv" else "jsonl"
        records = _iter_records(p.read_text(encoding="utf-8"), fmt)
        result = import_records(
            records,
            source=args.source,
            golden_ratio=args.golden_ratio,
            double_annotate_ratio=args.double_annotate_ratio,
            database=_db,
        )
        print(f"导入 {p}: 新增 {result['added']}，重复跳过 {result['duplicates']}")

    if args.no_serve:
        print(json.dumps(_db.stats(), ensure_ascii=False, indent=2))
        return 0

    import uvicorn

    s = _db.stats()
    print(f"待标注 {s['remaining']} / 共 {s['total_items']} 条")
    print(f"→ http://{args.host}:{args.port}   （标注指南 /guideline）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
