"""标注站的 SQLite 层。

只用标准库 sqlite3：标注是本地单机工具，不值得引入 ORM。

两张表：
  items       —— 待标注池（一条 = 一个待标注的对话轮）
  annotations —— 标注结果（一个 item 可有多条，用于双标一致性）
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "annotations.db"

STRATEGIES = ("neutral", "distress", "frustration", "positive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint       TEXT UNIQUE NOT NULL,   -- 去重键
    session_id        TEXT,
    turn_index        INTEGER,
    utterance         TEXT NOT NULL,
    last_agent_reply  TEXT DEFAULT '',
    source            TEXT NOT NULL,          -- real_session | seed | synthetic | distilled
    split             TEXT NOT NULL DEFAULT 'train',   -- train | golden
    double_annotate   INTEGER NOT NULL DEFAULT 0,
    meta              TEXT DEFAULT '{}',
    created_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    annotator     TEXT NOT NULL,
    strategy      TEXT,                       -- skip 时为 NULL
    valence       REAL,
    arousal       REAL,
    intensity     REAL,
    crisis_flag   INTEGER NOT NULL DEFAULT 0,
    skipped       INTEGER NOT NULL DEFAULT 0,
    note          TEXT DEFAULT '',
    suggestion_shown INTEGER NOT NULL DEFAULT 0,   -- 是否看过模型建议（审计锚定偏差）
    elapsed_ms    INTEGER,
    created_at    REAL NOT NULL,
    UNIQUE(item_id, annotator)
);

CREATE INDEX IF NOT EXISTS idx_items_split ON items(split);
CREATE INDEX IF NOT EXISTS idx_ann_item ON annotations(item_id);
CREATE INDEX IF NOT EXISTS idx_ann_annotator ON annotations(annotator);
"""


def fingerprint(utterance: str, last_agent_reply: str = "") -> str:
    import hashlib

    raw = f"{utterance.strip()}||{(last_agent_reply or '').strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class AnnotationDB:
    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ 导入
    def add_item(
        self,
        utterance: str,
        last_agent_reply: str = "",
        source: str = "real_session",
        split: str = "train",
        session_id: str = "",
        turn_index: int = 0,
        double_annotate: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> int | None:
        """返回新 item id；重复内容返回 None。"""
        utterance = (utterance or "").strip()
        if not utterance:
            return None
        fp = fingerprint(utterance, last_agent_reply)
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO items
                   (fingerprint, session_id, turn_index, utterance, last_agent_reply,
                    source, split, double_annotate, meta, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    fp,
                    session_id,
                    turn_index,
                    utterance,
                    (last_agent_reply or "").strip(),
                    source,
                    split,
                    int(double_annotate),
                    json.dumps(meta or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            return int(cur.lastrowid) if cur.rowcount else None

    # ------------------------------------------------------------------ 取题
    def next_item(self, annotator: str) -> dict[str, Any] | None:
        """取下一条待标注。

        优先级：
          1. 需要双标、已有 1 条别人的标注、且本人没标过的（补齐一致性样本）
          2. 完全没人标过的
        同优先级内按 id 升序，保证多人协作时不撞题的概率足够高（配合 UNIQUE 约束兜底）。
        """
        with self.connect() as con:
            row = con.execute(
                """
                SELECT i.*, (SELECT COUNT(*) FROM annotations a WHERE a.item_id = i.id) AS n_ann
                FROM items i
                WHERE i.double_annotate = 1
                  AND (SELECT COUNT(*) FROM annotations a WHERE a.item_id = i.id) = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM annotations a WHERE a.item_id = i.id AND a.annotator = ?
                  )
                ORDER BY i.id LIMIT 1
                """,
                (annotator,),
            ).fetchone()
            if row is None:
                row = con.execute(
                    """
                    SELECT i.*, 0 AS n_ann
                    FROM items i
                    WHERE NOT EXISTS (SELECT 1 FROM annotations a WHERE a.item_id = i.id)
                    ORDER BY i.id LIMIT 1
                    """
                ).fetchone()
            return dict(row) if row else None

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ 写标注
    def save_annotation(
        self,
        item_id: int,
        annotator: str,
        strategy: str | None,
        valence: float | None = None,
        arousal: float | None = None,
        intensity: float | None = None,
        crisis_flag: bool = False,
        skipped: bool = False,
        note: str = "",
        suggestion_shown: bool = False,
        elapsed_ms: int | None = None,
    ) -> int:
        if not skipped:
            if strategy not in STRATEGIES:
                raise ValueError(f"strategy 必须是 {STRATEGIES} 之一，得到 {strategy!r}")
            for name, v, lo, hi in (
                ("valence", valence, -1.0, 1.0),
                ("arousal", arousal, 0.0, 1.0),
                ("intensity", intensity, 0.0, 1.0),
            ):
                if v is None or not (lo <= float(v) <= hi):
                    raise ValueError(f"{name} 必须在 [{lo},{hi}]，得到 {v!r}")
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO annotations
                   (item_id, annotator, strategy, valence, arousal, intensity,
                    crisis_flag, skipped, note, suggestion_shown, elapsed_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(item_id, annotator) DO UPDATE SET
                     strategy=excluded.strategy, valence=excluded.valence,
                     arousal=excluded.arousal, intensity=excluded.intensity,
                     crisis_flag=excluded.crisis_flag, skipped=excluded.skipped,
                     note=excluded.note, suggestion_shown=excluded.suggestion_shown,
                     elapsed_ms=excluded.elapsed_ms, created_at=excluded.created_at""",
                (
                    item_id,
                    annotator,
                    strategy,
                    valence,
                    arousal,
                    intensity,
                    int(crisis_flag),
                    int(skipped),
                    note or "",
                    int(suggestion_shown),
                    elapsed_ms,
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def undo_last(self, annotator: str) -> int | None:
        """撤销该标注者最近一条标注，返回被撤销的 item_id。"""
        with self.connect() as con:
            row = con.execute(
                "SELECT id, item_id FROM annotations WHERE annotator = ? ORDER BY created_at DESC LIMIT 1",
                (annotator,),
            ).fetchone()
            if row is None:
                return None
            con.execute("DELETE FROM annotations WHERE id = ?", (row["id"],))
            return int(row["item_id"])

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict[str, Any]:
        with self.connect() as con:
            total = con.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
            labeled_items = con.execute(
                "SELECT COUNT(DISTINCT item_id) c FROM annotations WHERE skipped = 0"
            ).fetchone()["c"]
            skipped = con.execute(
                "SELECT COUNT(DISTINCT item_id) c FROM annotations WHERE skipped = 1"
            ).fetchone()["c"]
            by_label = {
                r["strategy"]: r["c"]
                for r in con.execute(
                    "SELECT strategy, COUNT(*) c FROM annotations WHERE skipped = 0 GROUP BY strategy"
                )
            }
            by_annotator = {
                r["annotator"]: r["c"]
                for r in con.execute(
                    "SELECT annotator, COUNT(*) c FROM annotations GROUP BY annotator"
                )
            }
            by_split = {
                f"{r['split']}:{r['source']}": r["c"]
                for r in con.execute(
                    """SELECT i.split, i.source, COUNT(*) c
                       FROM items i
                       WHERE EXISTS (SELECT 1 FROM annotations a
                                     WHERE a.item_id = i.id AND a.skipped = 0)
                       GROUP BY i.split, i.source"""
                )
            }
            crisis = con.execute(
                "SELECT COUNT(*) c FROM annotations WHERE crisis_flag = 1"
            ).fetchone()["c"]
            median_ms = con.execute(
                "SELECT AVG(elapsed_ms) m FROM annotations WHERE elapsed_ms IS NOT NULL"
            ).fetchone()["m"]
        return {
            "total_items": total,
            "labeled_items": labeled_items,
            "skipped_items": skipped,
            "remaining": max(0, total - labeled_items - skipped),
            "by_label": {k: by_label.get(k, 0) for k in STRATEGIES},
            "by_annotator": by_annotator,
            "by_split": by_split,
            "crisis_flagged": crisis,
            "avg_seconds_per_item": round((median_ms or 0) / 1000.0, 1),
            "agreement": self.agreement(),
        }

    def agreement(self) -> dict[str, Any]:
        """双标条目上的 Cohen's Kappa（§8.1 的天花板估计）。"""
        with self.connect() as con:
            rows = con.execute(
                """SELECT item_id, annotator, strategy FROM annotations
                   WHERE skipped = 0 AND item_id IN (
                       SELECT item_id FROM annotations WHERE skipped = 0
                       GROUP BY item_id HAVING COUNT(*) >= 2)
                   ORDER BY item_id, created_at"""
            ).fetchall()
        pairs: dict[int, list[str]] = {}
        for r in rows:
            pairs.setdefault(r["item_id"], []).append(r["strategy"])
        usable = [(v[0], v[1]) for v in pairs.values() if len(v) >= 2]
        n = len(usable)
        if n < 2:
            return {"n_double_annotated": n, "kappa": None, "raw_agreement": None}
        agree = sum(1 for a, b in usable if a == b) / n
        # Cohen's kappa
        labels = sorted({x for pair in usable for x in pair})
        p_e = 0.0
        for lab in labels:
            pa = sum(1 for a, _ in usable if a == lab) / n
            pb = sum(1 for _, b in usable if b == lab) / n
            p_e += pa * pb
        kappa = (agree - p_e) / (1 - p_e) if p_e < 1 else None
        confusion = {}
        for a, b in usable:
            if a != b:
                key = "|".join(sorted((a, b)))
                confusion[key] = confusion.get(key, 0) + 1
        return {
            "n_double_annotated": n,
            "kappa": None if kappa is None else round(kappa, 3),
            "raw_agreement": round(agree, 3),
            "top_disagreements": dict(
                sorted(confusion.items(), key=lambda kv: -kv[1])[:5]
            ),
        }

    # ------------------------------------------------------------------ 导出
    def export_rows(
        self, split: str | None = None, source: str | None = None
    ) -> list[dict[str, Any]]:
        """一条 item 一行。双标冲突的条目会带 `disagreement: true` 且**不导出标签**。"""
        where = ["a.skipped = 0"]
        params: list[Any] = []
        if split:
            where.append("i.split = ?")
            params.append(split)
        if source:
            where.append("i.source = ?")
            params.append(source)
        sql = f"""
            SELECT i.id, i.utterance, i.last_agent_reply, i.source, i.split,
                   i.session_id, i.turn_index, i.meta,
                   a.annotator, a.strategy, a.valence, a.arousal, a.intensity,
                   a.crisis_flag, a.note, a.suggestion_shown, a.created_at
            FROM items i JOIN annotations a ON a.item_id = i.id
            WHERE {' AND '.join(where)}
            ORDER BY i.id, a.created_at
        """
        with self.connect() as con:
            rows = [dict(r) for r in con.execute(sql, params)]

        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(r["id"], []).append(r)

        out: list[dict[str, Any]] = []
        for item_id, anns in grouped.items():
            first = anns[0]
            strategies = {a["strategy"] for a in anns}
            disagreement = len(strategies) > 1
            record: dict[str, Any] = {
                "item_id": item_id,
                "utterance": first["utterance"],
                "last_agent_reply": first["last_agent_reply"],
                "source": first["source"],
                "split": first["split"],
                "session_id": first["session_id"],
                "turn_index": first["turn_index"],
                "n_annotators": len(anns),
                "annotators": [a["annotator"] for a in anns],
                "disagreement": disagreement,
            }
            if disagreement:
                # §8.1：冲突条目不该带着任意一方的标签进训练/评估集
                record["candidate_strategies"] = sorted(strategies)
            else:
                record["strategy"] = first["strategy"]
                record["valence"] = round(
                    sum(a["valence"] for a in anns) / len(anns), 3
                )
                record["arousal"] = round(sum(a["arousal"] for a in anns) / len(anns), 3)
                record["intensity"] = round(
                    sum(a["intensity"] for a in anns) / len(anns), 3
                )
            record["crisis_flag"] = int(any(a["crisis_flag"] for a in anns))
            notes = [a["note"] for a in anns if a["note"]]
            if notes:
                record["notes"] = notes
            if any(a["suggestion_shown"] for a in anns):
                record["suggestion_shown"] = True
            out.append(record)
        return out

    # ------------------------------------------------------------ golden set 选取
    def select_golden(
        self, per_class: int = 60, seed: int = 42, force: bool = False
    ) -> dict[str, Any]:
        """标注完成后，按标签**均衡**挑选 golden set 并冻结。

        为什么不在导入时按哈希划分：macro-F1 与先验分布无关，而真实分布里 neutral 占
        60–80%，自然抽样会让 distress/positive 的 support 只有一二十条，
        per-class F1 的置信区间宽到无法比较两个模型（实测：自然分布 300 条 → ±0.075；
        均衡 200 条 → ±0.048）。

        约束：
          * 只从 source='real_session' 里挑（§8.1）
          * 跳过双标冲突的条目（标签本身不可靠，不该进评估集）
          * **一旦选定就冻结**。重复调用需要 force=True —— 否则每次迭代模型后重新抽
            golden set，等于慢慢把测试集调成对自己有利的样子。
        """
        with self.connect() as con:
            existing = con.execute(
                "SELECT COUNT(*) c FROM items WHERE split = 'golden'"
            ).fetchone()["c"]
            if existing and not force:
                raise ValueError(
                    f"已存在 {existing} 条 golden 条目。重新抽样会让评估集随模型迭代漂移；"
                    " 确认要重抽请传 force=True。"
                )

            rows = con.execute(
                """
                SELECT i.id, a.strategy, COUNT(*) n_ann,
                       COUNT(DISTINCT a.strategy) n_distinct
                FROM items i JOIN annotations a ON a.item_id = i.id
                WHERE a.skipped = 0 AND i.source = 'real_session'
                GROUP BY i.id
                HAVING n_distinct = 1
                """
            ).fetchall()

            by_label: dict[str, list[int]] = {}
            for r in rows:
                by_label.setdefault(r["strategy"], []).append(int(r["id"]))

            rng = random.Random(seed)
            chosen: list[int] = []
            report: dict[str, Any] = {}
            for label in STRATEGIES:
                pool = sorted(by_label.get(label, []))
                rng.shuffle(pool)
                take = pool[:per_class]
                chosen.extend(take)
                report[label] = {"available": len(pool), "selected": len(take)}

            if force:
                con.execute("UPDATE items SET split = 'train' WHERE split = 'golden'")
            con.executemany(
                "UPDATE items SET split = 'golden' WHERE id = ?", [(i,) for i in chosen]
            )

        short = {k: v for k, v in report.items() if v["selected"] < per_class}
        return {
            "per_class_target": per_class,
            "selected": len(chosen),
            "by_label": report,
            "underfilled": short,
            "warning": (
                f"这些类别不足 {per_class} 条，评估置信区间会偏宽：{sorted(short)}"
                if short
                else None
            ),
        }

    def skipped_with_notes(self) -> list[dict[str, Any]]:
        """skip + 备注 是最有价值的标签定义反馈（见 label_guideline §5）。"""
        with self.connect() as con:
            return [
                dict(r)
                for r in con.execute(
                    """SELECT i.id, i.utterance, a.annotator, a.note
                       FROM annotations a JOIN items i ON i.id = a.item_id
                       WHERE a.skipped = 1 AND a.note != ''
                       ORDER BY a.created_at DESC"""
                )
            ]
