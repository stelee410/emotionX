"""平台的 SQLite 存储：待标注池 + 成对比较 + 直接评分。

标注方式改为**成对比较为主**：
绝对数值标注的一致性历来很差（「这句话亲密度是 0.6 还是 0.7？」），
成对比较则容易得多（「A 和 B 哪句更亲密？」），每次判断 2–3 秒而不是 10–15 秒。

比较结果用 Bradley-Terry 还原成连续尺度（见 scale.py）。
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "platform.db"

# 可做成对比较的维度
COMPARABLE = ("intimacy_bid", "affiliation_bid", "dominance_bid", "distress_level")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint      TEXT UNIQUE NOT NULL,
    utterance        TEXT NOT NULL,
    last_agent_reply TEXT DEFAULT '',
    session_id       TEXT DEFAULT '',
    turn_index       INTEGER DEFAULT 0,
    source           TEXT NOT NULL,
    split            TEXT NOT NULL DEFAULT 'train',
    meta             TEXT DEFAULT '{}',
    created_at       REAL NOT NULL
);

-- 成对比较：在 dimension 上，winner 比 loser 更高
CREATE TABLE IF NOT EXISTS comparisons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dimension  TEXT NOT NULL,
    left_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    right_id   INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    winner     TEXT NOT NULL,          -- left | right | tie
    annotator  TEXT NOT NULL,
    elapsed_ms INTEGER,
    created_at REAL NOT NULL
);

-- 直接评分：用于锚定尺度、以及不适合比较的字段
CREATE TABLE IF NOT EXISTS ratings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id           INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    annotator         TEXT NOT NULL,
    affiliation_bid   REAL,
    dominance_bid     REAL,
    intimacy_bid      REAL,
    distress_level    REAL,
    intensity         REAL,
    directed_at_agent INTEGER,
    crisis_flag       INTEGER NOT NULL DEFAULT 0,
    note              TEXT DEFAULT '',
    skipped           INTEGER NOT NULL DEFAULT 0,
    elapsed_ms        INTEGER,
    created_at        REAL NOT NULL,
    UNIQUE(item_id, annotator)
);

CREATE INDEX IF NOT EXISTS idx_cmp_dim ON comparisons(dimension);
CREATE INDEX IF NOT EXISTS idx_cmp_ann ON comparisons(annotator);
CREATE INDEX IF NOT EXISTS idx_rat_item ON ratings(item_id);
CREATE INDEX IF NOT EXISTS idx_items_split ON items(split);
"""


def fingerprint(utterance: str, reply: str = "") -> str:
    raw = f"{utterance.strip()}||{(reply or '').strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def stable_ratio(key: str) -> float:
    """跨进程稳定的 [0,1) 哈希（内置 hash 对 str 加了随机盐）。"""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


class PlatformDB:
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

    # ---------------------------------------------------------------- 导入
    def add_item(
        self,
        utterance: str,
        last_agent_reply: str = "",
        source: str = "real_session",
        session_id: str = "",
        turn_index: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> int | None:
        utterance = (utterance or "").strip()
        if not utterance:
            return None
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO items
                   (fingerprint, utterance, last_agent_reply, session_id, turn_index,
                    source, split, meta, created_at)
                   VALUES (?,?,?,?,?,?,'train',?,?)""",
                (
                    fingerprint(utterance, last_agent_reply),
                    utterance,
                    (last_agent_reply or "").strip(),
                    session_id,
                    turn_index,
                    source,
                    json.dumps(meta or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            return int(cur.lastrowid) if cur.rowcount else None

    def import_records(
        self, records: list[dict[str, Any]], source: str = "real_session"
    ) -> dict[str, int]:
        added = dupes = 0
        for rec in records:
            text = str(rec.get("utterance") or rec.get("text") or rec.get("content") or "")
            if not text.strip():
                continue
            ok = self.add_item(
                utterance=text,
                last_agent_reply=str(rec.get("last_agent_reply") or ""),
                source=source,
                session_id=str(rec.get("session_id") or ""),
                turn_index=int(rec.get("turn_index") or 0),
                meta={k: v for k, v in rec.items() if k not in {"utterance", "text", "content"}},
            )
            if ok:
                added += 1
            else:
                dupes += 1
        return {"added": added, "duplicates": dupes, "total": len(records)}

    def items(self, limit: int = 200, source: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM items"
        params: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            params.append(source)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params)]

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------ 成对比较
    def next_pair(
        self, dimension: str, annotator: str, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """挑一对还没被这个人比较过的条目。

        优先挑**比较次数少**的条目 —— Bradley-Terry 需要每个条目有足够的
        比较边，否则它的尺度估计不稳。
        """
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT i.id, i.utterance, i.last_agent_reply, i.source,
                       (SELECT COUNT(*) FROM comparisons c
                        WHERE c.dimension = ? AND (c.left_id = i.id OR c.right_id = i.id)) AS n
                FROM items i
                ORDER BY n ASC, RANDOM()
                LIMIT 40
                """,
                (dimension,),
            ).fetchall()
            if len(rows) < 2:
                return None
            done = {
                (r["left_id"], r["right_id"])
                for r in con.execute(
                    "SELECT left_id, right_id FROM comparisons WHERE dimension = ? AND annotator = ?",
                    (dimension, annotator),
                )
            }

        rng = random.Random(seed)
        pool = [dict(r) for r in rows]
        for _ in range(60):
            a, b = rng.sample(pool, 2)
            if (a["id"], b["id"]) in done or (b["id"], a["id"]) in done:
                continue
            return a, b
        return None

    def save_comparison(
        self,
        dimension: str,
        left_id: int,
        right_id: int,
        winner: str,
        annotator: str,
        elapsed_ms: int | None = None,
    ) -> int:
        if dimension not in COMPARABLE:
            raise ValueError(f"维度必须是 {COMPARABLE} 之一，得到 {dimension!r}")
        if winner not in {"left", "right", "tie"}:
            raise ValueError(f"winner 必须是 left/right/tie，得到 {winner!r}")
        if left_id == right_id:
            raise ValueError("不能和自己比较")
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO comparisons
                   (dimension, left_id, right_id, winner, annotator, elapsed_ms, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (dimension, left_id, right_id, winner, annotator, elapsed_ms, time.time()),
            )
            return int(cur.lastrowid)

    def comparisons(self, dimension: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM comparisons WHERE dimension = ? ORDER BY id", (dimension,)
                )
            ]

    def undo_comparison(self, annotator: str) -> int | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT id FROM comparisons WHERE annotator = ? ORDER BY id DESC LIMIT 1",
                (annotator,),
            ).fetchone()
            if row is None:
                return None
            con.execute("DELETE FROM comparisons WHERE id = ?", (row["id"],))
            return int(row["id"])

    # -------------------------------------------------------------- 直接评分
    def next_unrated(self, annotator: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                """SELECT i.* FROM items i
                   WHERE NOT EXISTS (SELECT 1 FROM ratings r
                                     WHERE r.item_id = i.id AND r.annotator = ?)
                   ORDER BY i.id LIMIT 1""",
                (annotator,),
            ).fetchone()
            return dict(row) if row else None

    def save_rating(
        self, item_id: int, annotator: str, values: dict[str, Any], skipped: bool = False
    ) -> int:
        if not skipped:
            for k in ("affiliation_bid", "dominance_bid"):
                v = values.get(k)
                if v is None or not (-1.0 <= float(v) <= 1.0):
                    raise ValueError(f"{k} 必须在 [-1,1]，得到 {v!r}")
            for k in ("intimacy_bid", "distress_level", "intensity"):
                v = values.get(k)
                if v is None or not (0.0 <= float(v) <= 1.0):
                    raise ValueError(f"{k} 必须在 [0,1]，得到 {v!r}")
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO ratings
                   (item_id, annotator, affiliation_bid, dominance_bid, intimacy_bid,
                    distress_level, intensity, directed_at_agent, crisis_flag, note,
                    skipped, elapsed_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(item_id, annotator) DO UPDATE SET
                     affiliation_bid=excluded.affiliation_bid,
                     dominance_bid=excluded.dominance_bid,
                     intimacy_bid=excluded.intimacy_bid,
                     distress_level=excluded.distress_level,
                     intensity=excluded.intensity,
                     directed_at_agent=excluded.directed_at_agent,
                     crisis_flag=excluded.crisis_flag, note=excluded.note,
                     skipped=excluded.skipped, elapsed_ms=excluded.elapsed_ms,
                     created_at=excluded.created_at""",
                (
                    item_id,
                    annotator,
                    values.get("affiliation_bid"),
                    values.get("dominance_bid"),
                    values.get("intimacy_bid"),
                    values.get("distress_level"),
                    values.get("intensity"),
                    int(bool(values.get("directed_at_agent", True))),
                    int(bool(values.get("crisis_flag", False))),
                    str(values.get("note", "")),
                    int(skipped),
                    values.get("elapsed_ms"),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def ratings(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [
                dict(r)
                for r in con.execute(
                    """SELECT r.*, i.utterance, i.last_agent_reply, i.source, i.split
                       FROM ratings r JOIN items i ON i.id = r.item_id
                       WHERE r.skipped = 0 ORDER BY r.item_id"""
                )
            ]

    # -------------------------------------------------------------- golden
    def select_golden(self, n: int = 200, seed: int = 42, force: bool = False) -> dict[str, Any]:
        """挑选评估集并**冻结**。

        只从 source='real_session' 且已有评分的条目里挑。
        重复抽样会让评估集随模型迭代漂移 —— 等于慢慢把测试集调成对自己有利的样子，
        所以重抽必须显式 force。
        """
        with self.connect() as con:
            existing = con.execute(
                "SELECT COUNT(*) c FROM items WHERE split='golden'"
            ).fetchone()["c"]
            if existing and not force:
                raise ValueError(
                    f"已存在 {existing} 条 golden 条目。重抽会让评估集随模型迭代漂移；"
                    " 确认要重抽请传 force=true。"
                )
            rows = con.execute(
                """SELECT i.id FROM items i
                   WHERE i.source = 'real_session'
                     AND EXISTS (SELECT 1 FROM ratings r WHERE r.item_id = i.id AND r.skipped = 0)
                   ORDER BY i.id"""
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            rng = random.Random(seed)
            rng.shuffle(ids)
            chosen = ids[:n]
            if force:
                con.execute("UPDATE items SET split='train' WHERE split='golden'")
            con.executemany("UPDATE items SET split='golden' WHERE id = ?", [(i,) for i in chosen])
        return {
            "requested": n,
            "available": len(ids),
            "selected": len(chosen),
            "warning": (
                f"可用条目只有 {len(ids)} 条，少于请求的 {n} —— 评估置信区间会偏宽"
                if len(ids) < n
                else None
            ),
        }

    # ---------------------------------------------------------------- 统计
    def stats(self) -> dict[str, Any]:
        with self.connect() as con:
            total = con.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
            rated = con.execute(
                "SELECT COUNT(DISTINCT item_id) c FROM ratings WHERE skipped=0"
            ).fetchone()["c"]
            skipped = con.execute(
                "SELECT COUNT(DISTINCT item_id) c FROM ratings WHERE skipped=1"
            ).fetchone()["c"]
            by_dim = {
                r["dimension"]: r["c"]
                for r in con.execute(
                    "SELECT dimension, COUNT(*) c FROM comparisons GROUP BY dimension"
                )
            }
            by_source = {
                r["source"]: r["c"]
                for r in con.execute("SELECT source, COUNT(*) c FROM items GROUP BY source")
            }
            golden = con.execute("SELECT COUNT(*) c FROM items WHERE split='golden'").fetchone()["c"]
            annotators = [
                r["annotator"]
                for r in con.execute("SELECT DISTINCT annotator FROM comparisons")
            ]
            crisis = con.execute(
                "SELECT COUNT(*) c FROM ratings WHERE crisis_flag=1"
            ).fetchone()["c"]
        return {
            "items": total,
            "rated": rated,
            "skipped": skipped,
            "remaining": max(0, total - rated - skipped),
            "comparisons_by_dimension": {d: by_dim.get(d, 0) for d in COMPARABLE},
            "comparisons_total": sum(by_dim.values()),
            "by_source": by_source,
            "golden": golden,
            "annotators": annotators,
            "crisis_flagged": crisis,
        }
