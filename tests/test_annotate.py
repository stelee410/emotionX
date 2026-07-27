"""标注站的端到端测试：导入 → 取题 → 标注 → 双标一致性 → 撤销 → 导出。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from annotate import server as srv
from annotate.db import AnnotationDB


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    db = AnnotationDB(tmp_path / "test.db")
    monkeypatch.setattr(srv, "_db", db)
    monkeypatch.setattr(srv, "EXPORT_DIR", tmp_path / "exports")
    return TestClient(srv.app)


def _import(client: TestClient, utterances: list[str], **kw) -> dict:
    payload = {
        "content": "\n".join(json.dumps({"utterance": u}, ensure_ascii=False) for u in utterances),
        "format": "jsonl",
        "source": kw.pop("source", "real_session"),
        **kw,
    }
    r = client.post("/api/import", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def _label(client: TestClient, item_id: int, annotator: str, strategy: str, **kw) -> dict:
    body = {
        "item_id": item_id,
        "annotator": annotator,
        "strategy": strategy,
        "valence": kw.pop("valence", -0.5),
        "arousal": kw.pop("arousal", 0.5),
        "intensity": kw.pop("intensity", 0.6),
        **kw,
    }
    r = client.post("/api/annotate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------- 导入
def test_import_dedupes(client: TestClient) -> None:
    r1 = _import(client, ["我好难受", "帮我查一下"], double_annotate_ratio=0.0)
    assert r1["added"] == 2
    r2 = _import(client, ["我好难受", "新的一句"], double_annotate_ratio=0.0)
    assert r2["added"] == 1 and r2["duplicates"] == 1


def test_import_plain_lines(client: TestClient) -> None:
    r = client.post(
        "/api/import",
        json={"content": "第一句\n\n第二句\n", "format": "lines", "double_annotate_ratio": 0.0},
    )
    assert r.json()["added"] == 2


def test_golden_split_only_from_real_sessions(client: TestClient) -> None:
    """§8.1：golden set 绝不能包含开源/蒸馏数据。"""
    _import(client, [f"蒸馏句{i}" for i in range(40)], source="distilled", golden_ratio=0.9)
    rows = client.get("/api/stats").json()
    assert rows["total_items"] == 40
    with srv.db().connect() as con:
        goldens = con.execute("SELECT COUNT(*) c FROM items WHERE split='golden'").fetchone()["c"]
    assert goldens == 0, "非真实会话来源不得进入 golden split"


def test_golden_split_assigned_for_real_sessions(client: TestClient) -> None:
    _import(client, [f"真实句{i}" for i in range(60)], source="real_session", golden_ratio=0.5)
    with srv.db().connect() as con:
        goldens = con.execute("SELECT COUNT(*) c FROM items WHERE split='golden'").fetchone()["c"]
    assert 15 <= goldens <= 45, f"golden 划分比例异常: {goldens}/60"


def test_split_assignment_is_stable_across_runs(client: TestClient, tmp_path) -> None:
    """同一句话重复导入必须落到同一个 split（不能用内置 hash()）。"""
    a = srv._stable_ratio("我好难受")
    b = srv._stable_ratio("我好难受")
    assert a == b
    assert 0.0 <= a < 1.0


# --------------------------------------------------------------------- 取题/标注
def test_next_returns_unlabeled_then_advances(client: TestClient) -> None:
    _import(client, ["句子A", "句子B"], double_annotate_ratio=0.0)
    first = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    _label(client, first["id"], "s", "neutral")
    second = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    assert second["id"] != first["id"]


def test_pool_exhausted_returns_none(client: TestClient) -> None:
    _import(client, ["只有一句"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    _label(client, item["id"], "s", "positive")
    assert client.get("/api/next", params={"annotator": "s"}).json()["item"] is None


def test_suggestion_only_when_requested(client: TestClient) -> None:
    _import(client, ["又错了，说了多少遍了"], double_annotate_ratio=0.0)
    plain = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    assert "suggestion" not in plain
    withs = client.get(
        "/api/next", params={"annotator": "s", "with_suggestion": True}
    ).json()["item"]
    assert withs["suggestion"]["strategy"] == "frustration"


def test_crisis_hint_surfaced_to_annotator(client: TestClient) -> None:
    _import(client, ["我不想活了", "工作压力大快撑不下去了", "帮我查天气"], double_annotate_ratio=0.0)
    tiers = {}
    for _ in range(3):
        item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
        tiers[item["utterance"]] = item["crisis_hint_tier"]
        _label(client, item["id"], "s", "neutral")
    assert tiers["我不想活了"] == 2
    assert tiers["工作压力大快撑不下去了"] == 1
    assert tiers["帮我查天气"] == 0


def test_invalid_strategy_rejected(client: TestClient) -> None:
    _import(client, ["一句话"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    r = client.post(
        "/api/annotate",
        json={
            "item_id": item["id"],
            "annotator": "s",
            "strategy": "angry",
            "valence": 0,
            "arousal": 0.5,
            "intensity": 0.5,
        },
    )
    assert r.status_code == 422


def test_out_of_range_vad_rejected(client: TestClient) -> None:
    _import(client, ["一句话"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    r = client.post(
        "/api/annotate",
        json={
            "item_id": item["id"],
            "annotator": "s",
            "strategy": "neutral",
            "valence": 5.0,
            "arousal": 0.5,
            "intensity": 0.5,
        },
    )
    assert r.status_code == 422


def test_skip_requires_no_labels(client: TestClient) -> None:
    _import(client, ["乱码▓▓▓"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    r = client.post(
        "/api/annotate",
        json={"item_id": item["id"], "annotator": "s", "skipped": True, "note": "无法判断"},
    )
    assert r.status_code == 200
    assert client.get("/api/stats").json()["skipped_items"] == 1
    notes = client.get("/api/skipped").json()["items"]
    assert notes[0]["note"] == "无法判断"


def test_undo_restores_item_to_pool(client: TestClient) -> None:
    _import(client, ["唯一一句"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "s"}).json()["item"]
    _label(client, item["id"], "s", "distress")
    assert client.get("/api/next", params={"annotator": "s"}).json()["item"] is None
    r = client.post("/api/undo", params={"annotator": "s"})
    assert r.json()["item_id"] == item["id"]
    assert client.get("/api/next", params={"annotator": "s"}).json()["item"]["id"] == item["id"]


def test_undo_without_history_404(client: TestClient) -> None:
    assert client.post("/api/undo", params={"annotator": "nobody"}).status_code == 404


def test_annotate_unknown_item_404(client: TestClient) -> None:
    r = client.post(
        "/api/annotate",
        json={
            "item_id": 9999,
            "annotator": "s",
            "strategy": "neutral",
            "valence": 0,
            "arousal": 0.2,
            "intensity": 0.1,
        },
    )
    assert r.status_code == 404


# --------------------------------------------------------------- 双标一致性
def test_double_annotation_routing_and_kappa(client: TestClient) -> None:
    _import(client, [f"双标句{i}" for i in range(10)], double_annotate_ratio=1.0)
    # 标注者 A 标全部
    ids = []
    for _ in range(10):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        ids.append(item["id"])
        _label(client, item["id"], "A", "distress")
    # 标注者 B 应被优先分到这些已有 1 条标注的条目
    for i in range(10):
        item = client.get("/api/next", params={"annotator": "B"}).json()["item"]
        assert item["id"] in ids, "双标条目应优先分给第二个标注者"
        # 前 8 条一致，后 2 条不一致
        _label(client, item["id"], "B", "distress" if i < 8 else "frustration")

    ag = client.get("/api/stats").json()["agreement"]
    assert ag["n_double_annotated"] == 10
    assert ag["raw_agreement"] == pytest.approx(0.8)
    assert ag["kappa"] is not None
    assert "distress|frustration" in ag["top_disagreements"]


def test_same_annotator_cannot_double_label(client: TestClient) -> None:
    _import(client, ["一句"], double_annotate_ratio=1.0)
    item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
    _label(client, item["id"], "A", "neutral")
    # 同一人再取题不应再拿到同一条
    assert client.get("/api/next", params={"annotator": "A"}).json()["item"] is None
    # 第二个人可以
    assert client.get("/api/next", params={"annotator": "B"}).json()["item"]["id"] == item["id"]


def test_relabel_overwrites_not_duplicates(client: TestClient) -> None:
    _import(client, ["一句"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
    _label(client, item["id"], "A", "neutral")
    _label(client, item["id"], "A", "positive")
    with srv.db().connect() as con:
        n = con.execute(
            "SELECT COUNT(*) c FROM annotations WHERE item_id = ?", (item["id"],)
        ).fetchone()["c"]
    assert n == 1


# --------------------------------------------------------------------- 导出
def test_export_splits_and_excludes_conflicts(client: TestClient, tmp_path) -> None:
    _import(client, [f"句{i}" for i in range(6)], double_annotate_ratio=1.0, golden_ratio=0.0)
    ids = []
    for _ in range(6):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        ids.append(item["id"])
        _label(client, item["id"], "A", "distress", valence=-0.6, intensity=0.7)
    for i in range(6):
        item = client.get("/api/next", params={"annotator": "B"}).json()["item"]
        _label(
            client,
            item["id"],
            "B",
            "distress" if i < 5 else "positive",
            valence=-0.4,
            intensity=0.5,
        )

    r = client.post("/api/export").json()
    assert r["written"] == 5
    assert r["conflicts"] == 1
    with open(r["path"], encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    assert all(row["strategy"] == "distress" for row in rows)
    # 双标的连续值取平均
    assert rows[0]["valence"] == pytest.approx(-0.5)
    assert rows[0]["intensity"] == pytest.approx(0.6)
    assert rows[0]["n_annotators"] == 2
    # 冲突条目单独落盘，且不带标签
    with open(r["conflicts_path"], encoding="utf-8") as f:
        conflicts = [json.loads(x) for x in f]
    assert "strategy" not in conflicts[0]
    assert conflicts[0]["candidate_strategies"] == ["distress", "positive"]


def test_export_filters_by_split(client: TestClient) -> None:
    _import(client, [f"真实{i}" for i in range(30)], source="real_session", golden_ratio=1.0)
    for _ in range(30):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        _label(client, item["id"], "A", "neutral", valence=0.0, arousal=0.2, intensity=0.1)
    golden = client.post("/api/export", params={"split": "golden"}).json()
    train = client.post("/api/export", params={"split": "train"}).json()
    assert golden["written"] == 30
    assert train["written"] == 0


def test_crisis_flag_and_suggestion_audit_exported(client: TestClient) -> None:
    _import(client, ["我想自杀"], double_annotate_ratio=0.0)
    item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
    _label(client, item["id"], "A", "distress", crisis_flag=True, suggestion_shown=True)
    r = client.post("/api/export").json()
    with open(r["path"], encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["crisis_flag"] == 1
    assert row["suggestion_shown"] is True


def test_stats_tracks_label_distribution(client: TestClient) -> None:
    _import(client, ["a", "b", "c"], double_annotate_ratio=0.0)
    for strategy in ("neutral", "neutral", "positive"):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        _label(client, item["id"], "A", strategy)
    s = client.get("/api/stats").json()
    assert s["by_label"]["neutral"] == 2
    assert s["by_label"]["positive"] == 1
    assert s["remaining"] == 0


# --------------------------------------------------------- golden set 均衡抽样
def _annotate_pool(client: TestClient, spec: dict[str, int], annotator: str = "A") -> None:
    """按 {label: 条数} 造一批已标注的真实会话条目。"""
    utterances = [f"{lab}-{i}" for lab, n in spec.items() for i in range(n)]
    _import(client, utterances, source="real_session", golden_ratio=0.0, double_annotate_ratio=0.0)
    for _ in range(len(utterances)):
        item = client.get("/api/next", params={"annotator": annotator}).json()["item"]
        if item is None:
            break
        _label(client, item["id"], annotator, item["utterance"].rsplit("-", 1)[0])


def test_golden_selection_is_balanced(client: TestClient) -> None:
    """真实分布 neutral 占大头，但 golden 每类应取到目标条数。"""
    _annotate_pool(client, {"neutral": 60, "distress": 20, "frustration": 18, "positive": 12})
    r = client.post("/api/golden/select", params={"per_class": 10}).json()
    assert r["selected"] == 40
    assert all(v["selected"] == 10 for v in r["by_label"].values())
    assert r["warning"] is None

    exported = client.post("/api/export", params={"split": "golden"}).json()
    with open(exported["path"], encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["strategy"]] = counts.get(row["strategy"], 0) + 1
    assert counts == {"neutral": 10, "distress": 10, "frustration": 10, "positive": 10}


def test_golden_selection_warns_when_underfilled(client: TestClient) -> None:
    _annotate_pool(client, {"neutral": 30, "distress": 5, "frustration": 4, "positive": 2})
    r = client.post("/api/golden/select", params={"per_class": 20}).json()
    assert set(r["underfilled"]) == {"distress", "frustration", "positive"}
    assert "置信区间会偏宽" in r["warning"]


def test_golden_selection_is_frozen(client: TestClient) -> None:
    """重复抽样会让评估集随模型迭代漂移 —— 必须显式 force。"""
    _annotate_pool(client, {"neutral": 20, "distress": 20, "frustration": 20, "positive": 20})
    first = client.post("/api/golden/select", params={"per_class": 5}).json()
    assert first["selected"] == 20
    assert client.post("/api/golden/select", params={"per_class": 5}).status_code == 409
    forced = client.post("/api/golden/select", params={"per_class": 8, "force": True})
    assert forced.status_code == 200
    assert forced.json()["selected"] == 32


def test_golden_selection_excludes_non_real_sessions(client: TestClient) -> None:
    """§8.1 红线在选取环节也要生效。"""
    _import(client, [f"蒸馏{i}" for i in range(20)], source="distilled", double_annotate_ratio=0.0)
    for _ in range(20):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        _label(client, item["id"], "A", "distress")
    r = client.post("/api/golden/select", params={"per_class": 10}).json()
    assert r["selected"] == 0


def test_golden_selection_excludes_conflicted_items(client: TestClient) -> None:
    _import(client, [f"冲突{i}" for i in range(6)], source="real_session", double_annotate_ratio=1.0)
    ids = []
    for _ in range(6):
        item = client.get("/api/next", params={"annotator": "A"}).json()["item"]
        ids.append(item["id"])
        _label(client, item["id"], "A", "distress")
    for i in range(6):
        item = client.get("/api/next", params={"annotator": "B"}).json()["item"]
        _label(client, item["id"], "B", "distress" if i < 4 else "positive")
    r = client.post("/api/golden/select", params={"per_class": 10}).json()
    assert r["by_label"]["distress"]["available"] == 4, "双标冲突的 2 条不该进候选池"


def test_guideline_served(client: TestClient) -> None:
    md = client.get("/api/guideline").json()["markdown"]
    assert "StrategyLabel" in md and "frustration" in md
