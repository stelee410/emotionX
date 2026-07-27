"""M6 · 集成平台：API 与 Bradley-Terry 尺度还原。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from studio import server as srv
from studio.db import PlatformDB
from studio.scale import agreement, build_scale, fit_bradley_terry, theta_to_scale


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    db = PlatformDB(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_db", db)
    monkeypatch.setattr(srv, "EXPORTS", tmp_path / "exports")
    return TestClient(srv.app)


def _import(client: TestClient, texts: list[str], source: str = "real_session") -> dict:
    body = {
        "content": "\n".join(json.dumps({"utterance": t}, ensure_ascii=False) for t in texts),
        "format": "jsonl",
        "source": source,
    }
    r = client.post("/api/annotate/import", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ meta
def test_meta_exposes_everything_ui_needs(client: TestClient) -> None:
    m = client.get("/api/meta").json()
    assert len(m["channels"]) == 6
    assert len(m["relations"]) == 8
    assert m["personas"] and m["actions"] and m["moods"]
    assert set(m["comparable"]) <= {
        "intimacy_bid", "affiliation_bid", "dominance_bid", "distress_level"
    }
    for ch in m["channels"]:
        assert "gain" in ch and "half_life" in ch and "thresholds" in ch


# ------------------------------------------------------------- 对话测试台
def test_session_open_and_turn(client: TestClient) -> None:
    assert client.post("/api/session/open", json={"relation": "stranger"}).status_code == 200
    r = client.post("/api/session/turn", json={"utterance": "我想要你"}).json()
    assert r["state"]["threat"] > 0.5
    assert r["prompt"]["actions"]["chosen"]
    assert "危机" not in r["prompt"]["text"]


def test_same_utterance_flips_across_relations_through_api(client: TestClient) -> None:
    """★ 平台上能一眼看到的那件事，也得在 API 层成立。"""
    out = {}
    for rel in ("partner", "stranger"):
        client.post(
            "/api/session/open",
            json={"session_id": rel, "relation": rel, "age_verified": True},
        )
        out[rel] = client.post(
            "/api/session/turn", json={"session_id": rel, "utterance": "我想要你"}
        ).json()["state"]
    assert out["partner"]["affiliation"] > out["stranger"]["affiliation"]
    assert out["stranger"]["threat"] > out["partner"]["threat"]


def test_illegal_relation_combo_is_rejected(client: TestClient) -> None:
    r = client.post("/api/session/open", json={"relation": "partner", "age_verified": False})
    assert r.status_code == 409
    assert "年龄验证" in r.json()["detail"]


def test_turn_without_session_is_rejected(client: TestClient) -> None:
    r = client.post("/api/session/turn", json={"session_id": "nope", "utterance": "x"})
    assert r.status_code == 409


def test_move_override_bypasses_perception(client: TestClient) -> None:
    """手动给 UserMove，用于绕开感知桩单独测 L2/L3。"""
    client.post("/api/session/open", json={"relation": "stranger"})
    r = client.post(
        "/api/session/turn",
        json={
            "utterance": "无所谓的文本",
            "move_override": {"affiliation_bid": 0.9, "intimacy_bid": 0.95, "intensity": 0.9},
        },
    ).json()
    assert r["user_move"]["intimacy_bid"] == pytest.approx(0.95)
    assert r["state"]["threat"] > 0.5


def test_factual_content_neutralises_display(client: TestClient) -> None:
    client.post("/api/session/open", json={"relation": "friend", "display": True})
    r = client.post(
        "/api/session/turn", json={"utterance": "我快撑不住了", "factual_content": True}
    ).json()
    assert r["display"]["neutralised"] is True


def test_memory_is_manual_and_one_way(client: TestClient) -> None:
    client.post("/api/memory", json={"notes": ["他上周提过要换工作"]})
    client.post("/api/session/open", json={"relation": "friend"})
    r = client.post("/api/session/turn", json={"utterance": "最近怎么样"}).json()
    assert "他上周提过要换工作" in r["memory_notes"]
    assert "他上周提过要换工作" in r["prompt"]["text"]


def test_memory_keyword_filter(client: TestClient) -> None:
    client.post("/api/memory", json={"notes": ["工作|他在找新工作"]})
    client.post("/api/session/open", json={"relation": "friend"})
    hit = client.post("/api/session/turn", json={"utterance": "工作还顺利吗"}).json()
    miss = client.post("/api/session/turn", json={"utterance": "今天天气不错"}).json()
    assert hit["memory_notes"] and not miss["memory_notes"]


# --------------------------------------------------------------- 参数调校
def test_params_apply_runs_counterfactual(client: TestClient) -> None:
    r = client.post("/api/params", json={"params": {"breach_to_threat": 0.55}}).json()
    assert r["params"]["breach_to_threat"] == pytest.approx(0.55)
    assert "direction_accuracy" in r["counterfactual"]
    client.post("/api/params/reset")


def test_params_reject_contagion_config(client: TestClient) -> None:
    """★ 违反「共情≠镜像」的参数必须在应用前被拦下。"""
    r = client.post(
        "/api/params",
        json={"params": {"distress_to_concern": 0.05, "distress_to_valence": -0.6}},
    )
    assert r.status_code == 422
    assert "情绪传染" in r.json()["detail"]


def test_params_reject_boundary_bypass(client: TestClient) -> None:
    r = client.post("/api/params", json={"params": {"affiliation_inhibits_threat": 0.5}})
    assert r.status_code == 422
    assert "绕过边界机制" in r.json()["detail"]


def test_unknown_param_rejected(client: TestClient) -> None:
    assert client.post("/api/params", json={"params": {"cortisol": 1.0}}).status_code == 422


# ----------------------------------------------------------------- 反事实
def test_counterfactual_endpoint(client: TestClient) -> None:
    r = client.get("/api/counterfactual").json()
    assert r["summary"]["cases"] >= 30
    assert r["summary"]["direction_accuracy"] == 1.0
    assert r["results"][0]["assertions"]


def test_counterfactual_tag_filter(client: TestClient) -> None:
    r = client.get("/api/counterfactual?tag=core").json()
    assert 0 < r["summary"]["cases"] < 31


# ------------------------------------------------------------------ 标注
def test_import_and_dedupe(client: TestClient) -> None:
    assert _import(client, ["a", "b"])["added"] == 2
    assert _import(client, ["a", "c"])["duplicates"] == 1


def test_pair_selection_avoids_repeats(client: TestClient) -> None:
    _import(client, [f"句{i}" for i in range(6)])
    seen = set()
    for _ in range(5):
        pair = client.get(
            "/api/annotate/pair", params={"dimension": "intimacy_bid", "annotator": "A"}
        ).json()["pair"]
        assert pair is not None
        key = tuple(sorted((pair["left"]["id"], pair["right"]["id"])))
        assert key not in seen, "同一对不该重复分给同一个人"
        seen.add(key)
        client.post(
            "/api/annotate/compare",
            json={
                "dimension": "intimacy_bid",
                "left_id": pair["left"]["id"],
                "right_id": pair["right"]["id"],
                "winner": "left",
                "annotator": "A",
            },
        )


def test_compare_validation(client: TestClient) -> None:
    _import(client, ["x", "y"])
    bad_dim = client.post(
        "/api/annotate/compare",
        json={"dimension": "valence", "left_id": 1, "right_id": 2, "winner": "left", "annotator": "A"},
    )
    assert bad_dim.status_code == 422
    bad_winner = client.post(
        "/api/annotate/compare",
        json={"dimension": "intimacy_bid", "left_id": 1, "right_id": 2, "winner": "maybe", "annotator": "A"},
    )
    assert bad_winner.status_code == 422
    self_cmp = client.post(
        "/api/annotate/compare",
        json={"dimension": "intimacy_bid", "left_id": 1, "right_id": 1, "winner": "left", "annotator": "A"},
    )
    assert self_cmp.status_code == 422


def test_undo_comparison(client: TestClient) -> None:
    _import(client, ["p", "q"])
    client.post(
        "/api/annotate/compare",
        json={"dimension": "intimacy_bid", "left_id": 1, "right_id": 2, "winner": "left", "annotator": "A"},
    )
    assert client.post("/api/annotate/undo", params={"annotator": "A"}).status_code == 200
    assert client.post("/api/annotate/undo", params={"annotator": "A"}).status_code == 404


def test_rating_validation(client: TestClient) -> None:
    _import(client, ["一句"])
    bad = client.post(
        "/api/annotate/rate",
        json={"item_id": 1, "annotator": "A", "values": {"affiliation_bid": 5.0}},
    )
    assert bad.status_code == 422


def test_rating_and_export(client: TestClient) -> None:
    _import(client, ["甲", "乙"])
    for i in (1, 2):
        client.post(
            "/api/annotate/rate",
            json={
                "item_id": i,
                "annotator": "A",
                "values": {
                    "affiliation_bid": 0.3, "dominance_bid": -0.1, "intimacy_bid": 0.4,
                    "distress_level": 0.2, "intensity": 0.5, "directed_at_agent": True,
                },
            },
        )
    r = client.post("/api/annotate/export").json()
    assert r["written"] == 2
    with open(r["path"], encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    assert all("intimacy_bid" in row for row in rows)
    assert all(row["source"] == "real_session" for row in rows)


def test_golden_is_frozen(client: TestClient) -> None:
    _import(client, [f"g{i}" for i in range(5)])
    for i in range(1, 6):
        client.post(
            "/api/annotate/rate",
            json={
                "item_id": i, "annotator": "A",
                "values": {
                    "affiliation_bid": 0, "dominance_bid": 0, "intimacy_bid": 0.1,
                    "distress_level": 0, "intensity": 0.1,
                },
            },
        )
    first = client.post("/api/annotate/golden", params={"n": 3}).json()
    assert first["selected"] == 3
    assert client.post("/api/annotate/golden", params={"n": 3}).status_code == 409
    assert client.post("/api/annotate/golden", params={"n": 4, "force": True}).status_code == 200


# ---------------------------------------------------- Bradley-Terry 尺度
def test_bradley_terry_recovers_ordering() -> None:
    """构造一个明确的强弱顺序，BT 必须还原出来。"""
    comps = []
    for hi, lo in ((3, 2), (3, 1), (2, 1), (3, 2), (2, 1), (3, 1)):
        comps.append({"left_id": hi, "right_id": lo, "winner": "left"})
    theta = fit_bradley_terry(comps)
    assert theta[3] > theta[2] > theta[1]


def test_bradley_terry_handles_ties() -> None:
    comps = [{"left_id": 1, "right_id": 2, "winner": "tie"}] * 6
    theta = fit_bradley_terry(comps)
    assert abs(theta[1] - theta[2]) < 0.05


def test_undefeated_item_does_not_diverge() -> None:
    """没有正则化的话，从没输过的条目 θ 会跑到无穷大。"""
    comps = [{"left_id": 1, "right_id": i, "winner": "left"} for i in range(2, 8)]
    theta = fit_bradley_terry(comps)
    assert all(abs(v) < 20 for v in theta.values())


def test_theta_to_scale_uses_anchors() -> None:
    theta = {1: -1.0, 2: 0.0, 3: 1.0}
    scaled = theta_to_scale(theta, 0.0, 1.0, anchors={1: 0.2, 3: 0.8})
    assert scaled[1] == pytest.approx(0.2, abs=0.02)
    assert scaled[3] == pytest.approx(0.8, abs=0.02)
    assert 0.4 < scaled[2] < 0.6


def test_theta_to_scale_without_anchors_spreads_range() -> None:
    scaled = theta_to_scale({1: -2.0, 2: 0.0, 3: 2.0}, 0.0, 1.0)
    assert scaled[1] == pytest.approx(0.0)
    assert scaled[3] == pytest.approx(1.0)


def test_scale_warns_on_under_compared_items() -> None:
    comps = [{"left_id": 1, "right_id": 2, "winner": "left"}]
    out = build_scale(comps, "intimacy_bid", item_ids=[1, 2, 3])
    assert out["warning"] and "比较次数少于 3" in out["warning"]
    assert 3 in out["under_compared"]


def test_scale_range_depends_on_dimension() -> None:
    comps = [{"left_id": 1, "right_id": 2, "winner": "left"}]
    assert build_scale(comps, "intimacy_bid")["range"] == [0.0, 1.0]
    assert build_scale(comps, "affiliation_bid")["range"] == [-1.0, 1.0]


def test_agreement_normalises_pair_direction() -> None:
    """A 说左边高、B 说右边高（但左右调换过）—— 应算作一致。"""
    comps = [
        {"left_id": 1, "right_id": 2, "winner": "left", "annotator": "A"},
        {"left_id": 2, "right_id": 1, "winner": "right", "annotator": "B"},
    ]
    assert agreement(comps)["raw_agreement"] == 1.0


def test_agreement_detects_disagreement() -> None:
    comps = [
        {"left_id": 1, "right_id": 2, "winner": "left", "annotator": "A"},
        {"left_id": 1, "right_id": 2, "winner": "right", "annotator": "B"},
    ]
    assert agreement(comps)["raw_agreement"] == 0.0


# ------------------------------------------------------------------ 训练
def test_unknown_job_rejected(client: TestClient) -> None:
    assert client.post("/api/train/start", json={"kind": "rm -rf"}).status_code == 422


def test_job_log_404(client: TestClient) -> None:
    assert client.get("/api/train/log", params={"job_id": "nope"}).status_code == 404


def test_artifacts_endpoint(client: TestClient) -> None:
    assert "artifacts" in client.get("/api/train/artifacts").json()
