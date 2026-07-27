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
    assert {d["key"] for d in m["comparable"]} <= {
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


# ------------------------------------------------------------------ 中文词表
def test_every_param_has_a_chinese_label(client: TestClient) -> None:
    """★ 加一个参数就必须同时加一条中文说明 —— 少一条，UI 上就会露出英文字段名。"""
    m = client.get("/api/meta").json()
    missing = sorted(set(m["params"]) - set(m["labels"]["params"]))
    assert not missing, f"这些参数缺中文标签: {missing}"


def test_every_param_is_in_exactly_one_group(client: TestClient) -> None:
    """漏分组的参数在面板上根本不会显示出来。"""
    m = client.get("/api/meta").json()
    grouped: list[str] = [k for g in m["param_groups"] for k in g["keys"]]
    assert sorted(grouped) == sorted(m["params"]), "参数分组不完整或有重复"


def test_every_channel_and_mechanism_has_a_label(client: TestClient) -> None:
    m = client.get("/api/meta").json()
    for ch in m["channels"]:
        assert ch["zh"] and ch["zh"] != ch["name"], ch["name"]
    assert set(m["labels"]["buckets"]) == {"high", "medium", "low"}
    from affect.appraisal import RelationalAppraisal
    from affect.moves import TurnContext, UserMove
    from affect.relation import RelationType, preset

    # 把所有能触发的机制跑出来，逐个确认有中文
    eng = RelationalAppraisal()
    fired: set[str] = set()
    for move, ctx in (
        (UserMove(affiliation_bid=0.8, intimacy_bid=0.9, intensity=0.8), TurnContext()),
        (UserMove(affiliation_bid=-0.8, intimacy_bid=0.1, intensity=0.8), TurnContext()),
        (UserMove(distress_level=0.9, intensity=0.7), TurnContext()),
        (UserMove(dominance_bid=0.8, intensity=0.5), TurnContext(task_succeeded=True)),
        (UserMove(intensity=0.3), TurnContext(task_failed=True, user_repeated_query=True, latency_ms=9000)),
    ):
        for rt in (RelationType.PARTNER, RelationType.STRANGER):
            _, f, _ = eng.delta(move, ctx, preset(rt))
            fired.update(f)
    missing = sorted(fired - set(m["labels"]["mechanisms"]))
    assert not missing, f"这些机制缺中文标签: {missing}"


def test_comparable_dimensions_carry_help_text(client: TestClient) -> None:
    """下拉框里不能只有字段名 —— 标注者不该先去查字段含义。"""
    m = client.get("/api/meta").json()
    for d in m["comparable"]:
        assert d["zh"] != d["key"], d["key"]
        assert len(d["hint"]) > 10, d["key"]
    # 两个最容易标错的必须有额外提醒
    by_key = {d["key"]: d for d in m["comparable"]}
    assert by_key["intimacy_bid"]["note"]
    assert by_key["dominance_bid"]["note"]


# ------------------------------------------------------------------ 数据集
def test_datasets_report_availability(client: TestClient) -> None:
    d = client.get("/api/datasets").json()
    keys = {x["key"] for x in d["datasets"]}
    assert {"ewect", "simplifyweibo", "cped", "m3ed"} <= keys
    for x in d["datasets"]:
        assert x["status"] and x["detail"]
        # 自动下载的一定可用；需手动获取的只有本地存在才可用
        assert x["usable"] == (x["present"] or x["auto"])
    ewect = next(x for x in d["datasets"] if x["key"] == "ewect")
    assert ewect["auto"] and ewect["recommended"]


def test_seed_pool_autoloaded(tmp_path) -> None:
    """空库的标注面板没法用，也看不出该怎么用 —— 首次启动要自动灌种子。"""
    db = PlatformDB(tmp_path / "fresh.db")
    assert db.stats()["items"] == 0
    srv._autoload_seed(db)
    stats = db.stats()
    assert stats["items"] > 100
    assert stats["by_source"] == {"seed": stats["items"]}, "种子必须标成 seed，不能进评估集"
    # 幂等：再调一次不会重复灌
    srv._autoload_seed(db)
    assert db.stats()["items"] == stats["items"]


# ------------------------------------------------------------------ 训练
def test_unknown_job_rejected(client: TestClient) -> None:
    assert client.post("/api/train/start", json={"kind": "rm -rf"}).status_code == 422


def test_command_is_built_from_structured_config(client: TestClient) -> None:
    """面板配置 → 命令行。不接受自由文本参数：让人背参数是这个面板存在的反面。"""
    r = client.post(
        "/api/train/preview",
        json={
            "kind": "stage1",
            "config": {
                "datasets": ["ewect", "simplifyweibo"],
                "epochs": 5,
                "batch_size": 32,
                "max_per_dataset": 500,
            },
        },
    ).json()
    cmd = r["command"]
    assert "--datasets ewect simplifyweibo" in cmd
    assert "--epochs 5" in cmd and "--batch-size 32" in cmd
    assert "--max-per-dataset 500" in cmd


def test_stage2_source_switches_flags(client: TestClient) -> None:
    boot = client.post(
        "/api/train/preview",
        json={"kind": "stage2", "config": {"source": "bootstrap", "bootstrap": 8000}},
    ).json()["command"]
    assert "--bootstrap 8000" in boot and "--annotations" not in boot

    ann = client.post(
        "/api/train/preview",
        json={"kind": "stage2", "config": {"source": "annotations", "annotations": "x.jsonl"}},
    ).json()["command"]
    assert "--annotations" in ann and "--bootstrap" not in ann


def test_stage2_annotations_without_file_is_rejected(client: TestClient) -> None:
    r = client.post(
        "/api/train/preview", json={"kind": "stage2", "config": {"source": "annotations"}}
    )
    assert r.status_code == 422
    assert "先在标注面板导出" in r.json()["detail"]


def test_export_quantize_toggle(client: TestClient) -> None:
    on = client.post(
        "/api/train/preview", json={"kind": "export", "config": {"quantize": True}}
    ).json()["command"]
    off = client.post(
        "/api/train/preview", json={"kind": "export", "config": {"quantize": False}}
    ).json()["command"]
    assert "--no-quantize" not in on and "--no-quantize" in off


def test_eval_defaults_to_heuristic_baseline(client: TestClient) -> None:
    cmd = client.post("/api/train/preview", json={"kind": "eval", "config": {}}).json()["command"]
    assert "--heuristic" in cmd


def test_preview_does_not_execute(client: TestClient) -> None:
    """预览只翻译命令，不能真的跑起来。"""
    client.post("/api/train/preview", json={"kind": "stage1", "config": {}})
    assert client.get("/api/train/jobs").json()["jobs"] == []


def test_job_log_404(client: TestClient) -> None:
    assert client.get("/api/train/log", params={"job_id": "nope"}).status_code == 404


def test_artifacts_endpoint(client: TestClient) -> None:
    assert "artifacts" in client.get("/api/train/artifacts").json()
