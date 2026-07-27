"""数据加载与格式统一（§3.3 / §3.4）。

所有数据集在这里被压成同一个 `AffectRecord`，训练脚本不需要知道任何数据集的细节。

⚠️ §8.1 的红线在这里用代码强制：`load_golden_set()` 只接受 `source == "real_session"`
的人工标注条目。开源数据和蒸馏数据**永远进不了评估集**。
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from affect.targets import REGRESSION_TARGETS  # noqa: E402
from affect.text_format import build_l1_input  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


@dataclass
class AffectRecord:
    """统一记录。`text` 已经是 §3.1 的 L1 输入格式。"""

    text: str
    dataset: str
    native_label: str = ""
    valence: float | None = None
    arousal: float | None = None
    intensity: float | None = None
    # 仅 stage2 使用：UserMove 的回归目标 + 指向性
    targets: dict[str, float] | None = None
    directed_at_agent: bool | None = None
    teacher_logits: list[float] | None = None
    # VAD 监督是否来自人工标注。先验推来的 VAD 权重要压低，否则模型会去拟合一张查表。
    vad_is_human: bool = False
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 情感标签 → VAD / intensity 先验
#
# 来源：中文情感词典与 Russell 环形模型的常规取值，用于给 stage1 的回归头一个弱监督
# 信号。**这是先验不是真值**，因此 vad_is_human=False，训练时权重会被压到 0.3。
# ---------------------------------------------------------------------------
VAD_PRIORS: dict[str, tuple[float, float, float]] = {
    # label: (valence, arousal, intensity)
    "neutral": (0.00, 0.20, 0.15),
    "happy": (0.70, 0.55, 0.60),
    "happiness": (0.70, 0.55, 0.60),
    "joy": (0.70, 0.60, 0.62),
    "like": (0.55, 0.40, 0.45),
    "positive": (0.60, 0.45, 0.50),
    "angry": (-0.60, 0.85, 0.75),
    "anger": (-0.60, 0.85, 0.75),
    "sad": (-0.65, 0.30, 0.60),
    "sadness": (-0.65, 0.30, 0.60),
    "depress": (-0.60, 0.25, 0.55),
    "fear": (-0.60, 0.75, 0.70),
    "surprise": (0.10, 0.70, 0.50),
    "disgust": (-0.55, 0.60, 0.60),
    "negative": (-0.55, 0.50, 0.55),
    "worried": (-0.45, 0.55, 0.55),
    "anxious": (-0.50, 0.65, 0.62),
    "grateful": (0.60, 0.40, 0.50),
    "relaxed": (0.45, 0.20, 0.35),
    "astonished": (0.05, 0.75, 0.55),
}

# 开源情感标签 → StrategyLabel。**只用于 bootstrap（冷启动跑通训练链路）**，
# 不得用于 §8.1 的评估集：这个映射本身携带了系统性偏差（例如 angry 在客服场景里
# 多是 frustration，在微博语料里可能只是宣泄）。
# 开源情感标签 → UserMove 回归目标的弱映射。**只用于 bootstrap**。
# 这个映射本身携带系统性偏差：微博语料里的 angry 多是对世界的宣泄，
# 而客服语境下的 angry 是指向 agent 的敌意，两者的 directed_at_agent 完全不同。
BOOTSTRAP_MOVE_MAP: dict[str, dict[str, float]] = {
    #              affiliation dominance intimacy distress intensity
    "neutral":    {"affiliation_bid": 0.00, "dominance_bid": 0.05, "intimacy_bid": 0.05, "distress_level": 0.05, "intensity": 0.15},
    "surprise":   {"affiliation_bid": 0.05, "dominance_bid": 0.00, "intimacy_bid": 0.05, "distress_level": 0.15, "intensity": 0.55},
    "astonished": {"affiliation_bid": 0.05, "dominance_bid": 0.00, "intimacy_bid": 0.05, "distress_level": 0.15, "intensity": 0.55},
    "happy":      {"affiliation_bid": 0.55, "dominance_bid": 0.00, "intimacy_bid": 0.25, "distress_level": 0.00, "intensity": 0.60},
    "happiness":  {"affiliation_bid": 0.55, "dominance_bid": 0.00, "intimacy_bid": 0.25, "distress_level": 0.00, "intensity": 0.60},
    "joy":        {"affiliation_bid": 0.60, "dominance_bid": 0.00, "intimacy_bid": 0.28, "distress_level": 0.00, "intensity": 0.62},
    "like":       {"affiliation_bid": 0.65, "dominance_bid": -0.05, "intimacy_bid": 0.40, "distress_level": 0.00, "intensity": 0.50},
    "positive":   {"affiliation_bid": 0.55, "dominance_bid": 0.00, "intimacy_bid": 0.25, "distress_level": 0.00, "intensity": 0.50},
    "grateful":   {"affiliation_bid": 0.60, "dominance_bid": -0.20, "intimacy_bid": 0.25, "distress_level": 0.00, "intensity": 0.45},
    "relaxed":    {"affiliation_bid": 0.35, "dominance_bid": 0.00, "intimacy_bid": 0.15, "distress_level": 0.00, "intensity": 0.30},
    "angry":      {"affiliation_bid": -0.70, "dominance_bid": 0.55, "intimacy_bid": 0.05, "distress_level": 0.25, "intensity": 0.80},
    "anger":      {"affiliation_bid": -0.70, "dominance_bid": 0.55, "intimacy_bid": 0.05, "distress_level": 0.25, "intensity": 0.80},
    "disgust":    {"affiliation_bid": -0.65, "dominance_bid": 0.35, "intimacy_bid": 0.05, "distress_level": 0.20, "intensity": 0.70},
    "sad":        {"affiliation_bid": 0.05, "dominance_bid": -0.35, "intimacy_bid": 0.10, "distress_level": 0.80, "intensity": 0.65},
    "sadness":    {"affiliation_bid": 0.05, "dominance_bid": -0.35, "intimacy_bid": 0.10, "distress_level": 0.80, "intensity": 0.65},
    "depress":    {"affiliation_bid": 0.00, "dominance_bid": -0.40, "intimacy_bid": 0.10, "distress_level": 0.75, "intensity": 0.55},
    "fear":       {"affiliation_bid": 0.05, "dominance_bid": -0.45, "intimacy_bid": 0.10, "distress_level": 0.85, "intensity": 0.75},
    "worried":    {"affiliation_bid": 0.05, "dominance_bid": -0.30, "intimacy_bid": 0.10, "distress_level": 0.65, "intensity": 0.55},
    "anxious":    {"affiliation_bid": 0.05, "dominance_bid": -0.35, "intimacy_bid": 0.10, "distress_level": 0.72, "intensity": 0.62},
    "negative":   {"affiliation_bid": -0.35, "dominance_bid": 0.10, "intimacy_bid": 0.05, "distress_level": 0.50, "intensity": 0.55},
}


def _extract_targets(row: dict[str, Any]) -> dict[str, float] | None:
    """从标注 JSONL 里取回归目标。缺任何一项都视为未标注。"""
    if not all(k in row and row[k] is not None for k in REGRESSION_TARGETS):
        return None
    return {k: float(row[k]) for k in REGRESSION_TARGETS}


def _prior(label: str) -> tuple[float | None, float | None, float | None]:
    v = VAD_PRIORS.get(label.lower())
    return v if v else (None, None, None)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"下载 {url} → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "affect-system/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


# ---------------------------------------------------------------------------
# 阶段一数据集
# ---------------------------------------------------------------------------

EWECT_FILES = {
    "train": "usual_train.txt",
    "eval": "usual_eval_labeled.txt",
    "test": "usual_test_labeled.txt",
}
EWECT_BASE = "https://huggingface.co/datasets/hecongqing/EWECT_weibo_senti/resolve/main/"


def load_ewect(split: str = "train", auto_download: bool = True) -> list[AffectRecord]:
    """SMP2020-EWECT（usual 子集，6 类，train ~27.8k）。HF 上可直接下载。"""
    fname = EWECT_FILES[split]
    path = RAW_DIR / "ewect" / fname
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(f"缺少 {path}")
        _download(EWECT_BASE + fname, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[AffectRecord] = []
    for row in data:
        text = str(row.get("content") or "").strip()
        label = str(row.get("label") or "").strip()
        if not text or not label:
            continue
        v, a, i = _prior(label)
        out.append(
            AffectRecord(
                text=build_l1_input(text),
                dataset="ewect",
                native_label=label,
                valence=v,
                arousal=a,
                intensity=i,
            )
        )
    return out


def load_ocemotion(path: str | Path | None = None) -> list[AffectRecord]:
    """OCEMOTION（~3.5 万，7 类）。

    需要手动获取：天池 CCF「中文预训练模型泛化能力挑战赛」OCEMOTION 赛道，
    放到 data/raw/ocemotion/train.csv（TSV：id \\t text \\t label）。
    """
    p = Path(path) if path else RAW_DIR / "ocemotion" / "train.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"未找到 OCEMOTION：{p}\n"
            "  该数据集需要手动下载（天池赛题，无公开直链）。格式：id\\ttext\\tlabel"
        )
    label_zh = {
        "sadness": "sad",
        "happiness": "happy",
        "disgust": "disgust",
        "anger": "angry",
        "like": "like",
        "surprise": "surprise",
        "fear": "fear",
    }
    out: list[AffectRecord] = []
    with p.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            text, label = row[1].strip(), label_zh.get(row[2].strip(), row[2].strip())
            if not text:
                continue
            v, a, i = _prior(label)
            out.append(
                AffectRecord(
                    text=build_l1_input(text),
                    dataset="ocemotion",
                    native_label=label,
                    valence=v,
                    arousal=a,
                    intensity=i,
                )
            )
    return out


def load_cped(directory: str | Path | None = None) -> list[AffectRecord]:
    """CPED（12K+ 对话，13 类细粒度情感）。

    需要手动获取：https://github.com/scutcyr/CPED（HF 上的 scutcyr/CPED 是空仓库）。
    放到 data/raw/cped/{train,valid,test}.csv。

    CPED 是**对话**数据，因此这里会构造 §3.1 的双句输入（上一轮 agent 回复 = 对方上一句）。
    """
    d = Path(directory) if directory else RAW_DIR / "cped"
    files = sorted(d.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"未找到 CPED：{d}/*.csv\n"
            "  从 https://github.com/scutcyr/CPED 获取后放入该目录。"
        )
    out: list[AffectRecord] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            text_col = cols.get("utterance") or cols.get("text")
            emo_col = cols.get("emotion") or cols.get("sentiment")
            dlg_col = cols.get("dialogue_id") or cols.get("dialog_id")
            if not text_col or not emo_col:
                raise ValueError(f"{f} 缺少 Utterance/Emotion 列，实际列：{reader.fieldnames}")
            prev_by_dialogue: dict[str, str] = {}
            for row in reader:
                text = (row[text_col] or "").strip()
                label = (row[emo_col] or "").strip().lower()
                if not text:
                    continue
                dlg = (row.get(dlg_col) or "") if dlg_col else ""
                prev = prev_by_dialogue.get(dlg, "")
                v, a, i = _prior(label)
                out.append(
                    AffectRecord(
                        text=build_l1_input(text, prev),
                        dataset="cped",
                        native_label=label,
                        valence=v,
                        arousal=a,
                        intensity=i,
                    )
                )
                prev_by_dialogue[dlg] = text
    return out


def load_m3ed(directory: str | Path | None = None) -> list[AffectRecord]:
    """M3ED（24,449 句，7 类）。需要手动获取：https://github.com/AIM3-RUC/RUCM3ED

    只用文本模态（§0.3 明确排除多模态）。放到 data/raw/m3ed/*.json 或 *.jsonl。
    """
    d = Path(directory) if directory else RAW_DIR / "m3ed"
    files = sorted([*d.glob("*.json"), *d.glob("*.jsonl")])
    if not files:
        raise FileNotFoundError(
            f"未找到 M3ED：{d}/*.json[l]\n  从 https://github.com/AIM3-RUC/RUCM3ED 获取。"
        )
    out: list[AffectRecord] = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        rows: Iterable[dict[str, Any]]
        if f.suffix == ".jsonl":
            rows = (json.loads(x) for x in raw.splitlines() if x.strip())
        else:
            parsed = json.loads(raw)
            rows = parsed if isinstance(parsed, list) else parsed.values()
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or row.get("Utterance") or row.get("utterance") or "").strip()
            label = str(row.get("emotion") or row.get("Emotion") or "").strip().lower()
            if not text or not label:
                continue
            v, a, i = _prior(label)
            out.append(
                AffectRecord(
                    text=build_l1_input(text),
                    dataset="m3ed",
                    native_label=label,
                    valence=v,
                    arousal=a,
                    intensity=i,
                )
            )
    return out


def load_simplifyweibo(
    path: str | Path | None = None, auto_download: bool = True, limit: int | None = 40000
) -> list[AffectRecord]:
    """simplifyweibo_4_moods（36 万，4 类：喜悦/愤怒/厌恶/低落）。HF 可直接下载。

    体量远大于其他集，默认下采样到 4 万，避免它主导 stage1。
    """
    p = Path(path) if path else RAW_DIR / "simplifyweibo" / "simplifyweibo_4_moods.csv"
    if not p.exists():
        if not auto_download:
            raise FileNotFoundError(f"缺少 {p}")
        _download(
            "https://huggingface.co/datasets/dirtycomputer/simplifyweibo_4_moods/"
            "resolve/main/simplifyweibo_4_moods.csv",
            p,
        )
    id2label = {0: "happy", 1: "angry", 2: "disgust", 3: "depress"}
    out: list[AffectRecord] = []
    with p.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for n, row in enumerate(reader):
            if limit and len(out) >= limit:
                break
            # 每隔几条取一条，保证覆盖整个文件而不是只拿开头（文件按标签排序）
            if limit and n % 9:
                continue
            try:
                label = id2label[int(row.get("label", -1))]
            except (ValueError, KeyError):
                continue
            text = (row.get("review") or "").strip()
            if not text:
                continue
            v, a, i = _prior(label)
            out.append(
                AffectRecord(
                    text=build_l1_input(text),
                    dataset="simplifyweibo",
                    native_label=label,
                    valence=v,
                    arousal=a,
                    intensity=i,
                )
            )
    return out


STAGE1_LOADERS = {
    "ewect": load_ewect,
    "simplifyweibo": load_simplifyweibo,
    "ocemotion": load_ocemotion,
    "cped": load_cped,
    "m3ed": load_m3ed,
}

# 默认组合：这两个能直接下载，其余需手动获取（缺失时跳过并打印提示）
STAGE1_DEFAULT = ("ewect", "simplifyweibo")


def load_stage1(
    datasets: Iterable[str] = STAGE1_DEFAULT, strict: bool = False
) -> dict[str, list[AffectRecord]]:
    """返回 {dataset_name: records}。§3.3 阶段一「各数据集用各自原生标签」。"""
    out: dict[str, list[AffectRecord]] = {}
    for name in datasets:
        loader = STAGE1_LOADERS.get(name)
        if loader is None:
            raise KeyError(f"未知数据集 {name!r}，可用：{sorted(STAGE1_LOADERS)}")
        try:
            records = loader()
        except FileNotFoundError as exc:
            if strict:
                raise
            print(f"! 跳过 {name}：{exc}", file=sys.stderr)
            continue
        if records:
            out[name] = records
            labels = sorted({r.native_label for r in records})
            print(f"  {name}: {len(records)} 条, {len(labels)} 类 {labels}")
    if not out:
        raise RuntimeError("阶段一没有任何可用数据集")
    return out


# ---------------------------------------------------------------------------
# 阶段二数据集（自有标注 + 蒸馏）
# ---------------------------------------------------------------------------


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_annotations(
    path: str | Path,
    require_real_session: bool = False,
    allow_missing_vad: bool = True,
) -> list[AffectRecord]:
    """读标注站导出的 JSONL（data/exports/*.jsonl）。"""
    out: list[AffectRecord] = []
    for row in _read_jsonl(path):
        targets = _extract_targets(row)
        if targets is None:
            continue  # 冲突条目/未标注条目
        if require_real_session and row.get("source") != "real_session":
            raise ValueError(
                f"{path} 含非真实会话条目（source={row.get('source')!r}）。"
                " §8.1：评估集绝不能包含开源数据或蒸馏数据。"
            )
        v, a, i = row.get("valence"), row.get("arousal"), row.get("intensity")
        if not allow_missing_vad and None in (v, a, i):
            continue
        out.append(
            AffectRecord(
                text=build_l1_input(row["utterance"], row.get("last_agent_reply") or ""),
                dataset=str(row.get("source") or "annotated"),
                native_label="move",
                targets=targets,
                directed_at_agent=bool(row.get("directed_at_agent", True)),
                valence=v,
                arousal=a,
                intensity=i,
                teacher_logits=row.get("teacher_logits"),
                vad_is_human=v is not None,
                meta={"item_id": row.get("item_id"), "split": row.get("split")},
            )
        )
    return out


def load_golden_set(path: str | Path | None = None) -> list[AffectRecord]:
    """§8.1 评估集。**强制**只接受真实会话的人工标注。"""
    p = Path(path) if path else ROOT / "eval" / "fixtures" / "golden_set.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"未找到 golden set：{p}\n"
            "  用标注站标 300–500 条真实会话，导出 data/exports/golden_set.jsonl 后拷到这里。"
        )
    records = load_annotations(p, require_real_session=True)
    if not records:
        raise ValueError(f"{p} 里没有可用条目")
    return records


def load_distilled(path: str | Path) -> list[AffectRecord]:
    """教师蒸馏产出（含 soft logits，§3.4.3）。"""
    out: list[AffectRecord] = []
    for row in _read_jsonl(path):
        targets = _extract_targets(row)
        if targets is None:
            continue
        out.append(
            AffectRecord(
                text=build_l1_input(row["utterance"], row.get("last_agent_reply") or ""),
                dataset="distilled",
                native_label="move",
                targets=targets,
                directed_at_agent=bool(row.get("directed_at_agent", True)),
                valence=row.get("valence"),
                arousal=row.get("arousal"),
                intensity=row.get("intensity"),
                teacher_logits=row.get("teacher_logits"),
                vad_is_human=False,
                weight=float(row.get("weight", 1.0)),
                meta={"teacher": row.get("teacher")},
            )
        )
    return out


def bootstrap_stage2_from_stage1(
    records: Iterable[AffectRecord], limit: int | None = None
) -> list[AffectRecord]:
    """用开源情感标签弱映射出 UserMove 目标，仅用于**冷启动验证训练链路**。

    产出打上 dataset='bootstrap'，任何评估脚本看到这个来源都应拒绝使用 ——
    微博语料里的 angry 在客服语境下会被误映射，且模型会学得很自信。
    """
    out: list[AffectRecord] = []
    for r in records:
        t = BOOTSTRAP_MOVE_MAP.get(r.native_label.lower())
        if t is None:
            continue
        out.append(
            AffectRecord(
                text=r.text,
                dataset="bootstrap",
                native_label=r.native_label,
                targets=dict(t),
                directed_at_agent=True,
                vad_is_human=False,
                weight=0.6,  # 弱标签降权
                meta={"origin": r.dataset},
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def label_distribution(records: Iterable[AffectRecord], key: str = "native_label") -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        k = str(getattr(r, key) or "?")
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
