"""纯 Python WordPiece 必须与训练侧的 transformers tokenizer 逐 id 一致。

不一致会在生产上悄悄降精度（token id 错位 → encoder 看到的是另一句话），
且没有任何报错。所以这个测试是 L1 上线前的必过项。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from affect.text_format import build_l1_input
from affect.tokenization import WordPieceTokenizer

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    "你好",
    "我不想活了",
    "又错了，说了多少遍了？？",
    "帮我查一下 CT 报告，编号 A12345",
    "血糖12.8mg/dL算高吗",
    "Hello world，混合English和中文",
    "  多余的   空格  ",
    "emoji😊也要能处理",
    "特别难受，一整晚都没睡",
    "①②③ 特殊符号 ——「」《》",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "",
    "一",
    "ＡＢＣ全角字母",
    "Ｎｉ Hǎo 带声调的拼音",
]

PAIR_CASES = [
    ("我好难受", "要不要先聊聊今天发生了什么"),
    ("还是不行", ""),
    ("这个药一天几次", "建议随餐服用，具体请遵医嘱。" * 6),
]


@pytest.fixture(scope="module")
def hf_tokenizer():
    """训练侧真正用的那一个 tokenizer（含 [USER]/[AGENT] 特殊 token）。"""
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    from training.model import load_tokenizer

    return load_tokenizer()


@pytest.fixture(scope="module")
def vocab_file(hf_tokenizer, tmp_path_factory) -> Path:
    """从训练侧 tokenizer 导出 vocab.txt —— 与 export_onnx 走同一个函数。"""
    from training.model import write_vocab_file

    exported = ROOT / "artifacts" / "l1_onnx" / "vocab.txt"
    if exported.exists():
        return exported
    return write_vocab_file(hf_tokenizer, tmp_path_factory.mktemp("vocab") / "vocab.txt")


@pytest.fixture(scope="module")
def ours(vocab_file: Path) -> WordPieceTokenizer:
    return WordPieceTokenizer.from_vocab_file(vocab_file)


def test_exported_vocab_line_number_equals_token_id(vocab_file: Path, hf_tokenizer) -> None:
    """行号 == token id。这条断言保护的是「线上 token 全体错位」这个静默故障。"""
    ours = WordPieceTokenizer.from_vocab_file(vocab_file)
    hf_vocab = hf_tokenizer.get_vocab()
    for token, idx in hf_vocab.items():
        assert ours.vocab.get(token) == idx, f"{token!r} 的 id 不一致"
    # 空洞被占位符填上，总行数 = max_id + 1
    assert len(ours.vocab) == max(hf_vocab.values()) + 1


def test_embedding_size_covers_max_token_id(hf_tokenizer) -> None:
    """回归测试：ernie vocab 有空洞，len(tokenizer) 不等于所需 embedding 行数。"""
    from training.model import embedding_size

    max_id = max(hf_tokenizer.get_vocab().values())
    assert embedding_size(hf_tokenizer) == max_id + 1
    assert embedding_size(hf_tokenizer) >= len(hf_tokenizer)
    # [AGENT] 必须落在 embedding 范围内
    agent_id = hf_tokenizer.convert_tokens_to_ids("[AGENT]")
    assert agent_id < embedding_size(hf_tokenizer)


@pytest.mark.parametrize("text", CASES)
def test_matches_transformers(text: str, ours: WordPieceTokenizer, hf_tokenizer) -> None:
    mine = ours.encode(text, max_length=128)
    theirs = hf_tokenizer(
        text, max_length=128, truncation=True, padding="max_length", return_token_type_ids=True
    )
    assert mine["input_ids"] == theirs["input_ids"], f"token id 不一致: {text!r}"
    assert mine["attention_mask"] == theirs["attention_mask"]


@pytest.mark.parametrize(("utterance", "reply"), PAIR_CASES)
def test_matches_transformers_on_l1_format(
    utterance: str, reply: str, ours: WordPieceTokenizer, hf_tokenizer
) -> None:
    """§3.1 的完整输入格式（含 [USER]/[SEP]/[AGENT]）也要一致。"""
    text = build_l1_input(utterance, reply)
    mine = ours.encode(text, max_length=128)
    theirs = hf_tokenizer(
        text, max_length=128, truncation=True, padding="max_length"
    )
    assert mine["input_ids"] == theirs["input_ids"], text


def test_special_tokens_are_single_ids(ours: WordPieceTokenizer) -> None:
    assert "[USER]" in ours.vocab and "[AGENT]" in ours.vocab
    assert ours.tokenize("[USER] 你好") == ["[USER]", "你", "好"]
    assert ours.tokenize("[AGENT] 在") == ["[AGENT]", "在"]


def test_padding_and_truncation(ours: WordPieceTokenizer) -> None:
    long = "很难受" * 200
    enc = ours.encode(long, max_length=32)
    assert len(enc["input_ids"]) == 32
    assert enc["input_ids"][0] == ours.cls_id
    assert enc["input_ids"][-1] == ours.sep_id
    assert sum(enc["attention_mask"]) == 32

    short = ours.encode("嗯", max_length=16)
    assert len(short["input_ids"]) == 16
    assert sum(short["attention_mask"]) == 3  # [CLS] 嗯 [SEP]
    assert short["input_ids"][3:] == [ours.pad_id] * 13


def test_batch_encode_shape(ours: WordPieceTokenizer) -> None:
    b = ours.encode_batch(["你好", "我很难受"], max_length=16)
    assert len(b["input_ids"]) == 2
    assert all(len(x) == 16 for x in b["input_ids"])
    assert set(b) == {"input_ids", "attention_mask", "token_type_ids"}


def test_unknown_chars_map_to_unk(ours: WordPieceTokenizer, hf_tokenizer) -> None:
    rare = "\u9fff\U00020000"  # 已分配但不在 vocab 的生僻 CJK 字
    mine = ours.encode(rare, max_length=8)["input_ids"]
    theirs = hf_tokenizer(rare, max_length=8, truncation=True, padding="max_length")["input_ids"]
    assert ours.unk_id in mine
    assert mine == theirs


def test_build_l1_input_format() -> None:
    assert build_l1_input("你好") == "[USER] 你好"
    assert build_l1_input("你好", "在的") == "[USER] 你好 [SEP] [AGENT] 在的"
    # 上一轮 agent 回复过长时保留尾部
    long_reply = "".join(str(i % 10) for i in range(500))
    out = build_l1_input("嗯", long_reply)
    assert out.endswith(long_reply[-128:])
    assert len(out) < 200
    # 换行不得破坏格式
    assert "\n" not in build_l1_input("第一行\n第二行", "回复\n换行")
