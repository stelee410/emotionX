"""纯 Python 的 BERT/ERNIE WordPiece 分词器。

存在理由：§11 要求生产环境不安装 torch，也不想为了一个 tokenizer 装 transformers。
export_onnx.py 会把 `vocab.txt` 一并导出到模型目录，运行时只读这个文件。

行为与 `BertTokenizer(do_lower_case=True)` 对齐（CJK 逐字切分 + 贪心最长匹配子词），
并在 `training/datasets/text_format.py` 上有一致性测试。
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
USER_TOKEN = "[USER]"
AGENT_TOKEN = "[AGENT]"


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_control(ch: str) -> bool:
    """注意：Unicode 未分配码位（category 'Cn'）也会走这条分支被丢弃。

    HF 的 fast tokenizer 对未分配码位会给 [UNK] 而不是丢弃 —— 这是本实现与训练侧
    唯一已知的行为差异。真实用户输入里不会出现未分配码位，因此不做对齐；
    tests/test_tokenization.py 用**已分配**的生僻字做对齐验证。
    """
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


class WordPieceTokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        do_lower_case: bool = True,
        max_input_chars_per_word: int = 100,
        special_tokens: tuple[str, ...] = (CLS, SEP, PAD, UNK, USER_TOKEN, AGENT_TOKEN),
    ) -> None:
        self.vocab = vocab
        self.do_lower_case = do_lower_case
        self.max_input_chars_per_word = max_input_chars_per_word
        # 只保留真实存在于 vocab 的特殊 token，避免把 [USER] 切成字符
        self.special_tokens = tuple(t for t in special_tokens if t in vocab)
        self.pad_id = vocab.get(PAD, 0)
        self.cls_id = vocab.get(CLS, 1)
        self.sep_id = vocab.get(SEP, 2)
        self.unk_id = vocab.get(UNK, 17963)

    # ---- 构造 ----
    @classmethod
    def from_vocab_file(cls, path: str | Path, **kwargs: object) -> WordPieceTokenizer:
        vocab: dict[str, int] = {}
        with Path(path).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 后出现的覆盖先出现的 —— 与 transformers BertTokenizer 一致。
                # ernie 的 vocab.txt 确实存在重复 token（会留下 id 空洞），
                # 这里必须和训练侧用同一套规则，否则线上 token id 会错位。
                vocab[line.rstrip("\n")] = i
        return cls(vocab, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained_dir(cls, directory: str | Path, **kwargs: object) -> WordPieceTokenizer:
        d = Path(directory)
        vocab_file = d / "vocab.txt"
        if not vocab_file.exists():
            raise FileNotFoundError(f"{d} 下缺少 vocab.txt（export_onnx.py 应一并导出）")
        return cls.from_vocab_file(vocab_file, **kwargs)

    # ---- 分词 ----
    def _split_specials(self, text: str) -> list[str]:
        chunks = [text]
        for st in self.special_tokens:
            nxt: list[str] = []
            for chunk in chunks:
                if chunk in self.special_tokens:
                    nxt.append(chunk)
                    continue
                parts = chunk.split(st)
                for i, part in enumerate(parts):
                    if i:
                        nxt.append(st)
                    if part:
                        nxt.append(part)
            chunks = nxt
        return chunks

    def _basic_tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for chunk in self._split_specials(text):
            if chunk in self.special_tokens:
                out.append(chunk)
                continue
            cleaned = "".join(
                " " if (_is_control(c) or c in ("\t", "\n", "\r")) else c for c in chunk
            )
            buf = ""
            for ch in cleaned:
                if _is_cjk(ch) or _is_punct(ch):
                    if buf:
                        out.append(buf)
                        buf = ""
                    out.append(ch)
                elif ch.isspace():
                    if buf:
                        out.append(buf)
                        buf = ""
                else:
                    buf += ch
            if buf:
                out.append(buf)
        if self.do_lower_case:
            out = [
                t if t in self.special_tokens else _strip_accents(t.lower()) for t in out
            ]
        return [t for t in out if t]

    def _wordpiece(self, token: str) -> list[str]:
        if token in self.special_tokens or token in self.vocab:
            return [token]
        if len(token) > self.max_input_chars_per_word:
            return [UNK]
        sub_tokens: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            cur: str | None = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    cur = piece
                    break
                end -= 1
            if cur is None:
                return [UNK]
            sub_tokens.append(cur)
            start = end
        return sub_tokens

    def tokenize(self, text: str) -> list[str]:
        return [sub for tok in self._basic_tokenize(text) for sub in self._wordpiece(tok)]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    # ---- 编码 ----
    def encode(
        self, text: str, max_length: int = 128, add_special_tokens: bool = True
    ) -> dict[str, list[int]]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            keep = max_length - 2
            tokens = tokens[:keep]
            tokens = [CLS, *tokens, SEP]
        else:
            tokens = tokens[:max_length]
        ids = self.convert_tokens_to_ids(tokens)
        attention = [1] * len(ids)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            attention += [0] * pad_len
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "token_type_ids": [0] * max_length,
        }

    def encode_batch(
        self, texts: list[str], max_length: int = 128
    ) -> dict[str, list[list[int]]]:
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        return {k: [e[k] for e in encoded] for k in ("input_ids", "attention_mask", "token_type_ids")}
