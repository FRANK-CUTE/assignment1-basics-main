import json
import os
from collections import Counter, defaultdict
from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merges_priority = {pair: i for i, pair in enumerate(merges)}
        current_max_id = max(self.vocab.keys()) if self.vocab else -1

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            special_pattern = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            self.pattern = re.compile(f"(?:{special_pattern})|{PAT}")
        else:
            self.pattern = re.compile(PAT)

        self.vocab_inv = {v: k for k, v in vocab.items()}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.vocab_inv:
                current_max_id += 1
                self.vocab[current_max_id] = st_bytes
                self.vocab_inv[st_bytes] = current_max_id
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            v_raw = json.load(f)
            vocab = {int(k): bytes.fromhex(v) for k, v in v_raw.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p1, p2 = line.strip().split()
                    merges.append((bytes.fromhex(p1), bytes.fromhex(p2)))

        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, word_bytes):
        if word_bytes in self.cache:
            return self.cache[word_bytes]
        tokens = [word_bytes[i:i + 1] for i in range(len(word_bytes))]
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = min(pairs, key=lambda p: self.merges_priority.get(p, float("inf")))
            if best_pair not in self.merges_priority:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        ids = [self.vocab_inv[t] for t in tokens]
        self.cache[word_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for m in self.pattern.finditer(text):
            chunk = m.group()
            if chunk in self.special_tokens:
                ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                ids.extend(self._bpe_merge(chunk.encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        bytes_seg = [self.vocab[i] for i in ids]
        return b"".join(bytes_seg).decode("utf-8", errors="replace")


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_patterns = "|".join(re.escape(s) for s in special_tokens) if special_tokens else None
    special_regex = re.compile(f"(?:{special_patterns})") if special_patterns else None

    # Pre-tokenization
    counts = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # 如果有 special tokens，先把 line 切成非 special token 块
            chunks = [line]  # 默认整行
            if special_regex:
                chunks = [c for c in special_regex.split(line) if c and c not in special_tokens]

            for chunk in chunks:
                for m in re.finditer(PAT, chunk):
                    word = m.group()
                    byte_word = word.encode("utf-8")
                    byte_tuple = tuple(byte_word[i:i + 1] for i in range(len(byte_word)))
                    counts[byte_tuple] += 1

    # 利用 counts 计算 pair_frequency
    pair_frequency = Counter()
    pair_to_words = defaultdict(set)
    for token_tuple, freq in counts.items():
        for index in range(len(token_tuple) - 1):
            pair = (token_tuple[index], token_tuple[index + 1])
            pair_frequency[(token_tuple[index], token_tuple[index + 1])] += freq
            pair_to_words[pair].add(token_tuple)

    # Merges
    merges = []
    while len(vocab) < vocab_size:
        if not pair_frequency: break

        best_pair = max(pair_frequency, key=lambda x: (pair_frequency[x], x))
        if pair_frequency[best_pair] <= 0:
            break

        # 把 best_pair 加入 merges和vocab
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # 局部更新逻辑：只处理包含 best_pair 的 word_tuples
        affected_words = pair_to_words[best_pair]
        for old_word_tuple in list(affected_words):
            freq = counts[old_word_tuple]

            for i in range(len(old_word_tuple) - 1):
                p = (old_word_tuple[i], old_word_tuple[i + 1])
                pair_frequency[p] -= freq
                pair_to_words[p].discard(old_word_tuple)

            # 根据 best_pair 重构所有被best_pair影响的词元组
            new_word_list = []
            i = 0
            while i < len(old_word_tuple):
                if i < len(old_word_tuple) - 1 and (old_word_tuple[i], old_word_tuple[i + 1]) == best_pair:
                    new_word_list.append(new_token)
                    i += 2
                else:
                    new_word_list.append(old_word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_word_list)

            # counts更新
            del counts[old_word_tuple]
            counts[new_word_tuple] += freq

            # 更新 pair 的频率和 被best_pair影响的词元组
            for i in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[i], new_word_tuple[i + 1])
                pair_frequency[p] += freq
                pair_to_words[p].add(new_word_tuple)

        # best_pair是老pair，应该清理
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
        if best_pair in pair_frequency:
            del pair_frequency[best_pair]

    return vocab, merges
