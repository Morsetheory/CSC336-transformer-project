"""Regression tests for pretokenization across arbitrary stream boundaries."""

from io import StringIO

import pytest
import regex as re
import tiktoken

from cs336_basics.bpe import iter_pretokens_from_file
from cs336_basics.pretokenization import iter_pretokens
from .test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path


GPT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def whole_text_pretokens(text, specials):
    """Independent whole-text oracle: remove specials before the GPT regex."""
    if specials:
        split_re = re.compile("(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")")
        parts = split_re.split(text)
    else:
        parts = [text]
    result = []
    for part in parts:
        if part in specials:
            result.append((part, True))
        else:
            result.extend((token, False) for token in re.findall(GPT_PATTERN, part))
    return result


@pytest.mark.parametrize(
    "text,specials",
    [
        ("we'll they've you're I'm I'd can't 'l 'v 'r", []),
        ("hello\n\n   world \t\n\n next\t\t終わり 🙃 café ", []),
        ("hello.\n\n<|endoftext|> world!<|endoftext|> ", ["<|endoftext|>"]),
        ("x\n\n<|endoftext|><|endoftext|>next<|endoftext|>", ["<|endoftext|>", "<|endoftext|><|endoftext|>"]),
        ("zero abc abcd abcdef abc de done", ["ab", "abc", "abcdef", "de"]),
        (" \ta.bc rest a.b", ["a.bc"]),
        ("a\n\nb  c d e  c done", ["\n", "\n\n", " c", " c d"]),
        (".[a?]hi[a?]there\\[last", [".[", ".[a?]", "[", "\\["]),
        ("before<|unfinished", ["<|unfinished|>"]),
        ("", ["<|endoftext|>"]),
    ],
)
def test_pretokens_match_whole_text_at_every_split(text, specials):
    expected = whole_text_pretokens(text, specials)
    assert list(iter_pretokens(text, specials)) == expected  # One character per chunk.
    for split in range(len(text) + 1):
        assert list(iter_pretokens(["", text[:split], "", text[split:], ""], specials)) == expected


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 13, 64])
def test_bpe_byte_chunks_preserve_unicode_contractions_and_specials(tmp_path, chunk_size):
    text = "we'll they've you're\n\n<|endoftext|> café🙃 終わり.\n\n<|endoftext|>next"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    specials = ["<|endoftext|>"]
    expected = [token.encode("utf-8") for token, is_special in whole_text_pretokens(text, specials) if not is_special]
    assert list(iter_pretokens_from_file(path, specials, chunk_size)) == expected


@pytest.mark.parametrize("before,after", [("we'l", "l more"), (".<|endo", "ftext|>next")])
def test_bpe_default_megabyte_boundary(tmp_path, before, after):
    prefix_length = (1 << 20) - len(before)
    prefix = "x\n" * (prefix_length // 2) + " " * (prefix_length % 2)
    text = prefix + before + after
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    specials = ["<|endoftext|>"]
    expected = [token.encode("utf-8") for token, is_special in whole_text_pretokens(text, specials) if not is_special]
    assert list(iter_pretokens_from_file(path, specials)) == expected


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])


def test_encode_iterable_matches_tiktoken_across_lines_and_characters(gpt2_tokenizer):
    text = "we'll read\n\nnext\n\n<|endoftext|> café 🙃\n\n"
    expected = tiktoken.get_encoding("gpt2").encode(text, allowed_special={"<|endoftext|>"})
    assert list(gpt2_tokenizer.encode_iterable(StringIO(text))) == expected
    assert list(gpt2_tokenizer.encode_iterable(text)) == expected
    assert gpt2_tokenizer.encode(text) == expected


def test_encode_iterable_consumes_input_lazily(gpt2_tokenizer):
    consumed = []

    def chunks():
        consumed.append(1)
        yield "Hello world, this is a long enough first chunk. "
        consumed.append(2)
        yield "The next chunk."

    encoded = gpt2_tokenizer.encode_iterable(chunks())
    assert consumed == []
    first = next(encoded)
    assert consumed == [1]
    assert gpt2_tokenizer.decode([first] + list(encoded)) == "Hello world, this is a long enough first chunk. The next chunk."
