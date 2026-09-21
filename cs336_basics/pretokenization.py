"""Shared GPT-style pretokenization for text streams and BPE training."""

from collections import deque
from collections.abc import Iterable, Iterator
from itertools import chain

import regex as re


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pretok_re = re.compile(PATTERN)


def iter_pretokens(
    chunks: Iterable[str], special_tokens: list[str] | None = None
) -> Iterator[tuple[str, bool]]:
    """Yield (pretoken, is_special) as if the input strings were concatenated.

    Special tokens delimit ordinary text before the GPT regex is applied. Only
    an unresolved suffix is retained between chunks; the input is consumed lazily.
    """
    specials = sorted(set(special_tokens or []), key=len, reverse=True)
    if any(not token for token in specials):
        raise ValueError("Special tokens must be nonempty strings")
    special_re = re.compile("|".join(re.escape(token) for token in specials)) if specials else None
    max_special_length = max(map(len, specials), default=1)
    buffer = ""

    # The sentinel flushes the final suffix without fetching the input eagerly.
    for chunk in chain(chunks, (None,)):
        final = chunk is None
        if not final:
            if not chunk:
                continue
            buffer += chunk

        consumed = 0
        if special_re is not None:
            for match in special_re.finditer(buffer):
                # A longer overlapping special, or an earlier partial special,
                # may still be completed by the next chunk. Wait for lookahead.
                if not final and match.start() + max_special_length > len(buffer):
                    break
                for ordinary in pretok_re.finditer(buffer, consumed, match.start()):
                    yield ordinary.group(0), False
                yield match.group(0), True
                consumed = match.end()
        buffer = buffer[consumed:]

        if final:
            for match in pretok_re.finditer(buffer):
                yield match.group(0), False
            return

        # Do not emit text that may become the start of a special token.
        safe_end = len(buffer) - max_special_length + 1
        pending = deque()
        consumed = 0
        # Exclude unresolved special-token lookahead from the regex itself: a
        # possible special must not make preceding whitespace look nonterminal.
        for match in pretok_re.finditer(buffer, 0, max(0, safe_end)):
            pending.append(match)
            if len(pending) <= 2:
                continue
            ready = pending.popleft()
            if ready.end() > safe_end:
                break
            yield ready.group(0), False
            consumed = ready.end()

        # Keeping two regex matches also preserves incomplete contractions:
        # "'l" currently matches "'", "l", but the next "l" changes both.
        # Trailing whitespace likewise needs the following character to settle.
        buffer = buffer[consumed:]
