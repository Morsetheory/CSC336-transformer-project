import os
import heapq
import codecs
from typing import BinaryIO
from collections import defaultdict
from cs336_basics.pretokenization import PATTERN, iter_pretokens, pretok_re

def find_chunk_boundaries(
      file: BinaryIO,
      desired_num_chunks: int,
      split_special_tokens: list[bytes],
      ) -> list[int]:
      """
      Chunk the file into parts that can be counted independently.
      May return fewer chunks if the boundaries end up overlapping.
      """
      assert all(isinstance(tok, bytes) for tok in split_special_tokens), "Special tokens must be bytestrings"

      # Get total file size in bytes
      file.seek(0, os.SEEK_END)
      file_size = file.tell()
      file.seek(0)

      chunk_size = file_size // desired_num_chunks

      # Initial guesses for chunk boundary locations, uniformly spaced
      # Chunks start on previous index, don't include last index
      chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
      chunk_boundaries[-1] = file_size

      if not split_special_tokens:
          return sorted(set(chunk_boundaries))

      mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
      max_special_len = max(len(tok) for tok in split_special_tokens)

      for bi in range(1, len(chunk_boundaries) - 1):
          initial_position = chunk_boundaries[bi]
          file.seek(initial_position)
          scan_position = initial_position
          tail = b""
          while True:
              mini_chunk = file.read(mini_chunk_size)

              if mini_chunk == b"":
                  chunk_boundaries[bi] = file_size
                  break

              window = tail + mini_chunk
              window_start = scan_position - len(tail)

              found_positions = []
              for token in split_special_tokens:
                  pos = window.find(token)
                  if pos != -1:
                      found_positions.append(pos)

              if found_positions:
                  chunk_boundaries[bi] = window_start + min(found_positions)
                  break

              scan_position += len(mini_chunk)
              if max_special_len > 1:
                  tail = window[-(max_special_len - 1):]
              else:
                  tail = b""

      # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
      return sorted(set(chunk_boundaries))
    
def iter_pretokens_from_chunk_text(text: str, special_split_re):
      """
      Yield GPT-style pre-tokens from text, but split on special tokens first
      so no pretoken crosses a document boundary.
      """
      segments = [text] if special_split_re is None else special_split_re.split(text) # special tokens removed here
      for seg in segments:
          if not seg:
              continue
          for m in pretok_re.finditer(seg):
              yield m.group(0)


def iter_pretokens_from_file(
      input_path,
      special_tokens: list[str],
      chunk_size: int = 1 << 20,
      ):
      """
      Stream pre-tokens from a UTF-8 file without materializing large chunks in memory.
      Special tokens are recognized and skipped so BPE merges never cross them.
      """
      if chunk_size <= 0:
          raise ValueError("chunk_size must be positive")

      def text_chunks():
          decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
          with open(input_path, "rb") as file:
              while raw := file.read(chunk_size):
                  yield decoder.decode(raw)
              yield decoder.decode(b"", final=True)

      for token, is_special in iter_pretokens(text_chunks(), special_tokens):
          if not is_special:
              yield token.encode("utf-8")
              
def merge_pair_in_ids(ids, pair, new_id):
      a, b = pair
      out = []
      i = 0
      while i < len(ids):
          # if current and next symbol match the pair, merge them
          if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
              out.append(new_id)
              i += 2
          else:
              out.append(ids[i])
              i += 1
      return out

def build_pair_hist(ids):
      hist = defaultdict(int)
      for left, right in zip(ids[:-1], ids[1:]):
          hist[(left, right)] += 1
      return hist


def has_pair(ids, pair):
      for left, right in zip(ids[:-1], ids[1:]):
          if left == pair[0] and right == pair[1]:
              return True
      return False

def pop_best_pair(pair_counts, pair_heap, vocab):
      """
      Pop the best valid pair using lazy invalidation.
      Ordering must match max(pair_counts, key=lambda p: (count, vocab[a], vocab[b])).
      """
      while pair_heap:
          neg_count, pair = heapq.heappop(pair_heap)
          count = -neg_count
          current = pair_counts.get(pair)

          # Skip stale heap entries.
          if current is None or current != count:
              continue

          candidates = [pair]
          while pair_heap and -pair_heap[0][0] == count:
              neg_count2, pair2 = heapq.heappop(pair_heap)
              current2 = pair_counts.get(pair2)
              if current2 is None or current2 != -neg_count2:
                  continue
              candidates.append(pair2)

          best = max(candidates, key=lambda p: (count, vocab[p[0]], vocab[p[1]]))

          # Push back non-selected valid candidates for future rounds.
          for cand in candidates:
              if cand != best:
                  heapq.heappush(pair_heap, (-count, cand))

          return best

      return None

def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    pretok_freq = defaultdict(int)
    vocab = {x: bytes([x]) for x  in range(256)}
    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)
    file_size = os.path.getsize(input_path)
    low_memory = kwargs.get("low_memory")
    if low_memory is None:
        low_memory = file_size >= (512 << 20)

    for pretok in iter_pretokens_from_file(
        input_path=input_path,
        special_tokens=special_tokens,
        chunk_size=kwargs.get("chunk_size", 1 << 20),
    ):
        pretok_freq[pretok] += 1

    pretok_entries = list(pretok_freq.items())
    pretok_freqs = [freq for _, freq in pretok_entries]
    pretok_ids = [list(pretok) for pretok, _ in pretok_entries]
    del pretok_freq

    pair_counts = defaultdict(int)
    pair_to_pretoks = None if low_memory else defaultdict(set)
    pair_heap = []

    for pretok_index, ids in enumerate(pretok_ids):
        hist = build_pair_hist(ids)
        freq = pretok_freqs[pretok_index]
        for pair, count in hist.items():
            pair_counts[pair] += count * freq
            if pair_to_pretoks is not None:
                pair_to_pretoks[pair].add(pretok_index)

    for pair, count in pair_counts.items():
        heapq.heappush(pair_heap, (-count, pair))

    for i in range(num_merges):
        if not pair_counts:
            break
        pair = pop_best_pair(pair_counts, pair_heap, vocab)
        if pair is None:
            break
        a, b = pair
        merges.append((vocab[a], vocab[b]))
        new_index = 256 + i
        vocab[new_index] = vocab[a] + vocab[b]

        if pair_to_pretoks is None:
            affected_pretoks = [index for index, ids in enumerate(pretok_ids) if has_pair(ids, pair)]
        else:
            affected_pretoks = list(pair_to_pretoks.get(pair, ()))
        if not affected_pretoks:
            continue
        for pretok_index in affected_pretoks:
            current_ids = pretok_ids[pretok_index]
            old_hist = build_pair_hist(current_ids)
            freq = pretok_freqs[pretok_index]

            for old_pair, old_count in old_hist.items():
                pair_counts[old_pair] -= old_count * freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                if pair_to_pretoks is not None and old_pair in pair_to_pretoks:
                    pair_to_pretoks[old_pair].discard(pretok_index)
                    if not pair_to_pretoks[old_pair]:
                        del pair_to_pretoks[old_pair]
                if old_pair in pair_counts:
                    heapq.heappush(pair_heap, (-pair_counts[old_pair], old_pair))

            new_ids = merge_pair_in_ids(current_ids, pair, new_index)
            pretok_ids[pretok_index] = new_ids
            new_hist = build_pair_hist(new_ids)

            for new_pair, new_count in new_hist.items():
                pair_counts[new_pair] += new_count * freq
                if pair_to_pretoks is not None:
                    pair_to_pretoks[new_pair].add(pretok_index)
                heapq.heappush(pair_heap, (-pair_counts[new_pair], new_pair))

    next_special_id = 256 + len(merges)
    for tok in special_tokens:
        vocab[next_special_id] = tok.encode("utf-8")
        next_special_id += 1
    return vocab, merges
