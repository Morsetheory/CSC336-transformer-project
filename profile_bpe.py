from pathlib import Path
import cProfile
import pstats
from cs336_basics.bpe import train_bpe

def main():
    train_bpe(
        input_path=Path("tests/fixtures/corpus.en"),
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

if __name__ == "__main__":
    cProfile.run("main()", "bpe_profile.prof")
    stats = pstats.Stats("bpe_profile.prof")
    stats.sort_stats("cumulative").print_stats(40)
