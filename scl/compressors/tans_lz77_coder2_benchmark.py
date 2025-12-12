"""
Benchmark comparing LZ77 with empirical Huffman vs tANS entropy coding.

Tests two variants:
- Baseline: empirical Huffman (original LZ77StreamsEncoder)
- tANS: tANS entropy coding for all LZ77 streams

Usage: python tans_lz77_coder2_benchmark.py -i file1 file2 ...
       python tans_lz77_coder2_benchmark.py --data-folder path/to/testfiles/
       python tans_lz77_coder2_benchmark.py --plot-header-comparison output.png
"""

import argparse
import os
import tempfile
import time
from typing import List, Tuple
import lzma
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from scl.compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from scl.compressors.huffman_coder import HuffmanEncoder
from scl.compressors.lz77 import (
    LZ77Encoder,
    LZ77Decoder,
    LZ77StreamsEncoder,
    LZ77StreamsDecoder,
    LZ77Sequence,
    DEFAULT_MIN_MATCH_LEN,
    DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
)
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint

# Import tANS implementation (hybrid version with fallback)
from tans_lz77_coder2 import (
    LZ77TANSStreamsEncoder,
    LZ77TANSStreamsDecoder,
)

ENCODED_BLOCK_SIZE_HEADER_BITS = 32


# ---------------------------------------------------------------------------
# LZ77 Encoder/Decoder using tANS
# ---------------------------------------------------------------------------


class LZ77EncoderTANS(LZ77Encoder):
    """LZ77 encoder using tANS for entropy coding."""

    def __init__(
        self,
        min_match_length: int = DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered: int = DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        initial_window: List[int] = None,
        table_log: int = 10,
    ):
        super().__init__(
            min_match_length=min_match_length,
            max_num_matches_considered=max_num_matches_considered,
            initial_window=initial_window,
        )
        self.streams_encoder = LZ77TANSStreamsEncoder(table_log=table_log)


class LZ77DecoderTANS(LZ77Decoder):
    """LZ77 decoder using tANS for entropy decoding."""

    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77TANSStreamsDecoder(table_log=table_log)


# ---------------------------------------------------------------------------
# Header overhead measurement
# ---------------------------------------------------------------------------


def compute_literal_header_bits_huffman(literals: List[int]) -> int:
    """Count header bits for empirical Huffman (counts array)."""
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    counts = DataBlock(literals).get_counts()
    for i in range(256):
        if i not in counts:
            counts[i] = 0
    counts_list = [counts[i] for i in range(256)]

    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    header_bits = ENCODED_BLOCK_SIZE_HEADER_BITS + len(counts_encoding)
    return header_bits


def compute_literal_header_bits_tans(literals: List[int], table_log: int = 10) -> int:
    """
    Count header bits for tANS (frequency table).

    Format: [num_unique_symbols (16 bits)] + [(symbol, freq) pairs (32+32 bits each)]
    """
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    from collections import Counter
    freqs = Counter(literals)

    # 16 bits for number of unique symbols
    header_bits = 16

    # 64 bits (32+32) per unique symbol for (symbol, frequency) pairs
    header_bits += len(freqs) * 64

    return header_bits


# ---------------------------------------------------------------------------
# Progress bar wrapper
# ---------------------------------------------------------------------------


def encode_file_with_progress(encoder, input_path, output_path, block_size=100_000):
    """Wrap encoder.encode_file with a progress bar."""
    file_size = os.path.getsize(input_path)
    filename = os.path.basename(input_path)

    class ProgressFileWrapper:
        def __init__(self, file_obj, pbar):
            self.file_obj = file_obj
            self.pbar = pbar

        def read(self, size=-1):
            data = self.file_obj.read(size)
            self.pbar.update(len(data))
            return data

        def __getattr__(self, name):
            return getattr(self.file_obj, name)

    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f"Compressing {filename}") as pbar:

        original_open = open

        def tracked_open(path, mode, *args, **kwargs):
            f = original_open(path, mode, *args, **kwargs)
            if path == input_path and 'r' in mode:
                return ProgressFileWrapper(f, pbar)
            return f

        import builtins
        builtins.open = tracked_open

        try:
            encoder.encode_file(input_path, output_path, block_size=block_size)
        finally:
            builtins.open = original_open


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def run_single_file_benchmark(path: str, block_size: int = 100_000, table_log: int = 10) -> dict:
    """
    Run benchmark on a single file.

    Returns:
        dict with benchmark results
    """
    raw_size = os.path.getsize(path)

    print(f"\n=== Benchmark on file: {path} ===")
    print(f"Raw size: {raw_size} bytes ({raw_size / 1024 / 1024:.2f} MB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Baseline: empirical Huffman
        base_enc = LZ77Encoder()
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        start_time = time.time()
        encode_file_with_progress(base_enc, path, base_encoded_path, block_size=block_size)
        baseline_encode_time = time.time() - start_time

        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)
        baseline_speed = (raw_size / 1024 / 1024) / baseline_encode_time if baseline_encode_time > 0 else 0

        # tANS variant
        tans_enc = LZ77EncoderTANS(table_log=table_log)
        tans_dec = LZ77DecoderTANS(table_log=table_log)

        tans_encoded_path = os.path.join(tmpdir, "tans.lz77")
        tans_decoded_path = os.path.join(tmpdir, "tans.dec")

        start_time = time.time()
        encode_file_with_progress(tans_enc, path, tans_encoded_path, block_size=block_size)
        tans_encode_time = time.time() - start_time

        tans_dec.decode_file(tans_encoded_path, tans_decoded_path)

        with open(path, "rb") as f_in, open(tans_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "tANS LZ77 decode mismatch!"

        tans_size = os.path.getsize(tans_encoded_path)
        tans_speed = (raw_size / 1024 / 1024) / tans_encode_time if tans_encode_time > 0 else 0

    baseline_ratio = baseline_size / raw_size if raw_size > 0 else 0.0
    tans_ratio = tans_size / raw_size if raw_size > 0 else 0.0

    print("\nCompressed sizes (bytes):")
    print(f"  Baseline LZ77 (Huffman) : {baseline_size}")
    print(f"  tANS LZ77               : {tans_size}")

    print("\nCompression ratios (compressed/raw):")
    print(f"  Baseline LZ77 (Huffman) : {baseline_ratio:.4f}")
    print(f"  tANS LZ77               : {tans_ratio:.4f}")

    print("\nCompression speed (MB/s):")
    print(f"  Baseline LZ77 (Huffman) : {baseline_speed:.2f} MB/s ({baseline_encode_time:.2f}s)")
    print(f"  tANS LZ77               : {tans_speed:.2f} MB/s ({tans_encode_time:.2f}s)")

    # Compare header overhead for single-block parse
    with open(path, "rb") as f:
        data_bytes = list(f.read())
    data_block = DataBlock(data_bytes)

    parser = LZ77Encoder(
        min_match_length=DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
    )
    seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

    huffman_header_bits = compute_literal_header_bits_huffman(lits)
    tans_header_bits = compute_literal_header_bits_tans(lits, table_log=table_log)

    print("\nLiterals header overhead (single-block parse):")
    print(f"  #literals in stream           : {len(lits)}")
    print(f"  Huffman header bits           : {huffman_header_bits}")
    print(f"  tANS header bits              : {tans_header_bits}")
    if huffman_header_bits > 0:
        print(f"  tANS / Huffman header         : {tans_header_bits / huffman_header_bits:.4f}")

    return {
        'raw_size': raw_size,
        'baseline_size': baseline_size,
        'tans_size': tans_size,
        'baseline_ratio': baseline_ratio,
        'tans_ratio': tans_ratio,
        'baseline_speed': baseline_speed,
        'tans_speed': tans_speed,
        'baseline_time': baseline_encode_time,
        'tans_time': tans_encode_time,
        'num_literals': len(lits),
        'huffman_header_bits': huffman_header_bits,
        'tans_header_bits': tans_header_bits,
    }


def run_data_folder_benchmarks(data_folder: str, block_size: int = 100_000, table_log: int = 10):
    """Decompress all .xz files in folder and benchmark each."""
    if not os.path.isdir(data_folder):
        print(f"[Error] {data_folder} is not a directory.")
        return

    files = sorted(os.listdir(data_folder))
    xz_files = [f for f in files if f.endswith(".xz")]

    if not xz_files:
        print(f"[Warning] No .xz files found in {data_folder}")
        return

    print(f"\n=== Running benchmarks on folder: {data_folder} ===")
    print(f"Found {len(xz_files)} compressed files.\n")

    results = []

    for fname in xz_files:
        full_path = os.path.join(data_folder, fname)
        print(f"\n--- Decompressing {fname} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_out = os.path.join(tmpdir, fname.replace(".xz", ".raw"))

            with lzma.open(full_path, "rb") as f_in, open(raw_out, "wb") as f_out:
                f_out.write(f_in.read())

            result = run_single_file_benchmark(raw_out, block_size=block_size, table_log=table_log)
            result['filename'] = fname
            results.append(result)

    # Print summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        total_raw = sum(r['raw_size'] for r in results)
        total_baseline = sum(r['baseline_size'] for r in results)
        total_tans = sum(r['tans_size'] for r in results)

        avg_baseline_ratio = total_baseline / total_raw if total_raw > 0 else 0
        avg_tans_ratio = total_tans / total_raw if total_raw > 0 else 0

        avg_baseline_speed = np.mean([r['baseline_speed'] for r in results])
        avg_tans_speed = np.mean([r['tans_speed'] for r in results])

        print(f"\nTotal raw size: {total_raw} bytes ({total_raw / 1024 / 1024:.2f} MB)")
        print(f"\nAverage compression ratio:")
        print(f"  Baseline (Huffman): {avg_baseline_ratio:.4f}")
        print(f"  tANS:               {avg_tans_ratio:.4f}")
        print(f"  Improvement:        {(1 - avg_tans_ratio / avg_baseline_ratio) * 100:.2f}%")

        print(f"\nAverage compression speed:")
        print(f"  Baseline (Huffman): {avg_baseline_speed:.2f} MB/s")
        print(f"  tANS:               {avg_tans_speed:.2f} MB/s")


def plot_header_comparison(output_path: str = "tans_header_comparison.png", table_log: int = 10):
    """
    Generate files of different sizes and plot header overhead comparison.
    Tests sizes: 1KB, 10KB, 100KB, 1MB, 10MB
    """
    print("\n=== Generating Header Comparison Plot ===")

    # Target file sizes in bytes
    sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024]
    size_labels = ["1KB", "10KB", "100KB", "1MB", "10MB"]

    huffman_headers = []
    tans_headers = []
    ratios = []

    for size, label in zip(sizes, size_labels):
        print(f"\nProcessing {label} file...")

        # Generate synthetic text data with realistic distribution
        np.random.seed(42)

        # Create realistic text-like distribution
        char_probs = np.array([c % 26 + 1 for c in range(256)])
        char_probs = char_probs / char_probs.sum()

        data = np.random.choice(256, size=size, p=char_probs).tolist()
        data_block = DataBlock(data)

        # LZ77 parse to get literals
        parser = LZ77Encoder(
            min_match_length=DEFAULT_MIN_MATCH_LEN,
            max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        )
        seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

        # Compute header sizes
        huff_bits = compute_literal_header_bits_huffman(lits)
        tans_bits = compute_literal_header_bits_tans(lits, table_log=table_log)

        huffman_headers.append(huff_bits)
        tans_headers.append(tans_bits)
        ratios.append(tans_bits / huff_bits if huff_bits > 0 else 1.0)

        print(f"  Literals: {len(lits)}")
        print(f"  Huffman header: {huff_bits} bits ({huff_bits / 8:.1f} bytes)")
        print(f"  tANS header: {tans_bits} bits ({tans_bits / 8:.1f} bytes)")
        print(f"  Ratio (tANS/Huffman): {ratios[-1]:.4f}")

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(size_labels))
    width = 0.35

    # Plot 1: absolute header sizes
    bars1 = ax1.bar(x - width / 2, [b / 8 for b in huffman_headers], width,
                    label='Huffman', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, [b / 8 for b in tans_headers], width,
                    label='tANS', color='coral', alpha=0.8)

    ax1.set_xlabel('File Size', fontsize=12)
    ax1.set_ylabel('Header Size (bytes)', fontsize=12)
    ax1.set_title('Header Overhead Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontsize=9)

    # Plot 2: compression ratio (tANS/Huffman)
    line = ax2.plot(size_labels, ratios, marker='o', linewidth=2,
                    markersize=8, color='green', label='tANS/Huffman')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal (ratio=1.0)')

    ax2.set_xlabel('File Size', fontsize=12)
    ax2.set_ylabel('Header Size Ratio', fontsize=12)
    ax2.set_title('tANS vs Huffman Header Ratio', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add value labels
    for i, (label, ratio) in enumerate(zip(size_labels, ratios)):
        ax2.text(i, ratio + 0.01, f'{ratio:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n=== Plot saved to {output_path} ===")

    # Print summary
    print("\n=== Summary ===")
    if np.mean(ratios) < 1.0:
        print(f"Average header reduction: {(1 - np.mean(ratios)) * 100:.2f}%")
    else:
        print(f"Average header increase: {(np.mean(ratios) - 1) * 100:.2f}%")
    print(f"Best case (smallest ratio): {min(ratios):.4f} at {size_labels[ratios.index(min(ratios))]}")
    print(f"Worst case (largest ratio): {max(ratios):.4f} at {size_labels[ratios.index(max(ratios))]}")


def plot_header_comparison_from_file(
    file_path: str,
    output_path: str = "tans_header_comparison.png",
    table_log: int = 10
):
    """
    Use actual file and test different chunk sizes from it.
    """
    print(f"\n=== Generating Header Comparison Plot from {file_path} ===")

    with open(file_path, "rb") as f:
        full_data = list(f.read())

    total_size = len(full_data)
    print(f"Total file size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")

    # Define test sizes
    sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, min(10 * 1024 * 1024, total_size)]
    size_labels = ["1KB", "10KB", "100KB", "1MB",
                   "10MB" if total_size >= 10 * 1024 * 1024 else f"{total_size // 1024 // 1024}MB"]

    huffman_headers = []
    tans_headers = []
    ratios = []

    for size, label in zip(sizes, size_labels):
        if size > total_size:
            continue

        print(f"\nProcessing {label} chunk...")

        # Use first N bytes
        data = full_data[:size]
        data_block = DataBlock(data)

        # LZ77 parse
        parser = LZ77Encoder(
            min_match_length=DEFAULT_MIN_MATCH_LEN,
            max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        )
        seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

        huff_bits = compute_literal_header_bits_huffman(lits)
        tans_bits = compute_literal_header_bits_tans(lits, table_log=table_log)

        huffman_headers.append(huff_bits)
        tans_headers.append(tans_bits)
        ratios.append(tans_bits / huff_bits if huff_bits > 0 else 1.0)

        print(f"  Literals: {len(lits)}")
        print(f"  Huffman: {huff_bits} bits, tANS: {tans_bits} bits")
        print(f"  Ratio: {ratios[-1]:.4f}")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    used_labels = size_labels[:len(huffman_headers)]
    x = np.arange(len(used_labels))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, [b / 8 for b in huffman_headers], width,
                    label='Huffman', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, [b / 8 for b in tans_headers], width,
                    label='tANS', color='coral', alpha=0.8)

    ax1.set_xlabel('File Size', fontsize=12)
    ax1.set_ylabel('Header Size (bytes)', fontsize=12)
    ax1.set_title(f'Header Overhead - {os.path.basename(file_path)}',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(used_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontsize=9)

    ax2.plot(used_labels, ratios, marker='o', linewidth=2,
             markersize=8, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('File Size', fontsize=12)
    ax2.set_ylabel('Header Size Ratio (tANS/Huffman)', fontsize=12)
    ax2.set_title('Header Compression Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    for i, (label, ratio) in enumerate(zip(used_labels, ratios)):
        ax2.text(i, ratio + 0.01, f'{ratio:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n=== Plot saved to {output_path} ===")

    if np.mean(ratios) < 1.0:
        print(f"\nAverage reduction: {(1 - np.mean(ratios)) * 100:.2f}%")
    else:
        print(f"\nAverage increase: {(np.mean(ratios) - 1) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LZ77 with Huffman vs tANS entropy coding"
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input file(s) to benchmark",
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        help="Folder with .xz files for batch benchmarking",
    )

    parser.add_argument(
        "-b",
        "--block-size",
        type=int,
        default=100_000,
        help="LZ77 block size (default: 100000)",
    )

    parser.add_argument(
        "-t",
        "--table-log",
        type=int,
        default=10,
        help="tANS table log size (default: 10, table_size=1024)",
    )

    parser.add_argument(
        "--plot-header-comparison",
        type=str,
        metavar="OUTPUT_PNG",
        help="Generate header comparison plot (synthetic data)",
    )

    parser.add_argument(
        "--plot-from-file",
        type=str,
        nargs=2,
        metavar=("INPUT_FILE", "OUTPUT_PNG"),
        help="Generate header comparison plot from actual file",
    )

    args = parser.parse_args()

    # Plotting options
    if args.plot_header_comparison:
        plot_header_comparison(args.plot_header_comparison, table_log=args.table_log)
        return

    if args.plot_from_file:
        input_file, output_png = args.plot_from_file
        plot_header_comparison_from_file(input_file, output_png, table_log=args.table_log)
        return

    # Benchmarking
    if args.data_folder:
        run_data_folder_benchmarks(
            args.data_folder,
            block_size=args.block_size,
            table_log=args.table_log
        )

    if args.input:
        for path in args.input:
            if not os.path.isfile(path):
                print(f"Warning: {path} is not a file, skipping.")
                continue
            run_single_file_benchmark(
                path,
                block_size=args.block_size,
                table_log=args.table_log
            )


if __name__ == "__main__":
    main()