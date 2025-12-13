"""
Benchmark comparing LZ77 with empirical vs canonical Huffman.

Tests three variants:
- Baseline: empirical Huffman everywhere
- Canonical literals: canonical Huffman for byte values only
- Canonical all: canonical Huffman for all LZ77 streams

Usage: python lz77_canonical_benchmark.py -i file1 file2 ...
       python lz77_canonical_benchmark.py --data-folder path/to/testfiles/
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
    LogScaleBinnedIntegerEncoder,
    DEFAULT_MIN_MATCH_LEN,
    DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
)
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray
from scl.utils.test_utils import try_file_lossless_compression

from canonical_huffman_code import (
    CanonicalIntHuffmanEncoder,
    CanonicalIntHuffmanDecoder,
    ENCODED_BLOCK_SIZE_HEADER_BITS,
)



class CanonicalLogScaleBinnedIntegerEncoder(DataEncoder):
    """LogScaleBinned but with canonical Huffman for the bins."""

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset  # Values below this threshold are stored directly
        self.max_num_bins = max_num_bins + self.offset  # Total number of bins including offset
        # Use canonical Huffman to encode bin indices (more compact headers)
        self.canonical_huffman_encoder = CanonicalIntHuffmanEncoder(
            alphabet_size=self.max_num_bins
        )

    def encode_block(self, data_block: DataBlock) -> BitArray:
        import math

        # Log-scale binning: map large integers to (bin_index, residual) pairs
        bins: List[int] = []  # Bin indices for canonical Huffman encoding
        residuals: List[int] = []  # Residual values for fixed-width encoding
        residual_num_bits: List[int] = []  # Number of bits needed for each residual

        for val in data_block.data_list:
            assert val >= 0  # Only non-negative integers supported
            if val < self.offset:
                # Small values: store directly as bin index
                bins.append(val)
            else:
                # Large values: use log-scale binning
                val_minus_offset = val - self.offset
                val_plus_1 = val_minus_offset + 1  # Avoid log(0)
                log_val_plus_1 = int(math.log2(val_plus_1))  # Determine bin index
                if log_val_plus_1 >= self.max_num_bins:
                    raise ValueError(
                        f"Value {val} is too large to be encoded with {self.max_num_bins} bins"
                    )
                bins.append(log_val_plus_1 + self.offset)  # Store bin index
                residuals.append(val_plus_1 - 2**log_val_plus_1)  # Store remainder
                residual_num_bits.append(log_val_plus_1)  # Track bits needed

        # Encode bin indices using canonical Huffman (compact header)
        bins_encoding = self.canonical_huffman_encoder.encode_block(DataBlock(bins))

        from scl.utils.bitarray_utils import uint_to_bitarray

        # Encode residuals using fixed-width encoding
        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:  # No residual needed for smallest values in each bin
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        # Format: [canonical_huffman_bins][fixed_width_residuals]
        return bins_encoding + residuals_encoding


class CanonicalLogScaleBinnedIntegerDecoder(DataDecoder):

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.canonical_huffman_decoder = CanonicalIntHuffmanDecoder(
            alphabet_size=self.max_num_bins
        )

    def decode_block(self, encoded_bitarray: BitArray):
        from scl.utils.bitarray_utils import bitarray_to_uint

        # First, decode bin indices using canonical Huffman
        bins_decoded, num_bits_consumed = self.canonical_huffman_decoder.decode_block(
            encoded_bitarray
        )
        bins_decoded = bins_decoded.data_list  # Extract list from DataBlock
        encoded_bitarray = encoded_bitarray[num_bits_consumed:]  # Skip consumed bits

        # Reconstruct original values from (bin_index, residual) pairs
        decoded: List[int] = []
        for encoded_bin in bins_decoded:
            if encoded_bin < self.offset:
                # Small values: stored directly as bin index
                decoded.append(encoded_bin)
            else:
                # Large values: reconstruct from bin index and residual
                encoded_bin_minus_offset = encoded_bin - self.offset
                log_val_plus_1 = encoded_bin_minus_offset  # This is the log value
                num_bits = log_val_plus_1  # Number of bits for residual

                if num_bits == 0:
                    residual = 0  # No residual for smallest value in bin
                else:
                    # Read residual from fixed-width encoding
                    residual = bitarray_to_uint(encoded_bitarray[:num_bits])

                num_bits_consumed += num_bits  # Track total bits consumed
                encoded_bitarray = encoded_bitarray[num_bits:]  # Advance bit position


                decoded_val = self.offset + 2**log_val_plus_1 + residual - 1
                decoded.append(decoded_val)

        return DataBlock(decoded), num_bits_consumed




class LZ77StreamsEncoderCanonicalLiterals(LZ77StreamsEncoder):
    """Use canonical Huffman for literals, baseline for other streams."""

    def encode_literals(self, literals: List[int]) -> BitArray:
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=256)
        return encoder.encode_block(DataBlock(literals))


class LZ77StreamsDecoderCanonicalLiterals(LZ77StreamsDecoder):

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        decoder = CanonicalIntHuffmanDecoder(alphabet_size=256)
        decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
        return decoded_block.data_list, num_bits_consumed


class LZ77EncoderCanonicalLiterals(LZ77Encoder):

    def __init__(
        self,
        min_match_length: int = DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered: int = DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        initial_window: List[int] = None,
    ):
        super().__init__(
            min_match_length=min_match_length,
            max_num_matches_considered=max_num_matches_considered,
            initial_window=initial_window,
        )
        self.streams_encoder = LZ77StreamsEncoderCanonicalLiterals()


class LZ77DecoderCanonicalLiterals(LZ77Decoder):

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderCanonicalLiterals()


class LZ77StreamsEncoderCanonicalAll(LZ77StreamsEncoder):
    """Use canonical Huffman for all LZ77 streams."""

    def encode_lz77_sequences(self, lz77_sequences):
        # Use canonical Huffman for all LZ77 sequence streams
        coder = CanonicalLogScaleBinnedIntegerEncoder(
            offset=self.log_scale_binned_coder_offset
        )
        encoded_bitarray = BitArray()
        
        # Encode literal counts (how many literals before each match)
        encoded_bitarray += coder.encode_block(
            DataBlock([l.literal_count for l in lz77_sequences])
        )
        # Encode match lengths (length of each back-reference)
        encoded_bitarray += coder.encode_block(
            DataBlock([l.match_length for l in lz77_sequences])
        )
        # Encode match offsets (distance to back-reference)
        encoded_bitarray += coder.encode_block(
            DataBlock([l.match_offset for l in lz77_sequences])
        )
        return encoded_bitarray

    def encode_literals(self, literals: List[int]) -> BitArray:
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=256)
        return encoder.encode_block(DataBlock(literals))


class LZ77StreamsDecoderCanonicalAll(LZ77StreamsDecoder):

    def decode_lz77_sequences(self, encoded_bitarray: BitArray):
        # Use canonical Huffman decoder for all LZ77 sequence streams
        coder = CanonicalLogScaleBinnedIntegerDecoder(
            offset=self.log_scale_binned_coder_offset
        )

        num_bits_consumed = 0

        # Decode literal counts stream
        literal_counts, bits_lit = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_lit:]  # Advance bit position
        num_bits_consumed += bits_lit

        # Decode match lengths stream
        match_lengths, bits_len = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_len:]  # Advance bit position
        num_bits_consumed += bits_len

        # Decode match offsets stream
        match_offsets, bits_off = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_off:]  # Advance bit position
        num_bits_consumed += bits_off

        from scl.compressors.lz77 import LZ77Sequence

        # Reconstruct LZ77 sequences from decoded streams
        lz77_sequences = [
            LZ77Sequence(lc, ml, mo)
            for lc, ml, mo in zip(
                literal_counts.data_list,
                match_lengths.data_list,
                match_offsets.data_list,
            )
        ]
        return lz77_sequences, num_bits_consumed

    def decode_literals(self, encoded_bitarray: BitArray):
        decoder = CanonicalIntHuffmanDecoder(alphabet_size=256)
        decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
        return decoded_block.data_list, num_bits_consumed


class LZ77EncoderCanonicalAll(LZ77Encoder):

    def __init__(
        self,
        min_match_length: int = DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered: int = DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        initial_window: List[int] = None,
    ):
        super().__init__(
            min_match_length=min_match_length,
            max_num_matches_considered=max_num_matches_considered,
            initial_window=initial_window,
        )
        self.streams_encoder = LZ77StreamsEncoderCanonicalAll()


class LZ77DecoderCanonicalAll(LZ77Decoder):

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderCanonicalAll()



def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    """Count header bits for empirical Huffman (counts array)."""
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    # Build frequency counts for all 256 possible byte values
    counts = DataBlock(literals).get_counts()
    for i in range(256):
        if i not in counts:
            counts[i] = 0  # Ensure all 256 entries are present
    counts_list = [counts[i] for i in range(256)]  # Convert to ordered list

    # Encode frequency counts using Elias-Delta compression
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    # Header format: [size_header][elias_delta_counts]
    header_bits = ENCODED_BLOCK_SIZE_HEADER_BITS + len(counts_encoding)
    return header_bits


def compute_literal_header_bits_canonical(literals: List[int]) -> int:
    """Count header bits for canonical Huffman (code lengths)."""
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    # Build frequency distribution from literal bytes
    counts = DataBlock(literals).get_counts()
    prob_dist = ProbabilityDist.normalize_prob_dict(counts)

    # Generate standard Huffman codes to get code lengths
    huff_encoder = HuffmanEncoder(prob_dist)
    huff_table = huff_encoder.encoding_table

    # Extract code lengths for all 256 possible byte values
    code_lengths = [0] * 256  # 0 means symbol not used
    for sym, bits in huff_table.items():
        code_lengths[sym] = len(bits)  # Store code length for each symbol

    # Encode code lengths using Elias-Delta compression
    length_header_bits = EliasDeltaUintEncoder().encode_block(DataBlock(code_lengths))
    # Header format: [size_header][elias_delta_code_lengths]
    header_bits = ENCODED_BLOCK_SIZE_HEADER_BITS + len(length_header_bits)
    return header_bits



def run_single_file_benchmark(path: str, block_size: int = 100_000) -> None:
    """
    Benchmark a single file with three compression methods:
    1. Baseline LZ77 (empirical Huffman everywhere)
    2. Canonical literals (canonical Huffman for literals only)
    3. Canonical all (canonical Huffman for all streams)
    """
    raw_size = os.path.getsize(path)

    print(f"\n=== Benchmark on file: {path} ===")
    print(f"Raw size: {raw_size} bytes ({raw_size / 1024 / 1024:.2f} MB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_enc = LZ77Encoder()  
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        # Measure encoding time with progress bar
        start_time = time.time()
        encode_file_with_progress(base_enc, path, base_encoded_path, block_size=block_size)
        baseline_encode_time = time.time() - start_time

        # Verify lossless compression
        base_dec.decode_file(base_encoded_path, base_decoded_path)
        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)
        baseline_speed = (raw_size / 1024 / 1024) / baseline_encode_time if baseline_encode_time > 0 else 0

        # canonical literals only
        can_lit_enc = LZ77EncoderCanonicalLiterals()
        can_lit_dec = LZ77DecoderCanonicalLiterals()

        can_lit_encoded_path = os.path.join(tmpdir, "canonical_lit.lz77")
        can_lit_decoded_path = os.path.join(tmpdir, "canonical_lit.dec")

        start_time = time.time()
        encode_file_with_progress(can_lit_enc, path, can_lit_encoded_path, block_size=block_size)
        canonical_lit_encode_time = time.time() - start_time

        can_lit_dec.decode_file(can_lit_encoded_path, can_lit_decoded_path)

        with open(path, "rb") as f_in, open(can_lit_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Canonical-literals LZ77 decode mismatch!"

        canonical_lit_size = os.path.getsize(can_lit_encoded_path)
        canonical_lit_speed = (raw_size / 1024 / 1024) / canonical_lit_encode_time if canonical_lit_encode_time > 0 else 0

        # canonical everything
        can_all_enc = LZ77EncoderCanonicalAll()
        can_all_dec = LZ77DecoderCanonicalAll()

        can_all_encoded_path = os.path.join(tmpdir, "canonical_all.lz77")
        can_all_decoded_path = os.path.join(tmpdir, "canonical_all.dec")

        start_time = time.time()
        encode_file_with_progress(can_all_enc, path, can_all_encoded_path, block_size=block_size)
        canonical_all_encode_time = time.time() - start_time

        can_all_dec.decode_file(can_all_encoded_path, can_all_decoded_path)

        with open(path, "rb") as f_in, open(can_all_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Canonical-all LZ77 decode mismatch!"

        canonical_all_size = os.path.getsize(can_all_encoded_path)
        canonical_all_speed = (raw_size / 1024 / 1024) / canonical_all_encode_time if canonical_all_encode_time > 0 else 0

    baseline_ratio = baseline_size / raw_size if raw_size > 0 else 0.0
    canonical_lit_ratio = canonical_lit_size / raw_size if raw_size > 0 else 0.0
    canonical_all_ratio = canonical_all_size / raw_size if raw_size > 0 else 0.0

    print("\nCompressed sizes (bytes):")
    print(f"  Baseline LZ77        : {baseline_size}")
    print(f"  Canonical (literals) : {canonical_lit_size}")
    print(f"  Canonical (all)      : {canonical_all_size}")

    print("\nCompression ratios (compressed/raw):")
    print(f"  Baseline LZ77        : {baseline_ratio:.4f}")
    print(f"  Canonical (literals) : {canonical_lit_ratio:.4f}")
    print(f"  Canonical (all)      : {canonical_all_ratio:.4f}")

    print("\nCompression speed (MB/s):")
    print(f"  Baseline LZ77        : {baseline_speed:.2f} MB/s ({baseline_encode_time:.2f}s)")
    print(f"  Canonical (literals) : {canonical_lit_speed:.2f} MB/s ({canonical_lit_encode_time:.2f}s)")
    print(f"  Canonical (all)      : {canonical_all_speed:.2f} MB/s ({canonical_all_encode_time:.2f}s)")

    # compare header overhead for single-block parse
    with open(path, "rb") as f:
        data_bytes = list(f.read())
    data_block = DataBlock(data_bytes)

    parser = LZ77Encoder(
        min_match_length=DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
    )
    seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

    emp_header_bits = compute_literal_header_bits_empirical(lits)
    can_header_bits = compute_literal_header_bits_canonical(lits)

    print("\nLiterals header overhead (single-block parse):")
    print(f"  #literals in stream           : {len(lits)}")
    print(f"  Empirical Huffman header bits : {emp_header_bits}")
    print(f"  Canonical Huffman header bits : {can_header_bits}")
    if emp_header_bits > 0:
        print(
            f"  Canonical / Empirical header  : "
            f"{can_header_bits / emp_header_bits:.4f}"
        )

def run_data_folder_benchmarks(data_folder: str, block_size: int = 100_000):
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

    for fname in xz_files:
        full_path = os.path.join(data_folder, fname)
        print(f"\n--- Decompressing {fname} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_out = os.path.join(tmpdir, fname.replace(".xz", ".raw"))

            with lzma.open(full_path, "rb") as f_in, open(raw_out, "wb") as f_out:
                f_out.write(f_in.read())

            run_single_file_benchmark(raw_out, block_size=block_size)


def encode_file_with_progress(encoder, input_path, output_path, block_size=100_000):
    """Wrap encoder.encode_file with a progress bar."""
    file_size = os.path.getsize(input_path)

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
              desc=f"Compressing") as pbar:

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


def plot_header_comparison(output_path: str = "header_comparison.png"):
    """
    Generate files of different sizes and plot header overhead comparison.
    Tests sizes: 1KB, 10KB, 100KB, 1MB, 10MB
    """
    print("\n=== Generating Header Comparison Plot ===")

    # target file sizes in bytes
    sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024]
    size_labels = ["1KB", "10KB", "100KB", "1MB", "10MB"]

    empirical_headers = []
    canonical_headers = []
    ratios = []

    for size, label in zip(sizes, size_labels):
        print(f"\nProcessing {label} file...")

        # generate synthetic text data
        # mix of common English letters with some repetition
        np.random.seed(42)

        # create realistic text-like distribution
        # common letters appear more frequently
        char_probs = np.array([c % 26 + 1 for c in range(256)])  # 修复：直接用c而不是ord(c)
        char_probs = char_probs / char_probs.sum()

        data = np.random.choice(256, size=size, p=char_probs).tolist()
        data_block = DataBlock(data)

        # LZ77 parse to get literals
        parser = LZ77Encoder(
            min_match_length=DEFAULT_MIN_MATCH_LEN,
            max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        )
        seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

        # compute header sizes
        emp_bits = compute_literal_header_bits_empirical(lits)
        can_bits = compute_literal_header_bits_canonical(lits)

        empirical_headers.append(emp_bits)
        canonical_headers.append(can_bits)
        ratios.append(can_bits / emp_bits if emp_bits > 0 else 1.0)

        print(f"  Literals: {len(lits)}")
        print(f"  Empirical header: {emp_bits} bits ({emp_bits / 8:.1f} bytes)")
        print(f"  Canonical header: {can_bits} bits ({can_bits / 8:.1f} bytes)")
        print(f"  Ratio (can/emp): {ratios[-1]:.4f}")

    # create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(size_labels))
    width = 0.35

    # plot 1: absolute header sizes
    bars1 = ax1.bar(x - width / 2, [b / 8 for b in empirical_headers], width,
                    label='Empirical Huffman', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, [b / 8 for b in canonical_headers], width,
                    label='Canonical Huffman', color='coral', alpha=0.8)

    ax1.set_xlabel('File Size', fontsize=12)
    ax1.set_ylabel('Header Size (bytes)', fontsize=12)
    ax1.set_title('Header Overhead Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontsize=9)

    # plot 2: compression ratio (canonical/empirical)
    line = ax2.plot(size_labels, ratios, marker='o', linewidth=2,
                    markersize=8, color='green', label='Canonical/Empirical')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal (ratio=1.0)')

    ax2.set_xlabel('File Size', fontsize=12)
    ax2.set_ylabel('Header Size Ratio', fontsize=12)
    ax2.set_title('Canonical vs Empirical Header Ratio', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # add value labels
    for i, (label, ratio) in enumerate(zip(size_labels, ratios)):
        ax2.text(i, ratio + 0.01, f'{ratio:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n=== Plot saved to {output_path} ===")

    # print summary
    print("\n=== Summary ===")
    print(f"Average header reduction: {(1 - np.mean(ratios)) * 100:.2f}%")
    print(f"Best case (smallest ratio): {min(ratios):.4f} at {size_labels[ratios.index(min(ratios))]}")
    print(f"Worst case (largest ratio): {max(ratios):.4f} at {size_labels[ratios.index(max(ratios))]}")

def plot_header_comparison_from_file(file_path: str, output_path: str = "header_comparison.png"):
    """
    Use actual file and test different chunk sizes from it.
    """
    print(f"\n=== Generating Header Comparison Plot from {file_path} ===")

    with open(file_path, "rb") as f:
        full_data = list(f.read())

    total_size = len(full_data)
    print(f"Total file size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")

    # define test sizes
    sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, min(10 * 1024 * 1024, total_size)]
    size_labels = ["1KB", "10KB", "100KB", "1MB",
                   "10MB" if total_size >= 10 * 1024 * 1024 else f"{total_size // 1024 // 1024}MB"]

    empirical_headers = []
    canonical_headers = []
    ratios = []

    for size, label in zip(sizes, size_labels):
        if size > total_size:
            continue

        print(f"\nProcessing {label} chunk...")

        # use first N bytes
        data = full_data[:size]
        data_block = DataBlock(data)

        # LZ77 parse
        parser = LZ77Encoder(
            min_match_length=DEFAULT_MIN_MATCH_LEN,
            max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        )
        seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

        emp_bits = compute_literal_header_bits_empirical(lits)
        can_bits = compute_literal_header_bits_canonical(lits)

        empirical_headers.append(emp_bits)
        canonical_headers.append(can_bits)
        ratios.append(can_bits / emp_bits if emp_bits > 0 else 1.0)

        print(f"  Literals: {len(lits)}")
        print(f"  Empirical: {emp_bits} bits, Canonical: {can_bits} bits")
        print(f"  Ratio: {ratios[-1]:.4f}")

    # create plot (same as above)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    used_labels = size_labels[:len(empirical_headers)]
    x = np.arange(len(used_labels))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, [b / 8 for b in empirical_headers], width,
                    label='Empirical Huffman', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, [b / 8 for b in canonical_headers], width,
                    label='Canonical Huffman', color='coral', alpha=0.8)

    ax1.set_xlabel('File Size', fontsize=12)
    ax1.set_ylabel('Header Size (bytes)', fontsize=12)
    ax1.set_title(f'Header Overhead - {os.path.basename(file_path)}', fontsize=14, fontweight='bold')
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
    ax2.set_ylabel('Header Size Ratio (Canonical/Empirical)', fontsize=12)
    ax2.set_title('Header Compression Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    for i, (label, ratio) in enumerate(zip(used_labels, ratios)):
        ax2.text(i, ratio + 0.01, f'{ratio:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n=== Plot saved to {output_path} ===")

    print(f"\nAverage reduction: {(1 - np.mean(ratios)) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline LZ77 vs canonical Huffman variants"
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
        "--block_size",
        type=int,
        default=100_000,
        help="LZ77 block size (default: 100000)",
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

    # new plotting options
    if args.plot_header_comparison:
        plot_header_comparison(args.plot_header_comparison)
        return

    if args.plot_from_file:
        input_file, output_png = args.plot_from_file
        plot_header_comparison_from_file(input_file, output_png)
        return

    if args.data_folder:
        run_data_folder_benchmarks(args.data_folder, block_size=args.block_size)

    if args.input:
        for path in args.input:
            if not os.path.isfile(path):
                print(f"Warning: {path} is not a file, skipping.")
                continue
            run_single_file_benchmark(path, block_size=args.block_size)


if __name__ == "__main__":
    main()