"""
Benchmark script to compare baseline LZ77 (empirical Huffman)
vs. LZ77 with canonical Huffman on different streams.

Usage:
    python lz77_canonical_benchmark.py -i path/to/file1 path/to/file2 ...

This script does three things:
  1. Defines LZ77 stream encoders/decoders that use canonical Huffman:
        - only for literals
        - for literals + literal_count + match_length + match_offset
  2. For each input file, compresses it with:
        - baseline LZ77Encoder / LZ77Decoder
        - LZ77EncoderCanonicalLiterals / LZ77DecoderCanonicalLiterals
        - LZ77EncoderCanonicalAll / LZ77DecoderCanonicalAll
     and compares compressed sizes and compression ratios.
  3. For the LZ77 parsing of the whole file as a single block, computes
     the header overhead (in bits) of:
        - EmpiricalIntHuffmanEncoder (baseline)
        - CanonicalIntHuffmanEncoder
     for the literals stream only.
"""

import argparse
import os
import tempfile
from typing import List, Tuple
import lzma
from tqdm import tqdm


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


# ---------------------------------------------------------------------------
# Canonical log-scale-binned integer encoder/decoder
# (for literal_count, match_length, match_offset streams).
# ---------------------------------------------------------------------------


class CanonicalLogScaleBinnedIntegerEncoder(DataEncoder):
    """
    Same as LogScaleBinnedIntegerEncoder, but uses CanonicalIntHuffmanEncoder
    for the binned integers instead of EmpiricalIntHuffmanEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.canonical_huffman_encoder = CanonicalIntHuffmanEncoder(
            alphabet_size=self.max_num_bins
        )

    def encode_block(self, data_block: DataBlock) -> BitArray:
        import math

        bins: List[int] = []
        residuals: List[int] = []
        residual_num_bits: List[int] = []

        for val in data_block.data_list:
            assert val >= 0
            if val < self.offset:
                bins.append(val)
            else:
                val_minus_offset = val - self.offset
                val_plus_1 = val_minus_offset + 1
                log_val_plus_1 = int(math.log2(val_plus_1))
                if log_val_plus_1 >= self.max_num_bins:
                    raise ValueError(
                        f"Value {val} is too large to be encoded with {self.max_num_bins} bins"
                    )
                bins.append(log_val_plus_1 + self.offset)
                residuals.append(val_plus_1 - 2**log_val_plus_1)
                residual_num_bits.append(log_val_plus_1)

        # Encode bins with canonical Huffman
        bins_encoding = self.canonical_huffman_encoder.encode_block(DataBlock(bins))

        # Encode residuals as raw bits
        from scl.utils.bitarray_utils import uint_to_bitarray

        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        return bins_encoding + residuals_encoding


class CanonicalLogScaleBinnedIntegerDecoder(DataDecoder):
    """
    Decoder for CanonicalLogScaleBinnedIntegerEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.canonical_huffman_decoder = CanonicalIntHuffmanDecoder(
            alphabet_size=self.max_num_bins
        )

    def decode_block(self, encoded_bitarray: BitArray):
        from scl.utils.bitarray_utils import bitarray_to_uint

        # First decode the bin sequence (canonical Huffman)
        bins_decoded, num_bits_consumed = self.canonical_huffman_decoder.decode_block(
            encoded_bitarray
        )
        bins_decoded = bins_decoded.data_list
        encoded_bitarray = encoded_bitarray[num_bits_consumed:]

        decoded: List[int] = []
        for encoded_bin in bins_decoded:
            if encoded_bin < self.offset:
                decoded.append(encoded_bin)
            else:
                # Undo the log-scale binning
                encoded_bin_minus_offset = encoded_bin - self.offset
                log_val_plus_1 = encoded_bin_minus_offset
                num_bits = log_val_plus_1

                if num_bits == 0:
                    residual = 0
                else:
                    residual = bitarray_to_uint(encoded_bitarray[:num_bits])

                num_bits_consumed += num_bits
                encoded_bitarray = encoded_bitarray[num_bits:]

                decoded_val = self.offset + 2**log_val_plus_1 + residual - 1
                decoded.append(decoded_val)

        return DataBlock(decoded), num_bits_consumed


# ---------------------------------------------------------------------------
# Canonical LZ77 streams: literals only, and "all" streams.
# ---------------------------------------------------------------------------


class LZ77StreamsEncoderCanonicalLiterals(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses canonical Huffman for literals.

    Literal counts, match lengths, and match offsets are encoded exactly
    as in the baseline implementation (log-scale binned integers), but
    literals (byte values 0..255) use CanonicalIntHuffmanEncoder instead
    of EmpiricalIntHuffmanEncoder.
    """

    def encode_literals(self, literals: List[int]) -> BitArray:
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=256)
        return encoder.encode_block(DataBlock(literals))


class LZ77StreamsDecoderCanonicalLiterals(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderCanonicalLiterals."""

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        decoder = CanonicalIntHuffmanDecoder(alphabet_size=256)
        decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
        return decoded_block.data_list, num_bits_consumed


class LZ77EncoderCanonicalLiterals(LZ77Encoder):
    """LZ77 encoder with canonical Huffman for literals."""

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
    """LZ77 decoder matching LZ77EncoderCanonicalLiterals."""

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderCanonicalLiterals()


class LZ77StreamsEncoderCanonicalAll(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses canonical Huffman for
    all streams:
        - literal_count
        - match_length
        - match_offset
        - literals
    """

    def encode_lz77_sequences(self, lz77_sequences):
        coder = CanonicalLogScaleBinnedIntegerEncoder(
            offset=self.log_scale_binned_coder_offset
        )
        encoded_bitarray = BitArray()
        encoded_bitarray += coder.encode_block(
            DataBlock([l.literal_count for l in lz77_sequences])
        )
        encoded_bitarray += coder.encode_block(
            DataBlock([l.match_length for l in lz77_sequences])
        )
        encoded_bitarray += coder.encode_block(
            DataBlock([l.match_offset for l in lz77_sequences])
        )
        return encoded_bitarray

    def encode_literals(self, literals: List[int]) -> BitArray:
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=256)
        return encoder.encode_block(DataBlock(literals))


class LZ77StreamsDecoderCanonicalAll(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderCanonicalAll."""

    def decode_lz77_sequences(self, encoded_bitarray: BitArray):
        coder = CanonicalLogScaleBinnedIntegerDecoder(
            offset=self.log_scale_binned_coder_offset
        )

        num_bits_consumed = 0

        literal_counts, bits_lit = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_lit:]
        num_bits_consumed += bits_lit

        match_lengths, bits_len = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_len:]
        num_bits_consumed += bits_len

        match_offsets, bits_off = coder.decode_block(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_off:]
        num_bits_consumed += bits_off

        from scl.compressors.lz77 import LZ77Sequence

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
    """LZ77 encoder with canonical Huffman for all LZ77 streams."""

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
    """LZ77 decoder matching LZ77EncoderCanonicalAll."""

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderCanonicalAll()


# ---------------------------------------------------------------------------
# Header overhead computation for literals (same as之前)
# ---------------------------------------------------------------------------


def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    """Compute model header bits for empirical Huffman on literals.

    We mirror EmpiricalIntHuffmanEncoder's behavior but only count the
    model overhead:
        [32 bits: size_of_counts_encoding] + [counts_encoding_bits]

    The extra 32 bits for value-encoding size are shared by empirical
    and canonical encoders and are not counted here.
    """
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


def compute_literal_header_bits_canonical(literals: List[int]) -> int:
    """Compute model header bits for canonical Huffman on literals."""
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    counts = DataBlock(literals).get_counts()
    prob_dist = ProbabilityDist.normalize_prob_dict(counts)

    huff_encoder = HuffmanEncoder(prob_dist)
    huff_table = huff_encoder.encoding_table

    code_lengths = [0] * 256
    for sym, bits in huff_table.items():
        code_lengths[sym] = len(bits)

    length_header_bits = EliasDeltaUintEncoder().encode_block(DataBlock(code_lengths))
    header_bits = ENCODED_BLOCK_SIZE_HEADER_BITS + len(length_header_bits)
    return header_bits


# ---------------------------------------------------------------------------
# Benchmark: per-file compression & header comparison
# ---------------------------------------------------------------------------


def run_single_file_benchmark(path: str, block_size: int = 100_000) -> None:
    raw_size = os.path.getsize(path)

    print(f"\n=== Benchmark on file: {path} ===")
    print(f"Raw size: {raw_size} bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---------------- Baseline LZ77 ----------------
        base_enc = LZ77Encoder()
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        encode_file_with_progress(base_enc, path, base_encoded_path, block_size=block_size)
        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)

        # ------------- LZ77 with canonical literals only -------------
        can_lit_enc = LZ77EncoderCanonicalLiterals()
        can_lit_dec = LZ77DecoderCanonicalLiterals()

        can_lit_encoded_path = os.path.join(tmpdir, "canonical_lit.lz77")
        can_lit_decoded_path = os.path.join(tmpdir, "canonical_lit.dec")

        encode_file_with_progress(can_lit_enc, path, can_lit_encoded_path, block_size=block_size)
        can_lit_dec.decode_file(can_lit_encoded_path, can_lit_decoded_path)

        with open(path, "rb") as f_in, open(can_lit_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Canonical-literals LZ77 decode mismatch!"

        canonical_lit_size = os.path.getsize(can_lit_encoded_path)

        # ------------- LZ77 with canonical on all streams -------------
        can_all_enc = LZ77EncoderCanonicalAll()
        can_all_dec = LZ77DecoderCanonicalAll()

        can_all_encoded_path = os.path.join(tmpdir, "canonical_all.lz77")
        can_all_decoded_path = os.path.join(tmpdir, "canonical_all.dec")

        encode_file_with_progress(can_all_enc, path, can_all_encoded_path, block_size=block_size)
        can_all_dec.decode_file(can_all_encoded_path, can_all_decoded_path)

        with open(path, "rb") as f_in, open(can_all_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Canonical-all LZ77 decode mismatch!"

        canonical_all_size = os.path.getsize(can_all_encoded_path)

    baseline_ratio = baseline_size / raw_size if raw_size > 0 else 0.0
    canonical_lit_ratio = canonical_lit_size / raw_size if raw_size > 0 else 0.0
    canonical_all_ratio = canonical_all_size / raw_size if raw_size > 0 else 0.0

    print("Compressed sizes (bytes):")
    print(f"  Baseline LZ77        : {baseline_size}")
    print(f"  Canonical (literals) : {canonical_lit_size}")
    print(f"  Canonical (all)      : {canonical_all_size}")
    print("Compression ratios (compressed/raw):")
    print(f"  Baseline LZ77        : {baseline_ratio:.4f}")
    print(f"  Canonical (literals) : {canonical_lit_ratio:.4f}")
    print(f"  Canonical (all)      : {canonical_all_ratio:.4f}")

    # ---------------- Header overhead for literals ----------------
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
    """
    Scan a folder (e.g., testfiles/data/) for .xz files,
    decompress each to a temporary raw file, and run the benchmark.
    """
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

        # Decompress .xz to a temporary raw file
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_out = os.path.join(tmpdir, fname.replace(".xz", ".raw"))

            with lzma.open(full_path, "rb") as f_in, open(raw_out, "wb") as f_out:
                f_out.write(f_in.read())

            # Run benchmark on the decompressed file
            run_single_file_benchmark(raw_out, block_size=block_size)

def encode_file_with_progress(encoder, input_path, output_path, block_size=100_000):
    """
    A wrapper around encoder.encode_file that displays a progress bar
    based on how many bytes have been read from the input file.
    """
    file_size = os.path.getsize(input_path)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Compressing {os.path.basename(input_path)}")

    # We mimic chunk reading to update the progress bar,
    # but still rely on encoder.encode_file for actual compression.
    # So we manually read input and feed it chunk-by-chunk.
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        # But since LZ77Encoder expects to read the file path itself,
        # we simply track progress manually using a loop.
        while True:
            chunk = f_in.read(block_size)
            if not chunk:
                break
            pbar.update(len(chunk))

        # After simulating progress, call actual encoder
        encoder.encode_file(input_path, output_path, block_size=block_size)

    pbar.close()

def main_old():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline LZ77 vs. LZ77 with canonical Huffman "
            "on literals and on all LZ77 streams."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input file(s) to compress and benchmark.",
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=100_000,
        help="Block size used by LZ77 encode_file (default: 100000).",
    )
    args = parser.parse_args()

    for path in args.input:
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipping.")
            continue
        run_single_file_benchmark(path, block_size=args.block_size)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline LZ77 vs. LZ77 with canonical Huffman "
            "on literals and on all LZ77 streams."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input file(s) to compress and benchmark.",
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        help="Folder containing .xz files for batch benchmarking.",
    )

    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=100_000,
        help="Block size used by LZ77 encode_file (default: 100000).",
    )

    args = parser.parse_args()

    # If user passes --data-folder, run all files in that folder
    if args.data_folder:
        run_data_folder_benchmarks(args.data_folder, block_size=args.block_size)

    # If specific files are provided via -i, run them
    if args.input:
        for path in args.input:
            if not os.path.isfile(path):
                print(f"Warning: {path} is not a file, skipping.")
                continue
            run_single_file_benchmark(path, block_size=args.block_size)


if __name__ == "__main__":
    main()
