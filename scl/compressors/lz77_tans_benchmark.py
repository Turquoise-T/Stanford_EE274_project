"""
Benchmark script to compare baseline LZ77 (empirical Huffman)
vs. LZ77 with tANS on different streams.

Usage:
    python lz77_tans_benchmark.py -i path/to/file1 path/to/file2 ...
    python lz77_tans_benchmark.py -i path/to/file1 --table_log 8 10 12

This script does three things:
  1. Defines LZ77 stream encoders/decoders that use tANS:
        - only for literals
        - for literals + literal_count + match_length + match_offset (all streams)
  2. For each input file, compresses it with:
        - baseline LZ77Encoder / LZ77Decoder (empirical Huffman)
        - LZ77EncoderTANSLiterals / LZ77DecoderTANSLiterals
        - LZ77EncoderTANSAll / LZ77DecoderTANSAll
     and compares compressed sizes and compression ratios.
  3. For the LZ77 parsing of the whole file as a single block, computes
     the header overhead (in bits) of:
        - EmpiricalIntHuffmanEncoder (baseline)
        - TANSEncoder (with different table_log values)
     for the literals stream only.
"""

import argparse
import os
import tempfile
from typing import List, Tuple
from collections import Counter

from scl.compressors.elias_delta_uint_coder import (
    EliasDeltaUintDecoder,
    EliasDeltaUintEncoder,
)
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
from scl.utils.test_utils import try_file_lossless_compression

# Import tANS implementation
from tans_lz77_coder import (
    TANSEncoder,
    TANSDecoder,
    LZ77TANSStreamsEncoder,
    LZ77TANSStreamsDecoder,
)

ENCODED_BLOCK_SIZE_HEADER_BITS = 32  # Same as canonical Huffman


# ---------------------------------------------------------------------------
# tANS-based log-scale-binned integer encoder/decoder
# (for literal_count, match_length, match_offset streams).
# ---------------------------------------------------------------------------


class TANSLogScaleBinnedIntegerEncoder(DataEncoder):
    """
    Similar to LogScaleBinnedIntegerEncoder, but uses TANSEncoder
    for the binned integers instead of EmpiricalIntHuffmanEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32, table_log: int = 10):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.table_log = table_log
        self.tans_encoder = TANSEncoder(table_log=table_log)

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

        # Encode bins with tANS
        bins_encoding = self.tans_encoder.encode(bins)

        # Store frequency table for decoder
        freqs = Counter(bins)
        freq_encoding = self._encode_frequencies(freqs)

        # Encode residuals as raw bits
        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        # Format: [freq_table] + [bins_encoding] + [residuals]
        return freq_encoding + bins_encoding + residuals_encoding

    def _encode_frequencies(self, freqs: dict) -> BitArray:
        """Encode frequency table for decoder."""
        result = uint_to_bitarray(len(freqs), 16)
        for sym, freq in sorted(freqs.items()):
            result += uint_to_bitarray(sym, 32)
            result += uint_to_bitarray(freq, 32)
        return result


class TANSLogScaleBinnedIntegerDecoder(DataDecoder):
    """
    Decoder for TANSLogScaleBinnedIntegerEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32, table_log: int = 10):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.table_log = table_log
        self.tans_decoder = TANSDecoder(table_log=table_log)

    def decode_block(self, encoded_bitarray: BitArray):
        # Decode frequency table
        freqs, bits_consumed = self._decode_frequencies(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_consumed:]

        # Decode bins with tANS
        # Note: We need to know how many symbols to decode
        # This is stored implicitly in the total count of frequencies
        num_symbols = sum(freqs.values())
        bins_decoded, _ = self.tans_decoder.decode(
            encoded_bitarray, num_symbols, freqs
        )

        # Find where tANS encoding ends (simplified - need better stream delimiting)
        # For now, we'll track bits consumed during decode
        # This is a limitation of the current tANS implementation

        decoded: List[int] = []
        bit_position = 0  # Track position in remaining bitarray

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
                    # Read residual from the remaining stream
                    # This requires knowing where bins encoding ends
                    # Simplified: assume we can access residuals correctly
                    residual = 0  # Placeholder

                decoded_val = self.offset + 2**log_val_plus_1 + residual - 1
                decoded.append(decoded_val)

        return DataBlock(decoded), bits_consumed

    def _decode_frequencies(self, encoded_bitarray: BitArray) -> Tuple[dict, int]:
        """Decode frequency table."""
        num_unique = bitarray_to_uint(encoded_bitarray[0:16])
        bit_pos = 16
        freqs = {}
        for _ in range(num_unique):
            sym = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32
            freq = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32
            freqs[sym] = freq
        return freqs, bit_pos


# ---------------------------------------------------------------------------
# Helpers for literal counts header (shared by empirical Huffman and tANS)
# ---------------------------------------------------------------------------


def _build_literal_counts_list(literals: List[int]) -> List[int]:
    """Return a length-256 count vector for literal bytes."""
    counts = Counter(literals)
    return [counts.get(i, 0) for i in range(256)]


def _encode_literal_counts_header_from_counts(counts_list: List[int]) -> BitArray:
    """Encode counts header: [32 bits size] + [Elias–Delta(counts_list)]."""
    if not any(counts_list):
        # Mirror EmpiricalIntHuffmanEncoder behavior for empty streams.
        return uint_to_bitarray(0, ENCODED_BLOCK_SIZE_HEADER_BITS)

    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return (
        uint_to_bitarray(len(counts_encoding), ENCODED_BLOCK_SIZE_HEADER_BITS)
        + counts_encoding
    )


def _decode_literal_counts_header(encoded_bitarray: BitArray) -> Tuple[dict, int, int]:
    """
    Decode counts header produced by `_encode_literal_counts_header_from_counts`.

    Returns:
        freqs (dict): symbol -> count (only symbols with count > 0)
        num_literals (int): total number of literals
        bits_consumed (int): number of bits consumed from encoded_bitarray
    """
    counts_encoding_size = bitarray_to_uint(
        encoded_bitarray[0:ENCODED_BLOCK_SIZE_HEADER_BITS]
    )
    bit_pos = ENCODED_BLOCK_SIZE_HEADER_BITS

    if counts_encoding_size == 0:
        return {}, 0, bit_pos

    counts_block, num_bits_counts = EliasDeltaUintDecoder().decode_block(
        encoded_bitarray[bit_pos : bit_pos + counts_encoding_size]
    )
    assert num_bits_counts == counts_encoding_size

    counts_list = counts_block.data_list
    # For literals, we expect a full 256-entry vector.
    assert len(counts_list) == 256

    freqs = {i: c for i, c in enumerate(counts_list) if c > 0}
    num_literals = sum(counts_list)

    bit_pos += counts_encoding_size
    return freqs, num_literals, bit_pos


# ---------------------------------------------------------------------------
# tANS LZ77 streams: literals only, and "all" streams.
# ---------------------------------------------------------------------------


class LZ77StreamsEncoderTANSLiterals(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses tANS for literals.

    Literal counts, match lengths, and match offsets are encoded exactly
    as in the baseline implementation (log-scale binned integers with empirical Huffman),
    but literals (byte values 0..255) use TANSEncoder.
    """

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def encode_literals(self, literals: List[int]) -> BitArray:
        """Encode literals using tANS with a compact counts header.

        Layout:
            [counts header] + [tANS payload]

        The counts header matches the empirical Huffman implementation:
            [32 bits: size_of_counts_encoding] + [Elias–Delta(counts[0..255])].
        """
        counts_list = _build_literal_counts_list(literals)
        header = _encode_literal_counts_header_from_counts(counts_list)

        if not literals:
            # No payload when there are no literals.
            return header

        encoder = TANSEncoder(table_log=self.table_log)
        encoded_data = encoder.encode(literals)

        return header + encoded_data


class LZ77StreamsDecoderTANSLiterals(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderTANSLiterals."""

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        """Decode literals using tANS and the compact counts header."""
        freqs, num_literals, bit_pos = _decode_literal_counts_header(encoded_bitarray)

        if num_literals == 0:
            return [], bit_pos

        # Decode literals with tANS
        decoder = TANSDecoder(table_log=self.table_log)
        payload = encoded_bitarray[bit_pos:]
        literals, bits_used = decoder.decode(payload, num_literals, freqs)

        bits_consumed = bit_pos + bits_used
        return literals, bits_consumed


class LZ77EncoderTANSLiterals(LZ77Encoder):
    """LZ77 encoder with tANS for literals only."""

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
        # Use baseline LZ77 streams encoder with tANS only for literals.
        self.streams_encoder = LZ77StreamsEncoderTANSLiterals(table_log=table_log)


class LZ77DecoderTANSLiterals(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSLiterals."""

    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        # Matching streams decoder that uses tANS only for literals.
        self.streams_decoder = LZ77StreamsDecoderTANSLiterals(table_log=table_log)


class LZ77StreamsEncoderTANSAll(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses tANS for all streams:
        - literal_count
        - match_length
        - match_offset
        - literals
    """

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def encode_lz77_sequences(self, lz77_sequences):
        """Encode all LZ77 sequence components with tANS."""
        encoded_bitarray = BitArray()

        # Extract components
        literal_counts = [seq.literal_count for seq in lz77_sequences]
        match_lengths = [seq.match_length for seq in lz77_sequences]
        match_offsets = [seq.match_offset for seq in lz77_sequences]

        # Encode each stream
        for data_list in [literal_counts, match_lengths, match_offsets]:
            if not data_list:
                encoded_bitarray += uint_to_bitarray(0, 32)
                continue

            encoder = TANSEncoder(table_log=self.table_log)
            encoded = encoder.encode(data_list)

            freqs = Counter(data_list)
            freq_encoding = self._encode_frequencies(freqs)

            encoded_bitarray += uint_to_bitarray(len(data_list), 32)
            encoded_bitarray += freq_encoding
            encoded_bitarray += encoded

        return encoded_bitarray

    def _encode_frequencies(self, freqs: dict) -> BitArray:
        """Encode frequency table for decoder (integer streams)."""
        result = uint_to_bitarray(len(freqs), 16)
        for sym, freq in sorted(freqs.items()):
            result += uint_to_bitarray(sym, 32)
            result += uint_to_bitarray(freq, 32)
        return result

    def encode_literals(self, literals: List[int]) -> BitArray:
        """Encode literals using tANS with the same compact counts header."""
        counts_list = _build_literal_counts_list(literals)
        header = _encode_literal_counts_header_from_counts(counts_list)

        if not literals:
            return header

        encoder = TANSEncoder(table_log=self.table_log)
        encoded_data = encoder.encode(literals)

        return header + encoded_data


class LZ77StreamsDecoderTANSAll(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderTANSAll."""

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def _decode_frequencies(self, encoded_bitarray: BitArray) -> Tuple[dict, int]:
        """Decode frequency table for integer streams (matches encoder format)."""
        num_unique = bitarray_to_uint(encoded_bitarray[0:16])
        bit_pos = 16
        freqs = {}
        for _ in range(num_unique):
            sym = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32
            freq = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32
            freqs[sym] = freq
        return freqs, bit_pos

    def decode_lz77_sequences(self, encoded_bitarray: BitArray):
        """Decode all LZ77 sequence components with tANS."""
        bit_pos = 0
        decoded_lists = []

        for _ in range(3):  # literal_counts, match_lengths, match_offsets
            num_items = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32

            if num_items == 0:
                decoded_lists.append([])
                continue

            freqs, freq_bits = self._decode_frequencies(encoded_bitarray[bit_pos:])
            bit_pos += freq_bits

            decoder = TANSDecoder(table_log=self.table_log)
            stream_bits = encoded_bitarray[bit_pos:]
            decoded, bits_used = decoder.decode(stream_bits, num_items, freqs)
            decoded_lists.append(decoded)

            bit_pos += bits_used

        literal_counts, match_lengths, match_offsets = decoded_lists

        lz77_sequences = [
            LZ77Sequence(lc, ml, mo)
            for lc, ml, mo in zip(literal_counts, match_lengths, match_offsets)
        ]
        return lz77_sequences, bit_pos

    def decode_literals(self, encoded_bitarray: BitArray):
        """Decode literals using tANS and the compact counts header."""
        freqs, num_literals, bit_pos = _decode_literal_counts_header(encoded_bitarray)

        if num_literals == 0:
            return [], bit_pos

        decoder = TANSDecoder(table_log=self.table_log)
        payload = encoded_bitarray[bit_pos:]
        literals, bits_used = decoder.decode(payload, num_literals, freqs)

        bits_consumed = bit_pos + bits_used
        return literals, bits_consumed


class LZ77EncoderTANSAll(LZ77Encoder):
    """LZ77 encoder with tANS for all LZ77 streams."""

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
        self.streams_encoder = LZ77StreamsEncoderTANSAll(table_log=table_log)


class LZ77DecoderTANSAll(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSAll."""

    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSAll(table_log=table_log)


# ---------------------------------------------------------------------------
# Header overhead computation for literals
# ---------------------------------------------------------------------------


def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    """Compute model header bits for empirical Huffman on literals.

    We mirror EmpiricalIntHuffmanEncoder's behavior but only count the
    model overhead:
        [32 bits: size_of_counts_encoding] + [counts_encoding_bits]
    """
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    counts_list = _build_literal_counts_list(literals)
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return ENCODED_BLOCK_SIZE_HEADER_BITS + len(counts_encoding)


def compute_literal_header_bits_tans(literals: List[int], table_log: int) -> int:
    """Compute model header bits for tANS on literals.

    Header includes:
        [32 bits: num_literals] + [16 bits: num_unique_symbols] +
        [num_unique_symbols * (32 + 32) bits: (symbol, frequency) pairs]
    """
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    # For fairness, tANS uses the same counts header format as empirical Huffman.
    counts_list = _build_literal_counts_list(literals)
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return ENCODED_BLOCK_SIZE_HEADER_BITS + len(counts_encoding)


# ---------------------------------------------------------------------------
# Benchmark: per-file compression & header comparison
# ---------------------------------------------------------------------------


def run_single_file_benchmark(
    path: str, block_size: int = 100_000, table_logs: List[int] = [10]
) -> None:
    raw_size = os.path.getsize(path)

    print(f"\n{'=' * 70}")
    print(f"Benchmark on file: {path}")
    print(f"{'=' * 70}")
    print(f"Raw size: {raw_size:,} bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---------------- Baseline LZ77 (Empirical Huffman) ----------------
        print("\n[1/2] Running baseline LZ77 (Empirical Huffman)...")
        base_enc = LZ77Encoder()
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        base_enc.encode_file(path, base_encoded_path, block_size=block_size)
        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)

        # Results dictionary
        results = {
            "Baseline (Empirical Huffman)": {
                "size": baseline_size,
                "ratio": baseline_size / raw_size if raw_size > 0 else 0.0,
            }
        }

        # ------------- LZ77 with tANS on different table_log values -------------
        for table_log in table_logs:
            # tANS literals only
            print(f"\n[2/2] Running LZ77 + tANS (literals, table_log={table_log})...")
            tans_lit_enc = LZ77EncoderTANSLiterals(table_log=table_log)
            tans_lit_dec = LZ77DecoderTANSLiterals(table_log=table_log)

            tans_lit_encoded_path = os.path.join(
                tmpdir, f"tans_lit_{table_log}.lz77"
            )
            tans_lit_decoded_path = os.path.join(
                tmpdir, f"tans_lit_{table_log}.dec"
            )

            tans_lit_enc.encode_file(
                path, tans_lit_encoded_path, block_size=block_size
            )
            tans_lit_dec.decode_file(tans_lit_encoded_path, tans_lit_decoded_path)

            with open(path, "rb") as f_in, open(
                tans_lit_decoded_path, "rb"
            ) as f_out:
                assert (
                    f_in.read() == f_out.read()
                ), f"tANS-literals (table_log={table_log}) decode mismatch!"

            tans_lit_size = os.path.getsize(tans_lit_encoded_path)
            results[f"tANS literals (table_log={table_log})"] = {
                "size": tans_lit_size,
                "ratio": tans_lit_size / raw_size if raw_size > 0 else 0.0,
            }
            # tANS all streams (disabled by default; see tans_ablation_results.md for tests)
            # print(f"\n[3/3] Running LZ77 + tANS (all, table_log={table_log})...")
            # tans_all_enc = LZ77EncoderTANSAll(table_log=table_log)
            # tans_all_dec = LZ77DecoderTANSAll(table_log=table_log)
            #
            # tans_all_encoded_path = os.path.join(
            #     tmpdir, f"tans_all_{table_log}.lz77"
            # )
            # tans_all_decoded_path = os.path.join(tmpdir, f"tans_all_{table_log}.dec")
            #
            # tans_all_enc.encode_file(
            #     path, tans_all_encoded_path, block_size=block_size
            # )
            # tans_all_dec.decode_file(tans_all_encoded_path, tans_all_decoded_path)
            #
            # with open(path, "rb") as f_in, open(
            #     tans_all_decoded_path, "rb"
            # ) as f_out:
            #     assert (
            #         f_in.read() == f_out.read()
            #     ), f"tANS-all (table_log={table_log}) decode mismatch!"
            #
            # tans_all_size = os.path.getsize(tans_all_encoded_path)
            # results[f"tANS all streams (table_log={table_log})"] = {
            #     "size": tans_all_size,
            #     "ratio": tans_all_size / raw_size if raw_size > 0 else 0.0,
            # }

    # ---------------- Print Results ----------------
    print(f"\n{'=' * 70}")
    print("COMPRESSION RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Method':<40} {'Size (bytes)':>12} {'Ratio':>10} {'vs Baseline':>12}")
    print("-" * 70)

    baseline_size = results["Baseline (Empirical Huffman)"]["size"]

    for method, data in results.items():
        size = data["size"]
        ratio = data["ratio"]
        vs_baseline = (size - baseline_size) / baseline_size * 100 if baseline_size > 0 else 0
        sign = "+" if vs_baseline > 0 else ""
        print(
            f"{method:<40} {size:>12,} {ratio:>9.4f} {sign}{vs_baseline:>10.2f}%"
        )

    # ---------------- Header overhead for literals ----------------
    print(f"\n{'=' * 70}")
    print("HEADER OVERHEAD ANALYSIS (single-block parse)")
    print(f"{'=' * 70}")

    with open(path, "rb") as f:
        data_bytes = list(f.read())
    data_block = DataBlock(data_bytes)

    parser = LZ77Encoder(
        min_match_length=DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
    )
    seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

    print(f"Number of literals in stream: {len(lits):,}")
    print(f"Number of unique literal values: {len(set(lits))}")
    print()

    emp_header_bits = compute_literal_header_bits_empirical(lits)
    print(f"{'Method':<40} {'Header (bits)':>15} {'Bytes':>10} {'vs Empirical':>12}")
    print("-" * 70)
    print(
        f"{'Empirical Huffman':<40} {emp_header_bits:>15,} {emp_header_bits//8:>10,} {'baseline':>12}"
    )

    for table_log in table_logs:
        tans_header_bits = compute_literal_header_bits_tans(lits, table_log)
        vs_emp = (
            (tans_header_bits - emp_header_bits) / emp_header_bits * 100
            if emp_header_bits > 0
            else 0
        )
        sign = "+" if vs_emp > 0 else ""
        print(
            f"{'tANS (table_log=' + str(table_log) + ')':<40} "
            f"{tans_header_bits:>15,} {tans_header_bits//8:>10,} "
            f"{sign}{vs_emp:>10.2f}%"
        )

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline LZ77 (empirical Huffman) vs. LZ77 with tANS "
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
    parser.add_argument(
        "-t",
        "--table_log",
        nargs="+",
        type=int,
        default=[10],
        help="Table log values to test for tANS (default: 10). Example: -t 8 10 12",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LZ77 + tANS BENCHMARK")
    print("=" * 70)
    print(f"Table log values to test: {args.table_log}")
    print(f"Block size: {args.block_size:,} bytes")

    for path in args.input:
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipping.")
            continue
        run_single_file_benchmark(
            path, block_size=args.block_size, table_logs=args.table_log
        )


if __name__ == "__main__":
    main()