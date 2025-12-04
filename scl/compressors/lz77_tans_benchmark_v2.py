"""
Benchmark script to compare baseline LZ77 (empirical Huffman)
vs. LZ77 with tANS on different streams.

Usage:
    python lz77_tans_benchmark_v2.py -i path/to/file1 path/to/file2 ...
    python lz77_tans_benchmark_v2.py -i path/to/file1 --table_log 8 10 12

This script does three things:
  1. Defines LZ77 stream encoders/decoders that use tANS:
        - only for literals
        - for literals + literal_count + match_length + match_offset (all streams)
  2. For each input file, compresses it with:
        - baseline LZ77Encoder / LZ77Decoder
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

from scl.compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from scl.compressors.huffman_coder import HuffmanEncoder
from scl.compressors.lz77 import (
    LZ77Encoder,
    LZ77Decoder,
    LZ77StreamsEncoder,
    LZ77StreamsDecoder,
    DEFAULT_MIN_MATCH_LEN,
    DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
)
from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray

# Import tANS implementation
from tans_lz77_coder import (
    TANSEncoder,
    TANSDecoder,
    LZ77TANSStreamsEncoder,
    LZ77TANSStreamsDecoder,
)

ENCODED_BLOCK_SIZE_HEADER_BITS = 32


# ---------------------------------------------------------------------------
# tANS LZ77 streams: literals only, and "all" streams.
# ---------------------------------------------------------------------------


class LZ77StreamsEncoderTANSLiteralsOnly(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses tANS for literals only.

    Literal counts, match lengths, and match offsets are encoded exactly
    as in the baseline implementation, but literals use TANSEncoder.
    """

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log
        self.tans_encoder = TANSEncoder(table_log=table_log)

    def encode_literals(self, literals: List[int]) -> BitArray:
        """Encode literals using tANS with frequency table."""
        if not literals:
            return BitArray([])

        # Encode with tANS
        encoded_data = self.tans_encoder.encode(literals)

        # Store frequency table
        freqs = Counter(literals)
        freq_encoding = self._encode_frequencies(freqs)

        # Store number of literals and frequency table
        from scl.utils.bitarray_utils import uint_to_bitarray
        num_literals_bits = uint_to_bitarray(len(literals), 32)

        return num_literals_bits + freq_encoding + encoded_data

    def _encode_frequencies(self, freqs: dict) -> BitArray:
        """Encode frequency table for decoder."""
        from scl.utils.bitarray_utils import uint_to_bitarray
        result = uint_to_bitarray(len(freqs), 16)
        for sym, freq in sorted(freqs.items()):
            result += uint_to_bitarray(sym, 16)  # Use 16 bits for byte values (0-255)
            result += uint_to_bitarray(freq, 32)
        return result


class LZ77StreamsDecoderTANSLiteralsOnly(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderTANSLiteralsOnly."""

    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log
        self.tans_decoder = TANSDecoder(table_log=table_log)

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        """Decode literals using tANS."""
        from scl.utils.bitarray_utils import bitarray_to_uint

        # Read number of literals
        num_literals = bitarray_to_uint(encoded_bitarray[0:32])
        bit_pos = 32

        if num_literals == 0:
            return [], 32

        # Decode frequency table
        freqs, freq_bits = self._decode_frequencies(encoded_bitarray[bit_pos:])
        bit_pos += freq_bits

        # Decode literals with tANS
        literals = self.tans_decoder.decode(
            encoded_bitarray[bit_pos:], num_literals, freqs
        )

        # Return literals and total bits consumed
        return literals, len(encoded_bitarray)

    def _decode_frequencies(self, encoded_bitarray: BitArray) -> Tuple[dict, int]:
        """Decode frequency table."""
        from scl.utils.bitarray_utils import bitarray_to_uint
        num_unique = bitarray_to_uint(encoded_bitarray[0:16])
        bit_pos = 16
        freqs = {}
        for _ in range(num_unique):
            sym = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 16])
            bit_pos += 16
            freq = bitarray_to_uint(encoded_bitarray[bit_pos : bit_pos + 32])
            bit_pos += 32
            freqs[sym] = freq
        return freqs, bit_pos


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
        self.streams_encoder = LZ77StreamsEncoderTANSLiteralsOnly(table_log=table_log)


class LZ77DecoderTANSLiterals(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSLiterals."""

    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSLiteralsOnly(table_log=table_log)


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
        self.streams_encoder = LZ77TANSStreamsEncoder(table_log=table_log)


class LZ77DecoderTANSAll(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSAll."""

    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77TANSStreamsDecoder(table_log=table_log)


# ---------------------------------------------------------------------------
# Header overhead computation for literals
# ---------------------------------------------------------------------------


def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    """Compute model header bits for empirical Huffman on literals."""
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


def compute_literal_header_bits_tans(literals: List[int], table_log: int) -> int:
    """Compute model header bits for tANS on literals."""
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    freqs = Counter(literals)
    num_unique = len(freqs)

    # Format: [32 bits: num_literals] + [16 bits: num_unique] + 
    #         [num_unique * (16 + 32) bits: (symbol, frequency) pairs]
    header_bits = 32 + 16 + (num_unique * 48)
    return header_bits


# ---------------------------------------------------------------------------
# Benchmark: per-file compression & header comparison
# ---------------------------------------------------------------------------


def run_single_file_benchmark(
    path: str, block_size: int = 100_000, table_logs: List[int] = [10]
) -> None:
    raw_size = os.path.getsize(path)

    print(f"\n=== Benchmark on file: {path} ===")
    print(f"Raw size: {raw_size:,} bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---------------- Baseline LZ77 (Empirical Huffman) ----------------
        print("\n[1/3] Running baseline LZ77 (Empirical Huffman)...")
        base_enc = LZ77Encoder()
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        base_enc.encode_file(path, base_encoded_path, block_size=block_size)
        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)

        results = {}
        results["Baseline (Empirical Huffman)"] = {
            "size": baseline_size,
            "ratio": baseline_size / raw_size if raw_size > 0 else 0.0,
        }

        # ------------- LZ77 with tANS on different table_log values -------------
        for table_log in table_logs:
            # tANS literals only
            print(f"\n[2/3] Running LZ77 + tANS (literals, table_log={table_log})...")
            tans_lit_enc = LZ77EncoderTANSLiterals(table_log=table_log)
            tans_lit_dec = LZ77DecoderTANSLiterals(table_log=table_log)

            tans_lit_encoded_path = os.path.join(
                tmpdir, f"tans_lit_{table_log}.lz77"
            )
            tans_lit_decoded_path = os.path.join(tmpdir, f"tans_lit_{table_log}.dec")

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

            # tANS all streams
            print(f"\n[3/3] Running LZ77 + tANS (all, table_log={table_log})...")
            tans_all_enc = LZ77EncoderTANSAll(table_log=table_log)
            tans_all_dec = LZ77DecoderTANSAll(table_log=table_log)

            tans_all_encoded_path = os.path.join(tmpdir, f"tans_all_{table_log}.lz77")
            tans_all_decoded_path = os.path.join(tmpdir, f"tans_all_{table_log}.dec")

            tans_all_enc.encode_file(
                path, tans_all_encoded_path, block_size=block_size
            )
            tans_all_dec.decode_file(tans_all_encoded_path, tans_all_decoded_path)

            with open(path, "rb") as f_in, open(
                tans_all_decoded_path, "rb"
            ) as f_out:
                assert (
                    f_in.read() == f_out.read()
                ), f"tANS-all (table_log={table_log}) decode mismatch!"

            tans_all_size = os.path.getsize(tans_all_encoded_path)
            results[f"tANS all streams (table_log={table_log})"] = {
                "size": tans_all_size,
                "ratio": tans_all_size / raw_size if raw_size > 0 else 0.0,
            }

    # ---------------- Print Results ----------------
    baseline_ratio = baseline_size / raw_size if raw_size > 0 else 0.0

    print("\nCompressed sizes (bytes):")
    print(f"  Baseline LZ77                     : {baseline_size:,}")
    for method, data in results.items():
        if method != "Baseline (Empirical Huffman)":
            size = data["size"]
            print(f"  {method:35s}: {size:,}")

    print("\nCompression ratios (compressed/raw):")
    print(f"  Baseline LZ77                     : {baseline_ratio:.4f}")
    for method, data in results.items():
        if method != "Baseline (Empirical Huffman)":
            ratio = data["ratio"]
            print(f"  {method:35s}: {ratio:.4f}")

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

    print("\nLiterals header overhead (single-block parse):")
    print(f"  #literals in stream                : {len(lits):,}")
    print(f"  #unique literal values             : {len(set(lits))}")
    print(f"  Empirical Huffman header bits      : {emp_header_bits:,}")

    for table_log in table_logs:
        tans_header_bits = compute_literal_header_bits_tans(lits, table_log)
        if emp_header_bits > 0:
            ratio = tans_header_bits / emp_header_bits
            print(
                f"  tANS (table_log={table_log:2d}) header bits  : {tans_header_bits:,} "
                f"(ratio: {ratio:.4f})"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline LZ77 vs. LZ77 with tANS "
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

    for path in args.input:
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipping.")
            continue
        run_single_file_benchmark(
            path, block_size=args.block_size, table_logs=args.table_log
        )


if __name__ == "__main__":
    main()

