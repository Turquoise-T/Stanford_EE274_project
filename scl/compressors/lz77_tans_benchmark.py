"""
Benchmark script to compare baseline LZ77 (empirical Huffman)
vs. LZ77 with tANS on different streams.

Usage:
    python lz77_tans_benchmark.py -i path/to/file1 path/to/file2 ...

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
        - TANSEncoderByte (tANS)
     for the literals stream only.
"""

import argparse
import os
import tempfile
from typing import List, Tuple

from scl.compressors.lz77 import (
    LZ77Encoder,
    LZ77Decoder,
    LZ77StreamsEncoder,
    LZ77StreamsDecoder,
    LogScaleBinnedIntegerEncoder,
    LogScaleBinnedIntegerDecoder,
    DEFAULT_MIN_MATCH_LEN,
    DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
)
from tans_int_coder import TANSEncoderInt, TANSDecoderInt
from tans_byte_coder import TANSEncoderByte, TANSDecoderByte
from scl.compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.utils.bitarray_utils import BitArray

ENCODED_BLOCK_SIZE_HEADER_BITS = 32


# ---------------------------------------------------------------------------
# tANS log-scale-binned integer encoder/decoder
# (for literal_count, match_length, match_offset streams).
# ---------------------------------------------------------------------------


class TANSLogScaleBinnedIntegerEncoder(DataEncoder):
    """
    Same as LogScaleBinnedIntegerEncoder, but uses TANSEncoderInt
    for the binned integers instead of EmpiricalIntHuffmanEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset

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
        bins_encoding = TANSEncoderInt.encode(bins)

        # Encode residuals as raw bits
        from scl.utils.bitarray_utils import uint_to_bitarray

        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        return bins_encoding + residuals_encoding


class TANSLogScaleBinnedIntegerDecoder(DataDecoder):
    """
    Decoder for TANSLogScaleBinnedIntegerEncoder.
    """

    def __init__(self, offset: int = 0, max_num_bins: int = 32):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset

    def decode_block(self, encoded_bitarray: BitArray):
        from scl.utils.bitarray_utils import bitarray_to_uint

        # First decode the bin sequence (tANS)
        bins_decoded, num_bits_consumed = TANSDecoderInt.decode(encoded_bitarray, 0)
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
# tANS LZ77 streams: literals only, and "all" streams.
# ---------------------------------------------------------------------------


class LZ77StreamsEncoderTANSLiterals(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses tANS for literals.

    Literal counts, match lengths, and match offsets are encoded exactly
    as in the baseline implementation (log-scale binned integers), but
    literals (byte values 0..255) use TANSEncoderByte instead
    of EmpiricalIntHuffmanEncoder.
    """

    def encode_literals(self, literals: List[int]) -> BitArray:
        return TANSEncoderByte.encode(literals)


class LZ77StreamsDecoderTANSLiterals(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderTANSLiterals."""

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        literals, num_bits_consumed = TANSDecoderByte.decode(encoded_bitarray, 0)
        return literals, num_bits_consumed


class LZ77EncoderTANSLiterals(LZ77Encoder):
    """LZ77 encoder with tANS for literals."""

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
        self.streams_encoder = LZ77StreamsEncoderTANSLiterals()


class LZ77DecoderTANSLiterals(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSLiterals."""

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSLiterals()


class LZ77StreamsEncoderTANSAll(LZ77StreamsEncoder):
    """LZ77StreamsEncoder variant that uses tANS for all streams:
        - literal_count
        - match_length
        - match_offset
        - literals
    """

    def encode_lz77_sequences(self, lz77_sequences):
        coder = TANSLogScaleBinnedIntegerEncoder(
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
        return TANSEncoderByte.encode(literals)


class LZ77StreamsDecoderTANSAll(LZ77StreamsDecoder):
    """Decoder matching LZ77StreamsEncoderTANSAll."""

    def decode_lz77_sequences(self, encoded_bitarray: BitArray):
        coder = TANSLogScaleBinnedIntegerDecoder(
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
        literals, num_bits_consumed = TANSDecoderByte.decode(encoded_bitarray, 0)
        return literals, num_bits_consumed


class LZ77EncoderTANSAll(LZ77Encoder):
    """LZ77 encoder with tANS for all LZ77 streams."""

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
        self.streams_encoder = LZ77StreamsEncoderTANSAll()


class LZ77DecoderTANSAll(LZ77Decoder):
    """LZ77 decoder matching LZ77EncoderTANSAll."""

    def __init__(self, initial_window: List[int] = None):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSAll()


# ---------------------------------------------------------------------------
# Header overhead computation for literals
# ---------------------------------------------------------------------------


def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    """Compute model header bits for empirical Huffman on literals.

    We mirror EmpiricalIntHuffmanEncoder's behavior but only count the
    model overhead:
        [32 bits: size_of_counts_encoding] + [counts_encoding_bits]

    The extra 32 bits for value-encoding size are shared by empirical
    and tANS encoders and are not counted here.
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


def compute_literal_header_bits_tans(literals: List[int]) -> int:
    """Compute model header bits for tANS on literals.
    
    tANS header includes the frequency table serialization.
    We encode the full block to get the actual header size.
    """
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS
    
    from collections import Counter
    from tans_core import serialize_tables, build_tables
    
    # Build tables to get frequency info
    table_size, log_size, dec_table, encode_table, freq = build_tables(literals)
    
    # Serialize just the frequency table to get header size
    header_bits = len(serialize_tables(freq, dec_table, encode_table))
    
    # Add the state and bitstream size fields (32 bits each)
    header_bits += 32 + 32
    
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

        base_enc.encode_file(path, base_encoded_path, block_size=block_size)
        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)

        # ------------- LZ77 with tANS literals only -------------
        tans_lit_enc = LZ77EncoderTANSLiterals()
        tans_lit_dec = LZ77DecoderTANSLiterals()

        tans_lit_encoded_path = os.path.join(tmpdir, "tans_lit.lz77")
        tans_lit_decoded_path = os.path.join(tmpdir, "tans_lit.dec")

        tans_lit_enc.encode_file(path, tans_lit_encoded_path, block_size=block_size)
        tans_lit_dec.decode_file(tans_lit_encoded_path, tans_lit_decoded_path)

        with open(path, "rb") as f_in, open(tans_lit_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "tANS-literals LZ77 decode mismatch!"

        tans_lit_size = os.path.getsize(tans_lit_encoded_path)

        # ------------- LZ77 with tANS on all streams -------------
        tans_all_enc = LZ77EncoderTANSAll()
        tans_all_dec = LZ77DecoderTANSAll()

        tans_all_encoded_path = os.path.join(tmpdir, "tans_all.lz77")
        tans_all_decoded_path = os.path.join(tmpdir, "tans_all.dec")

        tans_all_enc.encode_file(path, tans_all_encoded_path, block_size=block_size)
        tans_all_dec.decode_file(tans_all_encoded_path, tans_all_decoded_path)

        with open(path, "rb") as f_in, open(tans_all_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "tANS-all LZ77 decode mismatch!"

        tans_all_size = os.path.getsize(tans_all_encoded_path)

    baseline_ratio = baseline_size / raw_size if raw_size > 0 else 0.0
    tans_lit_ratio = tans_lit_size / raw_size if raw_size > 0 else 0.0
    tans_all_ratio = tans_all_size / raw_size if raw_size > 0 else 0.0

    print("Compressed sizes (bytes):")
    print(f"  Baseline LZ77      : {baseline_size}")
    print(f"  tANS (literals)    : {tans_lit_size}")
    print(f"  tANS (all)         : {tans_all_size}")
    print("Compression ratios (compressed/raw):")
    print(f"  Baseline LZ77      : {baseline_ratio:.4f}")
    print(f"  tANS (literals)    : {tans_lit_ratio:.4f}")
    print(f"  tANS (all)         : {tans_all_ratio:.4f}")

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
    tans_header_bits = compute_literal_header_bits_tans(lits)

    print("\nLiterals header overhead (single-block parse):")
    print(f"  #literals in stream         : {len(lits)}")
    print(f"  Empirical Huffman header bits : {emp_header_bits}")
    print(f"  tANS header bits              : {tans_header_bits}")
    if emp_header_bits > 0:
        print(
            f"  tANS / Empirical header       : "
            f"{tans_header_bits / emp_header_bits:.4f}"
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
    args = parser.parse_args()

    for path in args.input:
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipping.")
            continue
        run_single_file_benchmark(path, block_size=args.block_size)


if __name__ == "__main__":
    main()

