import argparse
import os
import tempfile
import time
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
try:
    from scl.compressors.tans_lz77_coder import TANSEncoder, TANSDecoder
except ImportError:
    from tans_lz77_coder import TANSEncoder, TANSDecoder

ENCODED_BLOCK_SIZE_HEADER_BITS = 32  # Same as canonical Huffman


class TANSLogScaleBinnedIntegerEncoder(DataEncoder):
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

        bins_encoding = self.tans_encoder.encode(bins)

        freqs = Counter(bins)
        freq_encoding = self._encode_frequencies(freqs)

        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        return freq_encoding + bins_encoding + residuals_encoding

    def _encode_frequencies(self, freqs: dict) -> BitArray:
        result = uint_to_bitarray(len(freqs), 16)
        for sym, freq in sorted(freqs.items()):
            result += uint_to_bitarray(sym, 32)
            result += uint_to_bitarray(freq, 32)
        return result


class TANSLogScaleBinnedIntegerDecoder(DataDecoder):
    def __init__(self, offset: int = 0, max_num_bins: int = 32, table_log: int = 10):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.table_log = table_log
        self.tans_decoder = TANSDecoder(table_log=table_log)

    def decode_block(self, encoded_bitarray: BitArray):
        freqs, bits_consumed = self._decode_frequencies(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[bits_consumed:]

        num_symbols = sum(freqs.values())
        bins_decoded, bins_bits_used = self.tans_decoder.decode(encoded_bitarray, num_symbols, freqs)
        encoded_bitarray = encoded_bitarray[bins_bits_used:]

        decoded: List[int] = []
        bit_position = 0

        for encoded_bin in bins_decoded:
            if encoded_bin < self.offset:
                decoded.append(encoded_bin)
            else:
                encoded_bin_minus_offset = encoded_bin - self.offset
                log_val_plus_1 = encoded_bin_minus_offset
                num_bits = log_val_plus_1

                if num_bits == 0:
                    residual = 0
                else:
                    residual = bitarray_to_uint(encoded_bitarray[bit_position : bit_position + num_bits])
                    bit_position += num_bits

                decoded_val = self.offset + 2**log_val_plus_1 + residual - 1
                decoded.append(decoded_val)

        return DataBlock(decoded), bits_consumed + bins_bits_used + bit_position

    def _decode_frequencies(self, encoded_bitarray: BitArray) -> Tuple[dict, int]:
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


def _build_literal_counts_list(literals: List[int]) -> List[int]:
    counts = Counter(literals)
    return [counts.get(i, 0) for i in range(256)]


def _encode_literal_counts_header_from_counts(counts_list: List[int]) -> BitArray:
    if not any(counts_list):
        return uint_to_bitarray(0, ENCODED_BLOCK_SIZE_HEADER_BITS)

    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return (
        uint_to_bitarray(len(counts_encoding), ENCODED_BLOCK_SIZE_HEADER_BITS)
        + counts_encoding
    )


def _decode_literal_counts_header(encoded_bitarray: BitArray) -> Tuple[dict, int, int]:
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
    assert len(counts_list) == 256

    freqs = {i: c for i, c in enumerate(counts_list) if c > 0}
    num_literals = sum(counts_list)

    bit_pos += counts_encoding_size
    return freqs, num_literals, bit_pos


class LZ77StreamsEncoderTANSLiterals(LZ77StreamsEncoder):
    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def encode_literals(self, literals: List[int]) -> BitArray:
        counts_list = _build_literal_counts_list(literals)
        header = _encode_literal_counts_header_from_counts(counts_list)

        if not literals:
            return header

        encoder = TANSEncoder(table_log=self.table_log)
        encoded_data = encoder.encode(literals)

        return header + encoded_data


class LZ77StreamsDecoderTANSLiterals(LZ77StreamsDecoder):
    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def decode_literals(self, encoded_bitarray: BitArray) -> Tuple[List[int], int]:
        freqs, num_literals, bit_pos = _decode_literal_counts_header(encoded_bitarray)

        if num_literals == 0:
            return [], bit_pos

        decoder = TANSDecoder(table_log=self.table_log)
        payload = encoded_bitarray[bit_pos:]
        literals, bits_used = decoder.decode(payload, num_literals, freqs)

        bits_consumed = bit_pos + bits_used
        return literals, bits_consumed


class LZ77EncoderTANSLiterals(LZ77Encoder):
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
        self.streams_encoder = LZ77StreamsEncoderTANSLiterals(table_log=table_log)


class LZ77DecoderTANSLiterals(LZ77Decoder):
    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSLiterals(table_log=table_log)


class LZ77StreamsEncoderTANSAll(LZ77StreamsEncoder):
    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def encode_lz77_sequences(self, lz77_sequences):
        encoded_bitarray = BitArray()

        literal_counts = [seq.literal_count for seq in lz77_sequences]
        match_lengths = [seq.match_length for seq in lz77_sequences]
        match_offsets = [seq.match_offset for seq in lz77_sequences]

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
        result = uint_to_bitarray(len(freqs), 16)
        for sym, freq in sorted(freqs.items()):
            result += uint_to_bitarray(sym, 32)
            result += uint_to_bitarray(freq, 32)
        return result

    def encode_literals(self, literals: List[int]) -> BitArray:
        counts_list = _build_literal_counts_list(literals)
        header = _encode_literal_counts_header_from_counts(counts_list)

        if not literals:
            return header

        encoder = TANSEncoder(table_log=self.table_log)
        encoded_data = encoder.encode(literals)

        return header + encoded_data


class LZ77StreamsDecoderTANSAll(LZ77StreamsDecoder):
    def __init__(self, table_log: int = 10):
        super().__init__()
        self.table_log = table_log

    def _decode_frequencies(self, encoded_bitarray: BitArray) -> Tuple[dict, int]:
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
        bit_pos = 0
        decoded_lists = []

        for _ in range(3):
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
        freqs, num_literals, bit_pos = _decode_literal_counts_header(encoded_bitarray)

        if num_literals == 0:
            return [], bit_pos

        decoder = TANSDecoder(table_log=self.table_log)
        payload = encoded_bitarray[bit_pos:]
        literals, bits_used = decoder.decode(payload, num_literals, freqs)

        bits_consumed = bit_pos + bits_used
        return literals, bits_consumed


class LZ77EncoderTANSAll(LZ77Encoder):
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
    def __init__(self, initial_window: List[int] = None, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        self.streams_decoder = LZ77StreamsDecoderTANSAll(table_log=table_log)

def compute_literal_header_bits_empirical(literals: List[int]) -> int:
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS

    counts_list = _build_literal_counts_list(literals)
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return ENCODED_BLOCK_SIZE_HEADER_BITS + len(counts_encoding)


def compute_literal_header_bits_tans(literals: List[int], table_log: int) -> int:
    if not literals:
        return ENCODED_BLOCK_SIZE_HEADER_BITS
    return len(_encode_literal_counts_header_from_counts(_build_literal_counts_list(literals)))


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

        # Measure compression (encode) time for baseline
        start_time = time.perf_counter()
        base_enc.encode_file(path, base_encoded_path, block_size=block_size)
        baseline_encode_time = time.perf_counter() - start_time

        base_dec.decode_file(base_encoded_path, base_decoded_path)

        with open(path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)
        baseline_speed = (
            (raw_size / baseline_encode_time) / (1024 * 1024)
            if baseline_encode_time > 0 and raw_size > 0
            else 0.0
        )

        # Results dictionary
        results = {
            "Baseline (Empirical Huffman)": {
                "size": baseline_size,
                "ratio": baseline_size / raw_size if raw_size > 0 else 0.0,
                "enc_speed": baseline_speed,
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

            # Measure compression (encode) time for tANS literals-only
            start_time = time.perf_counter()
            tans_lit_enc.encode_file(
                path, tans_lit_encoded_path, block_size=block_size
            )
            tans_lit_encode_time = time.perf_counter() - start_time
            tans_lit_dec.decode_file(tans_lit_encoded_path, tans_lit_decoded_path)

            with open(path, "rb") as f_in, open(
                tans_lit_decoded_path, "rb"
            ) as f_out:
                assert (
                    f_in.read() == f_out.read()
                ), f"tANS-literals (table_log={table_log}) decode mismatch!"

            tans_lit_size = os.path.getsize(tans_lit_encoded_path)
            tans_lit_speed = (
                (raw_size / tans_lit_encode_time) / (1024 * 1024)
                if tans_lit_encode_time > 0 and raw_size > 0
                else 0.0
            )
            results[f"tANS literals (table_log={table_log})"] = {
                "size": tans_lit_size,
                "ratio": tans_lit_size / raw_size if raw_size > 0 else 0.0,
                "enc_speed": tans_lit_speed,
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

    # ---------------- Calculate header sizes first ----------------
    with open(path, "rb") as f:
        data_bytes = list(f.read())
    data_block = DataBlock(data_bytes)

    parser = LZ77Encoder(
        min_match_length=DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
    )
    seqs, lits = parser.lz77_parse_and_generate_sequences(data_block)

    # Calculate header sizes
    emp_header_bits = compute_literal_header_bits_empirical(lits)
    emp_header_bytes = emp_header_bits // 8
    
    tans_header_bytes_dict = {}
    for table_log in table_logs:
        tans_header_bits = compute_literal_header_bits_tans(lits, table_log)
        tans_header_bytes_dict[table_log] = tans_header_bits // 8

    # ---------------- Print Results with Header Info ----------------
    print(f"\n{'=' * 85}")
    print("COMPRESSION RESULTS")
    print(f"{'=' * 85}")
    print(
        f"{'Method':<40} {'Size (bytes)':>12} {'Header':>10} "
        f"{'Speed(MB/s)':>12} {'Ratio':>10} {'vs Baseline':>12}"
    )
    print("-" * 85)

    baseline_size = results["Baseline (Empirical Huffman)"]["size"]

    for method, data in results.items():
        size = data["size"]
        ratio = data["ratio"]
        speed = data.get("enc_speed", 0.0)
        vs_baseline = (size - baseline_size) / baseline_size * 100 if baseline_size > 0 else 0
        sign = "+" if vs_baseline > 0 else ""
        
        # Get header size
        if method == "Baseline (Empirical Huffman)":
            header_str = f"{emp_header_bytes:,}"
        else:
            # Extract table_log from method name
            for table_log in table_logs:
                if f"table_log={table_log}" in method:
                    header_str = f"{tans_header_bytes_dict[table_log]:,}"
                    break
        
        print(
            f"{method:<40} {size:>12,} {header_str:>10} "
            f"{speed:>12.2f} {ratio:>9.4f} {sign}{vs_baseline:>10.2f}%"
        )

    # ---------------- Detailed 4-dimensional comparison ----------------
    print(f"\n{'=' * 90}")
    print("DETAILED COMPARISON: tANS vs Baseline (single-block parse)")
    print(f"{'=' * 90}")

    print(f"Number of literals in stream: {len(lits):,}")
    print(f"Number of unique literal values: {len(set(lits))}")
    print()

    # Calculate metrics for each method
    comparison_data = []
    
    # Empirical Huffman (Baseline)
    emp_total_bytes = results["Baseline (Empirical Huffman)"]["size"]
    emp_ratio = results["Baseline (Empirical Huffman)"]["ratio"]
    
    comparison_data.append({
        "method": "Huffman (Empirical)",
        "header_bytes": emp_header_bytes,
        "total_bytes": emp_total_bytes,
        "ratio": emp_ratio
    })
    
    # tANS
    for table_log in table_logs:
        tans_header_bits = compute_literal_header_bits_tans(lits, table_log)
        tans_header_bytes = tans_header_bits // 8
        
        method_key = f"tANS literals (table_log={table_log})"
        tans_total_bytes = results[method_key]["size"]
        tans_ratio = results[method_key]["ratio"]
        
        comparison_data.append({
            "method": f"tANS (log={table_log})",
            "header_bytes": tans_header_bytes,
            "total_bytes": tans_total_bytes,
            "ratio": tans_ratio
        })
    
    # Print comparison table (4 dimensions: Header, Payload, Total, Ratio)
    print(f"{'Method':<25} {'Header (bytes)':>18} {'Payload (bytes)':>18} {'Total (bytes)':>16} {'Ratio':>10}")
    print("-" * 95)
    
    for data in comparison_data:
        payload_bytes = data['total_bytes'] - data['header_bytes']
        print(
            f"{data['method']:<25} "
            f"{data['header_bytes']:>18,} "
            f"{payload_bytes:>18,} "
            f"{data['total_bytes']:>16,} "
            f"{data['ratio']:>10.4f}"
        )
    
    print(f"{'=' * 95}")
    
    # Add payload efficiency analysis
    print("\nPAYLOAD EFFICIENCY ANALYSIS")
    print("=" * 95)
    
    # Find Huffman baseline payload
    huffman_data = None
    for data in comparison_data:
        if "Huffman" in data['method']:
            huffman_data = data
            break
    
    if huffman_data:
        huffman_payload = huffman_data['total_bytes'] - huffman_data['header_bytes']
        print(f"{'Method':<25} {'Payload (bytes)':>18} {'vs Huffman Payload':>20} {'Payload Efficiency':>18}")
        print("-" * 95)
        
        for data in comparison_data:
            payload_bytes = data['total_bytes'] - data['header_bytes']
            if huffman_payload > 0:
                payload_ratio = payload_bytes / huffman_payload
                efficiency_pct = (1 - payload_ratio) * 100 if payload_ratio != 1 else 0
                efficiency_str = f"{efficiency_pct:+.1f}%"
            else:
                efficiency_str = "N/A"
            
            vs_huffman_str = f"{payload_bytes - huffman_payload:+,}" if huffman_payload > 0 else "N/A"
            
            print(
                f"{data['method']:<25} "
                f"{payload_bytes:>18,} "
                f"{vs_huffman_str:>20} "
                f"{efficiency_str:>18}"
            )
        
        print(f"{'=' * 95}")
    
    print(f"\n{'=' * 95}\n")


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