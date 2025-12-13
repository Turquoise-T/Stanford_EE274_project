import argparse
import os
import tempfile
import time
import lzma
from dataclasses import dataclass
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


@dataclass
class BenchmarkRow:
    file: str
    raw_size: int
    baseline_size: int
    baseline_header: int
    baseline_ratio: float
    baseline_time_s: float
    baseline_speed_mb_s: float
    tans_size: int
    tans_header: int
    tans_ratio: float
    tans_time_s: float
    tans_speed_mb_s: float
    tans_vs_baseline_pct: float


def _timing_stat(values: List[float], stat: str) -> float:
    if not values:
        return 0.0
    stat = stat.lower().strip()
    if stat == "min":
        return min(values)
    if stat == "mean":
        return sum(values) / len(values)
    if stat == "median":
        xs = sorted(values)
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])
    raise ValueError(
        f"Unknown timing stat: {stat!r} (expected one of: median, mean, min)"
    )


def _measure_encode_time_s(
    encoder_factory,
    in_path: str,
    out_path: str,
    *,
    block_size: int,
    warmup: int = 0,
    trials: int = 1,
    stat: str = "median",
) -> float:
    """
    Measures encode_file runtime using perf_counter, with optional warmup runs and repeated trials.

    IMPORTANT: LZ77 encoders are stateful (they keep a window/index across calls),
    so we must create a fresh encoder per run/trial.

    We keep the last trial's encoded output at `out_path` so callers can measure size
    and verify decode without running an extra encode pass.
    """
    warmup = max(0, int(warmup))
    trials = max(1, int(trials))

    # Warmup runs (not recorded).
    for i in range(warmup):
        encoder = encoder_factory()
        tmp_out = f"{out_path}.warmup.{i}"
        encoder.encode_file(in_path, tmp_out, block_size=block_size)
        try:
            os.remove(tmp_out)
        except OSError:
            pass

    times: List[float] = []
    for i in range(trials):
        encoder = encoder_factory()
        tmp_out = out_path if i == trials - 1 else f"{out_path}.trial.{i}"
        start = time.perf_counter()
        encoder.encode_file(in_path, tmp_out, block_size=block_size)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i != trials - 1:
            try:
                os.remove(tmp_out)
            except OSError:
                pass

    return _timing_stat(times, stat)


def _format_suite_markdown_table(rows: List[BenchmarkRow]) -> str:
    header = (
        "| File | Raw Size (B) | Baseline Size (B) | Base Hdr (B) | Base Ratio | Base Time (s) | Base Speed (MB/s) | "
        "tANS Size (B) | tANS Hdr (B) | tANS Ratio | tANS Time (s) | tANS Speed (MB/s) | tANS vs Base (%) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.file} | {r.raw_size:,} | {r.baseline_size:,} | {r.baseline_header:,} | {r.baseline_ratio:.4f} | "
            f"{r.baseline_time_s:.3f} | {r.baseline_speed_mb_s:.2f} | {r.tans_size:,} | {r.tans_header:,} | "
            f"{r.tans_ratio:.4f} | {r.tans_time_s:.3f} | {r.tans_speed_mb_s:.2f} | {r.tans_vs_baseline_pct:+.2f}% |\n"
        )
    return "".join(lines) + "\n"


def _format_suite_csv(rows: List[BenchmarkRow]) -> str:
    cols = [
        "file",
        "raw_size_bytes",
        "baseline_size_bytes",
        "baseline_header_bytes",
        "baseline_ratio",
        "baseline_time_s",
        "baseline_speed_mb_s",
        "tans_size_bytes",
        "tans_header_bytes",
        "tans_ratio",
        "tans_time_s",
        "tans_speed_mb_s",
        "tans_vs_baseline_pct",
    ]
    out = [",".join(cols) + "\n"]
    for r in rows:
        out.append(
            f"{r.file},{r.raw_size},{r.baseline_size},{r.baseline_header},{r.baseline_ratio:.6f},"
            f"{r.baseline_time_s:.6f},{r.baseline_speed_mb_s:.6f},{r.tans_size},{r.tans_header},"
            f"{r.tans_ratio:.6f},{r.tans_time_s:.6f},{r.tans_speed_mb_s:.6f},{r.tans_vs_baseline_pct:.6f}\n"
        )
    return "".join(out)

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
    path: str,
    block_size: int = 100_000,
    table_logs: List[int] = [10],
    encode_warmup: int = 0,
    encode_trials: int = 1,
    encode_stat: str = "median",
) -> dict:
    """
    Per-file benchmark. Supports .xz inputs by decompressing once to a temporary file.
    Returns a dict containing a list of BenchmarkRow (one per table_log).
    """

    def _prepare_input_file(src_path: str, tmpdir: str) -> Tuple[str, int, str]:
        # Returns: (path_to_use, raw_size_bytes, display_name)
        display_name = os.path.basename(src_path)
        if src_path.lower().endswith(".xz"):
            out_name = os.path.basename(src_path)[:-3]  # strip .xz
            out_path = os.path.join(tmpdir, out_name)
            with lzma.open(src_path, "rb") as fin, open(out_path, "wb") as fout:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            raw_size_local = os.path.getsize(out_path)
            return out_path, raw_size_local, out_name
        raw_size_local = os.path.getsize(src_path)
        return src_path, raw_size_local, display_name

    summary_rows: List[BenchmarkRow] = []

    print(f"\n{'=' * 70}")
    print(f"Benchmark on file: {path}")
    print(f"{'=' * 70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path, raw_size, display_name = _prepare_input_file(path, tmpdir)
        print(f"Input: {display_name}")
        print(f"Raw size: {raw_size:,} bytes")

        # ---------------- Calculate header sizes (single-block parse) ----------------
        with open(in_path, "rb") as f:
            data_bytes = list(f.read())
        data_block = DataBlock(data_bytes)

        parser = LZ77Encoder(
            min_match_length=DEFAULT_MIN_MATCH_LEN,
            max_num_matches_considered=DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        )
        _, lits = parser.lz77_parse_and_generate_sequences(data_block)

        emp_header_bits = compute_literal_header_bits_empirical(lits)
        emp_header_bytes = emp_header_bits // 8

        tans_header_bytes_dict = {}
        for table_log in table_logs:
            tans_header_bits = compute_literal_header_bits_tans(lits, table_log)
            tans_header_bytes_dict[table_log] = tans_header_bits // 8

        # ---------------- Baseline LZ77 (Empirical Huffman) ----------------
        print("\n[1/2] Running baseline LZ77 (Empirical Huffman)...")
        base_dec = LZ77Decoder()

        base_encoded_path = os.path.join(tmpdir, "baseline.lz77")
        base_decoded_path = os.path.join(tmpdir, "baseline.dec")

        baseline_encode_time = _measure_encode_time_s(
            lambda: LZ77Encoder(),
            in_path,
            base_encoded_path,
            block_size=block_size,
            warmup=encode_warmup,
            trials=encode_trials,
            stat=encode_stat,
        )

        base_dec.decode_file(base_encoded_path, base_decoded_path)
        with open(in_path, "rb") as f_in, open(base_decoded_path, "rb") as f_out:
            assert f_in.read() == f_out.read(), "Baseline LZ77 decode mismatch!"

        baseline_size = os.path.getsize(base_encoded_path)
        baseline_speed = (
            (raw_size / baseline_encode_time) / (1024 * 1024)
            if baseline_encode_time > 0 and raw_size > 0
            else 0.0
        )

        results = {
            "Baseline (Empirical Huffman)": {
                "size": baseline_size,
                "ratio": baseline_size / raw_size if raw_size > 0 else 0.0,
                "enc_time_s": baseline_encode_time,
                "enc_speed": baseline_speed,
            }
        }

        # ------------- LZ77 with tANS on different table_log values -------------
        for table_log in table_logs:
            print(f"\n[2/2] Running LZ77 + tANS (literals, table_log={table_log})...")
            tans_lit_dec = LZ77DecoderTANSLiterals(table_log=table_log)

            tans_lit_encoded_path = os.path.join(tmpdir, f"tans_lit_{table_log}.lz77")
            tans_lit_decoded_path = os.path.join(tmpdir, f"tans_lit_{table_log}.dec")

            tans_lit_encode_time = _measure_encode_time_s(
                lambda tl=table_log: LZ77EncoderTANSLiterals(table_log=tl),
                in_path,
                tans_lit_encoded_path,
                block_size=block_size,
                warmup=encode_warmup,
                trials=encode_trials,
                stat=encode_stat,
            )

            tans_lit_dec.decode_file(tans_lit_encoded_path, tans_lit_decoded_path)
            with open(in_path, "rb") as f_in, open(tans_lit_decoded_path, "rb") as f_out:
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
                "enc_time_s": tans_lit_encode_time,
                "enc_speed": tans_lit_speed,
            }

        # ---------------- Print Results with Header Info ----------------
        print(f"\n{'=' * 85}")
        print("COMPRESSION RESULTS")
        print(f"{'=' * 85}")
        print(
            f"{'Method':<40} {'Size (bytes)':>12} {'Header':>10} "
            f"{'Time(s)':>10} {'Speed(MB/s)':>12} {'Ratio':>10} {'vs Baseline':>12}"
        )
        print("-" * 85)

        baseline_size_ref = results["Baseline (Empirical Huffman)"]["size"]

        for method, data in results.items():
            size = data["size"]
            ratio = data["ratio"]
            enc_time_s = data.get("enc_time_s", 0.0)
            speed = data.get("enc_speed", 0.0)
            vs_baseline = (
                (size - baseline_size_ref) / baseline_size_ref * 100
                if baseline_size_ref > 0
                else 0.0
            )
            sign = "+" if vs_baseline > 0 else ""

            if method == "Baseline (Empirical Huffman)":
                header_str = f"{emp_header_bytes:,}"
            else:
                header_str = "0"
                for table_log in table_logs:
                    if f"table_log={table_log}" in method:
                        header_str = f"{tans_header_bytes_dict[table_log]:,}"
                        break

            print(
                f"{method:<40} {size:>12,} {header_str:>10} "
                f"{enc_time_s:>10.3f} {speed:>12.2f} {ratio:>9.4f} {sign}{vs_baseline:>10.2f}%"
            )

        # Build suite rows (Baseline vs tANS literals only)
        for table_log in table_logs:
            method_key = f"tANS literals (table_log={table_log})"
            tans_size = results[method_key]["size"]
            tans_ratio = results[method_key]["ratio"]
            tans_time = results[method_key].get("enc_time_s", 0.0)
            tans_speed = results[method_key].get("enc_speed", 0.0)
            vs_baseline_pct = (
                (tans_size - baseline_size_ref) / baseline_size_ref * 100
                if baseline_size_ref > 0
                else 0.0
            )
            summary_rows.append(
                BenchmarkRow(
                    file=display_name,
                    raw_size=raw_size,
                    baseline_size=baseline_size_ref,
                    baseline_header=emp_header_bytes,
                    baseline_ratio=results["Baseline (Empirical Huffman)"]["ratio"],
                    baseline_time_s=results["Baseline (Empirical Huffman)"].get(
                        "enc_time_s", 0.0
                    ),
                    baseline_speed_mb_s=results["Baseline (Empirical Huffman)"].get(
                        "enc_speed", 0.0
                    ),
                    tans_size=tans_size,
                    tans_header=tans_header_bytes_dict[table_log],
                    tans_ratio=tans_ratio,
                    tans_time_s=tans_time,
                    tans_speed_mb_s=tans_speed,
                    tans_vs_baseline_pct=vs_baseline_pct,
                )
            )

    return {"file": display_name, "raw_size": raw_size, "rows": summary_rows}


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
    parser.add_argument(
        "--encode_warmup",
        type=int,
        default=0,
        help="Number of untimed warmup encode runs per method/file (default: 0).",
    )
    parser.add_argument(
        "--encode_trials",
        type=int,
        default=1,
        help="Number of timed encode trials per method/file (default: 1).",
    )
    parser.add_argument(
        "--encode_stat",
        type=str,
        default="median",
        choices=["median", "mean", "min"],
        help="Statistic used to summarize trials into a single encode time (default: median).",
    )
    parser.add_argument(
        "--suite_out",
        type=str,
        default="",
        help="If provided, write a suite summary table to this path (.md recommended).",
    )
    parser.add_argument(
        "--suite_csv_out",
        type=str,
        default="",
        help="If provided, write a suite summary CSV to this path.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LZ77 + tANS BENCHMARK")
    print("=" * 70)
    print(f"Table log values to test: {args.table_log}")
    print(f"Block size: {args.block_size:,} bytes")

    suite_rows: List[BenchmarkRow] = []
    for path in args.input:
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipping.")
            continue
        result = run_single_file_benchmark(
            path,
            block_size=args.block_size,
            table_logs=args.table_log,
            encode_warmup=args.encode_warmup,
            encode_trials=args.encode_trials,
            encode_stat=args.encode_stat,
        )
        suite_rows.extend(result["rows"])

    if len(suite_rows) > 1:
        print(f"{'=' * 95}")
        print("SUITE SUMMARY (Baseline vs tANS literals)")
        print(f"{'=' * 95}")
        md = _format_suite_markdown_table(suite_rows)
        print(md)

        if args.suite_out:
            os.makedirs(os.path.dirname(args.suite_out) or ".", exist_ok=True)
            with open(args.suite_out, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Wrote suite summary markdown to: {args.suite_out}")

        if args.suite_csv_out:
            os.makedirs(os.path.dirname(args.suite_csv_out) or ".", exist_ok=True)
            with open(args.suite_csv_out, "w", encoding="utf-8") as f:
                f.write(_format_suite_csv(suite_rows))
            print(f"Wrote suite summary CSV to: {args.suite_csv_out}")


if __name__ == "__main__":
    main()