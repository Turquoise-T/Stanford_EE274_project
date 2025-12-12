"""
System benchmark for native (table-header) LZ77+tANS codec that exercises:
- per-stream table_log selection
- match_length gating
- table reuse across blocks
- literal contexts (class4)

Usage:
  python lz77_tans_native_benchmark.py -i ../testfiles/alice_in_wonderland.txt \
    --block-size 32768 --literal-model class4 --reuse --matchlen-thresh 200
"""

import argparse
import os
import tempfile
import time

from scl.compressors.lz77 import LZ77Decoder, LZ77Encoder
from scl.compressors.lz77_tans_native import LZ77DecoderTANSNative, LZ77EncoderTANSNative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input file path")
    ap.add_argument("--block-size", type=int, default=32_768)
    ap.add_argument("--table-log", type=int, default=10, help="Default table_log hint")
    ap.add_argument("--matchlen-thresh", type=int, default=200)
    ap.add_argument("--literal-model", choices=["flat", "class4"], default="class4")
    ap.add_argument("--reuse", action="store_true", help="Enable frequency-table reuse across blocks")
    ap.add_argument("--reuse-min-samples", type=int, default=256)
    ap.add_argument("--reuse-max-rel-l1", type=float, default=0.05)
    args = ap.parse_args()

    raw_size = os.path.getsize(args.input)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_out = os.path.join(tmpdir, "baseline.lz77")
        base_dec = os.path.join(tmpdir, "baseline.dec")
        tans_out = os.path.join(tmpdir, "tans_native.lz77")
        tans_dec = os.path.join(tmpdir, "tans_native.dec")

        # Baseline
        enc0 = LZ77Encoder()
        dec0 = LZ77Decoder()
        t0 = time.perf_counter()
        enc0.encode_file(args.input, base_out, block_size=args.block_size)
        dt0 = time.perf_counter() - t0
        dec0.decode_file(base_out, base_dec)
        with open(args.input, "rb") as f_in, open(base_dec, "rb") as f_out:
            assert f_in.read() == f_out.read()
        base_size = os.path.getsize(base_out)
        base_speed = (raw_size / dt0) / (1024 * 1024) if dt0 > 0 else 0.0

        # Native tANS (all improvements)
        enc1 = LZ77EncoderTANSNative(
            table_log=args.table_log,
            match_length_tans_threshold=args.matchlen_thresh,
            literal_model=args.literal_model,
            reuse_tables=args.reuse,
            reuse_min_samples=args.reuse_min_samples,
            reuse_max_rel_l1=args.reuse_max_rel_l1,
        )
        dec1 = LZ77DecoderTANSNative(table_log=args.table_log)
        t1 = time.perf_counter()
        enc1.encode_file(args.input, tans_out, block_size=args.block_size)
        dt1 = time.perf_counter() - t1
        dec1.decode_file(tans_out, tans_dec)
        with open(args.input, "rb") as f_in, open(tans_dec, "rb") as f_out:
            assert f_in.read() == f_out.read()
        tans_size = os.path.getsize(tans_out)
        tans_speed = (raw_size / dt1) / (1024 * 1024) if dt1 > 0 else 0.0

    print("\nMETHOD                         SIZE(bytes)   SPEED(MB/s)   RATIO     vs baseline")
    print("-" * 78)
    print(
        f"Baseline (Empirical Huffman)   {base_size:>10,}   {base_speed:>10.2f}   "
        f"{base_size/raw_size:>6.4f}   {0:>8.2f}%"
    )
    vs = (tans_size - base_size) / base_size * 100 if base_size else 0.0
    print(
        f"Native tANS (all improvements) {tans_size:>10,}   {tans_speed:>10.2f}   "
        f"{tans_size/raw_size:>6.4f}   {vs:>+8.2f}%"
    )


if __name__ == "__main__":
    main()


