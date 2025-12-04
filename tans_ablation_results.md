## tANS Ablation Results

This file logs one-by-one experiments comparing the baseline empirical Huffman LZ77
against different tANS variants, all using `../testfiles/test_canonical.txt`
and `table_log=10` unless otherwise noted.

### Baseline configuration (current code)

- **Script**: `scl/compressors/lz77_tans_benchmark.py`
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 10`
- **Behavior**:
  - LZ77 parsing as in `lz77.py`.
  - Literals: empirical Huffman (baseline) vs tANS-on-literals with counts header.
  - LZ77 integer streams: empirical Huffman-based log-binned coder in all cases.
- **Results**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=10): 305 bytes (ratio 1.1296, **+0.99%** vs baseline).
  - Literal headers (single-block parse):
    - Empirical Huffman: 596 bits (74 bytes).
    - tANS literals: 596 bits (74 bytes, **0.00%** vs empirical).

### Test 1 – tANS payload layout without length field (now reverted)

- **Change**:
  - In `tans_lz77_coder.py`, changed tANS payload from
    `[final_state (32)][bitstream_length (32)][bitstream]`
    to `[final_state (32)][bitstream]`, and had `TANSDecoder.decode` return
    `(symbols, bits_consumed)`.
  - All tANS callers in `lz77_tans_benchmark.py` updated to use the new API.
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 10`
- **Results**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=10): 301 bytes (ratio 1.1148, **−0.33%** vs baseline).
  - Literal headers:
    - Empirical Huffman: 596 bits.
    - tANS literals: 596 bits.
- **Status**:
  - Change reverted to restore the shared baseline before subsequent tests.

### Test 2 – Enable `LZ77EncoderTANSAll` / `LZ77DecoderTANSAll` (tANS all streams)

- **Change**:
  - In `run_single_file_benchmark`, uncommented the "tANS all streams" path so that
    `LZ77EncoderTANSAll` / `LZ77DecoderTANSAll` are exercised.
  - Completed the integer-stream header format for `LZ77StreamsEncoderTANSAll` /
    `LZ77StreamsDecoderTANSAll` by:
    - Adding `_encode_frequencies` / `_decode_frequencies` using a simple
      `[16 bits: num_unique] + (sym:32, freq:32)*num_unique` layout.
    - Making `decode_lz77_sequences` advance `bit_pos` based on the stored
      `[final_state][bitstream_length][bitstream]` structure for each tANS-encoded
      integer stream.
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 10`
- **Results**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=10): 305 bytes (ratio 1.1296, **+0.99%** vs baseline).
  - tANS all streams (table_log=10): 327 bytes (ratio 1.2111, **+8.28%** vs baseline).
  - Literal headers (single-block parse):
    - Empirical Huffman: 596 bits.
    - tANS literals: 596 bits.
- **Notes**:
  - On this tiny file, applying tANS to all streams increases overall size
    (expected due to heavy headers and small sample size).
  - This test mainly validates correctness and provides a reference point for
    later improvements to integer modeling and header compression.

### Test 3 – Improved tANS frequency normalization (largest remainder)

- **Change**:
  - In `TANSEncoder.build_table` (`tans_lz77_coder.py`), replaced the simple
    rounding-and-while-loop normalization with a largest-remainder scheme:
    - Compute scaled frequencies `freq[sym] * table_size / total`.
    - Take an integer base allocation (at least 1) per symbol.
    - Use sorted remainders to distribute leftover states or remove extras while
      keeping each symbol's normalized frequency ≥ 1.
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 10`
- **Results**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=10): 307 bytes (ratio 1.1370, **+1.66%** vs baseline).
  - Literal headers:
    - Empirical Huffman: 596 bits.
    - tANS literals: 596 bits.
- **Notes**:
  - On this very small file, the new normalization slightly hurts compression,
    likely because the more balanced normalized frequencies reduce skew that
    Huffman/tANS were previously exploiting.
  - This change was therefore reverted to keep the shared baseline behavior.

### Test 4 – `table_log` sweep (8, 10, 12) for literals-only tANS

- **Change**:
  - No code changes; just run the benchmark with multiple `table_log` values.
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 8 10 12`
- **Results**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=8): 306 bytes (ratio 1.1333, **+1.32%** vs baseline).
  - tANS literals (table_log=10): 305 bytes (ratio 1.1296, **+0.99%** vs baseline).
  - tANS literals (table_log=12): 306 bytes (ratio 1.1333, **+1.32%** vs baseline).
  - Literal headers:
    - Empirical Huffman: 596 bits.
    - tANS literals: 596 bits for all `table_log` values.
- **Notes**:
  - For this tiny file, `table_log=10` remains the best among {8, 10, 12}, but all are
    still slightly worse than the empirical Huffman baseline.

### Test 5 – Adopt payload-layout optimization as new default

- **Change**:
  - Re-applied Test 1 (payload-layout optimization) and kept it:
    - `TANSEncoder.encode` now emits `[final_state (32)][bitstream]` without
      a separate length field.
    - `TANSDecoder.decode` returns `(symbols, bits_consumed)` and callers
      (`TANSLogScaleBinnedIntegerDecoder`, `LZ77StreamsDecoderTANSLiterals`,
      `LZ77StreamsDecoderTANSAll`) advance their bit positions based on
      `bits_consumed` instead of reading a stored length.
- **Command**:
  - `python lz77_tans_benchmark.py -i ../testfiles/test_canonical.txt -t 10`
- **Results (now the default behavior)**:
  - Baseline (Empirical Huffman): 302 bytes (ratio 1.1185).
  - tANS literals (table_log=10): 301 bytes (ratio 1.1148, **−0.33%** vs baseline).
  - Literal headers:
    - Empirical Huffman: 596 bits.
    - tANS literals: 596 bits.
- **Notes**:
  - This change strictly reduces tANS payload bits without changing the model
    header, giving a small but real win even on this tiny file.
  - We now treat this configuration as the new baseline for future experiments.


