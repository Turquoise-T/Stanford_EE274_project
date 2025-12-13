## Stanford EE274 Project: LZ77 Compression with Advanced Entropy Coding

This repository contains comprehensive implementations and benchmarking suites for **LZ77 compression** with multiple entropy coding methods: **tANS (Table-based Asymmetric Numeral Systems)** and **Canonical Huffman**, developed as part of Stanford EE274 coursework.

## Project Structure

### Core Implementation
- **`scl/compressors/lz77.py`** - Base LZ77 implementation with empirical Huffman
- **`scl/compressors/tans_lz77_coder.py`** - Core tANS encoder/decoder implementation
- **`scl/compressors/lz77_tans.py`** - Complete LZ77+tANS compressor (all streams)
- **`scl/compressors/canonical_huffman_code.py`** - Canonical Huffman encoder/decoder
- **`scl/compressors/elias_delta_uint_coder.py`** - Elias-Delta encoding for headers

### Benchmarking & Analysis
- **`scl/compressors/lz77_tans_benchmark.py`** - tANS vs empirical Huffman comparison
- **`scl/compressors/lz77_canonical_benchmark.py`** - Canonical vs empirical Huffman comparison
- **`scl/testfiles/`** - Test corpus (Canterbury, Silesia, synthetic files)

### Testing
- **`scl/compressors/test_tans_lz77_coder.py`** - Unit tests for tANS implementation
- **`scl/compressors/test_lz77_tans_benchmark.py`** - Integration tests for benchmarks

## Quick Start

### Prerequisites
```bash
# Activate virtual environment
source ee274_env/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Basic Usage

#### 1. Run Unit Tests
```bash
# Quick test suite
python -m scl.compressors.run_tests --quick

# Full tANS tests
python -m scl.compressors.run_tests --tans-only

# All tests
python -m scl.compressors.run_tests
```

#### 2. Single File Compression Benchmarks

**Empirical Huffman Baseline (LZ77 default):**
```bash
python -m scl.compressors.lz77_canonical_benchmark -i scl/testfiles/big.txt
```

**Canonical vs Empirical Huffman:**
```bash
python -m scl.compressors.lz77_canonical_benchmark -i scl/testfiles/big.txt --plot-header-comparison header_plot.png
```

**tANS vs Empirical Huffman:**
```bash
python -m scl.compressors.lz77_tans_benchmark -i scl/testfiles/big.txt -t 10
```

**Complete LZ77+tANS (all streams):**
```bash
python -m scl.compressors.lz77_tans scl/testfiles/alice_in_wonderland.txt
```

#### 3. Comprehensive Suite Benchmarks

**tANS Suite (multiple table_log values):**
```bash
python -m scl.compressors.lz77_tans_benchmark \
  -i "scl/testfiles/alice_in_wonderland.txt" \
      "scl/testfiles/big.txt" \
      "scl/testfiles/data/jquery-2.1.4.min.js.xz" \
  -t 8 10 12 \
  --suite_out "results_tans.md" \
  --suite_csv_out "results_tans.csv"
```

**Canonical Huffman Suite:**
```bash
python -m scl.compressors.lz77_canonical_benchmark \
  -i "scl/testfiles/alice_in_wonderland.txt" \
      "scl/testfiles/big.txt" \
      "scl/testfiles/data/bootstrap-3.3.6.min.css.xz"
```

**Batch Processing (.xz files):**
```bash
python -m scl.compressors.lz77_canonical_benchmark --data-folder scl/testfiles/data/
```

## Implementation Details

### Entropy Coding Methods

#### tANS Algorithm
- **Table-based ANS**: Uses state tables for symbol encoding/decoding
- **Frequency normalization**: Largest remainder method for optimal state allocation  
- **Renormalization**: Automatic bit output when state exceeds threshold
- **Header format**: Native tANS with `(symbol, frequency)` pairs

#### Canonical Huffman
- **Code length storage**: Stores only code lengths instead of full trees
- **Canonical ordering**: Deterministic code assignment by (length, symbol)
- **Header compression**: Uses Elias-Delta encoding for code lengths
- **Variants**: Literals-only and all-streams implementations

### LZ77 Integration Options
- **Empirical Huffman** (baseline): Standard frequency-based Huffman coding
- **Canonical Huffman**: Reduced header overhead through code length encoding
- **tANS literals**: tANS for literals, empirical Huffman for sequences  
- **tANS all streams**: tANS for literals, counts, lengths, offsets
- **Block-based processing**: Configurable block sizes for large file handling

### Performance Features
- **Multiple table_log values**: 6-16 (table size = 2^table_log) for tANS
- **Header analysis**: Detailed comparison of header overhead across methods
- **Statistical timing**: Warmup runs, multiple trials, configurable statistics
- **Memory optimization**: Lookup tables and frequency caching
- **Visualization**: Header comparison plots and compression analysis
- **Lossless verification**: Automatic encode/decode validation

## Benchmark Output

### Metrics Reported
- **Compression ratio**: Compressed size / original size
- **Header overhead**: Frequency table storage cost
- **Encoding speed**: MB/s throughput
- **Size comparison**: Percentage vs baseline methods

### Output Formats
- **Console**: Real-time progress and summary tables
- **Markdown**: Publication-ready tables (`--suite_out`)
- **CSV**: Data analysis format (`--suite_csv_out`)

## Configuration Options

### Algorithm Parameters
- **`table_log`** (tANS): Controls compression/speed tradeoff (8-12 recommended)
- **`alphabet_size`** (Canonical): Symbol alphabet size (256 for bytes)
- **`block_size`**: Memory usage vs compression efficiency
- **`min_freq`**: Minimum symbol frequency (usually 1)

### Benchmark Parameters  
- **`--encode_trials`**: Number of timing runs per test
- **`--encode_warmup`**: Untimed warmup iterations
- **`--encode_stat`**: Timing statistic (median/mean/min)
- **`--data-folder`**: Batch process .xz files in directory
- **`--plot-header-comparison`**: Generate header analysis plots