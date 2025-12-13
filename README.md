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