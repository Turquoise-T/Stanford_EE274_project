## Stanford EE274 Project: LZ77 Compression with Advanced Entropy Coding

This repository implements LZ77 compression with **tANS** and **Canonical Huffman** entropy coding methods for Stanford EE274 coursework.

## Installation

```bash
pip install -e .
pip install matplotlib tqdm  # Optional: for plotting
```

## Usage

### Run Tests

```bash
python -m scl.compressors.test_tans_lz77_coder
python -m scl.compressors.test_lz77_tans_benchmark
python -m unittest discover -s scl/compressors -p "test_*.py"
```

### Baseline LZ77

```bash
python -m scl.compressors.lz77 -i scl/testfiles/big.txt -o output.lz77
```

### Canonical Huffman Benchmark

```bash
python -m scl.compressors.lz77_canonical_benchmark -i scl/testfiles/big.txt
python -m scl.compressors.lz77_canonical_benchmark --plot-header-comparison header_plot.png
python -m scl.compressors.lz77_canonical_benchmark --plot-from-file scl/testfiles/big.txt header_plot.png
python -m scl.compressors.lz77_canonical_benchmark --data-folder scl/testfiles/data/
```

### tANS Benchmark

```bash
python -m scl.compressors.lz77_tans_benchmark -i scl/testfiles/big.txt -t 10
python -m scl.compressors.lz77_tans_benchmark -i scl/testfiles/big.txt -t 8 10 12
python -m scl.compressors.lz77_tans_benchmark \
  -i "scl/testfiles/alice_in_wonderland.txt" "scl/testfiles/big.txt" "scl/testfiles/data/jquery-2.1.4.min.js.xz" \
  -t 8 10 12 --suite_out "results_tans.md" --suite_csv_out "results_tans.csv"
```

## Project Files

- `scl/compressors/lz77.py` - Base LZ77 with empirical Huffman
- `scl/compressors/tans_lz77_coder.py` - tANS implementation (`TANSEncoder/Decoder`, `LZ77TANSStreamsEncoder/Decoder`)
- `scl/compressors/canonical_huffman_code.py` - Canonical Huffman encoder/decoder
- `scl/compressors/lz77_tans_benchmark.py` - tANS benchmarking (`LZ77EncoderTANSLiterals/DecoderTANSLiterals`)
- `scl/compressors/lz77_canonical_benchmark.py` - Canonical Huffman benchmarking
