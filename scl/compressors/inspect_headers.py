"""
Script to inspect and compare header contents for Huffman vs tANS.

This script shows exactly what information is stored in each header format.
"""

import sys
from collections import Counter

from scl.core.data_block import DataBlock
from scl.compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from scl.compressors.huffman_coder import HuffmanEncoder
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint


def inspect_huffman_header(literals):
    """
    Inspect Empirical Huffman header contents
    """
    print("=" * 80)
    print("EMPIRICAL HUFFMAN HEADER")
    print("=" * 80)
    
    # Step 1: Calculate counts
    counts = Counter(literals)
    counts_list = [counts.get(i, 0) for i in range(256)]
    
    print(f"\n1. Raw data statistics:")
    print(f"   - Total symbols: {len(literals)}")
    print(f"   - Unique symbols: {len(counts)}")
    print()
    
    # Show top 10 most frequent symbols
    print("   Top 10 most frequent symbols:")
    for i, (sym, count) in enumerate(counts.most_common(10)):
        char = chr(sym) if 32 <= sym < 127 else f"\\x{sym:02x}"
        percent = count / len(literals) * 100
        print(f"      {i+1}. '{char}' (ASCII {sym}): {count:,} times ({percent:.2f}%)")
    
    print(f"\n2. Counts vector (what is stored):")
    print(f"   - Format: counts[0..255] - list of 256 numbers")
    print(f"   - Content: each position stores occurrence count of corresponding byte value")
    print(f"   - Examples:")
    
    # Show some key counts
    print(f"      counts[32] (space): {counts_list[32]}")
    print(f"      counts[65] ('A'): {counts_list[65]}")
    print(f"      counts[101] ('e'): {counts_list[101]}")
    print(f"      counts[0] (null): {counts_list[0]}")
    
    # Calculate sparsity
    num_zeros = sum(1 for c in counts_list if c == 0)
    print(f"\n   - Zero values: {num_zeros}/256 ({num_zeros/256*100:.1f}%)")
    print(f"   - Non-zero values: {256-num_zeros}/256")
    
    # Step 2: Elias-Delta encoding
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    
    print(f"\n3. After Elias-Delta compression:")
    print(f"   - Raw size: 256 × 32 bits = 1,024 bytes (if storing each count with 32 bits)")
    print(f"   - Compressed: {len(counts_encoding)} bits = {len(counts_encoding)//8} bytes")
    print(f"   - Compression ratio: {len(counts_encoding)/(256*32)*100:.2f}%")
    print(f"   - Saved: {1024 - len(counts_encoding)//8} bytes")
    
    # Step 3: Complete header
    header = uint_to_bitarray(len(counts_encoding), 32) + counts_encoding
    
    print(f"\n4. Complete header format:")
    print(f"   [32 bits: size] + [Elias-Delta(counts)]")
    print(f"   ↓")
    print(f"   [32 bits: {len(counts_encoding)}] + [{len(counts_encoding)} bits]")
    print(f"\n5. Total header size:")
    print(f"   - {len(header)} bits = {len(header)//8} bytes")
    
    return len(header)


def inspect_tans_header(literals):
    """
    Inspect tANS header contents
    """
    print("\n" + "=" * 80)
    print("tANS NATIVE HEADER")
    print("=" * 80)
    
    # Step 1: Calculate frequencies
    freqs = Counter(literals)
    
    print(f"\n1. Raw data statistics:")
    print(f"   - Total symbols: {len(literals)}")
    print(f"   - Unique symbols: {len(freqs)}")
    print()
    
    # Show top 10 most frequent symbols
    print("   Top 10 most frequent symbols:")
    for i, (sym, freq) in enumerate(freqs.most_common(10)):
        char = chr(sym) if 32 <= sym < 127 else f"\\x{sym:02x}"
        percent = freq / len(literals) * 100
        print(f"      {i+1}. '{char}' (ASCII {sym}): {freq:,} times ({percent:.2f}%)")
    
    print(f"\n2. Frequency table (what is stored):")
    print(f"   - Format: (symbol, frequency) pairs - only stores symbols that appear")
    print(f"   - Content: each pair contains [symbol ID, occurrence count]")
    print(f"   - Examples (first 5 pairs):")
    
    for i, (sym, freq) in enumerate(sorted(freqs.items())[:5]):
        char = chr(sym) if 32 <= sym < 127 else f"\\x{sym:02x}"
        print(f"      {i+1}. (symbol={sym} '{char}', freq={freq})")
    
    print(f"      ...")
    
    # Step 2: Build header
    result = uint_to_bitarray(len(literals), 32)
    result += uint_to_bitarray(len(freqs), 16)
    
    for sym, freq in sorted(freqs.items()):
        result += uint_to_bitarray(sym, 16)
        result += uint_to_bitarray(freq, 32)
    
    print(f"\n3. Header component sizes:")
    print(f"   - num_literals (32 bits): 4 bytes")
    print(f"   - num_unique (16 bits): 2 bytes")
    print(f"   - (symbol, freq) pairs:")
    print(f"      {len(freqs)} pairs × (16 + 32) bits = {len(freqs)} × 48 bits")
    print(f"      = {len(freqs) * 48} bits = {len(freqs) * 48 // 8} bytes")
    
    print(f"\n4. Complete header format:")
    print(f"   [32 bits: num_literals] + [16 bits: num_unique] + [(16+32) bits] × {len(freqs)}")
    print(f"   ↓")
    print(f"   [32 bits: {len(literals)}] + [16 bits: {len(freqs)}] + [{len(freqs) * 48} bits]")
    
    print(f"\n5. Total header size:")
    print(f"   - {len(result)} bits = {len(result)//8} bytes")
    print(f"   - Calculation: 32 + 16 + ({len(freqs)} × 48) = {32 + 16 + len(freqs) * 48} bits")
    
    return len(result)


def compare_headers(literals):
    """
    Compare two header formats
    """
    print("\n" + "=" * 80)
    print("HEADER COMPARISON SUMMARY")
    print("=" * 80)
    
    counts = Counter(literals)
    
    # Huffman header
    counts_list = [counts.get(i, 0) for i in range(256)]
    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    huffman_header_bits = 32 + len(counts_encoding)
    huffman_header_bytes = huffman_header_bits // 8
    
    # tANS header
    tans_header_bits = 32 + 16 + (len(counts) * 48)
    tans_header_bytes = tans_header_bits // 8
    
    print(f"\nTotal data: {len(literals)} symbols, {len(counts)} unique values")
    print()
    print(f"{'Method':<20} {'Header (bits)':>15} {'Header (bytes)':>18} {'Difference':>15}")
    print("-" * 80)
    print(f"{'Huffman (Empirical)':<20} {huffman_header_bits:>15,} {huffman_header_bytes:>18,} {'baseline':>15}")
    print(f"{'tANS (Native)':<20} {tans_header_bits:>15,} {tans_header_bytes:>18,} {f'+{tans_header_bytes - huffman_header_bytes}':>15}")
    
    ratio = tans_header_bytes / huffman_header_bytes if huffman_header_bytes > 0 else 0
    print()
    print(f"tANS header is {ratio:.2f}x larger than Huffman")
    print(f"Difference: {tans_header_bytes - huffman_header_bytes} bytes (+{(ratio-1)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Why are the sizes different?")
    print("=" * 80)
    
    print("\nHuffman (Elias-Delta):")
    print("  Advantage: Exploits sparsity - compresses many zeros in 256-element array efficiently")
    print(f"  Reality: {256 - len(counts)} zero values are efficiently compressed")
    print("  Result: Very compact")
    
    print("\ntANS (Symbol-Freq Pairs):")
    print(f"  Disadvantage: Must store full 48 bits for each unique symbol")
    print(f"  Reality: {len(counts)} symbols × 48 bits = {len(counts) * 48} bits")
    print("  Result: Cannot exploit sparsity")
    
    print("\nConclusion:")
    if ratio > 1:
        print(f"  → Huffman header is more efficient ({ratio:.2f}x difference)")
    else:
        print(f"  → tANS header is more efficient")


def main():
    """
    Main function - read file and inspect headers
    """
    if len(sys.argv) < 2:
        print("Usage: python inspect_headers.py <input_file>")
        print("Example: python inspect_headers.py ../testfiles/test_canonical.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Read file
    with open(file_path, "rb") as f:
        data_bytes = list(f.read())
    
    print(f"\nFile: {file_path}")
    print(f"Size: {len(data_bytes)} bytes")
    print()
    
    # Inspect Huffman header
    huffman_bits = inspect_huffman_header(data_bytes)
    
    # Inspect tANS header  
    tans_bits = inspect_tans_header(data_bytes)
    
    # Compare
    compare_headers(data_bytes)
    
    print("\n" + "=" * 80)
    print("Additional Information: Payload Encoding")
    print("=" * 80)
    
    # Calculate theoretical entropy
    import math
    counts = Counter(data_bytes)
    total = len(data_bytes)
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    print(f"\nData entropy: {entropy:.4f} bits/byte")
    print(f"Theoretical optimum: {len(data_bytes) * entropy / 8:.0f} bytes")
    print()
    print("Huffman payload: Close to entropy, but at least 1 bit per symbol")
    print("tANS payload: Closer to entropy limit, especially for high-frequency symbols")
    print()
    print("Header comparison:")
    print(f"  Huffman: {huffman_bits//8} bytes (Elias-Delta compressed)")
    print(f"  tANS:    {tans_bits//8} bytes (raw storage)")
    print()
    print("Summary:")
    print("  - Huffman has advantage in header size")
    print("  - tANS may have slight advantage in payload")
    print("  - Small files: Huffman wins (header dominates)")
    print("  - Large files: tANS may win slightly (payload dominates)")


if __name__ == "__main__":
    main()

