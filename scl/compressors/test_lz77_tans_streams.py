#!/usr/bin/env python
"""Test LZ77TANSStreamsEncoder and LZ77TANSStreamsDecoder"""
import sys
sys.path.insert(0, '/Users/jiayuchang/Desktop/Stanford/ee274/ee274_LZ77 ')

from scl.compressors.tans_lz77_coder import LZ77TANSStreamsEncoder, LZ77TANSStreamsDecoder
from scl.compressors.lz77 import LZ77Sequence

print("="*70)
print("Testing LZ77TANSStreamsEncoder and LZ77TANSStreamsDecoder")
print("="*70)

# Create test data
print("\n[1] Creating test LZ77 sequences and literals...")
lz77_sequences = [
    LZ77Sequence(literal_count=5, match_length=3, match_offset=10),
    LZ77Sequence(literal_count=2, match_length=5, match_offset=8),
    LZ77Sequence(literal_count=4, match_length=0, match_offset=0),
    LZ77Sequence(literal_count=3, match_length=7, match_offset=15),
]

# Create some literal bytes
literals = bytearray([72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 33, 33])

print(f"  Sequences: {len(lz77_sequences)}")
for i, seq in enumerate(lz77_sequences):
    print(f"    [{i}] literal_count={seq.literal_count}, match_length={seq.match_length}, match_offset={seq.match_offset}")
print(f"  Literals: {len(literals)} bytes")
print(f"  Literals content: {literals}")
print(f"  Literals as text: {literals.decode('ascii', errors='ignore')}")

# Encode
print("\n[2] Encoding with LZ77TANSStreamsEncoder...")
encoder = LZ77TANSStreamsEncoder(table_log=10)
try:
    encoded = encoder.encode_block(lz77_sequences, literals)
    print(f"  ✓ Encoded successfully: {len(encoded)} bits")
except Exception as e:
    print(f"  ✗ Encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Decode
print("\n[3] Decoding with LZ77TANSStreamsDecoder...")
decoder = LZ77TANSStreamsDecoder(table_log=10)
try:
    (decoded_sequences, decoded_literals), bits_consumed = decoder.decode_block(encoded)
    print(f"  ✓ Decoded successfully")
    print(f"  Bits consumed: {bits_consumed}")
    print(f"  Decoded sequences: {len(decoded_sequences)}")
    print(f"  Decoded literals: {len(decoded_literals)} bytes")
except Exception as e:
    print(f"  ✗ Decoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare sequences
print("\n[4] Comparing sequences...")
if len(lz77_sequences) != len(decoded_sequences):
    print(f"  ❌ Length mismatch: {len(lz77_sequences)} vs {len(decoded_sequences)}")
else:
    all_match = True
    for i, (orig, dec) in enumerate(zip(lz77_sequences, decoded_sequences)):
        if (orig.literal_count != dec.literal_count or 
            orig.match_length != dec.match_length or 
            orig.match_offset != dec.match_offset):
            print(f"  ❌ Sequence {i} mismatch:")
            print(f"    Original: lc={orig.literal_count}, ml={orig.match_length}, mo={orig.match_offset}")
            print(f"    Decoded:  lc={dec.literal_count}, ml={dec.match_length}, mo={dec.match_offset}")
            all_match = False
    
    if all_match:
        print(f"  ✅ All {len(lz77_sequences)} sequences match!")

# Compare literals
print("\n[5] Comparing literals...")
if literals == decoded_literals:
    print(f"  ✅ Literals match! ({len(literals)} bytes)")
else:
    print(f"  ❌ Literals mismatch!")
    print(f"    Original: {len(literals)} bytes - {literals}")
    print(f"    Decoded:  {len(decoded_literals)} bytes - {decoded_literals}")
    
    if len(decoded_literals) > 0:
        print(f"    Original text: {literals.decode('ascii', errors='ignore')}")
        print(f"    Decoded text:  {decoded_literals.decode('ascii', errors='ignore')}")

print("\n" + "="*70)
print("Test complete!")
print("="*70)

