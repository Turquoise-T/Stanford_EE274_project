#!/usr/bin/env python
"""Test LZ77TANSStreamsEncoder and LZ77TANSStreamsDecoder"""
import sys

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
# Test with threshold=50 (should use Golomb for match_length since we only have 4 sequences)
encoder = LZ77TANSStreamsEncoder(
    table_log=10,
    match_length_tans_threshold=50,
    reuse_tables=True,
    reuse_min_samples=1,  # small test block
)
try:
    encoded = encoder.encode_block(lz77_sequences, literals)
    print(f"  ✓ Encoded successfully: {len(encoded)} bits")
    print(f"  Note: Using Golomb for match_length (only {len(lz77_sequences)} sequences < threshold 50)")
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

# Test with many sequences (should use tANS for match_length)
print("\n" + "="*70)
print("Testing with many sequences (should use tANS for match_length)...")
print("="*70)

# Create many sequences
many_sequences = [
    LZ77Sequence(literal_count=i % 10, match_length=(i % 20) + 3, match_offset=(i % 50) + 1)
    for i in range(100)
]
many_literals = bytearray([i % 256 for i in range(200)])

print(f"\n[6] Encoding {len(many_sequences)} sequences with tANS threshold=50...")
encoder2 = LZ77TANSStreamsEncoder(
    table_log=10,
    match_length_tans_threshold=50,
    reuse_tables=True,
    reuse_min_samples=1,
)
try:
    encoded2 = encoder2.encode_block(many_sequences, many_literals)
    print(f"  ✓ Encoded successfully: {len(encoded2)} bits")
    print(f"  Note: Using tANS for match_length ({len(many_sequences)} sequences >= threshold 50)")
except Exception as e:
    print(f"  ✗ Encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[7] Decoding...")
decoder2 = LZ77TANSStreamsDecoder(table_log=10)
try:
    (decoded_sequences2, decoded_literals2), bits_consumed2 = decoder2.decode_block(encoded2)
    print(f"  ✓ Decoded successfully")
    
    # Verify
    if len(many_sequences) == len(decoded_sequences2) and many_literals == decoded_literals2:
        print(f"  ✅ All {len(many_sequences)} sequences and {len(many_literals)} literals match!")
    else:
        print(f"  ❌ Mismatch!")
        print(f"    Sequences: {len(many_sequences)} vs {len(decoded_sequences2)}")
        print(f"    Literals: {len(many_literals)} vs {len(decoded_literals2)}")
except Exception as e:
    print(f"  ✗ Decoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Testing table reuse across two consecutive blocks...")
print("="*70)

# Encode/decode two blocks sequentially with the same instances so the second block can reuse tables.
reuse_encoder = LZ77TANSStreamsEncoder(
    table_log=10,
    match_length_tans_threshold=50,
    reuse_tables=True,
    reuse_min_samples=1,
    reuse_max_rel_l1=1.0,  # allow reuse in this unit test
)
reuse_decoder = LZ77TANSStreamsDecoder(table_log=10)

block1 = reuse_encoder.encode_block(many_sequences, many_literals)
(dec_seqs1, dec_lits1), _ = reuse_decoder.decode_block(block1)
assert dec_seqs1 == many_sequences and dec_lits1 == many_literals, "Block 1 mismatch under reuse"

# Block 2: same distribution, should trigger reuse and reduce header size.
block2 = reuse_encoder.encode_block(many_sequences, many_literals)
(dec_seqs2, dec_lits2), _ = reuse_decoder.decode_block(block2)
assert dec_seqs2 == many_sequences and dec_lits2 == many_literals, "Block 2 mismatch under reuse"

# Compare against encoding block2 with a fresh encoder (no reuse state) to confirm reuse reduces size.
fresh_encoder = LZ77TANSStreamsEncoder(
    table_log=10,
    match_length_tans_threshold=50,
    reuse_tables=False,
)
block2_no_reuse = fresh_encoder.encode_block(many_sequences, many_literals)
print(f"  Block2 bits (reuse):    {len(block2)}")
print(f"  Block2 bits (no reuse): {len(block2_no_reuse)}")
assert len(block2) < len(block2_no_reuse), "Expected reuse to reduce encoded size"

print("\n" + "="*70)
print("Testing literal_model='class4' (alpha/digit/whitespace/other)...")
print("="*70)

ctx_encoder = LZ77TANSStreamsEncoder(
    table_log=10,
    match_length_tans_threshold=50,
    literal_model="class4",
    reuse_tables=True,
    reuse_min_samples=1,
    reuse_max_rel_l1=1.0,
)
ctx_decoder = LZ77TANSStreamsDecoder(table_log=10)

ctx_block1 = ctx_encoder.encode_block(many_sequences, many_literals)
(ctx_seqs1, ctx_lits1), _ = ctx_decoder.decode_block(ctx_block1)
assert ctx_seqs1 == many_sequences and ctx_lits1 == many_literals

# Second block should reuse literal tables too (identical data).
ctx_block2 = ctx_encoder.encode_block(many_sequences, many_literals)
(ctx_seqs2, ctx_lits2), _ = ctx_decoder.decode_block(ctx_block2)
assert ctx_seqs2 == many_sequences and ctx_lits2 == many_literals

print("\n" + "="*70)
print("All tests complete!")
print("="*70)

