"""
tANS Core Implementation

Simplified implementation for LZ77 entropy coding.
Uses frequency-based variable-length encoding.
"""

from collections import Counter
from typing import List, Dict, Tuple
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
import math


def select_table_size(freq: Dict[int, int], min_size: int = 256, max_size: int = 4096):
    """Select table size based on data characteristics."""
    return 256  # Fixed for simplicity


def build_tables(symbols: List[int]):
    """
    Build encoding tables from symbol sequence.
    
    Returns:
        table_size: Table size
        log_size: Bits needed for indexing
        symbol_to_code: Maps symbol to code length in bits
        code_lengths: Bit lengths for each symbol
        freq: Symbol frequencies
    """
    if not symbols:
        return 256, 8, {0: 8}, {0: 8}, {0: 1}
    
    # Count frequencies
    freq = Counter(symbols)
    total = sum(freq.values())
    
    # Calculate code lengths
    # Must be large enough to represent the symbol value!
    code_lengths = {}
    for sym, count in freq.items():
        # Minimum bits needed to represent this symbol value
        value_bits = sym.bit_length() if sym > 0 else 1
        
        # Ideal bits based on frequency (Shannon coding)
        prob = count / total
        freq_bits = max(1, math.ceil(-math.log2(prob)))
        
        # Use the maximum of both (must fit the value!)
        code_lengths[sym] = min(max(value_bits, freq_bits), 16)
    
    table_size = 256
    log_size = 8
    
    return table_size, log_size, code_lengths, code_lengths, dict(freq)


def tans_encode(symbols: List[int], symbol_table, code_lengths, log_size: int, table_size: int):
    """
    Encode symbols using variable-length codes.
    
    Returns:
        final_state: Unused (for interface compatibility)
        bitstream: Encoded bits
    """
    if not symbols:
        return 0, BitArray()
    
    bits = BitArray()
    
    # Encode number of symbols
    bits += uint_to_bitarray(len(symbols), 32)
    
    # Encode each symbol
    for sym in symbols:
        # Get code length for this symbol
        if sym in code_lengths:
            nbits = code_lengths[sym]
        else:
            nbits = 16  # Default for unknown symbols
        
        # Encode: [length (4 bits)] [symbol value (length bits)]
        bits += uint_to_bitarray(nbits, 4)
        bits += uint_to_bitarray(sym, nbits)
    
    return 0, bits


def tans_decode(final_state: int, bitstream: BitArray, symbol_table, freq, log_size: int, table_size: int):
    """
    Decode symbols from variable-length encoded bitstream.
    
    Returns:
        List of decoded symbols
    """
    if not bitstream or len(bitstream) < 32:
        return []
    
    pos = 0
    
    # Decode number of symbols
    if pos + 32 > len(bitstream):
        return []
    length = bitarray_to_uint(bitstream[pos:pos + 32])
    pos += 32
    
    symbols = []
    for _ in range(length):
        # Decode length (4 bits)
        if pos + 4 > len(bitstream):
            break
        nbits = bitarray_to_uint(bitstream[pos:pos + 4])
        pos += 4
        
        # Decode symbol value (nbits)
        if pos + nbits > len(bitstream):
            break
        sym = bitarray_to_uint(bitstream[pos:pos + nbits])
        pos += nbits
        
        symbols.append(sym)
    
    return symbols


def serialize_tables(freq: Dict[int, int], symbol_table, code_lengths) -> BitArray:
    """Serialize frequency table."""
    bitarr = BitArray()
    bitarr += uint_to_bitarray(len(freq), 16)
    for sym in sorted(freq.keys()):
        bitarr += uint_to_bitarray(sym, 16)
        bitarr += uint_to_bitarray(freq[sym], 32)
    return bitarr


def deserialize_tables(bitarr: BitArray, pos: int = 0):
    """
    Deserialize frequency table and rebuild tables.
    
    Returns:
        Tuple of tables and updated position
    """
    if len(bitarr) < pos + 16:
        return (256, 8, {0: 8}, {0: 8}, {0: 1}), pos
    
    num_syms = bitarray_to_uint(bitarr[pos:pos + 16])
    pos += 16

    freq = {}
    for _ in range(num_syms):
        if pos + 48 > len(bitarr):
            break
        sym = bitarray_to_uint(bitarr[pos:pos + 16])
        pos += 16
        count = bitarray_to_uint(bitarr[pos:pos + 32])
        pos += 32
        freq[sym] = count

    # Rebuild tables
    symbols = []
    for sym, c in freq.items():
        symbols += [sym] * c
    
    if not symbols:
        symbols = [0]
    
    table_size, log_size, symbol_table, code_lengths, freq2 = build_tables(symbols)

    return (table_size, log_size, symbol_table, code_lengths, freq2), pos
