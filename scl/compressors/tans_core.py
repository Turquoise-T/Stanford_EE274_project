"""
tANS Core Implementation

A working table-based ANS (Asymmetric Numeral Systems) implementation.
Based on the tANS variant described in Fabian Giesen's blog posts.
"""

from collections import Counter
from typing import List, Dict, Tuple
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint


def select_table_size(freq: Dict[int, int], min_size: int = 256, max_size: int = 4096):
    """Select optimal tANS table size (power of 2) based on symbol frequencies."""
    total = sum(freq.values())
    
    # Start with a reasonable minimum
    size = min_size
    
    # Increase table size if we have many symbols or high frequency variation
    num_symbols = len(freq)
    if num_symbols > 128:
        size = 1024
    if total > 10000:
        size = 2048
    
    # Cap at max_size
    size = min(size, max_size)
    
    return size


def build_tables(symbols: List[int]):
    """
    Build tANS state tables.
    
    Returns:
        table_size: Size of the state table (power of 2)
        log_size: log2(table_size)
        state_table: Maps state -> (symbol, next_state_base)
        cumul: Cumulative frequency for each symbol
        freq: Symbol frequencies normalized to table_size
    """
    if not symbols:
        return 256, 8, [(0, 0)] * 256, {0: 0}, {0: 256}
    
    # Count raw frequencies
    raw_freq = Counter(symbols)
    table_size = select_table_size(raw_freq)
    log_size = table_size.bit_length() - 1
    
    # Normalize frequencies to sum to table_size
    total = len(symbols)
    freq = {}
    assigned = 0
    
    for sym in sorted(raw_freq.keys()):
        # Proportional allocation, minimum 1
        f = max(1, round(raw_freq[sym] * table_size / total))
        freq[sym] = f
        assigned += f
    
    # Adjust to exact table_size
    while assigned != table_size:
        # Adjust most frequent symbol
        adjust_sym = max(freq.keys(), key=lambda s: raw_freq[s])
        if assigned < table_size:
            freq[adjust_sym] += 1
            assigned += 1
        elif freq[adjust_sym] > 1:
            freq[adjust_sym] -= 1
            assigned -= 1
        else:
            break
    
    # Build cumulative frequency table
    cumul = {}
    c = 0
    for sym in sorted(freq.keys()):
        cumul[sym] = c
        c += freq[sym]
    
    # Build state table for decoding
    # state_table[i] = (symbol, next_state_base)
    state_table = []
    next_state = {sym: 0 for sym in freq}
    
    for state in range(table_size):
        # Spread symbols across states proportionally
        slot = state * len(symbols) // table_size
        sym = symbols[sorted(enumerate(symbols), key=lambda x: (raw_freq[x[1]], x[0]))[slot][0]]
        
        # Find which symbol this state belongs to
        for s in sorted(cumul.keys()):
            if state >= cumul[s] and (s == max(cumul.keys()) or state < cumul.get(list(sorted(cumul.keys()))[list(sorted(cumul.keys())).index(s) + 1], table_size)):
                sym = s
                break
        
        state_table.append((sym, next_state[sym]))
        next_state[sym] += 1
    
    return table_size, log_size, state_table, cumul, freq


def tans_encode(symbols: List[int], state_table, cumul, log_size: int, table_size: int):
    """
    Encode symbols using tANS.
    
    Returns:
        final_state: Final encoder state
        bitstream: Encoded bits (reversed, prepended to output)
    """
    if not symbols:
        return 0, BitArray()
    
    # Start at a valid state
    state = table_size
    bits = BitArray()
    
    # Process symbols in reverse order
    for sym in reversed(symbols):
        if sym not in cumul:
            continue
        
        # Calculate how many bits to output
        freq_sym = 0
        for s, (st_sym, _) in enumerate(state_table):
            if st_sym == sym:
                freq_sym += 1
        
        if freq_sym == 0:
            continue
        
        # Renormalize: output bits if state is too large
        while state >= freq_sym * table_size:
            bits += uint_to_bitarray(state & ((1 << log_size) - 1), log_size)
            state >>= log_size
        
        # Encode symbol: state maps to cumul[sym] + (state % freq[sym])
        slot = state % freq_sym
        state = cumul[sym] + slot + (state // freq_sym) * table_size
    
    return state, bits


def tans_decode(final_state: int, bitstream: BitArray, state_table, freq, log_size: int, table_size: int):
    """
    Decode symbols from tANS bitstream.
    
    Returns:
        List of decoded symbols in original order
    """
    if not bitstream:
        return []
    
    state = final_state
    pos = len(bitstream)
    symbols = []
    
    # Decode symbols in forward order (they were encoded in reverse)
    while pos > 0:
        # Get symbol from current state
        slot = state % table_size
        if slot >= len(state_table):
            break
        
        sym, next_base = state_table[slot]
        symbols.append(sym)
        
        # Get frequency for this symbol
        freq_sym = sum(1 for s, _ in state_table if s == sym)
        if freq_sym == 0:
            break
        
        # Reconstruct previous state
        state = next_base + (state // table_size) * freq_sym
        
        # Read more bits if needed
        while state < table_size and pos >= log_size:
            pos -= log_size
            new_bits = bitarray_to_uint(bitstream[pos:pos + log_size])
            state = (state << log_size) | new_bits
    
    return list(reversed(symbols))


def serialize_tables(freq: Dict[int, int], state_table, cumul) -> BitArray:
    """Serialize frequency table to bitstream."""
    bitarr = BitArray()
    bitarr += uint_to_bitarray(len(freq), 16)
    for sym in sorted(freq.keys()):
        bitarr += uint_to_bitarray(sym, 16)
        bitarr += uint_to_bitarray(freq[sym], 16)
    return bitarr


def deserialize_tables(bitarr: BitArray, pos: int = 0):
    """Deserialize frequency table and rebuild tANS tables."""
    if len(bitarr) < pos + 16:
        # Return default tables
        return (256, 8, [(0, 0)] * 256, {0: 0}, {0: 256}), pos
    
    num_syms = bitarray_to_uint(bitarr[pos:pos + 16])
    pos += 16

    freq = {}
    for _ in range(num_syms):
        if pos + 32 > len(bitarr):
            break
        sym = bitarray_to_uint(bitarr[pos:pos + 16])
        pos += 16
        count = bitarray_to_uint(bitarr[pos:pos + 16])
        pos += 16
        freq[sym] = count

    # Rebuild symbol sequence
    symbols = []
    for sym, c in freq.items():
        symbols += [sym] * c
    
    if not symbols:
        symbols = [0]
    
    table_size, log_size, state_table, cumul, freq2 = build_tables(symbols)

    return (table_size, log_size, state_table, cumul, freq2), pos
