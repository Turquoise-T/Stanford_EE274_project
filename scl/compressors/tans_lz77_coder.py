"""
tANS (tabled Asymmetric Numeral Systems) implementation for LZ77 entropy coding.

This module provides tANS-based entropy encoding/decoding as a drop-in replacement
for the `LZ77StreamsEncoder`/`LZ77StreamsDecoder` used in `lz77.py`.

High level:
- Build a symbol table from empirical frequencies
- Encode symbols by transitioning through ANS states
- Decode by reversing the state transitions

The integration here keeps the LZ77 parser unchanged and only swaps the entropy
coding backend. Header formats are designed to be compact and symmetric with the
baseline empirical Huffman implementation.
"""

from collections import Counter

import numpy as np

from scl.compressors.elias_delta_uint_coder import (
    EliasDeltaUintDecoder,
    EliasDeltaUintEncoder,
)
from scl.compressors.lz77 import LZ77Sequence
from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray


# Number of bits used for the size prefix of Elias–Delta encoded count vectors.
# This matches `ENCODED_BLOCK_SIZE_HEADER_BITS` in `lz77.py`.
COUNTS_SIZE_HEADER_BITS = 32


def _encode_literal_counts_header(literals):
    """
    Encode literal counts using the same scheme as EmpiricalIntHuffmanEncoder.

    Header layout:
        [32 bits: size_of_counts_encoding] + [counts_encoding_bits]

    where counts_encoding_bits is Elias–Delta coding of a length-256 count vector.
    """
    if not literals:
        # Mirror EmpiricalIntHuffmanEncoder behavior for empty blocks.
        return uint_to_bitarray(0, COUNTS_SIZE_HEADER_BITS)

    counts = Counter(literals)
    # Ensure all 256 byte values are represented.
    counts_list = [counts.get(i, 0) for i in range(256)]

    counts_encoding = EliasDeltaUintEncoder().encode_block(DataBlock(counts_list))
    return uint_to_bitarray(len(counts_encoding), COUNTS_SIZE_HEADER_BITS) + counts_encoding


def _decode_literal_counts_header(encoded_bitarray):
    """
    Decode literal counts header encoded by `_encode_literal_counts_header`.

    Returns:
        freqs (dict): symbol -> count (only symbols with count > 0)
        num_literals (int): total number of literals in the stream
        bits_consumed (int): number of bits consumed from encoded_bitarray
    """
    counts_encoding_size = bitarray_to_uint(
        encoded_bitarray[0:COUNTS_SIZE_HEADER_BITS]
    )
    bit_pos = COUNTS_SIZE_HEADER_BITS

    if counts_encoding_size == 0:
        return {}, 0, bit_pos

    counts_block, num_bits_counts = EliasDeltaUintDecoder().decode_block(
        encoded_bitarray[bit_pos : bit_pos + counts_encoding_size]
    )
    assert num_bits_counts == counts_encoding_size

    counts_list = counts_block.data_list
    # For literals we expect a full 256-long count vector.
    assert len(counts_list) == 256

    freqs = {i: c for i, c in enumerate(counts_list) if c > 0}
    num_literals = sum(counts_list)

    bit_pos += counts_encoding_size
    return freqs, num_literals, bit_pos


class TANSEncoder:
    """
    tANS encoder implementing tabled Asymmetric Numeral Systems.
    
    Parameters:
    - table_log: Log2 of table size (table_size = 2^table_log)
                 Larger values give better compression but use more memory
                 Typical values: 8-12
    """
    
    def __init__(self, table_log=10):
        self.table_log = table_log
        self.table_size = 1 << table_log  # 2^table_log
        self.table = None
        self.symbol_info = None
    
    def build_table(self, freqs):
        """
        Build the tANS encoding table from symbol frequencies.
        
        Args:
        - freqs: Dictionary mapping symbols to their frequencies
        
        Returns:
        - table: Encoding table with state transition info
        - symbol_info: Info needed for encoding each symbol
        """
        if not freqs:
            return None, None
        
        # Normalize frequencies to sum to table_size
        total = sum(freqs.values())
        symbols = sorted(freqs.keys())

        # Allocate states proportionally to frequencies
        # Ensure each symbol gets at least 1 state
        normalized_freqs = {}
        remaining = self.table_size

        for sym in symbols:
            # Round to ensure we use exactly table_size states
            allocated = max(1, int(freqs[sym] * self.table_size / total))
            normalized_freqs[sym] = allocated
            remaining -= allocated

        # Distribute remaining states to most frequent symbols
        while remaining != 0:
            for sym in symbols:
                if remaining == 0:
                    break
                if remaining > 0:
                    normalized_freqs[sym] += 1
                    remaining -= 1
                else:
                    if normalized_freqs[sym] > 1:
                        normalized_freqs[sym] -= 1
                        remaining += 1
        
        # Build the state table
        # table[state] = (symbol, next_state_base, num_bits_to_output)
        table = [None] * self.table_size
        symbol_info = {}
        
        position = 0
        for sym in symbols:
            num_states = normalized_freqs[sym]
            symbol_info[sym] = {
                'start': position,
                'freq': num_states
            }
            
            for i in range(num_states):
                table[position + i] = sym
            
            position += num_states
        
        self.table = table
        self.symbol_info = symbol_info
        return table, symbol_info
    
    def encode_symbol(self, state, symbol):
        """
        Encode a single symbol and update the state.
        
        Returns:
        - new_state: Updated state after encoding
        - bits_to_output: Bits to write to the stream
        """
        if symbol not in self.symbol_info:
            raise ValueError(f"Symbol {symbol} not in frequency table")
        
        info = self.symbol_info[symbol]
        freq = info['freq']
        start = info['start']
        
        # Renormalize: ensure state stays in valid range
        # We want state to be in [table_size, 2*table_size) before encoding
        bits_to_output = []
        while state >= (self.table_size * freq):
            # Output LSB and shift state right
            bits_to_output.append(state & 1)
            state >>= 1
        
        # Compute next state: (state // freq) * table_size + start + (state % freq)
        next_state = start + (state % freq)
        state = state // freq
        next_state = (state * self.table_size) + (next_state - start + start)
        
        # Simplified: spread state across symbol's allocated range
        next_state = start + (state % freq)
        
        return next_state, bits_to_output
    
    def encode(self, symbols):
        """
        Encode a sequence of symbols using TRUE tANS algorithm.
        
        Args:
        - symbols: List of symbols to encode
        
        Returns:
        - BitArray with encoded data (NO HEADERS - those are added by caller)
        """
        if not symbols:
            return BitArray([])
        
        # Count frequencies
        freqs = Counter(symbols)
        self.build_table(freqs)
        
        # Rebuild cumul_freq from symbol_info
        cumul_freq = {}
        for sym, info in self.symbol_info.items():
            cumul_freq[sym] = info['start']
        
        # Get freq dict from symbol_info
        freq = {sym: info['freq'] for sym, info in self.symbol_info.items()}
        
        # Initialize state
        state = self.table_size
        bits_list = []  # Use list for efficiency
        
        # Process symbols in REVERSE order (tANS property)
        for sym in reversed(symbols):
            if sym not in freq or freq[sym] == 0:
                continue
            
            # Renormalization: output bits if state is too large
            threshold = freq[sym] * self.table_size
            while state >= threshold:
                # Output lower log_size bits
                bits_list.append(uint_to_bitarray(state & ((1 << self.table_log) - 1), self.table_log))
                state >>= self.table_log
            
            # State transition (TRUE tANS formula)
            slot = state % freq[sym]
            state = cumul_freq[sym] + slot + (state // freq[sym]) * self.table_size
        
        # Reverse and concatenate efficiently
        bits = sum(reversed(bits_list), BitArray())

        # Layout: [final_state (32 bits)] + [bitstream]
        result = BitArray()
        result += uint_to_bitarray(state, 32)
        result += bits

        return result


class TANSDecoder:
    """
    tANS decoder - reverses the encoding process.
    """
    
    def __init__(self, table_log=10):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self.table = None
        self.symbol_info = None
    
    def build_table(self, freqs):
        """Build decoding table from frequencies (same as encoder)."""
        encoder = TANSEncoder(self.table_log)
        self.table, self.symbol_info = encoder.build_table(freqs)
    
    def decode(self, bitarray, num_symbols, freqs):
        """
        Decode a bitarray back to symbols using TRUE tANS algorithm.

        Args:
        - bitarray: Encoded BitArray (format: [final_state][bitstream])
        - num_symbols: Number of symbols to decode
        - freqs: Frequency dictionary (needed to rebuild table)

        Returns:
        - (symbols, bits_consumed): list of decoded symbols and bits consumed
        """
        if num_symbols == 0:
            return [], 0

        self.build_table(freqs)

        # Read final state
        state = bitarray_to_uint(bitarray[0:32])
        bits = bitarray[32:]

        # Rebuild cumul_freq and freq from symbol_info
        cumul_freq = {}
        freq = {}
        for sym, info in self.symbol_info.items():
            cumul_freq[sym] = info['start']
            freq[sym] = info['freq']
        
        # Decode symbols
        bit_pos = 0
        symbols = []

        for _ in range(num_symbols):
            # Get symbol from current state
            slot = state % self.table_size
            if slot >= len(self.table):
                break
            sym = self.table[slot]
            symbols.append(sym)

            # Recover previous state (TRUE tANS formula)
            slot_in_sym = (slot - cumul_freq[sym]) % freq[sym]
            quot = (state - slot) // self.table_size
            prev_state = quot * freq[sym] + slot_in_sym
            state = prev_state

            # Renormalize: read bits if state is too small
            while state < self.table_size and bit_pos + self.table_log <= len(bits):
                new_bits = bitarray_to_uint(bits[bit_pos:bit_pos + self.table_log])
                state = (state << self.table_log) | new_bits
                bit_pos += self.table_log

        bits_consumed = 32 + bit_pos
        return symbols, bits_consumed

class LZ77TANSStreamsEncoder:
    """
    Replacement for LZ77StreamsEncoder using tANS instead of the original entropy coder.
    
    Encodes LZ77 sequences (literals_count, match_length, match_offset) and literal bytes
    using separate tANS streams for better compression.
    """
    
    def __init__(self, table_log=10):
        self.table_log = table_log
    
    def encode_block(self, lz77_sequences, literals):
        """
        Encode LZ77 sequences and literals.
        
        Args:
        - lz77_sequences: List of LZ77Sequence objects
        - literals: bytearray of literal bytes
        
        Returns:
        - BitArray with encoded data
        """
        # Extract components from sequences
        literal_counts = [seq.literal_count for seq in lz77_sequences]
        match_lengths = [seq.match_length for seq in lz77_sequences]
        match_offsets = [seq.match_offset for seq in lz77_sequences]
        
        # Encode each stream with tANS
        encoder = TANSEncoder(self.table_log)
        
        # Encode literal bytes
        literals_encoded = encoder.encode(list(literals)) if literals else BitArray([])
        
        # Encode literal counts
        encoder_lc = TANSEncoder(self.table_log)
        literal_counts_encoded = encoder_lc.encode(literal_counts) if literal_counts else BitArray([])
        
        # Encode match lengths
        encoder_ml = TANSEncoder(self.table_log)
        match_lengths_encoded = encoder_ml.encode(match_lengths) if match_lengths else BitArray([])
        
        # Encode match offsets
        encoder_mo = TANSEncoder(self.table_log)
        match_offsets_encoded = encoder_mo.encode(match_offsets) if match_offsets else BitArray([])
        
        # Store metadata (frequencies and counts)
        # Format: [num_sequences][num_literals][freq_data][encoded_streams]
        
        result = BitArray([])
        
        # Store counts
        result += uint_to_bitarray(len(lz77_sequences), 32)
        result += uint_to_bitarray(len(literals), 32)
        
        # Store frequency tables (simplified - in production, use more compact encoding)
        # For each stream, store: num_unique_symbols, then (symbol, freq) pairs
        
        def encode_freqs(symbols):
            if not symbols:
                return BitArray([]) + uint_to_bitarray(0, 16)
            freqs = Counter(symbols)
            result = uint_to_bitarray(len(freqs), 16)
            for sym, freq in sorted(freqs.items()):
                result += uint_to_bitarray(sym, 32)
                result += uint_to_bitarray(freq, 32)
            return result
        
        result += encode_freqs(list(literals))
        result += encode_freqs(literal_counts)
        result += encode_freqs(match_lengths)
        result += encode_freqs(match_offsets)
        
        # Append encoded streams
        result += literals_encoded
        result += literal_counts_encoded
        result += match_lengths_encoded
        result += match_offsets_encoded
        
        return result


class LZ77TANSStreamsDecoder:
    """
    Replacement for LZ77StreamsDecoder using tANS.
    """
    
    def __init__(self, table_log=10):
        self.table_log = table_log
    
    def decode_block(self, encoded_bitarray):
        """
        Decode LZ77 sequences and literals from encoded bitarray.
        
        Returns:
        - tuple: (lz77_sequences, literals), num_bits_consumed
        """
        bit_pos = 0
        
        # Read counts
        num_sequences = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
        bit_pos += 32
        
        num_literals = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
        bit_pos += 32
        
        # Read frequency tables
        def decode_freqs():
            nonlocal bit_pos
            num_unique = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 16])
            bit_pos += 16
            if num_unique == 0:
                return {}
            freqs = {}
            for _ in range(num_unique):
                sym = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
                bit_pos += 32
                freq = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
                bit_pos += 32
                freqs[sym] = freq
            return freqs
        
        literals_freqs = decode_freqs()
        literal_counts_freqs = decode_freqs()
        match_lengths_freqs = decode_freqs()
        match_offsets_freqs = decode_freqs()
        
        # Now decode each stream using tANS
        decoder = TANSDecoder(self.table_log)
        
        # Decode literals
        if num_literals > 0 and literals_freqs:
            literals_bitarray = encoded_bitarray[bit_pos:]
            literals = decoder.decode(literals_bitarray, num_literals, literals_freqs)
            # Skip past the encoded literals stream
            # Format: [final_state (32)][bitstream_length (32)][bitstream]
            state_bits = 32
            length_bits = 32
            bitstream_length = bitarray_to_uint(literals_bitarray[state_bits:state_bits + length_bits])
            bit_pos += state_bits + length_bits + bitstream_length
        else:
            literals = []
        
        # Decode literal_counts
        if num_sequences > 0 and literal_counts_freqs:
            lc_bitarray = encoded_bitarray[bit_pos:]
            literal_counts = decoder.decode(lc_bitarray, num_sequences, literal_counts_freqs)
            state_bits = 32
            length_bits = 32
            bitstream_length = bitarray_to_uint(lc_bitarray[state_bits:state_bits + length_bits])
            bit_pos += state_bits + length_bits + bitstream_length
        else:
            literal_counts = []
        
        # Decode match_lengths
        if num_sequences > 0 and match_lengths_freqs:
            ml_bitarray = encoded_bitarray[bit_pos:]
            match_lengths = decoder.decode(ml_bitarray, num_sequences, match_lengths_freqs)
            state_bits = 32
            length_bits = 32
            bitstream_length = bitarray_to_uint(ml_bitarray[state_bits:state_bits + length_bits])
            bit_pos += state_bits + length_bits + bitstream_length
        else:
            match_lengths = []
        
        # Decode match_offsets
        if num_sequences > 0 and match_offsets_freqs:
            mo_bitarray = encoded_bitarray[bit_pos:]
            match_offsets = decoder.decode(mo_bitarray, num_sequences, match_offsets_freqs)
            state_bits = 32
            length_bits = 32
            bitstream_length = bitarray_to_uint(mo_bitarray[state_bits:state_bits + length_bits])
            bit_pos += state_bits + length_bits + bitstream_length
        else:
            match_offsets = []
        
        # Reconstruct sequences
        lz77_sequences = []
        for i in range(num_sequences):
            lz77_sequences.append(LZ77Sequence(
                literal_counts[i] if i < len(literal_counts) else 0,
                match_lengths[i] if i < len(match_lengths) else 0,
                match_offsets[i] if i < len(match_offsets) else 0
            ))
        
        return (lz77_sequences, bytearray(literals)), bit_pos

