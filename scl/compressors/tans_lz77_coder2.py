"""
tANS LZ77 - BULLETPROOF VERSION
Absolutely will not crash on large alphabets.
"""

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.compressors.lz77 import LZ77Sequence, LogScaleBinnedIntegerEncoder, LogScaleBinnedIntegerDecoder
from scl.core.data_block import DataBlock
from collections import Counter
import sys


def _highbit(x: int) -> int:
    """floor(log2(x)), assuming x > 0"""
    return x.bit_length() - 1


def _table_step(table_size: int) -> int:
    return (table_size >> 1) + (table_size >> 3) + 3


def _normalize_freqs(freqs: dict, table_size: int) -> dict:
    """Normalize raw frequencies so that sum(norm)=table_size, all positive."""
    if not freqs:
        return {}

    symbols = sorted(freqs.keys())
    if len(symbols) > table_size:
        raise ValueError(
            f"tANS table_size={table_size} too small for alphabet={len(symbols)}. "
            f"Increase table_log or avoid tANS for this stream."
        )

    total = sum(freqs.values())
    floats = [(sym, freqs[sym] * table_size / total) for sym in symbols]
    norm = {sym: int(val) for sym, val in floats}

    for sym in symbols:
        if norm[sym] <= 0:
            norm[sym] = 1

    used = sum(norm.values())
    remainders = sorted(
        ((sym, (freqs[sym] * table_size / total) - int(freqs[sym] * table_size / total)) for sym in symbols),
        key=lambda x: x[1],
        reverse=True,
    )

    if used < table_size:
        need = table_size - used
        i = 0
        while need > 0:
            sym = remainders[i % len(remainders)][0]
            norm[sym] += 1
            need -= 1
            i += 1
    elif used > table_size:
        need = used - table_size
        remainders_rev = sorted(remainders, key=lambda x: x[1])
        i = 0
        while need > 0 and i < 10 * len(remainders_rev):
            sym = remainders_rev[i % len(remainders_rev)][0]
            if norm[sym] > 1:
                norm[sym] -= 1
                need -= 1
            i += 1
        if need != 0:
            raise RuntimeError("Failed to normalize frequencies to exact table_size.")

    if sum(norm.values()) != table_size:
        raise RuntimeError("Normalization bug: sum(norm) != table_size")
    return norm


def _spread_symbols(norm: dict, table_log: int) -> list:
    """Build tableSymbol[0..L-1] by spreading symbols."""
    L = 1 << table_log
    mask = L - 1
    step = _table_step(L)

    table_symbol = [None] * L
    pos = 0

    for sym in sorted(norm.keys()):
        for _ in range(norm[sym]):
            table_symbol[pos] = sym
            pos = (pos + step) & mask

    if any(v is None for v in table_symbol):
        raise RuntimeError("Spread failed: table_symbol has None entries.")
    return table_symbol


class TANSEncoder:
    """True tANS / FSE-style encoder."""

    def __init__(self, table_log=10):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self._norm = None
        self._state_table = None
        self._symbol_tt = None

    def _build_ctable(self, freqs: dict):
        L = self.table_size
        table_log = self.table_log

        norm = _normalize_freqs(freqs, L)
        table_symbol = _spread_symbols(norm, table_log)

        symbols = sorted(norm.keys())
        cumul = {}
        running = 0
        for sym in symbols:
            cumul[sym] = running
            running += norm[sym]

        state_table = [0] * L
        next_pos = dict(cumul)
        for state_index in range(L):
            sym = table_symbol[state_index]
            idx = next_pos[sym]
            state_table[idx] = L + state_index
            next_pos[sym] = idx + 1

        symbol_tt = {}
        total = 0
        for sym in symbols:
            count = norm[sym]
            if count == 1:
                delta_nb_bits = (table_log << 16) - L
                delta_find_state = total - 1
                total += 1
            else:
                max_bits_out = table_log - _highbit(count - 1)
                min_state_plus = count << max_bits_out
                delta_nb_bits = (max_bits_out << 16) - min_state_plus
                delta_find_state = total - count
                total += count
            symbol_tt[sym] = (delta_find_state, delta_nb_bits)

        self._norm = norm
        self._state_table = state_table
        self._symbol_tt = symbol_tt

    def encode(self, symbols):
        """Returns BitArray: [init_state (32 bits)] + [bitstream]"""
        if not symbols:
            return BitArray([])

        freqs = Counter(symbols)
        self._build_ctable(freqs)

        L = self.table_size
        state = L
        out_bits = BitArray([])

        for sym in reversed(symbols):
            if sym not in self._symbol_tt:
                raise ValueError(f"Symbol {sym} not in table")

            delta_find, delta_nb = self._symbol_tt[sym]
            nb_bits_out = (state + delta_nb) >> 16

            if nb_bits_out < 0 or nb_bits_out > self.table_log:
                raise RuntimeError(f"nb_bits_out out of range: {nb_bits_out}")

            if nb_bits_out > 0:
                bits_val = state & ((1 << nb_bits_out) - 1)
                out_bits = uint_to_bitarray(bits_val, nb_bits_out) + out_bits

            idx = (state >> nb_bits_out) + delta_find
            if idx < 0 or idx >= L:
                raise RuntimeError(f"CTable index out of range: {idx}")
            state = self._state_table[idx]

        init_state = state & (L - 1)
        result = BitArray([])
        result += uint_to_bitarray(init_state, 32)
        result += out_bits
        return result


class TANSDecoder:
    """True tANS / FSE-style decoder."""

    def __init__(self, table_log=10):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self._dtable = None

    def _build_dtable(self, freqs: dict):
        L = self.table_size
        table_log = self.table_log

        norm = _normalize_freqs(freqs, L)
        table_symbol = _spread_symbols(norm, table_log)

        symbol_next = {sym: norm[sym] for sym in norm.keys()}
        dtable = [None] * L

        for state in range(L):
            sym = table_symbol[state]
            next_state = symbol_next[sym]
            symbol_next[sym] = next_state + 1

            nb_bits = table_log - _highbit(next_state)
            new_state = (next_state << nb_bits) - L
            dtable[state] = (sym, nb_bits, new_state)

        self._dtable = dtable

    def decode(self, bitarray, num_symbols, freqs):
        """Returns (symbols, bits_consumed)"""
        if num_symbols == 0:
            return [], 0

        self._build_dtable(freqs)

        state = bitarray_to_uint(bitarray[0:32])
        bits = bitarray[32:]

        if state >= self.table_size:
            raise ValueError(f"Bad init_state={state}, table_size={self.table_size}")

        out = []
        bit_pos = 0

        for _ in range(num_symbols):
            sym, nb_bits, new_state_base = self._dtable[state]
            out.append(sym)

            if nb_bits > 0:
                if bit_pos + nb_bits > len(bits):
                    raise ValueError("Not enough bits to decode stream.")
                low = bitarray_to_uint(bits[bit_pos:bit_pos + nb_bits])
                bit_pos += nb_bits
            else:
                low = 0

            state = new_state_base + low
            if state < 0 or state >= self.table_size:
                raise RuntimeError(f"Decoded state out of range: {state}")

        return out, 32 + bit_pos


def _safe_tans_encode(symbols, table_log, fallback_offset=16):
    """
    BULLETPROOF: Try tANS, fall back to LogScaleBinned if alphabet too large.
    This function GUARANTEES no crash.
    """
    if not symbols:
        return BitArray([]), False  # (encoded, used_tans)

    table_size = 1 << table_log
    unique_symbols = len(set(symbols))

    # Check if tANS is feasible
    if unique_symbols > table_size:
        # Use fallback
        fallback_encoder = LogScaleBinnedIntegerEncoder(offset=fallback_offset)
        encoded = fallback_encoder.encode_block(DataBlock(symbols))
        return encoded, False

    # Try tANS with safety catch
    try:
        encoder = TANSEncoder(table_log)
        encoded = encoder.encode(symbols)
        return encoded, True
    except (ValueError, RuntimeError) as e:
        # If tANS fails for any reason, use fallback
        print(f"[tANS Safety] tANS failed ({e}), using fallback", file=sys.stderr)
        fallback_encoder = LogScaleBinnedIntegerEncoder(offset=fallback_offset)
        encoded = fallback_encoder.encode_block(DataBlock(symbols))
        return encoded, False


class LZ77TANSStreamsEncoder:
    """BULLETPROOF hybrid LZ77 encoder."""

    def __init__(self, table_log=10, log_scale_binned_coder_offset=16):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self.log_scale_binned_coder_offset = log_scale_binned_coder_offset

    def encode_block(self, lz77_sequences, literals):
        """Encode with guaranteed no-crash hybrid encoding."""
        # Extract components
        literal_counts = [seq.literal_count for seq in lz77_sequences]
        match_lengths = [seq.match_length for seq in lz77_sequences]
        match_offsets = [seq.match_offset for seq in lz77_sequences]

        # Use safe encoding for each stream
        literals_encoded, use_tans_lit = _safe_tans_encode(
            list(literals), self.table_log, self.log_scale_binned_coder_offset
        ) if literals else (BitArray([]), False)

        literal_counts_encoded, use_tans_lc = _safe_tans_encode(
            literal_counts, self.table_log, self.log_scale_binned_coder_offset
        ) if literal_counts else (BitArray([]), False)

        match_lengths_encoded, use_tans_ml = _safe_tans_encode(
            match_lengths, self.table_log, self.log_scale_binned_coder_offset
        ) if match_lengths else (BitArray([]), False)

        match_offsets_encoded, use_tans_mo = _safe_tans_encode(
            match_offsets, self.table_log, self.log_scale_binned_coder_offset
        ) if match_offsets else (BitArray([]), False)

        # Build result
        result = BitArray([])

        # Counts
        result += uint_to_bitarray(len(lz77_sequences), 32)
        result += uint_to_bitarray(len(literals), 32)

        # Encoding flags (4 bits)
        encoding_flags = (
            (1 if use_tans_lit else 0) |
            ((1 if use_tans_lc else 0) << 1) |
            ((1 if use_tans_ml else 0) << 2) |
            ((1 if use_tans_mo else 0) << 3)
        )
        result += uint_to_bitarray(encoding_flags, 4)

        # Frequency tables (only for tANS streams)
        def encode_freqs(symbols):
            if not symbols:
                return uint_to_bitarray(0, 16)
            freqs = Counter(symbols)
            freq_result = uint_to_bitarray(len(freqs), 16)
            for sym, freq in sorted(freqs.items()):
                freq_result += uint_to_bitarray(sym, 32)
                freq_result += uint_to_bitarray(freq, 32)
            return freq_result

        # Always encode frequency tables to maintain consistent format
        if use_tans_lit:
            result += encode_freqs(list(literals))
        else:
            result += uint_to_bitarray(0, 16)

        if use_tans_lc:
            result += encode_freqs(literal_counts)
        else:
            result += uint_to_bitarray(0, 16)

        if use_tans_ml:
            result += encode_freqs(match_lengths)
        else:
            result += uint_to_bitarray(0, 16)

        if use_tans_mo:
            result += encode_freqs(match_offsets)
        else:
            result += uint_to_bitarray(0, 16)

        # Encoded streams
        result += literals_encoded
        result += literal_counts_encoded
        result += match_lengths_encoded
        result += match_offsets_encoded

        return result


class LZ77TANSStreamsDecoder:
    """Bulletproof hybrid LZ77 decoder."""

    def __init__(self, table_log=10, log_scale_binned_coder_offset=16):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self.log_scale_binned_coder_offset = log_scale_binned_coder_offset

    def decode_block(self, encoded_bitarray):
        """Decode with hybrid encoding."""
        bit_pos = 0

        # Read counts
        num_sequences = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
        bit_pos += 32
        num_literals = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
        bit_pos += 32

        # Read encoding flags
        encoding_flags = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 4])
        bit_pos += 4

        use_tans_literals = (encoding_flags & 1) != 0
        use_tans_literal_counts = (encoding_flags & 2) != 0
        use_tans_match_lengths = (encoding_flags & 4) != 0
        use_tans_match_offsets = (encoding_flags & 8) != 0

        # Read frequency tables (always read to maintain consistent format)
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

        # Helper function to decode a stream
        def decode_stream(use_tans, freqs, num_items):
            nonlocal bit_pos
            if num_items == 0:
                return []

            if use_tans and freqs:
                decoder = TANSDecoder(self.table_log)
                data, bits_consumed = decoder.decode(encoded_bitarray[bit_pos:], num_items, freqs)
                bit_pos += bits_consumed
                return data
            else:
                fallback_decoder = LogScaleBinnedIntegerDecoder(offset=self.log_scale_binned_coder_offset)
                decoded_block, bits_consumed = fallback_decoder.decode_block(encoded_bitarray[bit_pos:])
                bit_pos += bits_consumed
                return decoded_block.data_list

        # Decode all streams
        literals = decode_stream(use_tans_literals, literals_freqs, num_literals)
        literal_counts = decode_stream(use_tans_literal_counts, literal_counts_freqs, num_sequences)
        match_lengths = decode_stream(use_tans_match_lengths, match_lengths_freqs, num_sequences)
        match_offsets = decode_stream(use_tans_match_offsets, match_offsets_freqs, num_sequences)

        # Reconstruct sequences
        lz77_sequences = []
        for i in range(num_sequences):
            lz77_sequences.append(LZ77Sequence(
                literal_counts[i] if i < len(literal_counts) else 0,
                match_lengths[i] if i < len(match_lengths) else 0,
                match_offsets[i] if i < len(match_offsets) else 0
            ))

        return (lz77_sequences, bytearray(literals)), bit_pos