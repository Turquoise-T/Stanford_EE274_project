"""
tANS (tabled Asymmetric Numeral Systems) implementation for LZ77 entropy coding.

This module provides tANS-based entropy encoding/decoding as a drop-in replacement
for the `LZ77StreamsEncoder`/`LZ77StreamsDecoder` used in `lz77.py`.

High level:
- Build a symbol table from empirical frequencies
- Encode symbols by transitioning through ANS states
- Decode by reversing the state transitions

The implementation uses native tANS headers with (symbol, frequency) pairs
for maximum compatibility with standard tANS implementations.
"""

from collections import Counter
import math

# Elias-Delta imports removed - using native tANS headers
from scl.compressors.lz77 import LZ77Sequence
from scl.compressors.golomb_coder import GolombUintEncoder, GolombUintDecoder
# DataBlock import removed - not needed for native tANS headers
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray


# This implementation uses native tANS headers with (symbol, frequency) pairs
# rather than Elias-Delta compressed count vectors


# Note: Elias-Delta encoding functions removed as this implementation
# uses native tANS header format with (symbol, frequency) pairs


# Canonical stream order (bit positions 0..len(_STREAMS)-1)
_STREAMS = ("literals", "literal_counts", "match_lengths", "match_offsets")

# Header format constants (protocol-level; encoder/decoder must match)
_COUNT_BITS = 32
_TABLE_LOG_BITS = 4
_ENCODING_METHOD_BITS = 1
_GOLOMB_M_BITS = 16
_REUSE_MASK_BITS = len(_STREAMS)  # one bit per stream
_LITERAL_MODEL_BITS = 1  # 0=flat, 1=class4
_LITERAL_CLASS_TABLE_LOG_BITS = 4  # 4 extra table_logs for class byte substreams
_FREQ_TABLE_SIZE_BITS = 16
_FREQ_ENTRY_BITS = 32  # both symbol and freq are encoded as 32-bit uints
_TANS_STATE_BITS = 32

_ASCII_WS = set(b" \t\r\n\v\f")


def _byte_class(b: int) -> int:
    # 0=alpha, 1=digit, 2=whitespace, 3=other
    if 65 <= b <= 90 or 97 <= b <= 122:
        return 0
    if 48 <= b <= 57:
        return 1
    if b in _ASCII_WS:
        return 2
    return 3


def _can_reuse_freqs(prev: Counter, cur: Counter) -> bool:
    # Safe reuse requires the previous table to cover all symbols we will encode.
    return set(cur.keys()).issubset(prev.keys())


def select_optimal_table_log(
    symbols,
    *,
    min_table_log: int = 8,
    max_table_log: int = 12,
    default_table_log: int = 10,
    small_alphabet: int = 16,
    small_n: int = 100,
    large_alphabet: int = 256,
    large_n: int = 10000,
):
    """
    Select optimal table_log for a stream based on alphabet size and symbol count.

    Heuristic:
    - Small alphabet or few symbols: use smaller table_log (min_table_log)
    - Large alphabet or many symbols: use larger table_log (max_table_log)
    - Otherwise: use default_table_log

    Args:
        symbols: List of symbols to encode
        min_table_log: Minimum table_log to consider (default 8)
        max_table_log: Maximum table_log to consider (default 12)

    Returns:
        Optimal table_log value
    """
    if not symbols:
        return min_table_log

    # Avoid building a full Counter here; we only need alphabet size.
    alphabet_size = len(set(symbols))
    num_symbols = len(symbols)

    # Ensure the table can represent the alphabet (table_size >= alphabet_size).
    min_feasible = max(0, math.ceil(math.log2(max(1, alphabet_size))))

    if alphabet_size < small_alphabet or num_symbols < small_n:
        return max(min_table_log, min_feasible)
    if alphabet_size >= large_alphabet or num_symbols >= large_n:
        return max(max_table_log, min_feasible)
    return max(default_table_log, min_feasible)


def _normalize_freqs_largest_remainder(freqs: dict, table_size: int, *, min_freq: int = 1) -> dict:
    """Normalize positive freqs to integer counts summing to table_size (largest remainder).

    Guarantees:
    - If a symbol appears in freqs, its normalized count is >= min_freq.
    - Sum of normalized counts is exactly table_size.
    - Deterministic given freqs (ties broken by symbol id).
    """
    if not freqs:
        return {}

    symbols = sorted(freqs.keys())
    if table_size < len(symbols) * min_freq:
        raise ValueError(
            f"table_size={table_size} too small for alphabet={len(symbols)} with min_freq={min_freq}"
        )

    total = sum(freqs[s] for s in symbols)
    scaled = {s: (freqs[s] * table_size) / total for s in symbols}

    base = {}
    frac = {}
    for s in symbols:
        flo = int(math.floor(scaled[s]))
        base[s] = max(min_freq, flo)
        frac[s] = scaled[s] - flo

    cur = sum(base.values())
    if cur < table_size:
        # Add 1 to the largest fractional parts until we hit table_size.
        for s in sorted(symbols, key=lambda x: (frac[x], freqs[x], x), reverse=True)[: table_size - cur]:
            base[s] += 1
    elif cur > table_size:
        # Remove 1 from the smallest fractional parts (only where > min_freq) until we hit table_size.
        need = cur - table_size
        candidates = [s for s in symbols if base[s] > min_freq]
        if need > len(candidates) * (max(base[s] for s in candidates) - min_freq if candidates else 0):
            # Conservative guard; should not happen unless inputs are pathological.
            raise ValueError("Could not renormalize to requested table_size without violating min_freq")

        for s in sorted(candidates, key=lambda x: (frac[x], freqs[x], x)):
            if need == 0:
                break
            take = min(need, base[s] - min_freq)
            base[s] -= take
            need -= take
        if need != 0:
            raise ValueError("Could not renormalize to requested table_size without violating min_freq")

    return base


def _freqs_close_enough(
    prev_freqs: Counter,
    cur_freqs: Counter,
    *,
    min_samples: int,
    max_rel_l1: float,
) -> bool:
    """Return True if two empirical distributions are close enough to reuse the previous table.

    Uses relative L1 distance between probability vectors:
        0.5 * sum_s |p(s) - q(s)| <= max_rel_l1
    """
    prev_n = sum(prev_freqs.values())
    cur_n = sum(cur_freqs.values())
    if prev_n < min_samples or cur_n < min_samples:
        return False

    keys = set(prev_freqs.keys()) | set(cur_freqs.keys())
    if not keys:
        return True

    l1 = 0.0
    for k in keys:
        l1 += abs((prev_freqs.get(k, 0) / prev_n) - (cur_freqs.get(k, 0) / cur_n))
    return 0.5 * l1 <= max_rel_l1


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

        symbols = sorted(freqs.keys())
        normalized_freqs = _normalize_freqs_largest_remainder(freqs, self.table_size, min_freq=1)

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

    def encode_symbol(self, state, symbol, cumul_freq):
        """
        Encode a single symbol and update the state.

        Args:
        - state: Current ANS state
        - symbol: Symbol to encode
        - cumul_freq: Cumulative frequency mapping for symbols

        Returns:
        - new_state: Updated state after encoding
        - bits_to_output: BitArray of bits to write to the stream
        """
        if symbol not in self.symbol_info:
            raise ValueError(f"Symbol {symbol} not in frequency table")

        freq = self.symbol_info[symbol]['freq']

        # Renormalization: output bits if state is too large
        bits_list = []
        threshold = freq * self.table_size
        while state >= threshold:
            # Output lower table_log bits
            bits_list.append(uint_to_bitarray(state & ((1 << self.table_log) - 1), self.table_log))
            state >>= self.table_log

        # State transition (TRUE tANS formula)
        slot = state % freq
        new_state = cumul_freq[symbol] + slot + (state // freq) * self.table_size

        # Combine all output bits
        bits_to_output = sum(reversed(bits_list), BitArray())

        return new_state, bits_to_output

    def encode(self, symbols, freqs: Counter = None):
        """
        Encode a sequence of symbols using TRUE tANS algorithm.

        Args:
        - symbols: List of symbols to encode

        Returns:
        - BitArray with encoded data (NO HEADERS - those are added by caller)
        """
        if not symbols:
            return BitArray([])

        # Build table from provided freqs (for table reuse) or from empirical freqs.
        freqs = Counter(symbols) if freqs is None else freqs
        self.build_table(freqs)

        # Build cumulative frequency mapping
        cumul_freq = {}
        for sym, info in self.symbol_info.items():
            cumul_freq[sym] = info['start']

        # Initialize state
        state = self.table_size
        all_bits = BitArray()

        # Process symbols in REVERSE order (tANS property)
        for sym in reversed(symbols):
            if sym not in self.symbol_info:
                continue

            # Encode single symbol using the dedicated method
            state, bits_output = self.encode_symbol(state, sym, cumul_freq)
            all_bits = bits_output + all_bits  # Prepend bits (reverse order)

        # Layout: [final_state] + [bitstream]
        result = BitArray()
        result += uint_to_bitarray(state, _TANS_STATE_BITS)
        result += all_bits

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
        state = bitarray_to_uint(bitarray[0:_TANS_STATE_BITS])
        bits = bitarray[_TANS_STATE_BITS:]

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

        bits_consumed = _TANS_STATE_BITS + bit_pos
        return symbols, bits_consumed

class LZ77TANSStreamsEncoder:
    """
    Replacement for LZ77StreamsEncoder using tANS instead of the original entropy coder.

    Encodes LZ77 sequences (literals_count, match_length, match_offset) and literal bytes
    using separate tANS streams for better compression.

    Features:
    - Per-stream table_log selection based on alphabet size and symbol count
    - Match-length tANS gating: only uses tANS for match_length if #sequences > threshold
    """

    def __init__(
        self,
        table_log=10,
        match_length_tans_threshold=50,
        literal_model: str = "flat",
        *,
        reuse_tables: bool = True,
        reuse_min_samples: int = 256,
        reuse_max_rel_l1: float = 0.05,
    ):
        """
        Args:
            table_log: Default table_log (used as fallback, but per-stream selection overrides)
            match_length_tans_threshold: Minimum number of sequences to use tANS for match_length
        """
        self.default_table_log = table_log
        self.match_length_tans_threshold = match_length_tans_threshold
        if literal_model not in ("flat", "class4"):
            raise ValueError("literal_model must be one of: 'flat', 'class4'")
        self.literal_model = literal_model
        self.reuse_tables = reuse_tables
        self.reuse_min_samples = reuse_min_samples
        self.reuse_max_rel_l1 = reuse_max_rel_l1
        self._prev_freqs = {name: None for name in _STREAMS}  # Counters for non-literal or flat-literal tables
        self._prev_literals = None  # (model_id:int, tables:dict[str, Counter])

    def encode_block(self, lz77_sequences, literals):
        """
        Encode LZ77 sequences and literals.

        Args:
        - lz77_sequences: List of LZ77Sequence objects
        - literals: bytearray of literal bytes

        Returns:
        - BitArray with encoded data
        """
        literal_counts = [seq.literal_count for seq in lz77_sequences]
        match_lengths = [seq.match_length for seq in lz77_sequences]
        match_offsets = [seq.match_offset for seq in lz77_sequences]

        use_tans_for_match_length = len(lz77_sequences) >= self.match_length_tans_threshold

        match_length_encoding_method = 0  # 0=tANS, 1=Golomb
        golomb_M = None
        if match_lengths and not use_tans_for_match_length:
            mean_length = sum(match_lengths) / len(match_lengths)
            M = max(1, int(mean_length / math.log(2))) if mean_length > 0 else 4
            golomb_M = 2 ** max(0, int(math.log2(M)))
            match_length_encoding_method = 1

        literal_model_id = 0 if self.literal_model == "flat" else 1

        # Table logs
        literals_table_log = select_optimal_table_log(list(literals)) if literals else self.default_table_log
        literal_counts_table_log = select_optimal_table_log(literal_counts) if literal_counts else self.default_table_log
        match_lengths_table_log = select_optimal_table_log(match_lengths) if match_lengths else self.default_table_log
        match_offsets_table_log = select_optimal_table_log(match_offsets) if match_offsets else self.default_table_log

        class_byte_table_logs = [self.default_table_log] * 4
        if literal_model_id == 1 and literals:
            # class stream uses literals_table_log; store per-class byte table_logs in header too
            class_bytes = [[] for _ in range(4)]
            for b in literals:
                class_bytes[_byte_class(b)].append(b)
            for i in range(4):
                class_byte_table_logs[i] = select_optimal_table_log(class_bytes[i]) if class_bytes[i] else self.default_table_log

        # Current freqs
        cur_freqs = {
            "literal_counts": Counter(literal_counts) if literal_counts else Counter(),
            "match_lengths": Counter(match_lengths) if match_lengths else Counter(),
            "match_offsets": Counter(match_offsets) if match_offsets else Counter(),
        }
        if literal_model_id == 0:
            cur_literal_tables = {"flat": Counter(list(literals)) if literals else Counter()}
        else:
            lit_classes = [_byte_class(b) for b in literals]
            class_bytes = [[] for _ in range(4)]
            for b in literals:
                class_bytes[_byte_class(b)].append(b)
            cur_literal_tables = {
                "class": Counter(lit_classes) if lit_classes else Counter(),
                "b0": Counter(class_bytes[0]),
                "b1": Counter(class_bytes[1]),
                "b2": Counter(class_bytes[2]),
                "b3": Counter(class_bytes[3]),
            }

        # Reuse decisions + chosen freqs for encoding
        reuse = {name: False for name in _STREAMS}
        chosen_freqs = {}

        if match_length_encoding_method == 1:
            reuse["match_lengths"] = True

        if self.reuse_tables:
            # literals (all-or-nothing for its model)
            if literals and self._prev_literals is not None:
                prev_model_id, prev_tables = self._prev_literals
                if prev_model_id == literal_model_id and all(
                    _can_reuse_freqs(prev_tables[k], cur_literal_tables[k]) and _freqs_close_enough(
                        prev_tables[k],
                        cur_literal_tables[k],
                        min_samples=self.reuse_min_samples,
                        max_rel_l1=self.reuse_max_rel_l1,
                    )
                    for k in cur_literal_tables.keys()
                ):
                    reuse["literals"] = True

            for name in ("literal_counts", "match_offsets"):
                prev = self._prev_freqs[name]
                if prev is not None and _can_reuse_freqs(prev, cur_freqs[name]) and _freqs_close_enough(
                    prev,
                    cur_freqs[name],
                    min_samples=self.reuse_min_samples,
                    max_rel_l1=self.reuse_max_rel_l1,
                ):
                    reuse[name] = True

            if match_length_encoding_method == 0:
                prev = self._prev_freqs["match_lengths"]
                if prev is not None and _can_reuse_freqs(prev, cur_freqs["match_lengths"]) and _freqs_close_enough(
                    prev,
                    cur_freqs["match_lengths"],
                    min_samples=self.reuse_min_samples,
                    max_rel_l1=self.reuse_max_rel_l1,
                ):
                    reuse["match_lengths"] = True

        # Choose frequency tables for encoding and update caches when not reused
        if reuse["literals"] and self._prev_literals is not None:
            _, chosen_literal_tables = self._prev_literals
        else:
            chosen_literal_tables = cur_literal_tables
            self._prev_literals = (literal_model_id, cur_literal_tables)

        for name in ("literal_counts", "match_offsets"):
            chosen_freqs[name] = self._prev_freqs[name] if reuse[name] else cur_freqs[name]
            if not reuse[name]:
                self._prev_freqs[name] = cur_freqs[name]

        if match_length_encoding_method == 0:
            chosen_freqs["match_lengths"] = self._prev_freqs["match_lengths"] if reuse["match_lengths"] else cur_freqs["match_lengths"]
            if not reuse["match_lengths"]:
                self._prev_freqs["match_lengths"] = cur_freqs["match_lengths"]

        # Encode payloads using the chosen tables
        literals_payload = BitArray([])
        if literals:
            if literal_model_id == 0:
                literals_payload = TANSEncoder(literals_table_log).encode(list(literals), freqs=chosen_literal_tables["flat"])
            else:
                lit_classes = [_byte_class(b) for b in literals]
                class_bytes = [[] for _ in range(4)]
                for b in literals:
                    class_bytes[_byte_class(b)].append(b)
                literals_payload += TANSEncoder(literals_table_log).encode(lit_classes, freqs=chosen_literal_tables["class"])
                for i, key in enumerate(("b0", "b1", "b2", "b3")):
                    literals_payload += TANSEncoder(class_byte_table_logs[i]).encode(class_bytes[i], freqs=chosen_literal_tables[key]) if class_bytes[i] else BitArray([])

        literal_counts_payload = TANSEncoder(literal_counts_table_log).encode(literal_counts, freqs=chosen_freqs["literal_counts"]) if literal_counts else BitArray([])

        if match_length_encoding_method == 0:
            match_lengths_payload = TANSEncoder(match_lengths_table_log).encode(match_lengths, freqs=chosen_freqs["match_lengths"]) if match_lengths else BitArray([])
        else:
            match_lengths_payload = BitArray([])
            if match_lengths:
                golomb_encoder = GolombUintEncoder(golomb_M)
                for ml in match_lengths:
                    match_lengths_payload += golomb_encoder.encode_symbol(ml)

        match_offsets_payload = TANSEncoder(match_offsets_table_log).encode(match_offsets, freqs=chosen_freqs["match_offsets"]) if match_offsets else BitArray([])

        # Header
        result = BitArray([])
        result += uint_to_bitarray(len(lz77_sequences), _COUNT_BITS)
        result += uint_to_bitarray(len(literals), _COUNT_BITS)
        result += uint_to_bitarray(literals_table_log, _TABLE_LOG_BITS)
        result += uint_to_bitarray(literal_counts_table_log, _TABLE_LOG_BITS)
        result += uint_to_bitarray(match_lengths_table_log, _TABLE_LOG_BITS)
        result += uint_to_bitarray(match_offsets_table_log, _TABLE_LOG_BITS)
        result += uint_to_bitarray(literal_model_id, _LITERAL_MODEL_BITS)
        if literal_model_id == 1:
            for tl in class_byte_table_logs:
                result += uint_to_bitarray(tl, _LITERAL_CLASS_TABLE_LOG_BITS)
        result += uint_to_bitarray(match_length_encoding_method, _ENCODING_METHOD_BITS)
        if match_length_encoding_method == 1:
            result += uint_to_bitarray(golomb_M, _GOLOMB_M_BITS)

        reuse_mask = 0
        for i, name in enumerate(_STREAMS):
            reuse_mask |= (1 if reuse[name] else 0) << i
        result += uint_to_bitarray(reuse_mask, _REUSE_MASK_BITS)

        def encode_freq_table(c: Counter) -> BitArray:
            if not c:
                return uint_to_bitarray(0, _FREQ_TABLE_SIZE_BITS)
            out = uint_to_bitarray(len(c), _FREQ_TABLE_SIZE_BITS)
            for sym, freq in sorted(c.items()):
                out += uint_to_bitarray(sym, _FREQ_ENTRY_BITS)
                out += uint_to_bitarray(freq, _FREQ_ENTRY_BITS)
            return out

        if not reuse["literals"]:
            if literal_model_id == 0:
                result += encode_freq_table(cur_literal_tables["flat"])
            else:
                result += encode_freq_table(cur_literal_tables["class"])
                for key in ("b0", "b1", "b2", "b3"):
                    result += encode_freq_table(cur_literal_tables[key])

        if not reuse["literal_counts"]:
            result += encode_freq_table(cur_freqs["literal_counts"])

        if match_length_encoding_method == 0 and not reuse["match_lengths"]:
            result += encode_freq_table(cur_freqs["match_lengths"])

        if not reuse["match_offsets"]:
            result += encode_freq_table(cur_freqs["match_offsets"])

        # Payload order: literals, literal_counts, match_lengths, match_offsets
        result += literals_payload
        result += literal_counts_payload
        result += match_lengths_payload
        result += match_offsets_payload
        return result


class LZ77TANSStreamsDecoder:
    """
    Replacement for LZ77StreamsDecoder using tANS.
    """

    def __init__(self, table_log=10):
        # Kept for API compatibility; per-stream table_log values are carried in the block header.
        _ = table_log
        self._prev_freqs = {name: None for name in _STREAMS}
        self._prev_literals = None  # (model_id:int, tables:dict[str, Counter])

    def decode_block(self, encoded_bitarray):
        """
        Decode LZ77 sequences and literals from encoded bitarray.

        Returns:
        - tuple: (lz77_sequences, literals), num_bits_consumed
        """
        bit_pos = 0

        # Read counts
        num_sequences = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _COUNT_BITS])
        bit_pos += _COUNT_BITS

        num_literals = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _COUNT_BITS])
        bit_pos += _COUNT_BITS

        # Read per-stream table_log values
        literals_table_log = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _TABLE_LOG_BITS])
        bit_pos += _TABLE_LOG_BITS
        literal_counts_table_log = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _TABLE_LOG_BITS])
        bit_pos += _TABLE_LOG_BITS
        match_lengths_table_log = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _TABLE_LOG_BITS])
        bit_pos += _TABLE_LOG_BITS
        match_offsets_table_log = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _TABLE_LOG_BITS])
        bit_pos += _TABLE_LOG_BITS

        # Literal model id
        literal_model_id = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _LITERAL_MODEL_BITS])
        bit_pos += _LITERAL_MODEL_BITS

        class_byte_table_logs = [None] * 4
        if literal_model_id == 1:
            for i in range(4):
                class_byte_table_logs[i] = bitarray_to_uint(
                    encoded_bitarray[bit_pos:bit_pos + _LITERAL_CLASS_TABLE_LOG_BITS]
                )
                bit_pos += _LITERAL_CLASS_TABLE_LOG_BITS

        # Read match_length encoding method (0 = tANS, 1 = Golomb)
        match_length_encoding_method = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _ENCODING_METHOD_BITS])
        bit_pos += _ENCODING_METHOD_BITS

        # Read Golomb M parameter if using Golomb for match_length
        golomb_M = None
        if match_length_encoding_method == 1:
            golomb_M = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _GOLOMB_M_BITS])
            bit_pos += _GOLOMB_M_BITS

        # Read reuse mask
        reuse_mask = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _REUSE_MASK_BITS])
        bit_pos += _REUSE_MASK_BITS
        reuse = {name: bool(reuse_mask & (1 << i)) for i, name in enumerate(_STREAMS)}

        # Read frequency tables (only when not reused)
        def decode_freqs():
            nonlocal bit_pos
            num_unique = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _FREQ_TABLE_SIZE_BITS])
            bit_pos += _FREQ_TABLE_SIZE_BITS
            if num_unique == 0:
                return {}
            freqs = {}
            for _ in range(num_unique):
                sym = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _FREQ_ENTRY_BITS])
                bit_pos += _FREQ_ENTRY_BITS
                freq = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + _FREQ_ENTRY_BITS])
                bit_pos += _FREQ_ENTRY_BITS
                freqs[sym] = freq
            return freqs

        # Literals tables
        if reuse["literals"]:
            if self._prev_literals is None:
                raise ValueError("Header requested literals table reuse, but no previous table exists")
            prev_model_id, literals_tables = self._prev_literals
            if prev_model_id != literal_model_id:
                raise ValueError("Header requested literals table reuse, but literal model changed")
        else:
            if literal_model_id == 0:
                literals_tables = {"flat": Counter(decode_freqs())}
            else:
                literals_tables = {"class": Counter(decode_freqs())}
                for key in ("b0", "b1", "b2", "b3"):
                    literals_tables[key] = Counter(decode_freqs())
            self._prev_literals = (literal_model_id, literals_tables)

        # Other stream tables
        def get_table(name: str) -> Counter:
            if name == "match_lengths" and match_length_encoding_method == 1:
                return Counter()
            if reuse[name]:
                if self._prev_freqs[name] is None:
                    raise ValueError(f"Header requested {name} table reuse, but no previous table exists")
                return self._prev_freqs[name]
            c = Counter(decode_freqs())
            self._prev_freqs[name] = c
            return c

        literal_counts_freqs = get_table("literal_counts")
        match_lengths_freqs = get_table("match_lengths") if match_length_encoding_method == 0 else Counter()
        match_offsets_freqs = get_table("match_offsets")

        # Decode literals
        literals = []
        if num_literals > 0:
            if literal_model_id == 0:
                decoder_lit = TANSDecoder(literals_table_log)
                payload = encoded_bitarray[bit_pos:]
                literals, bits_used = decoder_lit.decode(payload, num_literals, literals_tables["flat"])
                bit_pos += bits_used
            else:
                # Decode class stream then class byte streams.
                decoder_cls = TANSDecoder(literals_table_log)
                payload = encoded_bitarray[bit_pos:]
                classes, bits_used = decoder_cls.decode(payload, num_literals, literals_tables["class"])
                bit_pos += bits_used

                counts = [0, 0, 0, 0]
                for c in classes:
                    counts[c] += 1

                class_bytes = []
                for i, key in enumerate(("b0", "b1", "b2", "b3")):
                    n = counts[i]
                    if n == 0:
                        class_bytes.append([])
                        continue
                    dec = TANSDecoder(class_byte_table_logs[i])
                    payload = encoded_bitarray[bit_pos:]
                    vals, used = dec.decode(payload, n, literals_tables[key])
                    bit_pos += used
                    class_bytes.append(vals)

                idx = [0, 0, 0, 0]
                for c in classes:
                    literals.append(class_bytes[c][idx[c]])
                    idx[c] += 1

        # Decode literal_counts
        literal_counts = []
        if num_sequences > 0 and literal_counts_freqs:
            decoder_lc = TANSDecoder(literal_counts_table_log)
            lc_bitarray = encoded_bitarray[bit_pos:]
            literal_counts, bits_used = decoder_lc.decode(lc_bitarray, num_sequences, literal_counts_freqs)
            bit_pos += bits_used

        # Decode match_lengths (with gating)
        match_lengths = []
        if num_sequences > 0:
            if match_length_encoding_method == 0:
                # Use tANS
                decoder_ml = TANSDecoder(match_lengths_table_log)
                ml_bitarray = encoded_bitarray[bit_pos:]
                match_lengths, bits_used = decoder_ml.decode(ml_bitarray, num_sequences, match_lengths_freqs)
                bit_pos += bits_used
            else:
                # Use Golomb: decode symbol by symbol
                golomb_decoder = GolombUintDecoder(golomb_M)
                for _ in range(num_sequences):
                    ml, bits_consumed = golomb_decoder.decode_symbol(encoded_bitarray[bit_pos:])
                    match_lengths.append(ml)
                    bit_pos += bits_consumed

        # Decode match_offsets
        match_offsets = []
        if num_sequences > 0 and match_offsets_freqs:
            decoder_mo = TANSDecoder(match_offsets_table_log)
            mo_bitarray = encoded_bitarray[bit_pos:]
            match_offsets, bits_used = decoder_mo.decode(mo_bitarray, num_sequences, match_offsets_freqs)
            bit_pos += bits_used

        # Reconstruct sequences
        lz77_sequences = []
        for i in range(num_sequences):
            lz77_sequences.append(LZ77Sequence(
                literal_counts[i] if i < len(literal_counts) else 0,
                match_lengths[i] if i < len(match_lengths) else 0,
                match_offsets[i] if i < len(match_offsets) else 0
            ))

        return (lz77_sequences, bytearray(literals)), bit_pos
