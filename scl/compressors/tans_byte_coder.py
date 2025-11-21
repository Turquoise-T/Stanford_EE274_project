"""
tANS Byte Stream Coder

Encodes and decodes byte sequences (0-255) using tANS entropy coding.
Optimized for dense frequency distributions over 256-symbol alphabet.
"""

from typing import List, Tuple
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from tans_core import (
    build_tables,
    tans_encode,
    tans_decode,
    serialize_tables,
    deserialize_tables
)


class TANSEncoderByte:
    """Encoder for byte sequences using tANS."""

    @staticmethod
    def encode(values: List[int]) -> BitArray:
        """Encode a byte sequence using tANS."""
        if len(values) == 0:
            return BitArray()

        table_size, log_size, dec_table, encode_table, freq = build_tables(values)
        final_state, bitstream = tans_encode(values, dec_table, encode_table, log_size, table_size)

        out = BitArray()
        out += serialize_tables(freq, dec_table, encode_table)
        out += uint_to_bitarray(final_state, 32)
        out += uint_to_bitarray(len(bitstream), 32)
        out += bitstream

        return out


class TANSDecoderByte:
    """Decoder for byte sequences using tANS."""

    @staticmethod
    def decode(bitarr: BitArray, pos: int = 0) -> Tuple[List[int], int]:
        """Decode a byte sequence from tANS-encoded bitstream."""
        (table_size, log_size, dec_table, encode_table, freq), pos2 = deserialize_tables(bitarr, pos)

        final_state = bitarray_to_uint(bitarr[pos2:pos2 + 32])
        pos2 += 32

        bitstream_len = bitarray_to_uint(bitarr[pos2:pos2 + 32])
        pos2 += 32

        bitstream = bitarr[pos2:pos2 + bitstream_len]
        pos2 += bitstream_len

        values = tans_decode(final_state, bitstream, dec_table, freq, log_size, table_size)

        return values, (pos2 - pos)
