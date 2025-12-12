"""
Canonical Huffman coding for integer symbols.

Uses canonical Huffman to avoid storing full code trees in the header.
Instead we just store code lengths and reconstruct the tree on decode.

Format: [length_header_size][length_data][value_size][encoded_values]
"""

from typing import Dict, List, Tuple

from scl.compressors.elias_delta_uint_coder import (
    EliasDeltaUintDecoder,
    EliasDeltaUintEncoder,
)
from scl.compressors.huffman_coder import HuffmanEncoder
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.utils.test_utils import get_random_data_block, try_lossless_compression

ENCODED_BLOCK_SIZE_HEADER_BITS = 32


def _build_canonical_codebook_from_lengths(
    code_lengths: List[int],
) -> Tuple[Dict[int, BitArray], Dict[Tuple[int, int], int], int]:
    """Build canonical Huffman tables from code lengths."""

    symbols = [s for s, L in enumerate(code_lengths) if L > 0]
    if not symbols:
        return {}, {}, 0

    # canonical order: sort by (length, symbol)
    symbols.sort(key=lambda s: (code_lengths[s], s))

    encode_table: Dict[int, BitArray] = {}
    decode_table: Dict[Tuple[int, int], int] = {}

    code_int = 0
    prev_length = code_lengths[symbols[0]]

    for idx, sym in enumerate(symbols):
        length = code_lengths[sym]
        if idx > 0:
            code_int += 1
            if length > prev_length:
                code_int <<= (length - prev_length)
            prev_length = length

        code_bits = uint_to_bitarray(code_int, length)
        encode_table[sym] = code_bits
        decode_table[(length, code_int)] = sym

    max_len = max(code_lengths)
    return encode_table, decode_table, max_len


class CanonicalIntHuffmanEncoder(DataEncoder):
    """
    Canonical Huffman encoder for integers in [0, alphabet_size).

    Stores only code lengths in header instead of full tree.
    """

    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size

    def encode_block(self, data_block: DataBlock) -> BitArray:
        vals: List[int] = data_block.data_list
        if not vals:
            return uint_to_bitarray(0, ENCODED_BLOCK_SIZE_HEADER_BITS)

        assert all(0 <= v < self.alphabet_size for v in vals)

        # get code lengths from standard Huffman
        counts = DataBlock(vals).get_counts()
        prob_dist = ProbabilityDist.normalize_prob_dict(counts)
        huff_encoder = HuffmanEncoder(prob_dist)
        huff_table: Dict[int, BitArray] = huff_encoder.encoding_table

        code_lengths: List[int] = [0] * self.alphabet_size
        for sym, bits in huff_table.items():
            code_lengths[sym] = len(bits)

        # build canonical codes
        encode_table, _, _ = _build_canonical_codebook_from_lengths(code_lengths)

        # encode lengths with Elias-Delta
        length_header_encoder = EliasDeltaUintEncoder()
        length_header_bits = length_header_encoder.encode_block(DataBlock(code_lengths))

        # encode actual data
        values_bits = BitArray()
        for v in vals:
            values_bits += encode_table[v]

        # pack everything together
        out = BitArray()
        out += uint_to_bitarray(len(length_header_bits), ENCODED_BLOCK_SIZE_HEADER_BITS)
        out += length_header_bits
        out += uint_to_bitarray(len(values_bits), ENCODED_BLOCK_SIZE_HEADER_BITS)
        out += values_bits
        return out


class CanonicalIntHuffmanDecoder(DataDecoder):
    """Decoder for CanonicalIntHuffmanEncoder."""

    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size

    def decode_block(self, encoded_bitarray: BitArray) -> Tuple[DataBlock, int]:
        num_bits_consumed = 0

        # read length header size
        if len(encoded_bitarray) < ENCODED_BLOCK_SIZE_HEADER_BITS:
            raise ValueError("Encoded bitarray too short to contain header size")
        length_header_size = bitarray_to_uint(
            encoded_bitarray[:ENCODED_BLOCK_SIZE_HEADER_BITS]
        )
        num_bits_consumed += ENCODED_BLOCK_SIZE_HEADER_BITS

        if length_header_size == 0:
            return DataBlock([]), num_bits_consumed

        # decode code lengths
        start = num_bits_consumed
        end = start + length_header_size
        length_header_bits = encoded_bitarray[start:end]
        length_block, used_bits = EliasDeltaUintDecoder().decode_block(
            length_header_bits
        )
        assert used_bits == length_header_size
        num_bits_consumed = end

        code_lengths: List[int] = length_block.data_list
        if len(code_lengths) != self.alphabet_size:
            raise ValueError(
                f"Expected {self.alphabet_size} code lengths, "
                f"got {len(code_lengths)}"
            )

        # rebuild canonical table
        _, decode_table, max_len = _build_canonical_codebook_from_lengths(
            code_lengths
        )
        if max_len == 0:
            return DataBlock([]), num_bits_consumed

        # read value bitstream size
        start = num_bits_consumed
        end = start + ENCODED_BLOCK_SIZE_HEADER_BITS
        if end > len(encoded_bitarray):
            raise ValueError("Truncated stream: missing value size header")
        value_bits_size = bitarray_to_uint(encoded_bitarray[start:end])
        num_bits_consumed = end

        start = num_bits_consumed
        end = start + value_bits_size
        if end > len(encoded_bitarray):
            raise ValueError("Truncated stream: value bits shorter than header")
        value_bits = encoded_bitarray[start:end]
        num_bits_consumed = end

        # decode values
        decoded_vals: List[int] = []
        i = 0
        nbits = len(value_bits)

        while i < nbits:
            code_int = 0
            # read bits until we find a valid code
            for length in range(1, max_len + 1):
                if i >= nbits:
                    raise ValueError("Truncated canonical Huffman codeword")
                bit = int(value_bits[i])
                i += 1
                code_int = (code_int << 1) | bit
                key = (length, code_int)
                if key in decode_table:
                    decoded_vals.append(decode_table[key])
                    break
            else:
                raise ValueError("Invalid canonical Huffman codeword encountered")

        return DataBlock(decoded_vals), num_bits_consumed


def test_canonical_int_huffman_encoder_decoder():
    """Basic sanity check for encode/decode."""
    alphabet_size = 16
    num_samples = 1000

    prob_dists = [
        ProbabilityDist({i: 1.0 / alphabet_size for i in range(alphabet_size)}),
        ProbabilityDist.normalize_prob_dict({i: (i + 1) for i in range(alphabet_size)}),  # 用 normalize_prob_dict
        ProbabilityDist({0: 0.7, 1: 0.2, 2: 0.05, 3: 0.05}),
    ]

    for prob_dist in prob_dists:
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
        decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

        is_lossless, _, _ = try_lossless_compression(
            data_block,
            encoder,
            decoder,
            add_extra_bits_to_encoder_output=True,
        )
        assert is_lossless, "Canonical Huffman coding is not lossless for this test case"


def test_canonical_int_huffman_encoder_decoder():
    """Basic sanity check for encode/decode."""
    alphabet_size = 16
    num_samples = 1000

    prob_dists = [
        ProbabilityDist({i: 1.0 / alphabet_size for i in range(alphabet_size)}),
        ProbabilityDist.normalize_prob_dict({i: (i + 1) for i in range(alphabet_size)}),
        ProbabilityDist({0: 0.7, 1: 0.2, 2: 0.05, 3: 0.05}),
    ]

    for prob_dist in prob_dists:
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)
        encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
        decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

        is_lossless, _, _ = try_lossless_compression(
            data_block,
            encoder,
            decoder,
            add_extra_bits_to_encoder_output=True,
        )
        assert is_lossless, "Canonical Huffman coding is not lossless for this test case"


def test_empty_block():
    """Test encoding/decoding empty blocks."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    empty_block = DataBlock([])
    encoded = encoder.encode_block(empty_block)
    decoded, bits_consumed = decoder.decode_block(encoded)

    assert decoded.data_list == []
    assert bits_consumed == ENCODED_BLOCK_SIZE_HEADER_BITS


def test_single_symbol():
    """Test with only one unique symbol."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    # all same symbol
    data = [42] * 100
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    decoded, _ = decoder.decode_block(encoded)

    assert decoded.data_list == data


def test_two_symbols():
    """Test with exactly two symbols."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    # two symbols with different frequencies
    data = [0] * 70 + [1] * 30
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    decoded, _ = decoder.decode_block(encoded)

    assert sorted(decoded.data_list) == sorted(data)


def test_all_symbols_used():
    """Test when all symbols in alphabet are used."""
    alphabet_size = 16
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    # use every symbol at least once
    data = list(range(alphabet_size)) * 10
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    decoded, _ = decoder.decode_block(encoded)

    assert sorted(decoded.data_list) == sorted(data)


def test_highly_skewed_distribution():
    """Test with very skewed distribution (one symbol dominates)."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    # 99% one symbol, 1% distributed among others
    data = [0] * 990 + list(range(1, 11))
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    decoded, _ = decoder.decode_block(encoded)

    assert sorted(decoded.data_list) == sorted(data)


def test_large_alphabet():
    """Test with larger alphabet size."""
    alphabet_size = 512
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    # sparse usage of large alphabet
    data = [0] * 50 + [100] * 30 + [200] * 20
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    decoded, _ = decoder.decode_block(encoded)

    assert sorted(decoded.data_list) == sorted(data)


def test_deterministic_encoding():
    """Test that same input produces same output."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)

    data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    data_block = DataBlock(data)

    encoded1 = encoder.encode_block(data_block)
    encoded2 = encoder.encode_block(data_block)

    assert encoded1 == encoded2


def test_bits_consumed_accurate():
    """Verify that bits_consumed matches actual encoded length."""
    alphabet_size = 256
    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)
    decoder = CanonicalIntHuffmanDecoder(alphabet_size=alphabet_size)

    data = list(range(10)) * 50
    data_block = DataBlock(data)

    encoded = encoder.encode_block(data_block)
    _, bits_consumed = decoder.decode_block(encoded)

    assert bits_consumed == len(encoded)


def test_random_data_reproducibility():
    """Test that encoding is reproducible with same random seed."""
    alphabet_size = 16
    num_samples = 500

    prob_dist = ProbabilityDist.normalize_prob_dict({i: i + 1 for i in range(10)})

    # generate same data twice
    data_block1 = get_random_data_block(prob_dist, num_samples, seed=42)
    data_block2 = get_random_data_block(prob_dist, num_samples, seed=42)

    encoder = CanonicalIntHuffmanEncoder(alphabet_size=alphabet_size)

    encoded1 = encoder.encode_block(data_block1)
    encoded2 = encoder.encode_block(data_block2)

    assert encoded1 == encoded2