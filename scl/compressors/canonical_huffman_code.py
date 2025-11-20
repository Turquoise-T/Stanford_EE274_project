"""
Canonical Huffman integer encoder/decoder.

This module implements a canonical Huffman code for integer alphabets
{0, 1, ..., alphabet_size - 1}. It is intended as a drop-in replacement
for EmpiricalIntHuffmanEncoder/Decoder when we want to transmit only
code lengths in the header and reconstruct the Huffman code using the
canonical construction on the decoder side.

The pipeline is:

Encoder:
    - Given a block of integer symbols in [0, alphabet_size),
      count occurrences and build a standard Huffman code to obtain
      code lengths for all used symbols.
    - Convert these code lengths into a canonical Huffman codebook:
        * sort symbols by (length, symbol id)
        * assign codes in increasing integer order, left-shifting
          when the length increases
    - Encode the code-length array using Elias-Delta coding.
    - Encode the data using the canonical codebook.
    - Output:
        [32 bits: length_of_length_header] +
        [length_header_bits] +
        [32 bits: length_of_value_bits] +
        [value_bits]

Decoder:
    - Read the length header size; if zero, the block is empty.
    - Decode the code-length array using Elias-Delta.
    - Reconstruct the canonical codebook from lengths.
    - Read the size of the value bitstream and decode symbols
      using the reconstructed codebook.

This mirrors the header structure of EmpiricalIntHuffmanEncoder, but
uses canonical Huffman instead of transmitting raw counts.
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

ENCODED_BLOCK_SIZE_HEADER_BITS = 32  # keep consistent with lz77.py


def _build_canonical_codebook_from_lengths(
    code_lengths: List[int],
) -> Tuple[Dict[int, BitArray], Dict[Tuple[int, int], int], int]:
    """
    Given a list of code lengths (per symbol), build canonical Huffman
    encode/decode tables.

    Args:
        code_lengths: list of length alphabet_size, where code_lengths[s]
                      is the desired code length (0 for unused symbols).

    Returns:
        encode_table: dict mapping symbol -> BitArray code
        decode_table: dict mapping (length, code_int) -> symbol
        max_len:      maximum code length among all symbols
    """
    # Collect symbols that are actually used (length > 0)
    symbols = [s for s, L in enumerate(code_lengths) if L > 0]
    if not symbols:
        return {}, {}, 0

    # Sort by (length, symbol id) as per canonical Huffman definition
    symbols.sort(key=lambda s: (code_lengths[s], s))

    encode_table: Dict[int, BitArray] = {}
    decode_table: Dict[Tuple[int, int], int] = {}

    # Initialize with the first symbol of the shortest length
    first = symbols[0]
    current_length = code_lengths[first]
    code_int = 0  # all zeros of current_length

    # Assign codes in canonical order
    prev_length = current_length
    for idx, sym in enumerate(symbols):
        length = code_lengths[sym]
        if idx == 0:
            # First symbol: code_int is already 0 with length = current_length
            prev_length = length
        else:
            # Increment code; if we move to a longer length, left-shift
            code_int += 1
            if length > prev_length:
                code_int <<= (length - prev_length)
            prev_length = length

        # Store BitArray code for encoder and integer code for decoder
        code_bits = uint_to_bitarray(code_int, length)
        encode_table[sym] = code_bits
        decode_table[(length, code_int)] = sym

    max_len = max(code_lengths)
    return encode_table, decode_table, max_len


class CanonicalIntHuffmanEncoder(DataEncoder):
    """
    Canonical Huffman encoder for integer alphabets [0, alphabet_size).

    For each block, it:
      1. Builds a standard Huffman code from the empirical distribution
         (using HuffmanEncoder) to obtain code lengths.
      2. Canonicalizes the code using the algorithm described above.
      3. Encodes the code-length array with Elias-Delta.
      4. Encodes the data with the canonical codebook.

    The output format is:
      [32 bits: length_of_length_header] +
      [length_header_bits] +
      [32 bits: length_of_value_bits] +
      [value_bits]

    If the block is empty (no values), we only output a 32-bit zero.
    """

    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size

    def encode_block(self, data_block: DataBlock) -> BitArray:
        vals: List[int] = data_block.data_list
        if not vals:
            # No data: transmit a 0 length header (like EmpiricalIntHuffmanEncoder)
            return uint_to_bitarray(0, ENCODED_BLOCK_SIZE_HEADER_BITS)

        # Sanity check: all values must be in [0, alphabet_size)
        assert all(0 <= v < self.alphabet_size for v in vals)

        # 1. Empirical distribution -> Huffman code lengths
        counts = DataBlock(vals).get_counts()  # dict symbol -> count
        # Build probability distribution only over observed symbols
        prob_dist = ProbabilityDist.normalize_prob_dict(counts)

        # Use the existing HuffmanEncoder to get a code table,
        # and derive code lengths from it.
        huff_encoder = HuffmanEncoder(prob_dist)
        huff_table: Dict[int, BitArray] = huff_encoder.encoding_table

        # Initialize full length array (unused symbols have length 0)
        code_lengths: List[int] = [0] * self.alphabet_size
        for sym, bits in huff_table.items():
            code_lengths[sym] = len(bits)

        # 2. Canonicalize: build canonical encode table
        encode_table, _, _ = _build_canonical_codebook_from_lengths(code_lengths)

        # 3. Encode the code-length array via Elias-Delta
        length_header_encoder = EliasDeltaUintEncoder()
        length_header_bits = length_header_encoder.encode_block(DataBlock(code_lengths))

        # 4. Encode the actual data using canonical codes
        values_bits = BitArray()
        for v in vals:
            values_bits += encode_table[v]

        # Concatenate everything:
        #   [len(length_header_bits)] [length_header_bits]
        #   [len(values_bits)]        [values_bits]
        out = BitArray()
        out += uint_to_bitarray(len(length_header_bits), ENCODED_BLOCK_SIZE_HEADER_BITS)
        out += length_header_bits
        out += uint_to_bitarray(len(values_bits), ENCODED_BLOCK_SIZE_HEADER_BITS)
        out += values_bits
        return out


class CanonicalIntHuffmanDecoder(DataDecoder):
    """
    Canonical Huffman decoder matching CanonicalIntHuffmanEncoder.

    Given the encoded bitarray, it:
      1. Reads the 32-bit size of the length header; if zero, returns
         an empty block.
      2. Decodes the length header via Elias-Delta to recover the
         code-length array.
      3. Reconstructs the canonical codebook from the lengths.
      4. Reads the 32-bit size of the value bitstream and decodes
         symbols sequentially using the canonical codebook.
    """

    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size

    def decode_block(self, encoded_bitarray: BitArray) -> Tuple[DataBlock, int]:
        num_bits_consumed = 0

        # 1. Read size of the length header
        if len(encoded_bitarray) < ENCODED_BLOCK_SIZE_HEADER_BITS:
            raise ValueError("Encoded bitarray too short to contain header size")
        length_header_size = bitarray_to_uint(
            encoded_bitarray[:ENCODED_BLOCK_SIZE_HEADER_BITS]
        )
        num_bits_consumed += ENCODED_BLOCK_SIZE_HEADER_BITS

        # Empty block case: encoder wrote only a zero header
        if length_header_size == 0:
            return DataBlock([]), num_bits_consumed

        # 2. Decode the length header using Elias-Delta
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

        # 3. Rebuild canonical decode table
        _, decode_table, max_len = _build_canonical_codebook_from_lengths(
            code_lengths
        )
        if max_len == 0:
            # No active symbols despite non-zero header: treat as empty
            return DataBlock([]), num_bits_consumed

        # 4. Read size of the value bitstream
        start = num_bits_consumed
        end = start + ENCODED_BLOCK_SIZE_HEADER_BITS
        if end > len(encoded_bitarray):
            raise ValueError("Truncated stream: missing value size header")
        value_bits_size = bitarray_to_uint(encoded_bitarray[start:end])
        num_bits_consumed = end

        # Extract the value bitstream
        start = num_bits_consumed
        end = start + value_bits_size
        if end > len(encoded_bitarray):
            raise ValueError("Truncated stream: value bits shorter than header")
        value_bits = encoded_bitarray[start:end]
        num_bits_consumed = end

        # 5. Decode the value bitstream using canonical codes
        decoded_vals: List[int] = []
        i = 0
        nbits = len(value_bits)

        while i < nbits:
            code_int = 0
            # Read bits until we hit a valid (length, code) pair
            for length in range(1, max_len + 1):
                if i >= nbits:
                    # Encoded stream is malformed or truncated
                    raise ValueError("Truncated canonical Huffman codeword")
                bit = int(value_bits[i])
                i += 1
                code_int = (code_int << 1) | bit
                key = (length, code_int)
                if key in decode_table:
                    decoded_vals.append(decode_table[key])
                    break
            else:
                # If we exit the for-loop normally, no code matched
                raise ValueError("Invalid canonical Huffman codeword encountered")

        return DataBlock(decoded_vals), num_bits_consumed


def test_canonical_int_huffman_encoder_decoder():
    """
    Basic randomized test to ensure that CanonicalIntHuffmanEncoder and
    CanonicalIntHuffmanDecoder are lossless and behave sensibly on a
    few random distributions.
    """
    alphabet_size = 16
    num_samples = 1000

    # A few different random probability distributions over [0, alphabet_size)
    prob_dists = [
        ProbabilityDist({i: 1.0 / alphabet_size for i in range(alphabet_size)}),
        ProbabilityDist({i: (i + 1) for i in range(alphabet_size)}).normalize(),
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
