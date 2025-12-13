"""
LZ77 with tANS entropy coding implementation.

This is a modified version of lz77.py that replaces empirical Huffman coding 
with our optimized tANS implementation for all entropy coding tasks.

Key differences from lz77.py:
- EmpiricalIntHuffmanEncoder/Decoder replaced with TANSIntEncoder/Decoder
- All entropy coding (literals, literal_counts, match_lengths, match_offsets) uses tANS
- Same LZ77 parsing algorithm and sequence generation
- Same overall structure and API for easy comparison

The tANS implementation uses:
- Native tANS headers with (symbol, frequency) pairs
- Optimized lookup tables for fast encoding/decoding
- Configurable table_log parameter for compression vs speed tradeoff

Usage is identical to lz77.py:
    python lz77_tans.py -i input.txt -o output.lz77
    python lz77_tans.py -d -i output.lz77 -o decoded.txt
"""

import argparse
from dataclasses import dataclass
import os
import tempfile
from typing import List, Tuple
from collections import Counter
import math

from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.utils.test_utils import (
    create_random_binary_file,
    try_file_lossless_compression,
    try_lossless_compression,
)

# Import standard LZ77 with Huffman for comparison
from scl.compressors.lz77 import LZ77Encoder, LZ77Decoder

# Import our optimized tANS implementation
from tans_lz77_coder import TANSEncoder, TANSDecoder

ENCODED_BLOCK_SIZE_HEADER_BITS = 32  # number of bits used for block size headers
DEFAULT_MIN_MATCH_LEN = 6
DEFAULT_MAX_NUM_MATCHES_CONSIDERED = 64


@dataclass
class LZ77Sequence:
    """LZ77Sequence that determines a series of operations during decompression.
    - First copy `literal_count` literal characters to output.
    - Next copy `match_length` characters from `match_offset` back in output to the output.
    """

    literal_count: int = 0
    match_length: int = 0
    match_offset: int = 0


class TANSIntEncoder(DataEncoder):
    """
    Perform entropy encoding of integer values using tANS and return the encoded bitarray.

    This assumes the values range from 0 to alphabet_size-1 (known in advance to encoder
    and decoder).

    We apply tANS coding for the values and also store the frequency table to enable the
    decoder to construct the same tANS table.
    """

    def __init__(self, alphabet_size: int, table_log: int = 10):
        self.alphabet_size = alphabet_size
        self.table_log = table_log

    def encode_block(self, data_block: DataBlock):
        vals = data_block.data_list
        # verify that all values are in the range 0 to alphabet_size-1
        assert all([val >= 0 and val < self.alphabet_size for val in vals])

        if not vals:
            # if no values, just transmit 0
            return uint_to_bitarray(0, ENCODED_BLOCK_SIZE_HEADER_BITS)

        # Encode with tANS
        encoder = TANSEncoder(table_log=self.table_log)
        tans_encoded = encoder.encode(vals)

        # Create native tANS header with frequency table
        freqs = Counter(vals)
        header = BitArray()
        
        # Header format: [32 bits: num_values] + [16 bits: num_unique] + [(symbol, freq) pairs]
        header += uint_to_bitarray(len(vals), 32)
        header += uint_to_bitarray(len(freqs), 16)
        
        for sym, freq in sorted(freqs.items()):
            header += uint_to_bitarray(sym, 16)  # symbol (16 bits for values 0-65535)
            header += uint_to_bitarray(freq, 32)  # frequency (32 bits)

        # Combine header and tANS payload
        return header + tans_encoded


class TANSIntDecoder(DataDecoder):
    """
    Decoder for TANSIntEncoder using tANS.
    """
    
    def __init__(self, alphabet_size: int, table_log: int = 10):
        self.alphabet_size = alphabet_size
        self.table_log = table_log

    def decode_block(self, encoded_bitarray: BitArray):
        bit_pos = 0
        
        # Read number of values
        if len(encoded_bitarray) < 32:
            return DataBlock([]), len(encoded_bitarray)
        
        num_values = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
        bit_pos += 32
        
        if num_values == 0:
            return DataBlock([]), bit_pos
        
        # Read number of unique symbols
        num_unique = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 16])
        bit_pos += 16
        
        # Read frequency table
        freqs = {}
        for _ in range(num_unique):
            sym = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 16])
            bit_pos += 16
            freq = bitarray_to_uint(encoded_bitarray[bit_pos:bit_pos + 32])
            bit_pos += 32
            freqs[sym] = freq
        
        # Decode with tANS
        decoder = TANSDecoder(table_log=self.table_log)
        payload = encoded_bitarray[bit_pos:]
        decoded_vals, bits_used = decoder.decode(payload, num_values, freqs)
        
        total_bits_consumed = bit_pos + bits_used
        return DataBlock(decoded_vals), total_bits_consumed


class TANSLogScaleBinnedIntegerEncoder(DataEncoder):
    """
    Encodes a list of non-negative integers by binning in log scale and then
    encoding the logarithm with tANS coding, and the difference to
    2^logarithm (residual) as plain old bits.

    This is the tANS equivalent of LogScaleBinnedIntegerEncoder from lz77.py.
    """

    def __init__(self, offset=0, max_num_bins=32, table_log=10):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.table_log = table_log
        self.tans_encoder = TANSIntEncoder(alphabet_size=self.max_num_bins, table_log=table_log)

    def encode_block(self, data_block: DataBlock):
        bins = []
        residuals = []
        residual_num_bits = []
        
        for val in data_block.data_list:
            assert val >= 0
            if val < self.offset:
                bins.append(val)
            else:
                val -= self.offset
                val_plus_1 = val + 1
                log_val_plus_1 = int(math.log2(val_plus_1))
                if log_val_plus_1 >= self.max_num_bins:
                    raise ValueError(
                        f"Value {val} is too large to be encoded with {self.max_num_bins} bins"
                    )
                bins.append(log_val_plus_1 + self.offset)
                residuals.append(val_plus_1 - 2**log_val_plus_1)
                residual_num_bits.append(log_val_plus_1)

        # Encode bins with tANS
        bins_encoding = self.tans_encoder.encode_block(DataBlock(bins))
        
        # Encode residuals as raw bits
        residuals_encoding = BitArray()
        for residual, num_bits in zip(residuals, residual_num_bits):
            if num_bits == 0:
                continue
            residuals_encoding += uint_to_bitarray(residual, num_bits)

        return bins_encoding + residuals_encoding


class TANSLogScaleBinnedIntegerDecoder(DataDecoder):
    """
    Decoder for TANSLogScaleBinnedIntegerEncoder.
    """

    def __init__(self, offset=0, max_num_bins=32, table_log=10):
        self.offset = offset
        self.max_num_bins = max_num_bins + self.offset
        self.table_log = table_log
        self.tans_decoder = TANSIntDecoder(alphabet_size=self.max_num_bins, table_log=table_log)

    def decode_block(self, encoded_bitarray: BitArray):
        # Decode bins with tANS
        bins_decoded, num_bits_consumed = self.tans_decoder.decode_block(encoded_bitarray)
        bins_decoded = bins_decoded.data_list
        encoded_bitarray = encoded_bitarray[num_bits_consumed:]
        
        decoded = []
        for encoded_bin in bins_decoded:
            if encoded_bin < self.offset:
                decoded.append(encoded_bin)
            else:
                encoded_bin -= self.offset
                log_val_plus_1 = encoded_bin
                num_bits = log_val_plus_1
                if num_bits == 0:
                    residual = 0
                else:
                    residual = bitarray_to_uint(encoded_bitarray[:num_bits])
                num_bits_consumed += num_bits
                encoded_bitarray = encoded_bitarray[num_bits:]
                decoded.append(self.offset + 2**log_val_plus_1 + residual - 1)
        
        return DataBlock(decoded), num_bits_consumed


class LZ77TANSStreamsEncoder(DataEncoder):
    """
    LZ77 streams encoder using tANS for all entropy coding.
    
    This replaces LZ77StreamsEncoder from lz77.py, using tANS instead of Huffman.
    """
    
    def __init__(self, log_scale_binned_coder_offset=16, table_log=10):
        """LZ77TANSStreamsEncoder. Encode LZ77 sequences and literals using tANS.

        Args:
            log_scale_binned_coder_offset (int): offset for log scale binned integer encoder
            table_log (int): tANS table log parameter
        """
        self.log_scale_binned_coder_offset = log_scale_binned_coder_offset
        self.table_log = table_log

    def encode_lz77_sequences(self, lz77_sequences: List[LZ77Sequence]):
        """Perform entropy encoding of the LZ77 sequences using tANS."""
        tans_log_scale_coder = TANSLogScaleBinnedIntegerEncoder(
            offset=self.log_scale_binned_coder_offset,
            table_log=self.table_log
        )
        encoded_bitarray = BitArray()
        encoded_bitarray += tans_log_scale_coder.encode_block(
            DataBlock([l.literal_count for l in lz77_sequences])
        )
        encoded_bitarray += tans_log_scale_coder.encode_block(
            DataBlock([l.match_length for l in lz77_sequences])
        )
        encoded_bitarray += tans_log_scale_coder.encode_block(
            DataBlock([l.match_offset for l in lz77_sequences])
        )
        return encoded_bitarray

    def encode_literals(self, literals: List):
        """Perform entropy encoding of the literals using tANS."""
        tans_encoder = TANSIntEncoder(alphabet_size=256, table_log=self.table_log)
        encoded_bitarray = tans_encoder.encode_block(DataBlock(literals))
        return encoded_bitarray

    def encode_block(self, lz77_sequences: List[LZ77Sequence], literals: List):
        """Encode LZ77 sequences and literals into a bitarray using tANS."""
        lz77_sequences_encoding = self.encode_lz77_sequences(lz77_sequences)
        literals_encoding = self.encode_literals(literals)
        return lz77_sequences_encoding + literals_encoding


class LZ77TANSStreamsDecoder(DataDecoder):
    """
    LZ77 streams decoder using tANS for all entropy decoding.
    
    This replaces LZ77StreamsDecoder from lz77.py, using tANS instead of Huffman.
    """
    
    def __init__(self, log_scale_binned_coder_offset=16, table_log=10):
        """LZ77TANSStreamsDecoder. Decode LZ77 sequences and literals from tANS encoding.

        Args:
            log_scale_binned_coder_offset (int): offset for log scale binned integer encoder
            table_log (int): tANS table log parameter
        """
        self.log_scale_binned_coder_offset = log_scale_binned_coder_offset
        self.table_log = table_log

    def decode_lz77_sequences(self, encoded_bitarray: BitArray):
        """Perform entropy decoding of the LZ77 sequences using tANS."""
        tans_log_scale_coder = TANSLogScaleBinnedIntegerDecoder(
            offset=self.log_scale_binned_coder_offset,
            table_log=self.table_log
        )
        num_bits_consumed = 0
        
        # Decode literal counts
        literal_counts, num_bits_consumed_literal_counts = tans_log_scale_coder.decode_block(
            encoded_bitarray
        )
        encoded_bitarray = encoded_bitarray[num_bits_consumed_literal_counts:]
        num_bits_consumed += num_bits_consumed_literal_counts
        
        # Decode match lengths
        match_lengths, num_bits_consumed_match_lengths = tans_log_scale_coder.decode_block(
            encoded_bitarray
        )
        encoded_bitarray = encoded_bitarray[num_bits_consumed_match_lengths:]
        num_bits_consumed += num_bits_consumed_match_lengths
        
        # Decode match offsets
        match_offsets, num_bits_consumed_match_offsets = tans_log_scale_coder.decode_block(
            encoded_bitarray
        )
        encoded_bitarray = encoded_bitarray[num_bits_consumed_match_offsets:]
        num_bits_consumed += num_bits_consumed_match_offsets
        
        lz77_sequences = [
            LZ77Sequence(l[0], l[1], l[2])
            for l in zip(literal_counts.data_list, match_lengths.data_list, match_offsets.data_list)
        ]
        return lz77_sequences, num_bits_consumed

    def decode_literals(self, encoded_bitarray: BitArray):
        """Perform entropy decoding of the literals using tANS."""
        tans_decoder = TANSIntDecoder(alphabet_size=256, table_log=self.table_log)
        literals, num_bits_consumed = tans_decoder.decode_block(encoded_bitarray)
        return literals.data_list, num_bits_consumed

    def decode_block(self, encoded_bitarray: BitArray):
        """Decode LZ77 sequences and literals from a bitarray using tANS."""
        lz77_sequences, num_bits_consumed_sequences = self.decode_lz77_sequences(encoded_bitarray)
        encoded_bitarray = encoded_bitarray[num_bits_consumed_sequences:]
        literals, num_bits_consumed_literals = self.decode_literals(encoded_bitarray)
        num_bits_consumed = num_bits_consumed_sequences + num_bits_consumed_literals

        return (lz77_sequences, literals), num_bits_consumed


class LZ77TANSEncoder(DataEncoder):
    """
    LZ77 Encoder with tANS entropy coding.
    
    This is identical to LZ77Encoder from lz77.py except it uses tANS for entropy coding
    instead of empirical Huffman.
    """
    
    def __init__(
        self,
        min_match_length: int = DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered: int = DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        initial_window: List = None,
        table_log: int = 10,
    ):
        """LZ77TANSEncoder. See lz77.py documentation for LZ77 details.

        Args:
            min_match_length (int, optional): Minimum match length. Defaults to 6.
            max_num_matches_considered (int, optional): Max number of matches considered
                at a position to bound time complexity. Defaults to 64.
            initial_window (List, optional): initialize window (dictionary). 
                The same initial window should be used for the decoder.
            table_log (int, optional): tANS table log parameter. Defaults to 10.
        """
        self.min_match_length = min_match_length
        self.max_num_matches_considered = max_num_matches_considered
        self.table_log = table_log
        self.window = []
        self.substring_dict = {}  # map from substr to list of positions where it occurs
        self.window_indexed_till = 0

        # if initial_window is provided, update window and index it
        if initial_window is not None:
            self.window = list(initial_window)
            self.index_window_upto_pos(len(self.window))

        self.streams_encoder = LZ77TANSStreamsEncoder(table_log=table_log)

    def reset(self):
        # reset the window and the index
        self.window = []
        self.substring_dict = {}
        self.window_indexed_till = 0

    def insert_substring_into_dict(self, substr: Tuple, start_pos: int):
        """Insert substring into the substring_dict."""
        if substr in self.substring_dict:
            self.substring_dict[substr].append(start_pos)
        else:
            self.substring_dict[substr] = [start_pos]

    def index_window_upto_pos(self, end_pos: int):
        """Index all tuples of min_match_length in self.window[:end_pos] into the substring_dict."""
        for end_pos_substr in range(self.window_indexed_till, end_pos + 1):
            start_pos_substr = end_pos_substr - self.min_match_length
            if start_pos_substr < 0:
                continue
            substr = tuple(self.window[start_pos_substr:end_pos_substr])
            self.insert_substring_into_dict(substr, start_pos_substr)
        self.window_indexed_till = end_pos + 1

    def find_match_length(self, start_pos_1: int, start_pos_2: int):
        """Find the match length of window starting from start_pos_1 and start_pos_2.
           Note: start_pos_1 should be < start_pos_2 (start_pos_1 is the reference, start_pos_2 is current)
        """
        # Prevent matching with itself
        if start_pos_1 >= start_pos_2:
            return 0
            
        match_length = 0
        while (start_pos_1 + match_length < len(self.window) and 
               start_pos_2 + match_length < len(self.window)):
            if self.window[start_pos_1 + match_length] != self.window[start_pos_2 + match_length]:
                break
            else:
                match_length += 1
        return match_length

    def lz77_parse_and_generate_sequences(self, data_block: DataBlock):
        """Parse data using LZ77 and returns the LZ77 sequences and literals.
        
        This is identical to the method in lz77.py - only the entropy coding changes.
        """
        lz77_sequences = []
        literals = []

        pos_in_window = len(self.window)

        # put the entire data block in the window at once, we will find matches later
        self.window += data_block.data_list

        # now go over the window starting at pos_in_window and try to find matches in the past
        while True:
            match_start_pos = pos_in_window
            match_found = False
            
            # loop over start positions until we find a match
            for match_start_pos in range(
                pos_in_window, len(self.window) - self.min_match_length + 1
            ):
                match_substr = tuple(
                    self.window[match_start_pos : match_start_pos + self.min_match_length]
                )
                if match_substr not in self.substring_dict:
                    # substring not seen before
                    self.index_window_upto_pos(match_start_pos + 1)
                    continue
                else:
                    candidate_match_positions = self.substring_dict[match_substr]
                    best_match_pos = None
                    best_match_length = 0
                    num_candidates_considered = 0
                    
                    # iterate over candidate_match_positions in reverse order
                    for candidate_match_pos in reversed(candidate_match_positions):
                        # Only consider positions that are strictly before the current position
                        if candidate_match_pos >= match_start_pos:
                            continue
                        match_len = self.find_match_length(candidate_match_pos, match_start_pos)
                        assert match_len >= self.min_match_length
                        if match_len > best_match_length:
                            best_match_length = match_len
                            best_match_pos = candidate_match_pos
                            match_found = True
                        num_candidates_considered += 1
                        if num_candidates_considered == self.max_num_matches_considered:
                            break
                if match_found:
                    break
                else:
                    # if match not found, we index the current substr
                    self.index_window_upto_pos(match_start_pos + 1)

            if not match_found:
                # no match found anywhere so put everything else as a literal and break
                literals += self.window[pos_in_window:]
                # make sure entire window is indexed
                self.index_window_upto_pos(len(self.window))
                break
            else:
                # match was found so we appropriately insert into literals and sequences
                literal_count = match_start_pos - pos_in_window
                literals += self.window[pos_in_window:match_start_pos]
                # compute the offset
                match_offset = match_start_pos - best_match_pos
                lz77_sequences.append(LZ77Sequence(literal_count, best_match_length, match_offset))
                match_end_pos = match_start_pos + best_match_length
                # index the covered portion into the substring_dict
                self.index_window_upto_pos(match_end_pos)
                # update position in window
                pos_in_window = match_end_pos

        return lz77_sequences, literals

    def encode_block(self, data_block: DataBlock):
        # first do lz77 parsing
        lz77_sequences, literals = self.lz77_parse_and_generate_sequences(data_block)
        # now encode sequences and literals with tANS
        encoded_bitarray = self.streams_encoder.encode_block(lz77_sequences, literals)
        return encoded_bitarray

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = 10000):
        """utility wrapper around the encode function using Uint8FileDataStream"""
        # call the encode function and write to the binary file
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class LZ77TANSDecoder(DataDecoder):
    """
    LZ77 Decoder with tANS entropy decoding.
    
    This is identical to LZ77Decoder from lz77.py except it uses tANS for entropy decoding
    instead of empirical Huffman.
    """
    
    def __init__(self, initial_window: List = None, table_log: int = 10):
        """Initialize LZ77TANS decoder.

        Args:
            initial_window (List, optional): initialize window (dictionary).
                The same initial window should be used as in encoder.
            table_log (int, optional): tANS table log parameter. Defaults to 10.
        """
        self.table_log = table_log
        self.window = []
        if initial_window is not None:
            self.window = list(initial_window)

        self.streams_decoder = LZ77TANSStreamsDecoder(table_log=table_log)

    def execute_lz77_sequences(self, literals: List, lz77_sequences: List[LZ77Sequence]):
        """Executes the LZ77 sequences and the literals and returns the decoded bytes.
        
        This is identical to the method in lz77.py.
        """
        window_len_before = len(self.window)
        pos_in_literals = 0
        for seq in lz77_sequences:
            # first copy over the literals
            self.window += literals[pos_in_literals : pos_in_literals + seq.literal_count]
            pos_in_literals += seq.literal_count
            # now copy the match
            if seq.match_length < seq.match_offset:
                # if the match length is not bigger than the offset a normal copy works!
                self.window += self.window[-seq.match_offset : -seq.match_offset + seq.match_length]
            else:
                # the match length exceeds the offset, so we need to copy byte by byte
                for _ in range(seq.match_length):
                    self.window.append(self.window[-seq.match_offset])

        # copy over any leftover literals
        self.window += literals[pos_in_literals:]

        return self.window[window_len_before:]

    def decode_block(self, encoded_bitarray: BitArray):
        # first entropy decode the lz77 sequences and the literals using tANS
        (lz77_sequences, literals), num_bits_consumed = self.streams_decoder.decode_block(
            encoded_bitarray
        )

        # now execute the sequences to decode
        decoded_block = DataBlock(self.execute_lz77_sequences(literals, lz77_sequences))
        return decoded_block, num_bits_consumed

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream"""
        # decode data and write output to a text file
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


# Test functions (adapted from lz77.py)
def test_tans_int_encoder_decoder():
    import random

    encoder = TANSIntEncoder(alphabet_size=45, table_log=10)
    decoder = TANSIntDecoder(alphabet_size=45, table_log=10)
    data_list = [random.randint(0, 44) for _ in range(1000)]
    data_block = DataBlock(data_list)
    is_lossless, _, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )
    assert is_lossless


def test_tans_log_scale_binned_integer_encoder_decoder():
    """Test that tANS log scale binned integer encoder and decoder are inverses"""
    import random

    encoder = TANSLogScaleBinnedIntegerEncoder(offset=10, table_log=10)
    decoder = TANSLogScaleBinnedIntegerDecoder(offset=10, table_log=10)
    data_list = (
        [0, 1, 5, 9, 10, 11, 12]
        + [random.randint(0, 20) for _ in range(100)]
        + [random.randint(0, 1000) for _ in range(100)]
    )
    data_block = DataBlock(data_list)
    is_lossless, _, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )
    assert is_lossless


def test_lz77_tans_encode_decode():
    initial_window = [0, 0, 1, 1, 1]
    data_list = [
        1, 1, 1, 1, 0, 0, 1, 1, 1, 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44,
    ]
    data_block = DataBlock(data_list)

    for min_match_length in [1, 2, 3, 4, 5]:
        for max_num_matches_considered in [0, 1, 5]:
            for table_log in [8, 10, 12]:
                encoder = LZ77TANSEncoder(
                    min_match_length, max_num_matches_considered, 
                    initial_window=initial_window, table_log=table_log
                )
                decoder = LZ77TANSDecoder(initial_window=initial_window, table_log=table_log)
                is_lossless, _, _ = try_lossless_compression(
                    data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
                )
                assert is_lossless


def test_lz77_tans_sequence_generation():
    """Test that LZ77 with tANS produces expected sequences (same as lz77.py)"""
    min_match_len = 3
    initial_window = [0, 0, 1, 1, 1]
    encoder = LZ77TANSEncoder(min_match_length=min_match_len, initial_window=initial_window, table_log=10)

    data_list = [
        1, 1, 1, 1, 0, 0, 1, 1, 1, 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44,
    ]
    data_block = DataBlock(data_list)

    expected_lits = [255, 254, 255, 254, 2, 44]
    expected_seqs = [
        LZ77Sequence(0, 4, 3),
        LZ77Sequence(0, 5, 9),
        LZ77Sequence(4, 3, 4),
        LZ77Sequence(1, 6, 22),
    ]
    seqs, lits = encoder.lz77_parse_and_generate_sequences(data_block)

    assert encoder.window == initial_window + data_list
    assert (
        sum(len(v) for v in encoder.substring_dict.values())
        == len(encoder.window) - min_match_len + 1
    )
    assert lits == expected_lits
    assert seqs == expected_seqs


def test_lz77_tans_multiblock_file_encode_decode():
    """Full test for LZ77TANSEncoder and LZ77TANSDecoder"""
    initial_window = [44, 45, 46] * 5
    # define encoder, decoder
    encoder = LZ77TANSEncoder(initial_window=initial_window, table_log=10)
    decoder = LZ77TANSDecoder(initial_window=initial_window, table_log=10)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a file with some random data
        input_file_path = os.path.join(tmpdirname, "inp_file.txt")
        create_random_binary_file(
            input_file_path,
            file_size=500,
            prob_dist=ProbabilityDist({44: 0.5, 45: 0.25, 46: 0.2, 255: 0.05}),
        )

        # test lossless compression
        assert try_file_lossless_compression(
            input_file_path, encoder, decoder, encode_block_size=1000
        )


def estimate_header_sizes(literals, lz77_sequences, table_log):
    """Estimate header sizes for tANS vs Huffman encoding."""
    from collections import Counter
    
    # Analyze literals
    literal_freqs = Counter(literals)
    num_unique_literals = len(literal_freqs)
    
    # tANS literals header: [32: num_literals] + [16: num_unique] + [num_unique * (16+32): (symbol,freq) pairs]
    tans_literals_header_bits = 32 + 16 + num_unique_literals * (16 + 32)
    
    # Huffman literals header: [32: counts_size] + [Elias-Delta encoded counts] + [32: huffman_size]
    # Estimate Elias-Delta size for counts (rough approximation)
    elias_delta_size = 0
    for i in range(256):  # All possible byte values
        count = literal_freqs.get(i, 0)
        if count == 0:
            elias_delta_size += 1  # Elias-Delta(0) ≈ 1 bit
        else:
            # Elias-Delta(n) ≈ 2*log2(n) + 1
            elias_delta_size += max(1, int(2 * math.log2(count) + 1))
    
    huffman_literals_header_bits = 32 + elias_delta_size + 32
    
    # For LZ77 sequences (literal_counts, match_lengths, match_offsets)
    # Both use log-scale binned encoding, so headers are similar
    # Estimate based on unique values in each stream
    
    def estimate_log_binned_header(values, method="huffman"):
        if not values:
            return 32  # Just the size field
        
        # Convert to bins
        bins = []
        for val in values:
            if val < 16:  # offset
                bins.append(val)
            else:
                val -= 16
                val_plus_1 = val + 1
                log_val_plus_1 = int(math.log2(val_plus_1))
                bins.append(log_val_plus_1 + 16)
        
        bin_freqs = Counter(bins)
        num_unique_bins = len(bin_freqs)
        
        if method == "tans":
            # tANS header for bins
            return 32 + 16 + num_unique_bins * (16 + 32)
        else:
            # Huffman header for bins (Elias-Delta)
            elias_size = 0
            max_bins = 48  # 16 offset + 32 log bins
            for i in range(max_bins):
                count = bin_freqs.get(i, 0)
                if count == 0:
                    elias_size += 1
                else:
                    elias_size += max(1, int(2 * math.log2(count) + 1))
            return 32 + elias_size + 32
    
    # Estimate sequence headers
    literal_counts = [seq.literal_count for seq in lz77_sequences]
    match_lengths = [seq.match_length for seq in lz77_sequences]
    match_offsets = [seq.match_offset for seq in lz77_sequences]
    
    tans_sequences_header_bits = (
        estimate_log_binned_header(literal_counts, "tans") +
        estimate_log_binned_header(match_lengths, "tans") +
        estimate_log_binned_header(match_offsets, "tans")
    )
    
    huffman_sequences_header_bits = (
        estimate_log_binned_header(literal_counts, "huffman") +
        estimate_log_binned_header(match_lengths, "huffman") +
        estimate_log_binned_header(match_offsets, "huffman")
    )
    
    return {
        'tans_literals_header': tans_literals_header_bits,
        'huffman_literals_header': huffman_literals_header_bits,
        'tans_sequences_header': tans_sequences_header_bits,
        'huffman_sequences_header': huffman_sequences_header_bits,
        'tans_total_header': tans_literals_header_bits + tans_sequences_header_bits,
        'huffman_total_header': huffman_literals_header_bits + huffman_sequences_header_bits
    }


def compare_tans_vs_huffman(input_file: str, table_log: int = 10, initial_window: List = None):
    """Compare tANS vs Empirical Huffman compression performance."""
    import time
    from collections import Counter
    
    print(f"\n{'='*90}")
    print(f"LZ77 COMPRESSION COMPARISON: tANS vs Empirical Huffman")
    print(f"{'='*90}")
    print(f"Input file: {input_file}")
    print(f"tANS table_log: {table_log}")
    
    # Read input file
    with open(input_file, 'rb') as f:
        input_data = list(f.read())
    
    original_size = len(input_data)
    data_block = DataBlock(input_data)
    print(f"Original size: {original_size:,} bytes")
    
    # Test tANS
    print(f"\n{'-'*50}")
    print("Testing LZ77 + tANS...")
    print(f"{'-'*50}")
    
    tans_encoder = LZ77TANSEncoder(initial_window=initial_window, table_log=table_log)
    tans_decoder = LZ77TANSDecoder(initial_window=initial_window, table_log=table_log)
    
    start_time = time.time()
    tans_encoded = tans_encoder.encode_block(data_block)
    tans_encode_time = time.time() - start_time
    
    start_time = time.time()
    tans_decoded, _ = tans_decoder.decode_block(tans_encoded)
    tans_decode_time = time.time() - start_time
    
    tans_size_bits = len(tans_encoded)
    tans_size_bytes = (tans_size_bits + 7) // 8
    tans_is_lossless = tans_decoded.data_list == input_data
    
    # Test Empirical Huffman
    print(f"\n{'-'*50}")
    print("Testing LZ77 + Empirical Huffman...")
    print(f"{'-'*50}")
    
    huffman_encoder = LZ77Encoder(initial_window=initial_window)
    huffman_decoder = LZ77Decoder(initial_window=initial_window)
    
    start_time = time.time()
    huffman_encoded = huffman_encoder.encode_block(data_block)
    huffman_encode_time = time.time() - start_time
    
    start_time = time.time()
    huffman_decoded, _ = huffman_decoder.decode_block(huffman_encoded)
    huffman_decode_time = time.time() - start_time
    
    huffman_size_bits = len(huffman_encoded)
    huffman_size_bytes = (huffman_size_bits + 7) // 8
    huffman_is_lossless = huffman_decoded.data_list == input_data
    
    # Calculate improvements
    size_improvement = (huffman_size_bytes - tans_size_bytes) / huffman_size_bytes * 100
    encode_speed_ratio = huffman_encode_time / tans_encode_time
    decode_speed_ratio = huffman_decode_time / tans_decode_time
    
    # Analyze LZ77 parsing (should be identical for both)
    lz77_sequences, literals = tans_encoder.lz77_parse_and_generate_sequences(data_block)
    literal_counter = Counter(literals)
    
    # Estimate header sizes
    header_analysis = estimate_header_sizes(literals, lz77_sequences, table_log)
    
    # Calculate payload sizes (total - estimated headers)
    tans_payload_bits = tans_size_bits - header_analysis['tans_total_header']
    huffman_payload_bits = huffman_size_bits - header_analysis['huffman_total_header']
    tans_payload_bytes = (tans_payload_bits + 7) // 8
    huffman_payload_bytes = (huffman_payload_bits + 7) // 8
    
    # Output comparison results
    print(f"\n{'='*90}")
    print(f"COMPRESSION COMPARISON RESULTS")
    print(f"{'='*90}")
    
    print(f"{'Method':<25} {'Size (bytes)':<12} {'Size (bits)':<12} {'Ratio':<8} {'Savings':<8} {'Lossless':<8}")
    print(f"{'-'*85}")
    print(f"{'Original':<25} {original_size:<12,} {original_size*8:<12,} {'1.0000':<8} {'0.0%':<8} {'N/A':<8}")
    print(f"{'LZ77 + Huffman':<25} {huffman_size_bytes:<12,} {huffman_size_bits:<12,} "
          f"{huffman_size_bytes/original_size:<8.4f} {(1-huffman_size_bytes/original_size)*100:<8.1f}% "
          f"{'✓' if huffman_is_lossless else '✗':<8}")
    print(f"{'LZ77 + tANS':<25} {tans_size_bytes:<12,} {tans_size_bits:<12,} "
          f"{tans_size_bytes/original_size:<8.4f} {(1-tans_size_bytes/original_size)*100:<8.1f}% "
          f"{'✓' if tans_is_lossless else '✗':<8}")
    
    print(f"\n{'='*90}")
    print(f"HEADER vs PAYLOAD BREAKDOWN")
    print(f"{'='*90}")
    
    print(f"{'Component':<20} {'Huffman (bytes)':<15} {'tANS (bytes)':<15} {'Difference':<15} {'tANS vs Huffman':<15}")
    print(f"{'-'*85}")
    
    # Literals header
    huffman_lit_header_bytes = (header_analysis['huffman_literals_header'] + 7) // 8
    tans_lit_header_bytes = (header_analysis['tans_literals_header'] + 7) // 8
    lit_header_diff = tans_lit_header_bytes - huffman_lit_header_bytes
    print(f"{'Literals Header':<20} {huffman_lit_header_bytes:<15,} {tans_lit_header_bytes:<15,} "
          f"{lit_header_diff:+<15,} {tans_lit_header_bytes/huffman_lit_header_bytes if huffman_lit_header_bytes > 0 else 0:<15.2f}x")
    
    # Sequences header
    huffman_seq_header_bytes = (header_analysis['huffman_sequences_header'] + 7) // 8
    tans_seq_header_bytes = (header_analysis['tans_sequences_header'] + 7) // 8
    seq_header_diff = tans_seq_header_bytes - huffman_seq_header_bytes
    print(f"{'Sequences Header':<20} {huffman_seq_header_bytes:<15,} {tans_seq_header_bytes:<15,} "
          f"{seq_header_diff:+<15,} {tans_seq_header_bytes/huffman_seq_header_bytes if huffman_seq_header_bytes > 0 else 0:<15.2f}x")
    
    # Total header
    huffman_total_header_bytes = (header_analysis['huffman_total_header'] + 7) // 8
    tans_total_header_bytes = (header_analysis['tans_total_header'] + 7) // 8
    total_header_diff = tans_total_header_bytes - huffman_total_header_bytes
    print(f"{'Total Header':<20} {huffman_total_header_bytes:<15,} {tans_total_header_bytes:<15,} "
          f"{total_header_diff:+<15,} {tans_total_header_bytes/huffman_total_header_bytes if huffman_total_header_bytes > 0 else 0:<15.2f}x")
    
    # Payload
    payload_diff = tans_payload_bytes - huffman_payload_bytes
    print(f"{'Payload':<20} {huffman_payload_bytes:<15,} {tans_payload_bytes:<15,} "
          f"{payload_diff:+<15,} {tans_payload_bytes/huffman_payload_bytes if huffman_payload_bytes > 0 else 0:<15.2f}x")
    
    # Total
    total_diff = tans_size_bytes - huffman_size_bytes
    print(f"{'-'*85}")
    print(f"{'TOTAL':<20} {huffman_size_bytes:<15,} {tans_size_bytes:<15,} "
          f"{total_diff:+<15,} {tans_size_bytes/huffman_size_bytes:<15.2f}x")
    
    # Header overhead analysis
    print(f"\n{'='*90}")
    print(f"HEADER OVERHEAD ANALYSIS")
    print(f"{'='*90}")
    
    huffman_header_ratio = huffman_total_header_bytes / huffman_size_bytes * 100
    tans_header_ratio = tans_total_header_bytes / tans_size_bytes * 100
    
    print(f"Huffman header overhead: {huffman_total_header_bytes:,} bytes ({huffman_header_ratio:.1f}% of total)")
    print(f"tANS header overhead:    {tans_total_header_bytes:,} bytes ({tans_header_ratio:.1f}% of total)")
    print(f"Header overhead difference: {total_header_diff:+,} bytes ({tans_header_ratio - huffman_header_ratio:+.1f}%)")
    
    # Detailed header breakdown
    print(f"\nDetailed header breakdown:")
    print(f"  Literals - Unique symbols: {len(literal_counter)}")
    print(f"  Literals - Huffman header: {huffman_lit_header_bytes:,} bytes (Elias-Delta counts)")
    print(f"  Literals - tANS header:    {tans_lit_header_bytes:,} bytes (symbol,freq pairs)")
    print(f"  Sequences - Huffman header: {huffman_seq_header_bytes:,} bytes (3 log-binned streams)")
    print(f"  Sequences - tANS header:    {tans_seq_header_bytes:,} bytes (3 log-binned streams)")
    
    print(f"\n{'='*90}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*90}")
    
    print(f"{'Metric':<25} {'Huffman':<15} {'tANS':<15} {'tANS vs Huffman':<20}")
    print(f"{'-'*80}")
    print(f"{'Encoding time':<25} {huffman_encode_time:<15.3f} {tans_encode_time:<15.3f} "
          f"{encode_speed_ratio:<20.2f}x")
    print(f"{'Decoding time':<25} {huffman_decode_time:<15.3f} {tans_decode_time:<15.3f} "
          f"{decode_speed_ratio:<20.2f}x")
    print(f"{'Encoding speed (MB/s)':<25} {original_size/(1024*1024)/huffman_encode_time:<15.2f} "
          f"{original_size/(1024*1024)/tans_encode_time:<15.2f} "
          f"{(original_size/(1024*1024)/tans_encode_time)/(original_size/(1024*1024)/huffman_encode_time):<20.2f}x")
    print(f"{'Decoding speed (MB/s)':<25} {original_size/(1024*1024)/huffman_decode_time:<15.2f} "
          f"{original_size/(1024*1024)/tans_decode_time:<15.2f} "
          f"{(original_size/(1024*1024)/tans_decode_time)/(original_size/(1024*1024)/huffman_decode_time):<20.2f}x")
    
    print(f"\n{'='*90}")
    print(f"COMPRESSION EFFICIENCY ANALYSIS")
    print(f"{'='*90}")
    
    if size_improvement > 0:
        print(f"✓ tANS achieves {size_improvement:.2f}% better compression than Huffman")
        print(f"  Size reduction: {huffman_size_bytes - tans_size_bytes:,} bytes")
    elif size_improvement < 0:
        print(f"✗ tANS is {-size_improvement:.2f}% worse than Huffman")
        print(f"  Size increase: {tans_size_bytes - huffman_size_bytes:,} bytes")
    else:
        print(f"= tANS and Huffman achieve identical compression")
    
    print(f"\n{'='*90}")
    print(f"LZ77 PARSING STATISTICS (identical for both methods)")
    print(f"{'='*90}")
    print(f"Total literals:       {len(literals):,}")
    print(f"Unique literal values: {len(literal_counter)}")
    print(f"LZ77 sequences:       {len(lz77_sequences):,}")
    
    if lz77_sequences:
        avg_match_length = sum(seq.match_length for seq in lz77_sequences) / len(lz77_sequences)
        avg_match_offset = sum(seq.match_offset for seq in lz77_sequences) / len(lz77_sequences)
        print(f"Average match length: {avg_match_length:.2f}")
        print(f"Average match offset: {avg_match_offset:.2f}")
    
    # Top literal frequencies
    if literals:
        print(f"\nTop 5 most frequent literals:")
        for i, (byte_val, count) in enumerate(literal_counter.most_common(5), 1):
            percentage = count / len(literals) * 100
            char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
            print(f"  {i}. Byte {byte_val:3d} ('{char_repr}'): {count:,} times ({percentage:.1f}%)")
    
    return {
        'original_size': original_size,
        'huffman_size': huffman_size_bytes,
        'tans_size': tans_size_bytes,
        'size_improvement': size_improvement,
        'huffman_encode_time': huffman_encode_time,
        'tans_encode_time': tans_encode_time,
        'huffman_decode_time': huffman_decode_time,
        'tans_decode_time': tans_decode_time,
        'huffman_lossless': huffman_is_lossless,
        'tans_lossless': tans_is_lossless,
        'huffman_header_size': huffman_total_header_bytes,
        'tans_header_size': tans_total_header_bytes,
        'huffman_payload_size': huffman_payload_bytes,
        'tans_payload_size': tans_payload_bytes
    }


if __name__ == "__main__":
    # Provide a CLI interface for tANS vs Huffman comparison
    parser = argparse.ArgumentParser(description="LZ77 with tANS entropy coding - Compare with Huffman")
    parser.add_argument("-i", "--input", help="input file", required=True, type=str)
    parser.add_argument(
        "-w", "--window_init", help="initialize window from file (like zstd dictionary)", type=str
    )
    parser.add_argument(
        "-t", "--table_log", help="tANS table log parameter (default: 10)", type=int, default=10
    )
    parser.add_argument(
        "--table_sweep", help="compare different table_log values (8,10,12,14)", action="store_true"
    )

    args = parser.parse_args()

    initial_window = None
    if args.window_init is not None:
        with open(args.window_init, "rb") as f:
            initial_window = list(f.read())

    if args.table_sweep:
        # Compare different table_log values against Huffman
        print(f"Comparing tANS (different table_log) vs Huffman for: {args.input}")
        
        # First get Huffman baseline
        print(f"\n{'='*50}")
        print("Getting Huffman baseline...")
        print(f"{'='*50}")
        
        with open(args.input, 'rb') as f:
            input_data = list(f.read())
        data_block = DataBlock(input_data)
        
        huffman_encoder = LZ77Encoder(initial_window=initial_window)
        huffman_encoded = huffman_encoder.encode_block(data_block)
        huffman_size = (len(huffman_encoded) + 7) // 8
        
        print(f"Huffman baseline: {huffman_size:,} bytes")
        
        # Test different table_log values
        results = []
        for table_log in [8, 10, 12, 14]:
            print(f"\nTesting tANS with table_log={table_log}...")
            try:
                result = compare_tans_vs_huffman(args.input, table_log, initial_window)
                result['table_log'] = table_log
                results.append(result)
            except Exception as e:
                print(f"Error with table_log={table_log}: {e}")
        
        # Summary comparison
        if results:
            print(f"\n{'='*120}")
            print(f"TABLE_LOG SWEEP SUMMARY")
            print(f"{'='*120}")
            print(f"{'table_log':<10} {'Total Size':<12} {'Header':<10} {'Payload':<10} {'Improvement':<12} {'Header Diff':<12} {'Enc Ratio':<10} {'Dec Ratio':<10}")
            print(f"{'-'*115}")
            for result in results:
                improvement = (result['huffman_size'] - result['tans_size']) / result['huffman_size'] * 100
                header_diff = result['tans_header_size'] - result['huffman_header_size']
                enc_ratio = result['huffman_encode_time'] / result['tans_encode_time']
                dec_ratio = result['huffman_decode_time'] / result['tans_decode_time']
                print(f"{result['table_log']:<10} {result['tans_size']:<12,} {result['tans_header_size']:<10,} "
                      f"{result['tans_payload_size']:<10,} {improvement:<12.2f}% {header_diff:+<12,} "
                      f"{enc_ratio:<10.2f}x {dec_ratio:<10.2f}x")
    
    else:
        # Single comparison
        compare_tans_vs_huffman(args.input, args.table_log, initial_window)
