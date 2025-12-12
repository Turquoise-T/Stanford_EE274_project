from dataclasses import dataclass
import numpy as np
from typing import Tuple
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import (
    BitArray,
    get_bit_width,
    uint_to_bitarray,
    bitarray_to_uint,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, get_avg_neg_log_prob
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.utils.misc_utils import is_power_of_two
from scl.compressors.rANS import rANSParams, rANSEncoder, rANSDecoder


@dataclass
class tANSParams(rANSParams):
    def __post_init__(self):
        super().__post_init__()
        assert is_power_of_two(
            self.M
        ), "Please normalize self.M parameter (sum of frequencies) to be a power of two"

        assert self.NUM_BITS_OUT == 1, "only NUM_OUT_BITS = 1 supported for now"

        if self.RANGE_FACTOR > (1 << 16):
            print("WARNING: RANGE_FACTOR > 2^16 --> the lookup tables could be huge")


class tANSEncoder(DataEncoder):
    def __init__(self, tans_params: tANSParams):
        self.params = tans_params
        self.build_base_encode_step_table()
        self.build_shrink_num_out_bits_lookup_table()

    def shrink_state_num_out_bits_base(self, s):
        y = get_bit_width(self.params.max_shrunk_state[s])
        num_out_bits_base = self.params.NUM_STATE_BITS - y

        thresh_state = (self.params.max_shrunk_state[s] + 1) << num_out_bits_base
        return num_out_bits_base, thresh_state

    def build_base_encode_step_table(self):
        rans_encoder = rANSEncoder(self.params)
        self.base_encode_step_table = {}
        for s in self.params.freqs.alphabet:
            _min, _max = self.params.min_shrunk_state[s], self.params.max_shrunk_state[s]
            for x_shrunk in range(_min, _max + 1):
                self.base_encode_step_table[(s, x_shrunk)] = rans_encoder.rans_base_encode_step(
                    s, x_shrunk
                )

    def build_shrink_num_out_bits_lookup_table(self):
        self.shrink_state_num_out_bits_base_table = {}
        self.shrink_state_thresh_table = {}
        for s in self.params.freqs.alphabet:
            num_bits, thresh = self.shrink_state_num_out_bits_base(s)
            self.shrink_state_num_out_bits_base_table[s] = num_bits
            self.shrink_state_thresh_table[s] = thresh

    def encode_symbol(self, s, state: int) -> Tuple[int, BitArray]:
        symbol_bitarray = BitArray("")

        # Emit bits so the cached base step sees a valid shrunk state.
        num_out_bits = self.shrink_state_num_out_bits_base_table[s]
        if state >= self.shrink_state_thresh_table[s]:
            num_out_bits += 1

        out_bits = uint_to_bitarray(state)[-num_out_bits:] if num_out_bits else BitArray("")
        state = state >> num_out_bits

        symbol_bitarray = out_bits + symbol_bitarray

        state = self.base_encode_step_table[(s, state)]
        return state, symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        encoded_bitarray = BitArray("")

        state = self.params.INITIAL_STATE

        for s in data_block.data_list:
            state, symbol_bitarray = self.encode_symbol(s, state)
            encoded_bitarray = symbol_bitarray + encoded_bitarray

        encoded_bitarray = uint_to_bitarray(state, self.params.NUM_STATE_BITS) + encoded_bitarray

        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )

        return encoded_bitarray


class tANSDecoder(DataDecoder):
    def __init__(self, tans_params: tANSParams):
        self.params = tans_params

        self.build_rans_base_decode_table()
        self.build_expand_state_num_bits_table()

    def build_rans_base_decode_table(self):
        rans_decoder = rANSDecoder(self.params)
        self.base_decode_step_table = {}
        for state in range(self.params.L, self.params.H + 1):
            self.base_decode_step_table[state] = rans_decoder.rans_base_decode_step(state)

    def build_expand_state_num_bits_table(self):
        self.expand_state_num_bits_table = {}
        for s in self.params.freqs.alphabet:
            _min, _max = self.params.min_shrunk_state[s], self.params.max_shrunk_state[s]
            for x_shrunk in range(_min, _max + 1):
                num_bits = self.params.NUM_STATE_BITS - get_bit_width(x_shrunk)
                self.expand_state_num_bits_table[x_shrunk] = num_bits

    def decode_symbol(self, state: int, encoded_bitarray: BitArray):
        s, state_shrunk = self.base_decode_step_table[state]

        num_bits = self.expand_state_num_bits_table[state_shrunk]
        state_remainder = 0
        if num_bits:
            state_remainder = bitarray_to_uint(encoded_bitarray[:num_bits])
        state = (state_shrunk << num_bits) + state_remainder

        return s, state, num_bits

    def decode_block(self, encoded_bitarray: BitArray):
        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)
        num_bits_consumed = self.params.DATA_BLOCK_SIZE_BITS

        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + self.params.NUM_STATE_BITS]
        )
        num_bits_consumed += self.params.NUM_STATE_BITS

        decoded_data_list = []
        for _ in range(input_data_block_size):
            s, state, num_symbol_bits = self.decode_symbol(
                state, encoded_bitarray[num_bits_consumed:]
            )

            # rANS/tANS decode runs backwards.
            decoded_data_list = [s] + decoded_data_list
            num_bits_consumed += num_symbol_bits

        assert state == self.params.INITIAL_STATE

        return DataBlock(decoded_data_list), num_bits_consumed


def test_generated_lookup_tables():
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = tANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    expected_base_encode_step_table = {
        ("A", 3): 8,
        ("A", 4): 9,
        ("A", 5): 10,
        ("B", 3): 11,
        ("B", 4): 12,
        ("B", 5): 13,
        ("C", 2): 14,
        ("C", 3): 15,
    }
    expected_shrink_state_num_out_bits_base_table = {"A": 1, "B": 1, "C": 2}
    expected_shrink_state_thresh_table = {"A": 12, "B": 12, "C": 16}

    encoder = tANSEncoder(params)
    assert expected_base_encode_step_table == encoder.base_encode_step_table
    assert (
        expected_shrink_state_num_out_bits_base_table
        == encoder.shrink_state_num_out_bits_base_table
    )
    assert expected_shrink_state_thresh_table == encoder.shrink_state_thresh_table

    expected_base_decode_step_table = {
        8: ("A", 3),
        9: ("A", 4),
        10: ("A", 5),
        11: ("B", 3),
        12: ("B", 4),
        13: ("B", 5),
        14: ("C", 2),
        15: ("C", 3),
    }
    expected_expand_state_num_bits = {2: 2, 3: 2, 4: 1, 5: 1}

    decoder = tANSDecoder(params)
    assert expected_base_decode_step_table == decoder.base_decode_step_table
    assert expected_expand_state_num_bits == decoder.expand_state_num_bits_table


def test_check_encoded_bitarray():
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = tANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    encoder = tANSEncoder(params)

    expected_encoded_bitarray = BitArray("")

    x = 8
    assert params.INITIAL_STATE == 8

    x = 4
    expected_encoded_bitarray = BitArray("0") + expected_encoded_bitarray

    x = 9

    x = 2
    expected_encoded_bitarray = BitArray("01") + expected_encoded_bitarray

    x = 14

    x = 3
    expected_encoded_bitarray = BitArray("10") + expected_encoded_bitarray

    x = 11

    num_state_bits = 4
    assert params.NUM_STATE_BITS == num_state_bits
    expected_encoded_bitarray = BitArray("1011") + expected_encoded_bitarray

    expected_encoded_bitarray = BitArray("00011") + expected_encoded_bitarray

    encoded_bitarray = encoder.encode_block(data)
    assert expected_encoded_bitarray == encoded_bitarray


def test_tANS_coding():
    freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 3}),
        Frequencies({"A": 3, "B": 4, "C": 9}),
    ]
    params_list = [
        tANSParams(freqs_list[0], RANGE_FACTOR=1),
        tANSParams(freqs_list[1], RANGE_FACTOR=1 << 4),
        tANSParams(freqs_list[2]),
    ]

    DATA_SIZE = 10000
    SEED = 0
    for freq, tans_params in zip(freqs_list, params_list):
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)

        encoder = tANSEncoder(tans_params)
        decoder = tANSDecoder(tans_params)

        is_lossless, encode_len, _ = try_lossless_compression(
            data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
        )
        assert is_lossless
        avg_codelen = encode_len / data_block.size
        print(f"tANS coding: avg_log_prob={avg_log_prob:.3f}, tANS codelen: {avg_codelen:.3f}")
