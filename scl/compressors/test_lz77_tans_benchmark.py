#!/usr/bin/env python3
import unittest
import tempfile
import os
from collections import Counter

from scl.core.data_block import DataBlock
from scl.utils.bitarray_utils import BitArray
try:
    from scl.compressors.lz77 import LZ77Encoder, LZ77Sequence
    from scl.compressors.lz77_tans_benchmark import (
        _build_literal_counts_list,
        _encode_literal_counts_header_from_counts,
        _decode_literal_counts_header,
        compute_literal_header_bits_empirical,
        compute_literal_header_bits_tans,
        LZ77StreamsEncoderTANSLiterals,
        LZ77StreamsDecoderTANSLiterals,
        LZ77EncoderTANSLiterals,
        LZ77DecoderTANSLiterals,
    )
except ImportError:
    from lz77 import LZ77Encoder, LZ77Sequence
    from lz77_tans_benchmark import (
        _build_literal_counts_list,
        _encode_literal_counts_header_from_counts,
        _decode_literal_counts_header,
        compute_literal_header_bits_empirical,
        compute_literal_header_bits_tans,
        LZ77StreamsEncoderTANSLiterals,
        LZ77StreamsDecoderTANSLiterals,
        LZ77EncoderTANSLiterals,
        LZ77DecoderTANSLiterals,
    )


class TestHeaderFunctions(unittest.TestCase):
    def test_build_literal_counts_list_empty(self):
        counts = _build_literal_counts_list([])
        self.assertEqual(len(counts), 256)
        self.assertEqual(sum(counts), 0)
        self.assertTrue(all(c == 0 for c in counts))
    
    def test_build_literal_counts_list_basic(self):
        literals = [65, 66, 65, 67, 66, 65]  # A, B, A, C, B, A
        counts = _build_literal_counts_list(literals)
        
        self.assertEqual(len(counts), 256)
        self.assertEqual(counts[65], 3)  # 'A' appears 3 times
        self.assertEqual(counts[66], 2)  # 'B' appears 2 times
        self.assertEqual(counts[67], 1)  # 'C' appears 1 time
        self.assertEqual(sum(counts), 6)
    
    def test_build_literal_counts_list_all_bytes(self):
        literals = list(range(256))
        counts = _build_literal_counts_list(literals)
        
        self.assertEqual(len(counts), 256)
        self.assertTrue(all(c == 1 for c in counts))
        self.assertEqual(sum(counts), 256)
    
    def test_encode_decode_literal_counts_header_empty(self):
        counts = [0] * 256
        
        encoded = _encode_literal_counts_header_from_counts(counts)
        self.assertIsInstance(encoded, BitArray)
        
        freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
        
        self.assertEqual(freqs, {})
        self.assertEqual(num_literals, 0)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literal_counts_header_basic(self):
        counts = [0] * 256
        counts[65] = 3  # A
        counts[66] = 2  # B
        counts[67] = 1  # C
        
        encoded = _encode_literal_counts_header_from_counts(counts)
        
        freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
        
        expected_freqs = {65: 3, 66: 2, 67: 1}
        self.assertEqual(freqs, expected_freqs)
        self.assertEqual(num_literals, 6)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literal_counts_header_round_trip(self):
        test_cases = [
            [],  # Empty
            [65] * 10,  # Single repeated byte
            [65, 66, 67],  # Multiple different bytes
            list(range(256)),  # All bytes once
        ]
        
        for literals in test_cases:
            with self.subTest(literals=literals[:10]):
                counts = _build_literal_counts_list(literals)
                encoded = _encode_literal_counts_header_from_counts(counts)
                freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
                
                expected_freqs = {i: c for i, c in enumerate(counts) if c > 0}
                self.assertEqual(freqs, expected_freqs)
                self.assertEqual(num_literals, len(literals))
                self.assertEqual(bits_consumed, len(encoded))


class TestHeaderComputationFunctions(unittest.TestCase):
    def test_compute_literal_header_bits_empirical_empty(self):
        bits = compute_literal_header_bits_empirical([])
        self.assertGreater(bits, 0)  # Should have at least the size header
        self.assertEqual(bits % 8, 0)  # Should be byte-aligned
    
    def test_compute_literal_header_bits_empirical_basic(self):
        literals = [65, 66, 65, 67]  # A, B, A, C
        bits = compute_literal_header_bits_empirical(literals)
        
        self.assertGreater(bits, 32)  # More than just size header
        self.assertIsInstance(bits, int)
    
    def test_compute_literal_header_bits_tans_empty(self):
        bits = compute_literal_header_bits_tans([], table_log=10)
        self.assertGreater(bits, 0)
        self.assertEqual(bits % 8, 0)  # Should be byte-aligned
    
    def test_compute_literal_header_bits_tans_basic(self):
        literals = [65, 66, 65, 67]  # A, B, A, C
        bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        self.assertGreater(bits, 32)  # More than just size header
        self.assertIsInstance(bits, int)
    
    def test_header_computation_consistency(self):
        literals = [65, 66, 65, 67, 66, 65]  # A, B, A, C, B, A
        
        emp_bits = compute_literal_header_bits_empirical(literals)
        tans_bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        counts = _build_literal_counts_list(literals)
        actual_header = _encode_literal_counts_header_from_counts(counts)
        
        self.assertEqual(emp_bits, len(actual_header))
        self.assertEqual(tans_bits, len(actual_header))
    
    def test_header_computation_different_table_logs(self):
        literals = [65, 66, 67] * 10
        
        bits_8 = compute_literal_header_bits_tans(literals, table_log=8)
        bits_10 = compute_literal_header_bits_tans(literals, table_log=10)
        bits_12 = compute_literal_header_bits_tans(literals, table_log=12)
        
        self.assertEqual(bits_8, bits_10)
        self.assertEqual(bits_10, bits_12)


class TestLZ77StreamsEncoderTANSLiterals(unittest.TestCase):
    def setUp(self):
        self.encoder = LZ77StreamsEncoderTANSLiterals(table_log=8)
        self.decoder = LZ77StreamsDecoderTANSLiterals(table_log=8)
    
    def test_encode_decode_literals_empty(self):
        literals = []
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_basic(self):
        literals = [72, 101, 108, 108, 111]  # "Hello"
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_all_bytes(self):
        literals = list(range(256))
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_repeated(self):
        literals = [65] * 100  # 'A' repeated 100 times
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_full_lz77_streams_empty(self):
        sequences = []
        literals = []
        
        encoded = self.encoder.encode_block(sequences, literals)
        (decoded_sequences, decoded_literals), bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_sequences, sequences)
        self.assertEqual(decoded_literals, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_full_lz77_streams_basic(self):
        sequences = [
            LZ77Sequence(literal_count=2, match_length=3, match_offset=5),
            LZ77Sequence(literal_count=0, match_length=4, match_offset=8),
        ]
        literals = [65, 66, 67, 68]  # A, B, C, D
        
        encoded = self.encoder.encode_block(sequences, literals)
        (decoded_sequences, decoded_literals), bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig.literal_count, dec.literal_count)
            self.assertEqual(orig.match_length, dec.match_length)
            self.assertEqual(orig.match_offset, dec.match_offset)
        
        self.assertEqual(decoded_literals, literals)
        self.assertEqual(bits_consumed, len(encoded))


class TestLZ77EncoderTANSLiterals(unittest.TestCase):
    def setUp(self):
        self.encoder = LZ77EncoderTANSLiterals(table_log=8)
        self.decoder = LZ77DecoderTANSLiterals(table_log=8)
    
    def test_simple_data(self):
        data = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]  # "Hello World"
        data_block = DataBlock(data)
        
        encoded = self.encoder.encode_block(data_block)
        
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_repeated_data(self):
        data = [65, 66, 67] * 20  # "ABC" repeated 20 times
        data_block = DataBlock(data)
        
        encoded = self.encoder.encode_block(data_block)
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_random_data(self):
        import random
        random.seed(42)
        data = [random.randint(0, 255) for _ in range(100)]
        data_block = DataBlock(data)
        
        encoded = self.encoder.encode_block(data_block)
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_with_initial_window(self):
        initial_window = [65, 66, 67] * 5  # "ABC" repeated 5 times
        data = [65, 66, 67, 68, 69]  # "ABCDE"
        
        encoder = LZ77EncoderTANSLiterals(initial_window=initial_window, table_log=8)
        decoder = LZ77DecoderTANSLiterals(initial_window=initial_window, table_log=8)
        
        data_block = DataBlock(data)
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)


class TestBenchmarkIntegration(unittest.TestCase):
    def test_lz77_parsing_consistency(self):
        data = [65, 66, 67] * 10 + [68, 69, 70] * 5  # Some repeated patterns
        data_block = DataBlock(data)
        
        baseline_encoder = LZ77Encoder()
        baseline_sequences, baseline_literals = baseline_encoder.lz77_parse_and_generate_sequences(data_block)
        
        tans_encoder = LZ77EncoderTANSLiterals(table_log=8)
        tans_sequences, tans_literals = tans_encoder.lz77_parse_and_generate_sequences(data_block)
        
        self.assertEqual(len(baseline_sequences), len(tans_sequences))
        for base_seq, tans_seq in zip(baseline_sequences, tans_sequences):
            self.assertEqual(base_seq.literal_count, tans_seq.literal_count)
            self.assertEqual(base_seq.match_length, tans_seq.match_length)
            self.assertEqual(base_seq.match_offset, tans_seq.match_offset)
        
        self.assertEqual(baseline_literals, tans_literals)
    
    def test_header_size_computation_accuracy(self):
        data = list(range(50)) * 3  # Some variety in the data
        data_block = DataBlock(data)
        
        encoder = LZ77Encoder()
        sequences, literals = encoder.lz77_parse_and_generate_sequences(data_block)
        
        emp_header_bits = compute_literal_header_bits_empirical(literals)
        tans_header_bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        tans_encoder = LZ77StreamsEncoderTANSLiterals(table_log=10)
        encoded_literals = tans_encoder.encode_literals(literals)
        
        self.assertEqual(emp_header_bits, tans_header_bits)
        
        self.assertGreater(emp_header_bits, 0)
        self.assertLess(emp_header_bits, len(encoded_literals))


class TestEdgeCases(unittest.TestCase):
    def test_very_small_data(self):
        data = [65]  # Single 'A'
        data_block = DataBlock(data)
        
        encoder = LZ77EncoderTANSLiterals(table_log=8)
        decoder = LZ77DecoderTANSLiterals(table_log=8)
        
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
    
    def test_all_same_byte(self):
        data = [42] * 1000  # All the same byte
        data_block = DataBlock(data)
        
        encoder = LZ77EncoderTANSLiterals(table_log=8)
        decoder = LZ77DecoderTANSLiterals(table_log=8)
        
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
    
    def test_different_table_logs(self):
        data = [65, 66, 67] * 20
        data_block = DataBlock(data)
        
        for table_log in [6, 8, 10, 12]:
            with self.subTest(table_log=table_log):
                encoder = LZ77EncoderTANSLiterals(table_log=table_log)
                decoder = LZ77DecoderTANSLiterals(table_log=table_log)
                
                encoded = encoder.encode_block(data_block)
                decoded_block, _ = decoder.decode_block(encoded)
                
                self.assertEqual(decoded_block.data_list, data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
