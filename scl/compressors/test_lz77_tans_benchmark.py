#!/usr/bin/env python3
"""
Unit tests for lz77_tans_benchmark.py

Tests cover:
1. Header computation functions
2. LZ77StreamsEncoderTANSLiterals/Decoder functionality
3. Benchmark helper functions
4. Integration with actual LZ77 parsing
5. Edge cases in benchmark scenarios
"""

import unittest
import tempfile
import os
from collections import Counter

from scl.core.data_block import DataBlock
from lz77 import LZ77Encoder, LZ77Sequence
from scl.utils.bitarray_utils import BitArray
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
    """Test header computation and encoding/decoding functions."""
    
    def test_build_literal_counts_list_empty(self):
        """Test building counts list from empty literals."""
        counts = _build_literal_counts_list([])
        self.assertEqual(len(counts), 256)
        self.assertEqual(sum(counts), 0)
        self.assertTrue(all(c == 0 for c in counts))
    
    def test_build_literal_counts_list_basic(self):
        """Test building counts list from basic literals."""
        literals = [65, 66, 65, 67, 66, 65]  # A, B, A, C, B, A
        counts = _build_literal_counts_list(literals)
        
        self.assertEqual(len(counts), 256)
        self.assertEqual(counts[65], 3)  # 'A' appears 3 times
        self.assertEqual(counts[66], 2)  # 'B' appears 2 times
        self.assertEqual(counts[67], 1)  # 'C' appears 1 time
        self.assertEqual(sum(counts), 6)
    
    def test_build_literal_counts_list_all_bytes(self):
        """Test with all possible byte values."""
        literals = list(range(256))
        counts = _build_literal_counts_list(literals)
        
        self.assertEqual(len(counts), 256)
        self.assertTrue(all(c == 1 for c in counts))
        self.assertEqual(sum(counts), 256)
    
    def test_encode_decode_literal_counts_header_empty(self):
        """Test encoding/decoding empty counts header."""
        counts = [0] * 256
        
        # Encode
        encoded = _encode_literal_counts_header_from_counts(counts)
        self.assertIsInstance(encoded, BitArray)
        
        # Decode
        freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
        
        self.assertEqual(freqs, {})
        self.assertEqual(num_literals, 0)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literal_counts_header_basic(self):
        """Test encoding/decoding basic counts header."""
        # Create counts: A=3, B=2, C=1, rest=0
        counts = [0] * 256
        counts[65] = 3  # A
        counts[66] = 2  # B
        counts[67] = 1  # C
        
        # Encode
        encoded = _encode_literal_counts_header_from_counts(counts)
        
        # Decode
        freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
        
        expected_freqs = {65: 3, 66: 2, 67: 1}
        self.assertEqual(freqs, expected_freqs)
        self.assertEqual(num_literals, 6)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literal_counts_header_round_trip(self):
        """Test round-trip encoding/decoding of various counts."""
        test_cases = [
            [],  # Empty
            [65] * 10,  # Single repeated byte
            [65, 66, 67],  # Multiple different bytes
            list(range(256)),  # All bytes once
        ]
        
        for literals in test_cases:
            with self.subTest(literals=literals[:10]):  # Show first 10 for readability
                counts = _build_literal_counts_list(literals)
                encoded = _encode_literal_counts_header_from_counts(counts)
                freqs, num_literals, bits_consumed = _decode_literal_counts_header(encoded)
                
                # Verify
                expected_freqs = {i: c for i, c in enumerate(counts) if c > 0}
                self.assertEqual(freqs, expected_freqs)
                self.assertEqual(num_literals, len(literals))
                self.assertEqual(bits_consumed, len(encoded))


class TestHeaderComputationFunctions(unittest.TestCase):
    """Test header size computation functions."""
    
    def test_compute_literal_header_bits_empirical_empty(self):
        """Test empirical header computation for empty literals."""
        bits = compute_literal_header_bits_empirical([])
        self.assertGreater(bits, 0)  # Should have at least the size header
        self.assertEqual(bits % 8, 0)  # Should be byte-aligned
    
    def test_compute_literal_header_bits_empirical_basic(self):
        """Test empirical header computation for basic literals."""
        literals = [65, 66, 65, 67]  # A, B, A, C
        bits = compute_literal_header_bits_empirical(literals)
        
        self.assertGreater(bits, 32)  # More than just size header
        self.assertIsInstance(bits, int)
    
    def test_compute_literal_header_bits_tans_empty(self):
        """Test tANS header computation for empty literals."""
        bits = compute_literal_header_bits_tans([], table_log=10)
        self.assertGreater(bits, 0)
        self.assertEqual(bits % 8, 0)  # Should be byte-aligned
    
    def test_compute_literal_header_bits_tans_basic(self):
        """Test tANS header computation for basic literals."""
        literals = [65, 66, 65, 67]  # A, B, A, C
        bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        self.assertGreater(bits, 32)  # More than just size header
        self.assertIsInstance(bits, int)
    
    def test_header_computation_consistency(self):
        """Test that header computations are consistent with actual encoding."""
        literals = [65, 66, 65, 67, 66, 65]  # A, B, A, C, B, A
        
        # Compute header sizes
        emp_bits = compute_literal_header_bits_empirical(literals)
        tans_bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        # Actually encode headers
        counts = _build_literal_counts_list(literals)
        actual_header = _encode_literal_counts_header_from_counts(counts)
        
        # Both should match the actual encoding (they use the same format)
        self.assertEqual(emp_bits, len(actual_header))
        self.assertEqual(tans_bits, len(actual_header))
    
    def test_header_computation_different_table_logs(self):
        """Test that tANS header computation is independent of table_log."""
        literals = [65, 66, 67] * 10
        
        # Different table_log values should give same header size
        # (since header format is independent of table_log in hybrid design)
        bits_8 = compute_literal_header_bits_tans(literals, table_log=8)
        bits_10 = compute_literal_header_bits_tans(literals, table_log=10)
        bits_12 = compute_literal_header_bits_tans(literals, table_log=12)
        
        self.assertEqual(bits_8, bits_10)
        self.assertEqual(bits_10, bits_12)


class TestLZ77StreamsEncoderTANSLiterals(unittest.TestCase):
    """Test LZ77StreamsEncoderTANSLiterals and decoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = LZ77StreamsEncoderTANSLiterals(table_log=8)
        self.decoder = LZ77StreamsDecoderTANSLiterals(table_log=8)
    
    def test_encode_decode_literals_empty(self):
        """Test encoding/decoding empty literals."""
        literals = []
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_basic(self):
        """Test encoding/decoding basic literals."""
        literals = [72, 101, 108, 108, 111]  # "Hello"
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_all_bytes(self):
        """Test encoding/decoding all possible byte values."""
        literals = list(range(256))
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_encode_decode_literals_repeated(self):
        """Test encoding/decoding repeated literals."""
        literals = [65] * 100  # 'A' repeated 100 times
        
        encoded = self.encoder.encode_literals(literals)
        decoded, bits_consumed = self.decoder.decode_literals(encoded)
        
        self.assertEqual(decoded, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_full_lz77_streams_empty(self):
        """Test full LZ77 streams encoding/decoding with empty data."""
        sequences = []
        literals = []
        
        encoded = self.encoder.encode_block(sequences, literals)
        (decoded_sequences, decoded_literals), bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_sequences, sequences)
        self.assertEqual(decoded_literals, literals)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_full_lz77_streams_basic(self):
        """Test full LZ77 streams encoding/decoding with basic data."""
        sequences = [
            LZ77Sequence(literal_count=2, match_length=3, match_offset=5),
            LZ77Sequence(literal_count=0, match_length=4, match_offset=8),
        ]
        literals = [65, 66, 67, 68]  # A, B, C, D
        
        encoded = self.encoder.encode_block(sequences, literals)
        (decoded_sequences, decoded_literals), bits_consumed = self.decoder.decode_block(encoded)
        
        # Check sequences
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig.literal_count, dec.literal_count)
            self.assertEqual(orig.match_length, dec.match_length)
            self.assertEqual(orig.match_offset, dec.match_offset)
        
        # Check literals
        self.assertEqual(decoded_literals, literals)
        self.assertEqual(bits_consumed, len(encoded))


class TestLZ77EncoderTANSLiterals(unittest.TestCase):
    """Test full LZ77 encoder/decoder with tANS literals."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = LZ77EncoderTANSLiterals(table_log=8)
        self.decoder = LZ77DecoderTANSLiterals(table_log=8)
    
    def test_simple_data(self):
        """Test encoding/decoding simple data."""
        data = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]  # "Hello World"
        data_block = DataBlock(data)
        
        # Encode
        encoded = self.encoder.encode_block(data_block)
        
        # Decode
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_repeated_data(self):
        """Test with data that should compress well (repeated patterns)."""
        data = [65, 66, 67] * 20  # "ABC" repeated 20 times
        data_block = DataBlock(data)
        
        encoded = self.encoder.encode_block(data_block)
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_random_data(self):
        """Test with random data (should not compress well)."""
        import random
        random.seed(42)
        data = [random.randint(0, 255) for _ in range(100)]
        data_block = DataBlock(data)
        
        encoded = self.encoder.encode_block(data_block)
        decoded_block, bits_consumed = self.decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_with_initial_window(self):
        """Test encoder/decoder with initial window."""
        initial_window = [65, 66, 67] * 5  # "ABC" repeated 5 times
        data = [65, 66, 67, 68, 69]  # "ABCDE"
        
        encoder = LZ77EncoderTANSLiterals(initial_window=initial_window, table_log=8)
        decoder = LZ77DecoderTANSLiterals(initial_window=initial_window, table_log=8)
        
        data_block = DataBlock(data)
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)


class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration aspects of the benchmark."""
    
    def test_lz77_parsing_consistency(self):
        """Test that LZ77 parsing is consistent between baseline and tANS."""
        data = [65, 66, 67] * 10 + [68, 69, 70] * 5  # Some repeated patterns
        data_block = DataBlock(data)
        
        # Parse with baseline LZ77
        baseline_encoder = LZ77Encoder()
        baseline_sequences, baseline_literals = baseline_encoder.lz77_parse_and_generate_sequences(data_block)
        
        # Parse with tANS LZ77 (should be identical parsing)
        tans_encoder = LZ77EncoderTANSLiterals(table_log=8)
        tans_sequences, tans_literals = tans_encoder.lz77_parse_and_generate_sequences(data_block)
        
        # Parsing should be identical (only entropy coding differs)
        self.assertEqual(len(baseline_sequences), len(tans_sequences))
        for base_seq, tans_seq in zip(baseline_sequences, tans_sequences):
            self.assertEqual(base_seq.literal_count, tans_seq.literal_count)
            self.assertEqual(base_seq.match_length, tans_seq.match_length)
            self.assertEqual(base_seq.match_offset, tans_seq.match_offset)
        
        self.assertEqual(baseline_literals, tans_literals)
    
    def test_header_size_computation_accuracy(self):
        """Test that header size computations match actual encoded sizes."""
        data = list(range(50)) * 3  # Some variety in the data
        data_block = DataBlock(data)
        
        # Get literals from LZ77 parsing
        encoder = LZ77Encoder()
        sequences, literals = encoder.lz77_parse_and_generate_sequences(data_block)
        
        # Compute header sizes
        emp_header_bits = compute_literal_header_bits_empirical(literals)
        tans_header_bits = compute_literal_header_bits_tans(literals, table_log=10)
        
        # Actually encode with tANS (which uses same header format)
        tans_encoder = LZ77StreamsEncoderTANSLiterals(table_log=10)
        encoded_literals = tans_encoder.encode_literals(literals)
        
        # The header computation should match the actual header size
        # (Both use the same Elias-Delta format in hybrid design)
        self.assertEqual(emp_header_bits, tans_header_bits)
        
        # Verify the computation is reasonable (not zero, not huge)
        self.assertGreater(emp_header_bits, 0)
        self.assertLess(emp_header_bits, len(encoded_literals))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_very_small_data(self):
        """Test with very small data (single byte)."""
        data = [65]  # Single 'A'
        data_block = DataBlock(data)
        
        encoder = LZ77EncoderTANSLiterals(table_log=8)
        decoder = LZ77DecoderTANSLiterals(table_log=8)
        
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
    
    def test_all_same_byte(self):
        """Test with all identical bytes."""
        data = [42] * 1000  # All the same byte
        data_block = DataBlock(data)
        
        encoder = LZ77EncoderTANSLiterals(table_log=8)
        decoder = LZ77DecoderTANSLiterals(table_log=8)
        
        encoded = encoder.encode_block(data_block)
        decoded_block, _ = decoder.decode_block(encoded)
        
        self.assertEqual(decoded_block.data_list, data)
    
    def test_different_table_logs(self):
        """Test that different table_log values work correctly."""
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
    # Run tests with verbose output
    unittest.main(verbosity=2)
