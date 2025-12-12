#!/usr/bin/env python3
"""
Unit tests for tans_lz77_coder.py

Tests cover:
1. TANSEncoder/TANSDecoder basic functionality
2. Frequency table building and normalization
3. Symbol encoding/decoding correctness
4. LZ77TANSStreamsEncoder/Decoder integration
5. Edge cases (empty data, single symbols, etc.)
6. Round-trip lossless compression
"""

import unittest
import random
from collections import Counter

from scl.core.data_block import DataBlock
from lz77 import LZ77Sequence
from scl.utils.bitarray_utils import BitArray
from tans_lz77_coder import (
    TANSEncoder,
    TANSDecoder,
    LZ77TANSStreamsEncoder,
    LZ77TANSStreamsDecoder,
)


class TestTANSEncoder(unittest.TestCase):
    """Test TANSEncoder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = TANSEncoder(table_log=10)
        self.decoder = TANSDecoder(table_log=10)
    
    def test_build_table_basic(self):
        """Test basic table building functionality."""
        freqs = {'A': 3, 'B': 2, 'C': 1}
        table, symbol_info = self.encoder.build_table(freqs)
        
        # Check table size
        self.assertEqual(len(table), 1024)  # 2^10
        
        # Check symbol_info structure
        self.assertIn('A', symbol_info)
        self.assertIn('B', symbol_info)
        self.assertIn('C', symbol_info)
        
        # Check frequency allocation
        total_allocated = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total_allocated, 1024)
        
        # Check proportional allocation (A should get most states)
        self.assertGreater(symbol_info['A']['freq'], symbol_info['B']['freq'])
        self.assertGreater(symbol_info['B']['freq'], symbol_info['C']['freq'])
    
    def test_build_table_empty(self):
        """Test table building with empty frequencies."""
        table, symbol_info = self.encoder.build_table({})
        self.assertIsNone(table)
        self.assertIsNone(symbol_info)
    
    def test_build_table_single_symbol(self):
        """Test table building with single symbol."""
        freqs = {'X': 100}
        table, symbol_info = self.encoder.build_table(freqs)
        
        self.assertEqual(len(symbol_info), 1)
        self.assertEqual(symbol_info['X']['freq'], 1024)
        self.assertEqual(symbol_info['X']['start'], 0)
    
    def test_frequency_normalization(self):
        """Test that frequencies are properly normalized to table_size."""
        freqs = {'A': 1000, 'B': 2000, 'C': 3000}  # Large numbers
        table, symbol_info = self.encoder.build_table(freqs)
        
        total_allocated = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total_allocated, 1024)
        
        # Check that each symbol gets at least 1 state
        for info in symbol_info.values():
            self.assertGreaterEqual(info['freq'], 1)


class TestTANSDecoder(unittest.TestCase):
    """Test TANSDecoder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = TANSEncoder(table_log=8)  # Smaller for faster tests
        self.decoder = TANSDecoder(table_log=8)
    
    def test_decode_build_table(self):
        """Test that decoder can build the same table as encoder."""
        freqs = {'A': 10, 'B': 20, 'C': 5}
        
        enc_table, enc_symbol_info = self.encoder.build_table(freqs)
        self.decoder.build_table(freqs)
        
        # Tables should be identical
        self.assertEqual(enc_table, self.decoder.table)
        self.assertEqual(enc_symbol_info, self.decoder.symbol_info)


class TestTANSRoundTrip(unittest.TestCase):
    """Test round-trip encoding/decoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.table_log = 8  # Smaller for faster tests
        self.encoder = TANSEncoder(table_log=self.table_log)
        self.decoder = TANSDecoder(table_log=self.table_log)
    
    def test_simple_symbols(self):
        """Test encoding/decoding simple symbol sequences."""
        symbols = ['A', 'B', 'A', 'C', 'B', 'A']
        
        # Encode
        encoded = self.encoder.encode(symbols)
        self.assertIsInstance(encoded, BitArray)
        self.assertGreater(len(encoded), 0)
        
        # Decode
        freqs = Counter(symbols)
        decoded, bits_consumed = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_empty_sequence(self):
        """Test encoding/decoding empty sequences."""
        symbols = []
        
        encoded = self.encoder.encode(symbols)
        self.assertEqual(len(encoded), 0)
        
        decoded, bits_consumed = self.decoder.decode(encoded, 0, {})
        self.assertEqual(decoded, [])
        self.assertEqual(bits_consumed, 0)
    
    def test_single_symbol_repeated(self):
        """Test encoding/decoding repeated single symbol."""
        symbols = ['X'] * 100
        
        encoded = self.encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_integer_symbols(self):
        """Test encoding/decoding integer symbols (like bytes)."""
        symbols = [0, 1, 2, 0, 1, 255, 128, 0]
        
        encoded = self.encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_random_data(self):
        """Test with random data to catch edge cases."""
        random.seed(42)  # Reproducible
        
        for _ in range(10):  # Multiple random tests
            # Generate random symbols
            alphabet = list(range(10))  # 0-9
            length = random.randint(1, 100)
            symbols = [random.choice(alphabet) for _ in range(length)]
            
            # Use fresh encoder/decoder instances for each test to avoid state pollution
            encoder = TANSEncoder(table_log=self.table_log)
            decoder = TANSDecoder(table_log=self.table_log)
            
            encoded = encoder.encode(symbols)
            freqs = Counter(symbols)
            decoded, _ = decoder.decode(encoded, len(symbols), freqs)
            
            self.assertEqual(decoded, symbols, f"Failed on symbols: {symbols}")
    
    def test_different_table_logs(self):
        """Test with different table_log values."""
        symbols = ['A', 'B', 'C'] * 20
        
        for table_log in [6, 8, 10, 12]:
            with self.subTest(table_log=table_log):
                encoder = TANSEncoder(table_log=table_log)
                decoder = TANSDecoder(table_log=table_log)
                
                encoded = encoder.encode(symbols)
                freqs = Counter(symbols)
                decoded, _ = decoder.decode(encoded, len(symbols), freqs)
                
                self.assertEqual(decoded, symbols)


class TestLZ77TANSStreamsEncoder(unittest.TestCase):
    """Test LZ77TANSStreamsEncoder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = LZ77TANSStreamsEncoder(table_log=8)
        self.decoder = LZ77TANSStreamsDecoder(table_log=8)
    
    def test_empty_data(self):
        """Test encoding/decoding empty LZ77 data."""
        sequences = []
        literals = []
        
        encoded = self.encoder.encode_block(sequences, literals)
        decoded, bits_consumed = self.decoder.decode_block(encoded)
        
        decoded_sequences, decoded_literals = decoded
        self.assertEqual(decoded_sequences, sequences)
        self.assertEqual(list(decoded_literals), literals)
    
    def test_only_literals(self):
        """Test encoding/decoding only literals (no sequences)."""
        sequences = []
        literals = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        
        encoded = self.encoder.encode_block(sequences, literals)
        decoded, bits_consumed = self.decoder.decode_block(encoded)
        
        decoded_sequences, decoded_literals = decoded
        self.assertEqual(decoded_sequences, sequences)
        self.assertEqual(list(decoded_literals), literals)
    
    def test_only_sequences(self):
        """Test encoding/decoding only sequences (no literals)."""
        sequences = [
            LZ77Sequence(literal_count=0, match_length=5, match_offset=10),
            LZ77Sequence(literal_count=2, match_length=3, match_offset=7),
        ]
        literals = []
        
        encoded = self.encoder.encode_block(sequences, literals)
        decoded, bits_consumed = self.decoder.decode_block(encoded)
        
        decoded_sequences, decoded_literals = decoded
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig.literal_count, dec.literal_count)
            self.assertEqual(orig.match_length, dec.match_length)
            self.assertEqual(orig.match_offset, dec.match_offset)
        self.assertEqual(list(decoded_literals), literals)
    
    def test_mixed_data(self):
        """Test encoding/decoding mixed sequences and literals."""
        sequences = [
            LZ77Sequence(literal_count=3, match_length=4, match_offset=8),
            LZ77Sequence(literal_count=0, match_length=2, match_offset=5),
        ]
        literals = [65, 66, 67, 68, 69]  # "ABCDE"
        
        encoded = self.encoder.encode_block(sequences, literals)
        decoded, bits_consumed = self.decoder.decode_block(encoded)
        
        decoded_sequences, decoded_literals = decoded
        
        # Check sequences
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig.literal_count, dec.literal_count)
            self.assertEqual(orig.match_length, dec.match_length)
            self.assertEqual(orig.match_offset, dec.match_offset)
        
        # Check literals
        self.assertEqual(list(decoded_literals), literals)
    
    def test_large_values(self):
        """Test with large values in sequences."""
        sequences = [
            LZ77Sequence(literal_count=1000, match_length=500, match_offset=2000),
            LZ77Sequence(literal_count=0, match_length=10000, match_offset=5000),
        ]
        literals = list(range(256))  # All possible byte values
        
        encoded = self.encoder.encode_block(sequences, literals)
        decoded, bits_consumed = self.decoder.decode_block(encoded)
        
        decoded_sequences, decoded_literals = decoded
        
        # Verify sequences
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig.literal_count, dec.literal_count)
            self.assertEqual(orig.match_length, dec.match_length)
            self.assertEqual(orig.match_offset, dec.match_offset)
        
        # Verify literals
        self.assertEqual(list(decoded_literals), literals)


class TestTANSEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_invalid_table_log(self):
        """Test that invalid table_log values are handled."""
        # These should work
        TANSEncoder(table_log=6)
        TANSEncoder(table_log=16)
        
        # These might raise errors or warnings (implementation dependent)
        # Just ensure they don't crash
        try:
            TANSEncoder(table_log=3)  # Very small
            TANSEncoder(table_log=20)  # Very large
        except (ValueError, AssertionError):
            pass  # Expected for some implementations
    
    def test_symbol_not_in_table(self):
        """Test encoding symbol not in frequency table."""
        encoder = TANSEncoder(table_log=8)
        encoder.build_table({'A': 10, 'B': 5})
        
        # This should handle gracefully (skip or error)
        symbols = ['A', 'B', 'C']  # 'C' not in table
        encoded = encoder.encode(symbols, rebuild_table=False)  # Use existing table
        
        # Should encode A and B, skip C
        decoder = TANSDecoder(table_log=8)
        freqs = {'A': 10, 'B': 5}
        decoded, _ = decoder.decode(encoded, 2, freqs)  # Expect 2 symbols
        
        self.assertEqual(decoded, ['A', 'B'])
    
    def test_frequency_edge_cases(self):
        """Test edge cases in frequency handling."""
        encoder = TANSEncoder(table_log=6)  # Small table for edge cases
        
        # Very unbalanced frequencies
        freqs = {'A': 1, 'B': 1000000}
        table, symbol_info = encoder.build_table(freqs)
        
        # Both symbols should get at least 1 state
        self.assertGreaterEqual(symbol_info['A']['freq'], 1)
        self.assertGreaterEqual(symbol_info['B']['freq'], 1)
        
        # Total should equal table size
        total = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total, 64)  # 2^6


class TestTANSPerformance(unittest.TestCase):
    """Test performance-related aspects (not timing, but correctness under load)."""
    
    def test_large_alphabet(self):
        """Test with large alphabet (all 256 byte values)."""
        symbols = list(range(256)) * 4  # Each byte appears 4 times
        random.shuffle(symbols)
        
        encoder = TANSEncoder(table_log=10)
        decoder = TANSDecoder(table_log=10)
        
        encoded = encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_long_sequence(self):
        """Test with long sequences."""
        random.seed(123)
        symbols = [random.randint(0, 9) for _ in range(10000)]
        
        encoder = TANSEncoder(table_log=12)
        decoder = TANSDecoder(table_log=12)
        
        encoded = encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
