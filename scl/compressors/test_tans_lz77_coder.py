#!/usr/bin/env python3
import unittest
import random
from collections import Counter

from scl.utils.bitarray_utils import BitArray
try:
    from scl.compressors.tans_lz77_coder import TANSEncoder, TANSDecoder
except ImportError:
    from tans_lz77_coder import TANSEncoder, TANSDecoder


class TestTANSEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TANSEncoder(table_log=10)
        self.decoder = TANSDecoder(table_log=10)
    
    def test_build_table_basic(self):
        freqs = {'A': 3, 'B': 2, 'C': 1}
        table, symbol_info = self.encoder.build_table(freqs)
        self.assertEqual(len(table), 1024)  # 2^10
        self.assertIn('A', symbol_info)
        self.assertIn('B', symbol_info)
        self.assertIn('C', symbol_info)
        total_allocated = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total_allocated, 1024)
        self.assertGreater(symbol_info['A']['freq'], symbol_info['B']['freq'])
        self.assertGreater(symbol_info['B']['freq'], symbol_info['C']['freq'])
    
    def test_build_table_empty(self):
        table, symbol_info = self.encoder.build_table({})
        self.assertIsNone(table)
        self.assertIsNone(symbol_info)
    
    def test_build_table_single_symbol(self):
        freqs = {'X': 100}
        table, symbol_info = self.encoder.build_table(freqs)
        
        self.assertEqual(len(symbol_info), 1)
        self.assertEqual(symbol_info['X']['freq'], 1024)
        self.assertEqual(symbol_info['X']['start'], 0)
    
    def test_frequency_normalization(self):
        freqs = {'A': 1000, 'B': 2000, 'C': 3000}  # Large numbers
        table, symbol_info = self.encoder.build_table(freqs)
        
        total_allocated = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total_allocated, 1024)
        
        for info in symbol_info.values():
            self.assertGreaterEqual(info['freq'], 1)


class TestTANSDecoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TANSEncoder(table_log=8)  # Smaller for faster tests
        self.decoder = TANSDecoder(table_log=8)
    
    def test_decode_build_table(self):
        freqs = {'A': 10, 'B': 20, 'C': 5}
        
        enc_table, enc_symbol_info = self.encoder.build_table(freqs)
        self.decoder.build_table(freqs)
        
        self.assertEqual(enc_table, self.decoder.table)
        self.assertEqual(enc_symbol_info, self.decoder.symbol_info)


class TestTANSRoundTrip(unittest.TestCase):
    def setUp(self):
        self.table_log = 8  # Smaller for faster tests
        self.encoder = TANSEncoder(table_log=self.table_log)
        self.decoder = TANSDecoder(table_log=self.table_log)
    
    def test_simple_symbols(self):
        symbols = ['A', 'B', 'A', 'C', 'B', 'A']
        
        encoded = self.encoder.encode(symbols)
        self.assertIsInstance(encoded, BitArray)
        self.assertGreater(len(encoded), 0)
        
        freqs = Counter(symbols)
        decoded, bits_consumed = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
        self.assertEqual(bits_consumed, len(encoded))
    
    def test_empty_sequence(self):
        symbols = []
        
        encoded = self.encoder.encode(symbols)
        self.assertEqual(len(encoded), 0)
        
        decoded, bits_consumed = self.decoder.decode(encoded, 0, {})
        self.assertEqual(decoded, [])
        self.assertEqual(bits_consumed, 0)
    
    def test_single_symbol_repeated(self):
        symbols = ['X'] * 100
        
        encoded = self.encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_integer_symbols(self):
        symbols = [0, 1, 2, 0, 1, 255, 128, 0]
        
        encoded = self.encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = self.decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_random_data(self):
        random.seed(42)  # Reproducible
        
        for _ in range(10):  # Multiple random tests
            alphabet = list(range(10))  # 0-9
            length = random.randint(1, 100)
            symbols = [random.choice(alphabet) for _ in range(length)]
            
            encoder = TANSEncoder(table_log=self.table_log)
            decoder = TANSDecoder(table_log=self.table_log)
            
            encoded = encoder.encode(symbols)
            freqs = Counter(symbols)
            decoded, _ = decoder.decode(encoded, len(symbols), freqs)
            
            self.assertEqual(decoded, symbols, f"Failed on symbols: {symbols}")
    
    def test_different_table_logs(self):
        symbols = ['A', 'B', 'C'] * 20
        
        for table_log in [6, 8, 10, 12]:
            with self.subTest(table_log=table_log):
                encoder = TANSEncoder(table_log=table_log)
                decoder = TANSDecoder(table_log=table_log)
                
                encoded = encoder.encode(symbols)
                freqs = Counter(symbols)
                decoded, _ = decoder.decode(encoded, len(symbols), freqs)
                
                self.assertEqual(decoded, symbols)


class TestTANSEdgeCases(unittest.TestCase):
    def test_invalid_table_log(self):
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
        encoder = TANSEncoder(table_log=8)
        encoder.build_table({'A': 10, 'B': 5})
        
        symbols = ['A', 'B', 'C']  # 'C' not in table
        encoded = encoder.encode(symbols, rebuild_table=False)  # Use existing table
        
        decoder = TANSDecoder(table_log=8)
        freqs = {'A': 10, 'B': 5}
        decoded, _ = decoder.decode(encoded, 2, freqs)  # Expect 2 symbols
        
        self.assertEqual(decoded, ['A', 'B'])
    
    def test_frequency_edge_cases(self):
        encoder = TANSEncoder(table_log=6)  # Small table for edge cases
        
        freqs = {'A': 1, 'B': 1000000}
        table, symbol_info = encoder.build_table(freqs)
        
        self.assertGreaterEqual(symbol_info['A']['freq'], 1)
        self.assertGreaterEqual(symbol_info['B']['freq'], 1)
        
        total = sum(info['freq'] for info in symbol_info.values())
        self.assertEqual(total, 64)  # 2^6


class TestTANSPerformance(unittest.TestCase):
    def test_large_alphabet(self):
        symbols = list(range(256)) * 4  # Each byte appears 4 times
        random.shuffle(symbols)
        
        encoder = TANSEncoder(table_log=10)
        decoder = TANSDecoder(table_log=10)
        
        encoded = encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)
    
    def test_long_sequence(self):
        random.seed(123)
        symbols = [random.randint(0, 9) for _ in range(10000)]
        
        encoder = TANSEncoder(table_log=12)
        decoder = TANSDecoder(table_log=12)
        
        encoded = encoder.encode(symbols)
        freqs = Counter(symbols)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        
        self.assertEqual(decoded, symbols)


if __name__ == '__main__':
    unittest.main(verbosity=2)
