import unittest
from collections import Counter
import sys
import os

# Mock the dependencies if they're not available
try:
    from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
    from scl.compressors.lz77 import LZ77Sequence
except ImportError:
    # Create mock implementations for testing
    class BitArray:
        def __init__(self, data=None):
            if data is None:
                self.data = []
            elif isinstance(data, list):
                self.data = data
            else:
                self.data = list(data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return BitArray(self.data[key])
            return self.data[key]

        def __add__(self, other):
            result = BitArray()
            result.data = self.data + other.data
            return result

        def __eq__(self, other):
            return self.data == other.data

        def __repr__(self):
            return f"BitArray({self.data})"


    def uint_to_bitarray(value, num_bits):
        bits = []
        for i in range(num_bits):
            bits.append((value >> i) & 1)
        return BitArray(bits)


    def bitarray_to_uint(bitarray):
        value = 0
        for i, bit in enumerate(bitarray.data):
            value |= (bit << i)
        return value


    class LZ77Sequence:
        def __init__(self, literal_count, match_length, match_offset):
            self.literal_count = literal_count
            self.match_length = match_length
            self.match_offset = match_offset

        def __eq__(self, other):
            return (self.literal_count == other.literal_count and
                    self.match_length == other.match_length and
                    self.match_offset == other.match_offset)

        def __repr__(self):
            return f"LZ77Sequence({self.literal_count}, {self.match_length}, {self.match_offset})"

# Import the implementation to test
from tans_lz77_coder2 import (
    TANSEncoder, TANSDecoder,
    LZ77TANSStreamsEncoder, LZ77TANSStreamsDecoder,
    _normalize_freqs, _spread_symbols, _highbit
)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    def test_highbit(self):
        """Test _highbit function"""
        self.assertEqual(_highbit(1), 0)
        self.assertEqual(_highbit(2), 1)
        self.assertEqual(_highbit(3), 1)
        self.assertEqual(_highbit(4), 2)
        self.assertEqual(_highbit(7), 2)
        self.assertEqual(_highbit(8), 3)
        self.assertEqual(_highbit(255), 7)
        self.assertEqual(_highbit(256), 8)

    def test_normalize_freqs_basic(self):
        """Test frequency normalization"""
        freqs = {'a': 5, 'b': 3, 'c': 2}
        norm = _normalize_freqs(freqs, 16)

        # Sum should equal table_size
        self.assertEqual(sum(norm.values()), 16)

        # All values should be positive
        for v in norm.values():
            self.assertGreater(v, 0)

    def test_normalize_freqs_uniform(self):
        """Test normalization with uniform distribution"""
        freqs = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
        norm = _normalize_freqs(freqs, 16)

        self.assertEqual(sum(norm.values()), 16)
        # Should be roughly equal
        self.assertEqual(norm['a'], 4)
        self.assertEqual(norm['b'], 4)
        self.assertEqual(norm['c'], 4)
        self.assertEqual(norm['d'], 4)

    def test_normalize_freqs_skewed(self):
        """Test normalization with skewed distribution"""
        freqs = {'a': 100, 'b': 1}
        norm = _normalize_freqs(freqs, 64)

        self.assertEqual(sum(norm.values()), 64)
        # 'a' should get most of the table
        self.assertGreater(norm['a'], 60)
        self.assertGreaterEqual(norm['b'], 1)

    def test_spread_symbols(self):
        """Test symbol spreading"""
        norm = {'a': 8, 'b': 4, 'c': 4}
        table = _spread_symbols(norm, 4)  # table_log=4, size=16

        # Count occurrences
        counts = Counter(table)
        self.assertEqual(counts['a'], 8)
        self.assertEqual(counts['b'], 4)
        self.assertEqual(counts['c'], 4)
        self.assertEqual(len(table), 16)


class TestTANSEncoderDecoder(unittest.TestCase):
    """Test basic tANS encoder/decoder functionality"""

    def test_encode_decode_simple(self):
        """Test encoding and decoding a simple sequence"""
        symbols = [1, 2, 3, 1, 2, 3, 1, 2, 3]

        encoder = TANSEncoder(table_log=6)
        encoded = encoder.encode(symbols)

        # Should produce non-empty bitarray
        self.assertGreater(len(encoded), 0)

        # Decode and verify
        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=6)
        decoded, bits_consumed = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)
        self.assertEqual(bits_consumed, len(encoded))

    def test_encode_decode_single_symbol(self):
        """Test encoding a sequence with a single repeated symbol"""
        symbols = [5] * 20

        encoder = TANSEncoder(table_log=6)
        encoded = encoder.encode(symbols)

        # Decode
        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=6)
        decoded, bits_consumed = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_encode_decode_alphabet_large(self):
        """Test with larger alphabet"""
        symbols = list(range(20)) * 5  # 20 different symbols, repeated

        encoder = TANSEncoder(table_log=8)
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=8)
        decoded, bits_consumed = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_encode_decode_bytes(self):
        """Test encoding byte values (0-255)"""
        symbols = [65, 66, 67, 68, 69] * 10  # ASCII 'ABCDE' repeated

        encoder = TANSEncoder(table_log=8)
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=8)
        decoded, bits_consumed = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_encode_decode_skewed_distribution(self):
        """Test with highly skewed distribution"""
        symbols = [1] * 90 + [2] * 8 + [3] * 2

        encoder = TANSEncoder(table_log=7)
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=7)
        decoded, bits_consumed = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_encode_empty(self):
        """Test encoding empty sequence"""
        symbols = []

        encoder = TANSEncoder(table_log=6)
        encoded = encoder.encode(symbols)

        self.assertEqual(len(encoded), 0)

    def test_different_table_sizes(self):
        """Test with different table_log values"""
        symbols = [1, 2, 3, 4, 5] * 10

        for table_log in [6, 8, 10, 12]:
            encoder = TANSEncoder(table_log=table_log)
            encoded = encoder.encode(symbols)

            freqs = Counter(symbols)
            decoder = TANSDecoder(table_log=table_log)
            decoded, _ = decoder.decode(encoded, len(symbols), freqs)

            self.assertEqual(decoded, symbols, f"Failed with table_log={table_log}")

    def test_compression_ratio(self):
        """Test that tANS achieves some compression on biased data"""
        # Highly biased sequence
        symbols = [1] * 100 + [2] * 10 + [3] * 5

        encoder = TANSEncoder(table_log=8)
        encoded = encoder.encode(symbols)

        # Naive encoding would use ceil(log2(3)) = 2 bits per symbol
        naive_bits = len(symbols) * 2
        tans_bits = len(encoded)

        # tANS should do better than naive encoding
        self.assertLess(tans_bits, naive_bits)

        # Verify correctness
        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=8)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        self.assertEqual(decoded, symbols)


class TestLZ77TANSStreams(unittest.TestCase):
    """Test LZ77 + tANS integration"""

    def test_encode_decode_simple_lz77(self):
        """Test encoding/decoding simple LZ77 sequences"""
        sequences = [
            LZ77Sequence(5, 3, 10),
            LZ77Sequence(2, 5, 8),
            LZ77Sequence(1, 2, 3),
        ]
        literals = bytearray([65, 66, 67, 68, 69, 70, 71, 72])

        encoder = LZ77TANSStreamsEncoder(table_log=8)
        encoded = encoder.encode_block(sequences, literals)

        decoder = LZ77TANSStreamsDecoder(table_log=8)
        (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

        # Verify sequences
        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig, dec)

        # Verify literals
        self.assertEqual(decoded_literals, literals)

    def test_encode_decode_no_sequences(self):
        """Test with only literals, no sequences"""
        sequences = []
        literals = bytearray(b"Hello, World!")

        encoder = LZ77TANSStreamsEncoder(table_log=8)
        encoded = encoder.encode_block(sequences, literals)

        decoder = LZ77TANSStreamsDecoder(table_log=8)
        (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

        self.assertEqual(len(decoded_sequences), 0)
        self.assertEqual(decoded_literals, literals)

    def test_encode_decode_no_literals(self):
        """Test with only sequences, no literals"""
        sequences = [
            LZ77Sequence(0, 5, 10),
            LZ77Sequence(0, 3, 8),
        ]
        literals = bytearray()

        encoder = LZ77TANSStreamsEncoder(table_log=8)
        encoded = encoder.encode_block(sequences, literals)

        decoder = LZ77TANSStreamsDecoder(table_log=8)
        (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig, dec)
        self.assertEqual(len(decoded_literals), 0)

    def test_encode_decode_large_lz77(self):
        """Test with larger LZ77 data"""
        sequences = []
        for i in range(50):
            sequences.append(LZ77Sequence(
                literal_count=i % 10,
                match_length=(i * 3) % 15 + 1,
                match_offset=(i * 7) % 100 + 1
            ))

        literals = bytearray(range(256)) * 2  # All byte values repeated

        encoder = LZ77TANSStreamsEncoder(table_log=10)
        encoded = encoder.encode_block(sequences, literals)

        decoder = LZ77TANSStreamsDecoder(table_log=10)
        (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig, dec)
        self.assertEqual(decoded_literals, literals)

    def test_encode_decode_repeated_patterns(self):
        """Test with repeated patterns (realistic LZ77 scenario)"""
        # Simulate compressed text with repeated patterns
        sequences = [
            LZ77Sequence(10, 5, 10),  # Repeat previous 5 bytes
            LZ77Sequence(3, 8, 15),  # Repeat previous 8 bytes
            LZ77Sequence(10, 5, 10),  # Same pattern again
            LZ77Sequence(3, 8, 15),  # Same pattern again
        ]

        literals = bytearray(b"The quick brown fox jumps over")

        encoder = LZ77TANSStreamsEncoder(table_log=8)
        encoded = encoder.encode_block(sequences, literals)

        decoder = LZ77TANSStreamsDecoder(table_log=8)
        (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

        self.assertEqual(len(decoded_sequences), len(sequences))
        for orig, dec in zip(sequences, decoded_sequences):
            self.assertEqual(orig, dec)
        self.assertEqual(decoded_literals, literals)

    def test_roundtrip_various_sizes(self):
        """Test round-trip with various data sizes"""
        test_cases = [
            (1, 10),  # 1 sequence, 10 literals
            (10, 100),  # 10 sequences, 100 literals
            (100, 500),  # 100 sequences, 500 literals
        ]

        for num_seq, num_lit in test_cases:
            with self.subTest(sequences=num_seq, literals=num_lit):
                sequences = [
                    LZ77Sequence(i % 5, (i * 2) % 10 + 1, (i * 3) % 50 + 1)
                    for i in range(num_seq)
                ]
                literals = bytearray([i % 256 for i in range(num_lit)])

                encoder = LZ77TANSStreamsEncoder(table_log=10)
                encoded = encoder.encode_block(sequences, literals)

                decoder = LZ77TANSStreamsDecoder(table_log=10)
                (decoded_sequences, decoded_literals), _ = decoder.decode_block(encoded)

                self.assertEqual(len(decoded_sequences), len(sequences))
                for orig, dec in zip(sequences, decoded_sequences):
                    self.assertEqual(orig, dec)
                self.assertEqual(decoded_literals, literals)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_single_symbol_stream(self):
        """Test stream with only one unique symbol"""
        symbols = [42] * 100

        encoder = TANSEncoder(table_log=6)
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=6)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_alphabet_size_equals_table_size(self):
        """Test when alphabet size equals table size"""
        symbols = list(range(64))  # 64 unique symbols

        encoder = TANSEncoder(table_log=6)  # table_size = 64
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=6)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)

    def test_alphabet_too_large_raises_error(self):
        """Test that too large alphabet raises error"""
        symbols = list(range(128))  # 128 unique symbols

        encoder = TANSEncoder(table_log=6)  # table_size = 64, too small

        with self.assertRaises(ValueError):
            encoder.encode(symbols)

    def test_long_sequence(self):
        """Test with long sequence"""
        symbols = [i % 10 for i in range(10000)]

        encoder = TANSEncoder(table_log=10)
        encoded = encoder.encode(symbols)

        freqs = Counter(symbols)
        decoder = TANSDecoder(table_log=10)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)

        self.assertEqual(decoded, symbols)


class TestCompressionEfficiency(unittest.TestCase):
    """Test compression efficiency of tANS"""

    def test_entropy_lower_bound(self):
        """Test that tANS achieves compression close to entropy"""
        import math

        # Create a distribution with known entropy
        symbols = [0] * 700 + [1] * 200 + [2] * 100

        # Calculate entropy
        freqs = Counter(symbols)
        total = len(symbols)
        entropy = 0
        for count in freqs.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Expected bits using entropy
        expected_bits = entropy * len(symbols)

        # Encode with tANS
        encoder = TANSEncoder(table_log=10)
        encoded = encoder.encode(symbols)

        # tANS should be close to entropy (within reasonable overhead)
        # Allow some overhead for state initialization and table
        overhead_tolerance = 1.2  # 20% overhead
        self.assertLess(len(encoded), expected_bits * overhead_tolerance)

        # Verify correctness
        decoder = TANSDecoder(table_log=10)
        decoded, _ = decoder.decode(encoded, len(symbols), freqs)
        self.assertEqual(decoded, symbols)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestTANSEncoderDecoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLZ77TANSStreams))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestCompressionEfficiency))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return result


if __name__ == '__main__':
    run_tests()