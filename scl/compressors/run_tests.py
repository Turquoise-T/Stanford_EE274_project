#!/usr/bin/env python3
"""
Test runner for tANS and LZ77 benchmark unit tests.

Usage (from scl/compressors directory):
    python run_tests.py                    # Run all tests
    python run_tests.py --tans-only        # Run only tANS coder tests
    python run_tests.py --benchmark-only   # Run only benchmark tests
    python run_tests.py --quick            # Run quick tests only (skip slow ones)

Alternative usage (from project root):
    python -m scl.compressors.run_tests                    # Run all tests
    python -m scl.compressors.run_tests --tans-only        # Run only tANS coder tests
    python -m scl.compressors.run_tests --benchmark-only   # Run only benchmark tests
    python -m scl.compressors.run_tests --quick            # Run quick tests only (skip slow ones)
"""

import sys
import unittest
import argparse
import time


def run_tans_coder_tests(verbosity=2):
    """Run tests for tans_lz77_coder.py"""
    print("=" * 60)
    print("Running tANS Coder Tests")
    print("=" * 60)
    
    # Import test module (absolute import for direct execution)
    from test_tans_lz77_coder import (
        TestTANSEncoder,
        TestTANSDecoder,
        TestTANSRoundTrip,
        TestLZ77TANSStreamsEncoder,
        TestTANSEdgeCases,
        TestTANSPerformance,
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestTANSEncoder))
    suite.addTest(unittest.makeSuite(TestTANSDecoder))
    suite.addTest(unittest.makeSuite(TestTANSRoundTrip))
    suite.addTest(unittest.makeSuite(TestLZ77TANSStreamsEncoder))
    suite.addTest(unittest.makeSuite(TestTANSEdgeCases))
    suite.addTest(unittest.makeSuite(TestTANSPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_benchmark_tests(verbosity=2):
    """Run tests for lz77_tans_benchmark.py"""
    print("=" * 60)
    print("Running LZ77 tANS Benchmark Tests")
    print("=" * 60)
    
    # Import test module (absolute import for direct execution)
    from test_lz77_tans_benchmark import (
        TestHeaderFunctions,
        TestHeaderComputationFunctions,
        TestLZ77StreamsEncoderTANSLiterals,
        TestLZ77EncoderTANSLiterals,
        TestBenchmarkIntegration,
        TestEdgeCases,
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestHeaderFunctions))
    suite.addTest(unittest.makeSuite(TestHeaderComputationFunctions))
    suite.addTest(unittest.makeSuite(TestLZ77StreamsEncoderTANSLiterals))
    suite.addTest(unittest.makeSuite(TestLZ77EncoderTANSLiterals))
    suite.addTest(unittest.makeSuite(TestBenchmarkIntegration))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_quick_tests(verbosity=2):
    """Run only quick tests (skip performance tests)"""
    print("=" * 60)
    print("Running Quick Tests Only")
    print("=" * 60)
    
    # Import test modules (absolute imports for direct execution)
    from test_tans_lz77_coder import (
        TestTANSEncoder,
        TestTANSDecoder,
        TestTANSEdgeCases,
    )
    from test_lz77_tans_benchmark import (
        TestHeaderFunctions,
        TestHeaderComputationFunctions,
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add only quick test classes
    suite.addTest(unittest.makeSuite(TestTANSEncoder))
    suite.addTest(unittest.makeSuite(TestTANSDecoder))
    suite.addTest(unittest.makeSuite(TestTANSEdgeCases))
    suite.addTest(unittest.makeSuite(TestHeaderFunctions))
    suite.addTest(unittest.makeSuite(TestHeaderComputationFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run tANS and LZ77 benchmark tests")
    parser.add_argument("--tans-only", action="store_true", 
                       help="Run only tANS coder tests")
    parser.add_argument("--benchmark-only", action="store_true", 
                       help="Run only benchmark tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests (skip slow ones)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Quiet output")
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Track results
    all_passed = True
    start_time = time.time()
    
    try:
        if args.quick:
            # Run quick tests only
            success = run_quick_tests(verbosity)
            all_passed = all_passed and success
            
        elif args.tans_only:
            # Run only tANS coder tests
            success = run_tans_coder_tests(verbosity)
            all_passed = all_passed and success
            
        elif args.benchmark_only:
            # Run only benchmark tests
            success = run_benchmark_tests(verbosity)
            all_passed = all_passed and success
            
        else:
            # Run all tests
            print("Running all tests...\n")
            
            success = run_tans_coder_tests(verbosity)
            all_passed = all_passed and success
            
            print("\n")
            
            success = run_benchmark_tests(verbosity)
            all_passed = all_passed and success
    
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure you're running from the correct directory and have all dependencies installed.")
        return 1
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    
    if all_passed:
        print("✓ All tests PASSED")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
