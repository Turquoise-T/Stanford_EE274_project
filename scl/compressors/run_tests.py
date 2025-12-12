#!/usr/bin/env python3
import sys
import unittest
import argparse
import time
import importlib


def _import_test_module(module_basename: str):
    for name in (module_basename, f"scl.compressors.{module_basename}"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return importlib.import_module(module_basename)


def run_tans_coder_tests(verbosity=2):
    print("=" * 60)
    print("Running tANS Coder Tests")
    print("=" * 60)
    
    m = _import_test_module("test_tans_lz77_coder")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(m.TestTANSEncoder))
    suite.addTests(loader.loadTestsFromTestCase(m.TestTANSDecoder))
    suite.addTests(loader.loadTestsFromTestCase(m.TestTANSRoundTrip))
    suite.addTests(loader.loadTestsFromTestCase(m.TestTANSEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(m.TestTANSPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_benchmark_tests(verbosity=2):
    print("=" * 60)
    print("Running LZ77 tANS Benchmark Tests")
    print("=" * 60)
    
    m = _import_test_module("test_lz77_tans_benchmark")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(m.TestHeaderFunctions))
    suite.addTests(loader.loadTestsFromTestCase(m.TestHeaderComputationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(m.TestLZ77StreamsEncoderTANSLiterals))
    suite.addTests(loader.loadTestsFromTestCase(m.TestLZ77EncoderTANSLiterals))
    suite.addTests(loader.loadTestsFromTestCase(m.TestBenchmarkIntegration))
    suite.addTests(loader.loadTestsFromTestCase(m.TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_quick_tests(verbosity=2):
    print("=" * 60)
    print("Running Quick Tests Only")
    print("=" * 60)
    
    tans = _import_test_module("test_tans_lz77_coder")
    bench = _import_test_module("test_lz77_tans_benchmark")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add only quick test classes
    suite.addTests(loader.loadTestsFromTestCase(tans.TestTANSEncoder))
    suite.addTests(loader.loadTestsFromTestCase(tans.TestTANSDecoder))
    suite.addTests(loader.loadTestsFromTestCase(tans.TestTANSEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(bench.TestHeaderFunctions))
    suite.addTests(loader.loadTestsFromTestCase(bench.TestHeaderComputationFunctions))
    
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
