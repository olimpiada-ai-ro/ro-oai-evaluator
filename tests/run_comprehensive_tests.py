#!/usr/bin/env python3
"""
Comprehensive test runner for the AI Olympiad Evaluator.
Runs all tests with proper categorization and reporting.
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Exit code: {result.returncode}")
    
    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout}")
    
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive tests for AI Olympiad Evaluator")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend([
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    # Test selection based on arguments
    test_markers = []
    test_files = []
    
    if args.unit:
        test_markers.append("unit")
    elif args.integration:
        test_markers.append("integration")
    elif args.performance:
        test_markers.append("performance")
    else:
        # Run all tests by default
        pass
    
    # Skip slow tests if requested
    if args.fast:
        test_markers.append("not slow")
    
    # Add marker selection
    if test_markers:
        pytest_cmd.extend(["-m", " and ".join(test_markers)])
    
    # Add test directory
    pytest_cmd.append("tests/")
    
    # Run the tests
    print("AI Olympiad Evaluator - Comprehensive Test Suite")
    print("=" * 60)
    
    result = run_command(pytest_cmd, "Comprehensive Test Suite")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Exit code: {result.returncode}")
    
    # Additional test categories if running all tests
    if not any([args.unit, args.integration, args.performance]):
        print("\nRunning additional test categories...")
        
        # Unit tests
        unit_cmd = pytest_cmd.copy()
        if "-m" in unit_cmd:
            marker_idx = unit_cmd.index("-m") + 1
            unit_cmd[marker_idx] = "unit"
        else:
            unit_cmd.extend(["-m", "unit"])
        
        unit_result = run_command(unit_cmd, "Unit Tests Only")
        
        # Integration tests
        integration_cmd = pytest_cmd.copy()
        if "-m" in integration_cmd:
            marker_idx = integration_cmd.index("-m") + 1
            integration_cmd[marker_idx] = "integration"
        else:
            integration_cmd.extend(["-m", "integration"])
        
        integration_result = run_command(integration_cmd, "Integration Tests Only")
        
        # Performance tests (if not skipping slow tests)
        if not args.fast:
            perf_cmd = pytest_cmd.copy()
            if "-m" in perf_cmd:
                marker_idx = perf_cmd.index("-m") + 1
                perf_cmd[marker_idx] = "performance"
            else:
                perf_cmd.extend(["-m", "performance"])
            
            perf_result = run_command(perf_cmd, "Performance Tests Only")
    
    # Coverage report location
    if args.coverage:
        print(f"\n📊 Coverage report generated at: htmlcov/index.html")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())