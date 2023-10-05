#!/usr/bin/env python3

import unittest


def run_tests():
    """
    Discovers and runs all tests in the 'tests' directory.

    This function uses the unittest module to discover and run all tests in the 'tests' directory.
    It creates a test loader to load the tests, a test suite to hold the discovered tests, and a test runner to execute the tests.
    The 'tests' directory should contain all the unit test files.

    Returns:
    None
    """

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests")
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)


if __name__ == "__main__":
    run_tests()
