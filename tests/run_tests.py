import unittest
import os

if __name__ == "__main__":
    # Discover all test cases in the tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = unittest.defaultTestLoader.discover(test_dir)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
