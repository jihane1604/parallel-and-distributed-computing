"""
Tests for utility functions in utils.py.
These tests validate that the square function returns the correct squared value
for a variety of inputs.
"""

import pytest
from src import utils

def test_square():
    """
    Test the square function for several cases including:
    - Zero and positive integers.
    - Negative integers.
    """
    assert utils.square(0) == 0
    assert utils.square(2) == 4
    assert utils.square(10) == 100
    # Also check negative numbers.
    assert utils.square(-3) == 9
