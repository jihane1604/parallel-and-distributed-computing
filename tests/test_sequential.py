"""
Tests for the sequential processing function in sequential.py.
These tests verify that the function computes the correct squared values,
returns a valid elapsed time, and prints the expected last result.
"""

import pytest
from src import sequential

N = 3

def expected_last(n):
    """Return the expected last squared value for the range 0 to 10**n - 1."""
    return (10**n - 1) ** 2

def test_run_seq(capsys):
    """
    Test the run_seq function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = sequential.run_seq(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured
