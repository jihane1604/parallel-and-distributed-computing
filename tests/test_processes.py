"""
Tests for the multiprocessing functions in proceses.py.
These tests verify that the functions compute the correct squared values,
return a valid elapsed time, and print the expected last result.
"""

import pytest
from src import processes

# Use a smaller exponent for testing speed.
N = 3  # 10**3 numbers

def expected_last(n):
    """Return the expected last squared value for the range 0 to 10**n - 1."""
    return (10**n - 1) ** 2

def test_run_loop_processes(capsys):
    """
    Test the run_loop_processes function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_loop_processes(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_map(capsys):
    """
    Test the run_pool_map function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_map(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_map_async(capsys):
    """
    Test the run_pool_map_async function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_map_async(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_apply(capsys):
    """
    Test the run_pool_apply function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_apply(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_apply_chunk(capsys):
    """
    Test the run_pool_apply_chunk function with a smaller chunk size by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_apply_chunk(n=N, chunk_size=100)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_apply_async(capsys):
    """
    Test the run_pool_apply_async function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_apply_async(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_pool_apply_async_chunk(capsys):
    """
    Test the run_pool_apply_async_chunk function with chunking by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_pool_apply_async_chunk(n=N, chunk_size=100)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured

def test_run_executor(capsys):
    """
    Test the run_executor function by verifying:
    - It returns a float representing the elapsed time.
    - The printed output contains the correct last squared value.
    """
    elapsed = processes.run_executor(n=N)
    assert isinstance(elapsed, float)
    assert elapsed > 0
    captured = capsys.readouterr().out
    assert str(expected_last(N)) in captured
