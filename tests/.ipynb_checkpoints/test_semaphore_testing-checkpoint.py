"""
Tests for semaphore-based connection pooling in semaphore_testing.py.
These tests verify that the connection pool correctly manages the acquisition
and release of connections, and that the access_database function performs as expected.
"""

import pytest
import multiprocessing
from src import semaphore_testing

def test_connection_pool():
    """
    Test the ConnectionPool class to verify that:
    - The initial number of connections equals max_connections.
    - Connections are acquired and removed correctly.
    - Connections are released back to the pool correctly.
    """
    max_connections = 2
    pool = semaphore_testing.ConnectionPool(max_connections)
    
    # Initially, the pool should have max_connections connections.
    assert len(pool.connections) == max_connections

    # Acquire one connection.
    conn1 = pool.get_connection()
    assert conn1.startswith("Connection-")
    # Now the pool has one less connection.
    assert len(pool.connections) == max_connections - 1

    # Acquire a second connection.
    conn2 = pool.get_connection()
    assert len(pool.connections) == 0

    # Release one connection.
    pool.release_connection(conn1)
    assert len(pool.connections) == 1

    # Release the other connection.
    pool.release_connection(conn2)
    assert len(pool.connections) == max_connections

# Optionally, test the access_database function.
# Note: Because access_database uses random sleep and prints output,
# the following test spawns a process and captures printed output.
def test_access_database(capsys):
    """
    Test the access_database function by:
    - Running it in a separate process.
    - Capturing the printed output to verify that it indicates:
      waiting, acquiring, and releasing a connection.
    """
    max_connections = 1
    pool = semaphore_testing.ConnectionPool(max_connections)
    p = multiprocessing.Process(target=semaphore_testing.access_database, args=(pool,))
    p.start()
    p.join()
    captured = capsys.readouterr().out
    # Check that the expected status messages appear in the output.
    assert "Waiting for a connection" in captured
    assert "Acquired" in captured
    assert "Releasing" in captured
