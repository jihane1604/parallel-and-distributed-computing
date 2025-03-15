import multiprocessing
import random
import time

class ConnectionPool:
    """
    A connection pool that simulates a set of database connections.
    
    This pool uses a shared list (via a multiprocessing.Manager), a semaphore to limit
    concurrent access, and a lock to ensure safe modifications of the connection list.
    """
    
    def __init__(self, max_connections):
        """
        Initialize the ConnectionPool with a maximum number of connections.
        
        Args:
            max_connections (int): The maximum number of connections available in the pool.
        """
        
        manager = multiprocessing.Manager()
        # create a list of connection identifiers use manager to handle inteprocess communication
        self.connections = manager.list([f"Connection-{i+1}" for i in range(max_connections)])
        
        # the semaphore controls access to the pool.
        self.semaphore = multiprocessing.Semaphore(max_connections)
        # lock to safely update the shared connection list.
        self.lock = multiprocessing.Lock()

    def get_connection(self):
        """
        Acquire a connection from the pool.
        
        This method blocks until a connection is available, then removes and returns one
        connection from the pool.
        
        Returns:
            str: The acquired connection identifier.
        """
        
        # wait until a connection becomes available.
        self.semaphore.acquire()
        with self.lock:
            # remove and return a connection from the pool.
            connection = self.connections.pop()
        return connection

    def release_connection(self, connection):
        """
        Release a connection back to the pool.
        
        Args:
            connection (str): The connection identifier to be returned to the pool.
        """
        
        with self.lock:
            # return the connection back to the pool.
            self.connections.append(connection)

        # release the connection for other processes to use
        self.semaphore.release()

def access_database(pool):
    """
    Simulate a process performing a database operation.
    
    This function acquires a connection from the provided connection pool, prints messages
    indicating its status, simulates work by sleeping for a random duration, and then releases
    the connection.
    
    Args:
        pool (ConnectionPool): The connection pool from which to acquire and release connections.
    """
    
    process_name = multiprocessing.current_process().name
    print(f"{process_name}: Waiting for a connection...")
    
    connection = pool.get_connection()
    print(f"{process_name}: Acquired {connection}")
    
    # simulate a database operation by sleeping for a random time.
    time.sleep(random.randint(1, 5))
    
    print(f"{process_name}: Releasing {connection}")
    pool.release_connection(connection)

def main():
    """
    Main function to demonstrate the ConnectionPool with multiprocessing.
    
    This function creates a ConnectionPool with a limited number of connections, spawns multiple
    processes that attempt to access the database (simulate work), and ensures all processes complete.
    It prints messages indicating when a process is waiting, acquiring, and releasing a connection.
    """
    
    # define a pool with a limited number of connections
    max_connections = 3
    pool = ConnectionPool(max_connections)
    processes = []
    
    # create more processes than there are connections.
    for i in range(6):
        p = multiprocessing.Process(target = access_database, args = (pool,), name = f"Process-{i+1}")
        processes.append(p)
        p.start()

    # wait for processes to finish
    for p in processes:
        p.join()

    print("All processes have completed.")

# run main function
main()