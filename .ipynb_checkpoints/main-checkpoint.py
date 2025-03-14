from mpi4py import MPI
import numpy as np
import time

def square(arr):
    return arr ** 2  # Vectorized NumPy operation

start_time = time.time()
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

N = int(1e8)  # Compute squares up to 10^8

# Divide workload among processes
chunk_size = N // size  # Evenly distribute numbers
remainder = N % size  # Handle remainder if not evenly divisible

# Master process (rank 0) initializes data
if rank == 0:
    numbers = np.arange(N, dtype="i")  # Array from 0 to 10^8
    # Split into chunks for each process
    chunks = [numbers[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    if remainder:
        chunks[-1] = np.append(chunks[-1], numbers[-remainder:])  # Add remainder to the last chunk
else:
    chunks = None

# Scatter data to all processes
local_numbers = np.zeros(chunk_size + (remainder if rank == size - 1 else 0), dtype="i")
comm.Scatter(chunks, local_numbers, root=0)

# Compute the square of local chunk
local_results = square(local_numbers)

# Gather results at rank 0
if rank == 0:
    final_results = np.zeros(N, dtype="i")
else:
    final_results = None

comm.Gather(local_results, final_results, root=0)

# Master process prints the results
if rank == 0:
    print(f"Computation completed. Last squared number: {final_results[-1]}")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
