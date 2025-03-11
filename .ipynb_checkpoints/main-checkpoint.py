from mpi4py import MPI
import numpy as np
from src.square import square
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
results = None

print(f"which process is this: {rank}, size is {size}")

if rank == 0:
    numbers = np.arange(size, dtype = "i")
    
else:
    numbers = None

# print(numbers)
number = np.zeros(1, dtype = "i")
# broadcasting: one process sends to all the rest
comm.Scatter(numbers, number, root = 0)
# print(numbers)
# print(number)

result = square(number[0])
# print(result)
time.sleep(random.randint(1, 10))
# non blocking: can finish executing without waiting for it to be received
request = comm.isend(result, dest = 0, tag = rank)

if rank == 0:
    results = np.zeros(size, dtype = "i")
    for i in range(size):
        results[i] = comm.irecv(source = i, tag = i).wait()
    print(f"the results are {results}")
        
request.wait()





