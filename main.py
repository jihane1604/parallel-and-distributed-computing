from src.sequential import run_sequential
from src.threads import run_thread
from src.processes import run_process

# Measure the total time for each 
seq_time = run_sequential(10000000, 10000000)
print(f"Total time taken sequentially: {seq_time} seconds")
print('-----')

thread_time = run_thread(10000000, 10000000)
print(f"Total time taken using multithreading: {thread_time} seconds")
print('-----')

process_time = run_process(10000000, 10000000)
print(f"Total time taken using multiprocessing: {process_time} seconds")

# calculate speedup 
speedup = seq_time / thread_time
efficiency = speedup / 2
amdhal = 1 / (1/2)
gustaffson = 2

print(f"Speedup rate: {speedup} \nEfficiency: {efficiency} \nAmdhal: {amdhal} \nGustaffson: {gustaffson}")

 