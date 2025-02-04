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
print('-----')

# calculate speedup 
thread_speedup = seq_time / thread_time
thread_efficiency = thread_speedup / 2 # 2 is the number of processes / threads
thread_amdhal = 1 / ((1-1)+(1/2))
thread_gustaffson = 2

process_speedup = seq_time / process_time
process_efficiency = process_speedup / 2 # 2 is the number of processes / threads
process_amdhal = 1 / ((1-1)+(1/2))
process_gustaffson = 2


print(f"Thread speedup rate: {thread_speedup} \nThread efficiency: {thread_efficiency} \n--------\nProcess speedup rate: {process_speedup} \nProcess efficiency: {process_efficiency} \n--------")
