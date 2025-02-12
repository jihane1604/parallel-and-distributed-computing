from src.sequential import run_sequential
from src.threads import run_thread
from src.processes import run_process

# Measure the total time for each 
seq_time = run_sequential(10000000)
print(f"Total time taken sequentially: {seq_time} seconds")
print('-----')

thread_time = run_thread(4, 10000000)
print(f"Total time taken using multithreading: {thread_time} seconds")
print('-----')

process_time = run_process(4, 10000000)
print(f"Total time taken using multiprocessing: {process_time} seconds")
print('-----')

# calculate speedup 
thread_speedup = seq_time / thread_time
thread_efficiency = thread_speedup / 4 # 4 is the number of processes / threads
# estimating p and alpha from number of lines
# p = number of parallel lines / number of total lines 
thread_amdhal = 1 / ((1-0.87) + (0.87/4))
thread_gustaffson = 4 + 0.13 * (1-4)

process_speedup = seq_time / process_time
process_efficiency = process_speedup / 4 # 4 is the number of processes / threads
# estimating p and alpha from number of lines
# p = number of parallel lines / number of total lines 
process_amdhal = 1 / ((1-0.87) + (0.87/4))
process_gustaffson = 4 + 0.13 * (1-4)


print(f"Thread speedup rate: {thread_speedup} \nThread efficiency: {thread_efficiency} \nThread Amdhal {thread_amdhal} \nThread Gustaffson {thread_gustaffson} \n-------- \nProcess speedup rate: {process_speedup} \nProcess efficiency: {process_efficiency} \nAmdhal: {process_amdhal} \nGustaffson {process_gustaffson} \n--------")
