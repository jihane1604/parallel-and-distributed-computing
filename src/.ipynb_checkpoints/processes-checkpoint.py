import time
import multiprocessing
from src.utils import add_n_numbers

def run_process(num_processes = 4, num_numbers = 10000):
    """
    """
    # start processes in a loop
    processes = []
    step = num_numbers // num_processes
    # calculating time for processes 
    start_time = time.time()
    for i in range(num_processes):
        process = multiprocessing.Process(target=add_n_numbers, args=(i * step, ((i+1) * step) - 1))
        processes.append(process)
        process.start()
    # wait for threads to finish
    for process in processes:
        process.join()
    end_time = time.time()
    return end_time - start_time

