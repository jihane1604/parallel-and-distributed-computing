import time
import multiprocessing
from src.utils import add_n_numbers

def run_process(num_processes = 4, num_numbers = 10000):
    """
    Runs the add_n_numbers function using multiple processes and returns the execution time.
    
    Parameters:
    num_processes (int): The number of processes to use. Default is 4.
    num_numbers (int): The range of numbers to sum. Default is 10000.
    
    Returns:
    float: The execution time in seconds.
    """
    # calculating time for processes 
    start_time = time.time()
    
    # start processes in a loop
    processes = []
    results_queue = multiprocessing.Queue()
    step = num_numbers // num_processes
    total = 0
    
    for i in range(num_processes):
        start = i * step + 1
        end = num_numbers if i == num_processes - 1 else (i + 1) * step
        process = multiprocessing.Process(target=add_n_numbers, 
                                          args=(start, end, results_queue))
        processes.append(process)
        process.start()
    # wait for threads to finish
    for process in processes:
        process.join()
        
    for _ in range(num_processes):
        result = results_queue.get()
        total += result
        
    print(f'total sum is: {total}')
    end_time = time.time()
    return end_time - start_time
