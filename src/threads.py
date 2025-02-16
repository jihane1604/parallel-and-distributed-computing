import time
import threading
from src.utils import add_n_numbers
import queue

def run_thread(num_threads = 4, num_numbers = 10000):
    """
    Runs the add_n_numbers function using multiple threads and returns the execution time.
    
    Parameters:
    num_threads (int): The number of threads to use. Default is 4.
    num_numbers (int): The range of numbers to sum. Default is 10000.
    
    Returns:
    float: The execution time in seconds.
    """
    # calculating the time for threading 
    start_time = time.time()
    
    # start threads in a loop
    threads = []
    results_queue = queue.Queue()
    step = num_numbers // num_threads
    total = 0
    
   
    for i in range(num_threads):
        start = i * step + 1
        end = num_numbers if i == num_threads - 1 else (i + 1) * step
        thread = threading.Thread(target=add_n_numbers, 
                                  args=(start, end, results_queue))
        threads.append(thread)
        thread.start()
    # wait for threads to finish
    for thread in threads:
        thread.join()

    for _ in range(num_threads):
        total += results_queue.get()

    print(f'total sum is: {total}')
    end_time = time.time()    
    return end_time - start_time
