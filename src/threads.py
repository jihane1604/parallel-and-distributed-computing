import time
import threading
from src.utils import add_n_numbers

def run_thread(num_threads = 4, num_numbers = 10000):
    """
    Runs the add_n_numbers function using multiple threads and returns the execution time.
    
    Parameters:
    num_threads (int): The number of threads to use. Default is 4.
    num_numbers (int): The range of numbers to sum. Default is 10000.
    
    Returns:
    float: The execution time in seconds.
    """
    # start threads in a loop
    threads = []
    step = num_numbers // num_threads
    # calculating the time for threading 
    start_time = time.time()
    for i in range(num_threads):
        thread = threading.Thread(target=add_n_numbers, args=(i * step, ((i+1) * step) - 1))
        threads.append(thread)
        thread.start()
    # wait for threads to finish
    for thread in threads:
        thread.join()
    end_time = time.time()    
    return end_time - start_time
