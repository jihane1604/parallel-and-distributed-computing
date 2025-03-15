import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from src.utils import square

# helper function fro processes
def worker(num, results_queue):
    """
    Helper function for processing a single number in a separate process.

    Computes the square of the given number and puts the result in the provided queue.

    Args:
        num (int): The number to square.
        results_queue (multiprocessing.Queue): The queue to store the computed result.
    """
    results_queue.put(square(num))

# function to loop through the numbers and create a seperate process for each one
def run_loop_processes(n = 6):
    """
    Create a separate process for each number in a range of 10^n numbers,
    compute their square, and measure the total processing time.

    This function creates a new process for every number, uses a multiprocessing
    Queue to collect results, and then returns the elapsed time.

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]
    
    processes = []
    # queue to store the results of the squaring operations 
    results_queue = multiprocessing.Queue()

    # start processes
    for num in numbers:
        p = multiprocessing.Process(target = worker, args = (num, results_queue))
        processes.append(p)
        p.start()

    # wait for processes to finish
    for p in processes:
        p.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    # end time 
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time

# function to use process pools with map
def run_pool_map(n = 6):
    """
    Use a multiprocessing Pool with the map() method to compute the squares of
    numbers in a range of 10^n numbers, and measure the total processing time.

    The pool is configured with 6 worker processes.

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]
    
    with multiprocessing.Pool(processes = 6) as pool:
        results = pool.map(square, numbers)

    # end time
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time

# function to use process pools with map_async
def run_pool_map_async(n = 6):
    """
    Use a multiprocessing Pool with the asynchronous map (map_async) method to compute
    the squares of numbers in a range of 10^n numbers, and measure the total processing time.

    The pool is configured with 6 worker processes.

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]
    
    with multiprocessing.Pool(processes = 6) as pool:
        results = pool.map_async(square, numbers).get()

    # end time
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time

# function to use process pools with apply
def run_pool_apply(n = 6):
    """
    Use a multiprocessing Pool with the apply() method to compute the squares of
    numbers in a range of 10^n numbers, and measure the total processing time.

    The pool is configured with 6 worker processes. This method applies the function
    sequentially to each element, which may not be as efficient as map() for many tasks.

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]
    
    with multiprocessing.Pool(processes = 6) as pool:
        results = [pool.apply(square, (num,)) for num in numbers]

    # end time
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time

# function to use process pools with apply_async
def run_pool_apply_async(n = 6):
    """
    Use a multiprocessing Pool with the asynchronous apply (apply_async) method to compute
    the squares of numbers in a range of 10^n numbers, and measure the total processing time.

    The pool is configured with 6 worker processes. Each task is submitted asynchronously,
    and results are collected using get().

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]
    
    with multiprocessing.Pool(processes = 6) as pool:
        results = [pool.apply_async(square, (num,)).get() for num in numbers]

    # end time
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time

# function to use the process pool executor
def run_executor(n = 6):
    """
    Use a ProcessPoolExecutor to compute the squares of numbers in a range of 10^n numbers,
    and measure the total processing time.

    The executor is configured with a maximum of 6 worker processes.

    Args:
        n (int): The exponent to define the number range (10^n).

    Returns:
        float: The total time (in seconds) required to process all numbers.
    """
    # satrt time 
    start_time = time.time()
    # create a list of 10 ^ n numbers  
    numbers = [i for i in range(10 ** n)]

    with ProcessPoolExecutor(max_workers = 6) as executor:
        results = list(executor.map(square, numbers))
    # end time
    end_time = time.time()
    print(f"The last results is: {results[-1]}")
    return end_time - start_time
