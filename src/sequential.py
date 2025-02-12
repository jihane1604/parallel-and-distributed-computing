import time
from src.utils import add_n_numbers

def run_sequential(num_numbers = 10000):
    """
    Runs the add_n_numbers function sequentially and returns the execution time.
    
    Parameters:
    num_numbers (int): The range of numbers to sum. Default is 10000.
    
    Returns:
    float: The execution time in seconds.
    """
    total_start_time = time.time()
    add_n_numbers(0, num_numbers)
    total_end_time = time.time()
    return total_end_time - total_start_time
