import time
from src.utils import add_n_numbers

def run_sequential(num_numbers = 10000):
    """
    """
    total_start_time = time.time()
    add_n_numbers(0, num_numbers)
    total_end_time = time.time()
    return total_end_time - total_start_time
