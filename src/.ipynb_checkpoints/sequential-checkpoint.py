import time
from src.utils import join_random_letters
from src.utils import add_random_numbers

def run_sequential(num_letters = 10000, num_numbers = 10000):
    total_start_time = time.time()
    join_random_letters(0, num_letters)
    add_random_numbers(0, num_numbers)
    total_end_time = time.time()
    return total_end_time - total_start_time
