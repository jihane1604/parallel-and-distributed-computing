import time
from src.utils import join_random_letters
from src.utils import add_random_numbers

def run_sequential(num_letters = 10000, num_numbers = 10000):
    """
    Executes random letter generation and number summation sequentially.

    This function runs the following tasks one after the other:
    - `join_random_letters`: Generates and joins random letters.
    - `add_random_numbers`: Generates and sums random numbers.

    The function measures and returns the total execution time.

    Args:
        num_letters (int, optional): The total number of random letters to generate. Default is 10,000.
        num_numbers (int, optional): The total number of random numbers to generate. Default is 10,000.

    Returns:
        float: The total execution time in seconds.
    """
    total_start_time = time.time()
    join_random_letters(0, num_letters)
    add_random_numbers(0, num_numbers)
    total_end_time = time.time()
    return total_end_time - total_start_time
