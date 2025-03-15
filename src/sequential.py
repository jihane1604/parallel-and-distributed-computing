import time
from src.utils import square

def run_seq(n = 6):
    """
    Compute the squares of numbers from 0 to 10^n - 1 sequentially and measure the execution time.

    This function creates a list of numbers from 0 to 10^n - 1, computes the square of each number using
    the square() function, prints the last computed result, and returns the total time taken to perform the computations.

    Args:
        n (int, optional): The exponent that determines the upper bound of the range (10^n). Defaults to 6.

    Returns:
        float: The elapsed time in seconds for computing the squares.
    """
    # satrt time 
    start_time = time.time()
    numbers = [i for i in range(10 ** n)]

    results = []
    for num in numbers:
        results.append(square(num))
    # end time 
    end_time = time.time()

    print(f"The last results is: {results[-1]}")

    return end_time - start_time