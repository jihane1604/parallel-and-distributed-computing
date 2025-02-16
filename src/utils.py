# Function to add numbers from 1 to n
def add_n_numbers(start = 0, end = 1000, results_queue = None):
    """
    Computes the sum of numbers in the range from start to end (inclusive) and prints the total sum.
    
    Parameters:
    start (int): The starting number of the range. Default is 0.
    end (int): The ending number of the range. Default is 1000.
    """
    total_sum = 0
    for n in range(start, end +1):
        total_sum += n
    #print("Add Numbers Task Done")
    if results_queue:
        results_queue.put(total_sum)
    return total_sum
    # print(f"Total sum for range({start},{end}) is {total_sum}")