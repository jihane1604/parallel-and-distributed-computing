# Function to add numbers from 1 to n
def add_n_numbers(start = 0, end = 1000):
    """
    """
    numbers = [n for n in range(start, end+1)]
    total_sum = sum(numbers)
    #print("Add Numbers Task Done")
    print(f"Total sum for range({start},{end}) is {total_sum}")