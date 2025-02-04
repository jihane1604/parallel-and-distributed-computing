import random
import string
# Function to join a thousand random letters
def join_random_letters(start = 0, end = 1000):
    """
    Generates and joins random letters within a specified range.

    This function creates a list of random uppercase and lowercase letters 
    from the English alphabet and joins them into a single string.

    Args:
        start (int, optional): The starting index (default is 0).
        end (int, optional): The ending index, determining the number of letters generated (default is 1000).

    Returns:
        str: A string containing randomly generated letters.
    """
    letters = [random.choice(string.ascii_letters) for _ in range(start, end)]
    joined_letters = ''.join(letters)
    #print("Joined Letters Task Done")
    return joined_letters
# Function to add a thousand random numbers
def add_random_numbers(start = 0, end = 1000):
    """
    Generates random numbers within a specified range and returns their sum.

    This function creates a list of random integers between 1 and 100, then sums them.

    Args:
        start (int, optional): The starting index (default is 0).
        end (int, optional): The ending index, determining the number of numbers generated (default is 1000).

    Returns:
        int: The sum of randomly generated numbers.
    """
    numbers = [random.randint(1, 100) for _ in range(start, end)]
    total_sum = sum(numbers)
    #print("Add Numbers Task Done")
    return total_sum