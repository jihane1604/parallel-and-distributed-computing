import time
import multiprocessing
from src.utils import join_random_letters
from src.utils import add_random_numbers

def run_process(num_letters = 10000, num_numbers = 10000):
    # calculating time for processes 
    total_start_time = time.time()
    # Create processes for both functions
    process_letters1 = multiprocessing.Process(target=join_random_letters, args=(0, num_letters//2))
    process_letters2 = multiprocessing.Process(target=join_random_letters, args=(num_letters//2, (num_letters//2)*1))
    # process_letters3 = multiprocessing.Process(target=join_random_letters, args=((num_letters//3)*2, num_letters))
    
    process_numbers1 = multiprocessing.Process(target=add_random_numbers, args=(0, num_numbers//2))
    process_numbers2 = multiprocessing.Process(target=add_random_numbers, args=(num_numbers//2, (num_numbers//2)*1))
    # process_numbers3 = multiprocessing.Process(target=add_random_numbers, args=((num_numbers//3)*2, num_numbers))
    
    # Start the processes
    process_letters1.start()
    process_letters2.start()
    # process_letters3.start()
    
    process_numbers1.start()
    process_numbers2.start()
    # process_numbers3.start()
    # Wait for all processes to complete
    process_letters1.join()
    process_letters2.join()
    # process_letters3.join()
    
    process_numbers1.join()
    process_numbers2.join()
    # process_numbers3.join()
    
    total_end_time = time.time()
    return total_end_time - total_start_time