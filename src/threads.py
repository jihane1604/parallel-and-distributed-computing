import time
import threading
from src.utils import join_random_letters
from src.utils import add_random_numbers

def run_thread(num_threads, num_letters = 10000, num_numbers = 10000):
    # calculating the time for threading 
    total_start_time = time.time()
    # Create threads for both functions
    thread_letters1 = threading.Thread(target=join_random_letters, args=(0, num_letters//2))
    thread_letters2 = threading.Thread(target=join_random_letters, args=(num_letters//2, (num_letters//2)*1))
    # thread_letters3 = threading.Thread(target=join_random_letters, args=((num_letters//3)*2, num_letters))
    
    thread_numbers1 = threading.Thread(target=add_random_numbers, args=(0, num_numbers//2))
    thread_numbers2 = threading.Thread(target=add_random_numbers, args=(num_numbers//2, (num_numbers//2)*1))
    # thread_numbers3 = threading.Thread(target=add_random_numbers, args=((num_numbers//3)*2, num_numbers))
    
    # Start the threads
    thread_letters1.start()
    thread_letters2.start()
    # thread_letters3.start()
    
    thread_numbers1.start()
    thread_numbers2.start()
    # thread_numbers3.start()
    # Wait for all threads to complete
    thread_letters1.join()
    thread_letters2.join()
    # thread_letters3.join()
    
    thread_numbers1.join()
    thread_numbers2.join()
    # thread_numbers3.join()
    
    total_end_time = time.time()
    return total_end_time - total_start_time