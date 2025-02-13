from src.sequential import run_sequential
from src.threads import run_thread
from src.processes import run_process
from src.utils import prepare_data
import threading


# Define the parameter ranges
n_estimators_range = [10, 25, 50, 100, 200, 300, 400]
max_features_range = ['sqrt', 'log2', None]  # None means using all features
max_depth_range = [1, 2, 5, 10, 20, None]  # None means no limit

# Shared variables
best_rmse = float('inf')
best_mape = float('inf')
best_model = None
best_parameters = {}
lock = threading.Lock()
X_train_filled, X_val_filled, y_train, y_val = prepare_data()

# Measure the total time for each 
seq_time = run_sequential(n_estimators_range, max_features_range, max_depth_range)
print(f"Total time taken sequentially: {seq_time} seconds")
print('-----')

thread_time = run_thread(6)
print(f"Total time taken using multithreading: {thread_time} seconds")
print('-----')

process_time = run_process(6)
print(f"Total time taken using multiprocessing: {process_time} seconds")
print('-----')

# # calculate speedup 
# thread_speedup = seq_time / thread_time
# thread_efficiency = thread_speedup / 4 # 4 is the number of processes / threads
# # estimating p and alpha from number of lines
# # p = number of parallel lines / number of total lines 
# thread_amdhal = 1 / ((1-0.87) + (0.87/6))
# thread_gustaffson = 4 + 0.13 * (1-6)

# process_speedup = seq_time / process_time
# process_efficiency = process_speedup / 4 # 4 is the number of processes / threads
# # estimating p and alpha from number of lines
# # p = number of parallel lines / number of total lines 
# process_amdhal = 1 / ((1-0.87) + (0.87/6))
# process_gustaffson = 4 + 0.13 * (1-6)


# print(f"Thread speedup rate: {thread_speedup} \nThread efficiency: {thread_efficiency} \nThread Amdhal {thread_amdhal} \nThread Gustaffson {thread_gustaffson} \n-------- \nProcess speedup rate: {process_speedup} \nProcess efficiency: {process_efficiency} \nAmdhal: {process_amdhal} \nGustaffson {process_gustaffson} \n--------")
