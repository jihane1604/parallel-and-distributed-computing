import threading
import time
from src.utils import train_and_evaluate
from src.preprocessing import split_data
from queue import Queue

def run_thread(n):
    """
    """
    # start time 
    star_time = time.time()

    # Define the parameter ranges
    n_estimators_range = [10, 25, 50, 100, 200, 300, 400]
    max_features_range = ['sqrt', 'log2', None]  # None means using all features
    max_depth_range = [1, 2, 5, 10, 20, None]  # None means no limit
    # define best parameters
    best_rmse = float('inf')
    best_mape = float('inf')
    best_model = None
    best_params = {}

    X_train, X_val, y_train, y_val = split_data()
    
    # Loop over all possible combinations of parameters and put them in a queue
    param_queue = Queue()
    for n_estimators in n_estimators_range:
        for max_features in max_features_range:
            for max_depth in max_depth_range:
                param_queue.put((n_estimators, max_features, max_depth))
    
    # create a barrier for every n threads
    barrier = threading.Barrier(n)
    
    # launch threads
    threads = []
    while not param_queue.empty():
        for _ in range(n):  # Run n threads at a time
            if param_queue.empty():
                break
            n_estimators, max_features, max_depth = param_queue.get()
            thread = threading.Thread(target=train_and_evaluate, 
                                      args=(n_estimators, max_features, max_depth, X_train, y_train, X_val, y_val, barrier))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    print(f"Best Parameters: {best_params}, Best RMSE: {best_rmse}, Best MAPE: {best_mape}%")
    
    end_time = time.time()
    return end_time - start_time
