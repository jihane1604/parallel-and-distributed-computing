import threading
import time
from src.utils import train_and_evaluate
from queue import Queue

def run_thread(n):
    """
    """
    # global variables
    global n_estimators_range, max_features_range, max_depth_range, best_rmse, best_mape, best_model, best_parameters
    
    # start time 
    start_time = time.time()
    
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
            thread = threading.Thread(target=train_and_evaluate, args=(n_estimators, max_features, max_depth, barrier))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    print(f"Best Parameters: {best_parameters}, Best RMSE: {best_rmse}, Best MAPE: {best_mape}%")
    
    end_time = time.time()
    return end_time - start_time
