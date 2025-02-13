import time
import threading
from src.utils import train_and_evaluate

def run_thread(n):
    """
    """
    # global variable
    global param_queue
    start_time = time.time()
    # create a barrier for every n threads
    barrier = threading.Barrier(n)
    
    # launch threads
    threads = []
    while not param_queue.empty():
        for _ in range(n):  # Run 6 threads at a time
            if param_queue.empty():
                break
            n_estimators, max_features, max_depth = param_queue.get()
            thread = threading.Thread(target=train_and_evaluate, args=(n_estimators, max_features, max_depth))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    print(f"Best Parameters: {best_parameters}, Best RMSE: {best_rmse}, Best MAPE: {best_mape}%")
    end_time = time.time()
    return end_time - start_time
