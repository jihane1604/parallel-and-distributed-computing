import multiprocessing 
import time
from src.utils import train_and_evaluate
from src.preprocessing import split_data

def run_process(n):
    """
    """
    # start time 
    start_time = time.time()
    
    # Define the parameter ranges
    n_estimators_range = [10, 25, 50, 100, 200, 300, 400]
    max_features_range = ['sqrt', 'log2', None]  # None means using all features
    max_depth_range = [1, 2, 5, 10, 20, None]  # None means no limit
    # define best parameters
    manager = multiprocessing.Manager()
    best = manager.dict({
        "best_rmse": float('inf'),
        "best_mape": float('inf'),
        "best_model": None,
        "best_n_estimators": None,
        "best_max_features": None,
        "best_max_depth": None
    })

    X_train, X_val, y_train, y_val = split_data()
    
    # Loop over all possible combinations of parameters and put them in a queue
    param_queue = multiprocessing.Queue()
    for n_estimators in n_estimators_range:
        for max_features in max_features_range:
            for max_depth in max_depth_range:
                param_queue.put((n_estimators, max_features, max_depth))
    
    # create a barrier for every n processes
    barrier = multiprocessing.Barrier(n)
    
    # launch processes
    processes = []
    lock = multiprocessing.Lock()
    # sender, receiver = multiprocessing.pipe()
    while not param_queue.empty():
        for _ in range(n):  # Run n processes at a time
            if param_queue.empty():
                break
            params = param_queue.get()
            process = multiprocessing.Process(target=train_and_evaluate, 
                                              args=(*params, best, X_train, y_train, X_val, y_val, lock, barrier))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
            
    print(f"Best Parameters: n_estimators={best['best_n_estimators']}, "
          f"max_features={best['best_max_features']}, "
          f"max_depth={best['best_max_depth']}")
    print(f"Best RMSE: {best['best_rmse']:.2f}, "
          f"Best MAPE: {best['best_mape']:.2f}%")

    # end time
    end_time = time.time()
    return end_time - start_time