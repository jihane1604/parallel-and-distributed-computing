import time
from src.utils import train_and_evaluate

def run_sequential(n_estimators_range, max_features_range, max_depth_range):
    """
    """
    # global variables
    global best_rmse, best_mape, best_model, best_parameters
    
    # start time 
    star_time = time.time()
    
    # Loop over all possible combinations of parameters
    for n_estimators in n_estimators_range:
        for max_features in max_features_range:
            for max_depth in max_depth_range:
                train_and_evaluate(n_estimators, max_features, max_depth, None)

    print(f"Best Parameters: {best_parameters}, Best RMSE: {best_rmse}, Best MAPE: {best_mape}%")
                    
    end_time = time.time()
    return start_time - end_time
