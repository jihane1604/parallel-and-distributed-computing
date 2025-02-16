import time
from src.utils import train_and_evaluate
from src.preprocessing import split_data

def run_sequential():
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
    # Loop over all possible combinations of parameters
    for n_estimators in n_estimators_range:
        for max_features in max_features_range:
            for max_depth in max_depth_range:
                train_and_evaluate(n_estimators, max_features, max_depth, X_train, y_train, X_val, y_val, None)

    print(f"Best Parameters: {best_params}, Best RMSE: {best_rmse}, Best MAPE: {best_mape}%")

    # end time
    end_time = time.time()
    
    return start_time - end_time
