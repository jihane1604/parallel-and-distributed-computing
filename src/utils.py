import threading
import multiprocessing
from queue import Queue
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# function to train and evaluate the random forest model
def train_and_evaluate(n_estimators, max_features, max_depth, best, 
                       X_train, y_train, X_val, y_val, lock, barrier):
    """
    Trains and evaluates a Random Forest model with the given hyperparameters.

    This function initializes and fits a Random Forest model with the specified values of 
    n_estimators, max_features, and max_depth. It then evaluates the model on the validation set 
    and computes the RMSE and MAPE metrics. If the model's performance is better than the previous best,
    it updates the best parameters and metrics.

    Args:
        n_estimators (int): The number of trees in the Random Forest model.
        max_features (str or int or None): The number of features to consider for the best split.
        max_depth (int or None): The maximum depth of the trees in the model.
        best (dict): A dictionary that holds the best model parameters and performance metrics.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation labels.
        lock (multiprocessing.Lock or threading.Lock, optional): A lock to manage access to shared resources.
        barrier (multiprocessing.Barrier or threading.Barrier, optional): A barrier to synchronize multiple processes or threads.
    """  
    # train the model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # make predictions and compute RMSE
    y_val_pred = rf_model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    if barrier:
        with lock:
            if rmse < best["best_rmse"]:
                best['best_rmse'] = rmse
                best['best_mape'] = mape
                best['best_n_estimators'] = n_estimators
                best['best_max_features'] = max_features
                best['best_max_depth'] = max_depth
        
        # wait for other threads before continuing
        barrier.wait()
    else:
        if rmse < best["best_rmse"]:
            best['best_rmse'] = rmse
            best['best_mape'] = mape
            best['best_n_estimators'] = n_estimators
            best['best_max_features'] = max_features
            best['best_max_depth'] = max_depth