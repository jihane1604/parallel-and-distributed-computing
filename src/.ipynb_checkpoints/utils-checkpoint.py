import threading
import multiprocessing
from queue import Queue
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

lock = threading.Lock()
best_rmse = float('inf')
best_mape = float('inf')
best_model = None
best_parameters = {}

# function to train and evaluate the random forest model
def train_and_evaluate(n_estimators, max_features, max_depth, X_train, y_train, X_val, y_val, barrier):
    """
    """
    # global variables
    global best_rmse, best_mape, best_model, best_parameters
    
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
            if rmse < best_rmse:
                best_rmse = rmse
                best_mape = mape
                best_model = rf_model
                best_parameters = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth
                }
        
        # wait for other threads before continuing
        barrier.wait()
    else:
        if rmse < best_rmse:
            best_rmse = rmse
            best_mape = mape
            best_model = rf_model
            best_parameters = {
                'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth
            }