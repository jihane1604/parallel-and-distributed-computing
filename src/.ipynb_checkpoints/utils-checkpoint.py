# Function to add numbers from 1 to n
def train_and_evaluate(n_estimators, max_features, max_depth):
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
    rf_model.fit(X_train_filled, y_train)
    
    # make predictions and compute RMSE
    y_val_pred = rf_model.predict(X_val_filled)
    rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    
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