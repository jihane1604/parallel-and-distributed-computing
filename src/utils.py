from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data():
    """
    """
    # Load the train_dataset
    file_path = '../data/housing_prices_data/train.csv'
    train_data = pd.read_csv(file_path, index_col="Id")
    
    # Columns to be deleted
    columns_to_delete = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    
    # Delete the specified columns
    train_data_cleaned = train_data.drop(columns=columns_to_delete, axis=1)
    
    # Define the input features (X) and the output (y)
    X = train_data_cleaned.drop('SalePrice', axis=1)
    y = train_data_cleaned['SalePrice']
    
    # Identify the categorical columns in X
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Initialize a LabelEncoder for each categorical column
    label_encoders = {column: LabelEncoder() for column in categorical_columns}
    
    # Apply Label Encoding to each categorical column
    for column in categorical_columns:
        X[column] = label_encoders[column].fit_transform(X[column])

    # Split the first dataset (X, y) into train and test sets with a 70% - 30% split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Fill NaN values in X_train and X_val with the median of the respective columns
    X_train_filled = X_train.fillna(X_train.median())
    X_val_filled = X_val.fillna(X_val.median())

    return X_train_filled, X_val_filled, y_train, y_val


# function to train and evaluate the random forest model
def train_and_evaluate(n_estimators, max_features, max_depth, barrier):
    """
    """
    # global variables
    global best_rmse, best_mape, best_model, best_parameters, X_train_filled, X_val_filled, y_train, y_val, lock
    
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