from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.data_loader import load_data

def split_data():
    """
    Splits the dataset into training and validation sets.

    This function first calls the `clean_data` function to prepare the data by removing
    unwanted columns and encoding categorical variables. Then, it splits the dataset into
    training and validation sets (70% - 30%), and fills any missing values in the training
    and validation data using the median of each column.

    Returns:
        X_train_filled (pd.DataFrame): The cleaned and filled training feature set.
        X_val_filled (pd.DataFrame): The cleaned and filled validation feature set.
        y_train (pd.Series): The training labels (target values).
        y_val (pd.Series): The validation labels (target values
        """
    X, y = clean_data()
    # Split the first dataset (X, y) into train and test sets with a 70% - 30% split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Fill NaN values in X_train and X_val with the median of the respective columns
    X_train_filled = X_train.fillna(X_train.median())
    X_val_filled = X_val.fillna(X_val.median())
    return X_train_filled, X_val_filled, y_train, y_val

def clean_data():
    """
    Cleans the dataset by removing unwanted columns and encoding categorical features.

    This function loads the dataset, drops irrelevant columns, and separates the features (X)
    from the target variable (SalePrice). It also encodes categorical features using LabelEncoder.

    Returns:
        X (pd.DataFrame): The cleaned feature set with encoded categorical columns.
        y (pd.Series): The target variable (SalePrice) from the dataset.
    """
    data = load_data()
    columns_to_delete = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    data = data.drop(columns=columns_to_delete, axis=1)
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    # encoding categorical columns 
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col in categorical_columns:
        X[col] = label_encoders[col].fit_transform(X[col])
    return X, y
