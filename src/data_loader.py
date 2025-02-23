import pandas as pd

def load_data():
    """
    Loads the housing prices dataset from the specified file path.

    This function reads the CSV file located at "../data/housing_prices_data/train.csv"
    and returns a Pandas DataFrame with the "Id" column as the index.

    Returns:
        pd.DataFrame: The loaded dataset with "Id" as the index.
    """

    file_path = "../data/housing_prices_data/train.csv"
    return pd.read_csv(file_path, index_col="Id")