import pandas as pd

def load_data():
    """Loads dataset from the given file path."""
    file_path = "../data/housing_prices_data/train.csv"
    return pd.read_csv(file_path, index_col="Id")