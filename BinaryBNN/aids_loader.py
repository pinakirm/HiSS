import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_data( get_categorical_info=False):
    """
    Preprocess the AIDS Clinical Trials Group Study 175 dataset.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        get_categorical_info (bool): Whether to return categorical information.

    Returns:
        X_train, y_train, X_test, y_test: Training and test sets.
        (Optional) start_index, cat_length: Categorical feature information.
    """
    # Define column names
    columns = [
        'pidnum', 'cid', 'time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof',
        'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender', 'str2', 'strat',
        'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'cd820'
    ]

    # Load the dataset
    aids_data = fetch_ucirepo(id=890).data

    # Drop ID column
    X = aids_data.features

    # Separate features and target
    y = aids_data.targets   # Target column: censoring indicator (1=failure, 0=censoring)


    # One-hot encode categorical features
    categorical_columns = ['trt', 'hemo', 'homo', 'drugs', 'oprior', 'z30', 'zprior',
                           'race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt']
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    for col in X_encoded.select_dtypes(include=['bool']).columns:
        X_encoded[col] = X_encoded[col].astype(int)
    # Convert to NumPy arrays
    X_encoded = X_encoded.values
    y = y.values.ravel()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42)

    if get_categorical_info:
        # Gather categorical information
        start_index = []
        cat_length = []
        start = 0
        for col in X.columns:
            if col in categorical_columns:
                unique_vals = len(pd.get_dummies(X[col], drop_first=False).columns)
                start_index.append(start)
                cat_length.append(unique_vals)
                start += unique_vals
            else:
                start += 1
        return X_train, y_train, X_test, y_test, start_index, cat_length
    else:
        return X_train, y_train, X_test, y_test