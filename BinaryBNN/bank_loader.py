import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def one_hot_encode_features(X):
    """
    One-hot encodes categorical features in the dataset.
    """
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=False)
    return X_encoded.values, X_encoded.columns


def preprocess_bank_data(file_path, get_categorical_info=False):
    """
    Preprocesses the Bank Marketing dataset.
    """
    # Load the dataset
    bank_data = pd.read_csv(file_path, delimiter=';')

    # Separate features and labels
    y = bank_data['y']
    X = bank_data.drop(columns=['y'])

    # Encode the target variable as binary (no=0, yes=1)
    y = np.where(y == 'yes', 1, 0)

    # One-hot encode categorical features
    X_encoded, encoded_columns = one_hot_encode_features(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    if get_categorical_info:
        start_index = []
        cat_length = []
        start = 0
        for col in encoded_columns:
            start_index.append(start)
            cat_length.append(1)  # For one-hot encoding, each feature becomes a single column
            start += 1
        return X_train, y_train, X_test, y_test, start_index, cat_length
    else:
        return X_train, y_train, X_test, y_test


# File path to the dataset
file_path = './data/bank-full.csv'

# Preprocess the dataset
X_train, y_train, X_test, y_test = preprocess_bank_data(file_path)

# Display the shapes of the training and testing sets
print("Training features shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing features shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)