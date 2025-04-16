import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(get_categorical_info=False):
    """
    Load and preprocess the Credit Card Fraud dataset.
    """

    # Load data
    data = pd.read_csv('./data/creditcard.csv', header=0)

    # Define features and labels
    X = data.drop(columns=['Class']) .values # Features
    y = np.where(data['Class'] == 1, 1, 0) # Labels


    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if get_categorical_info:
        start_index = []
        cat_length = []
        for i in range(X.shape[1]):  # Each feature in the dataset
            start_index.append(i)
            cat_length.append(1)
        return X_train, y_train, X_test, y_test, start_index, cat_length
    else:
        return X_train, y_train, X_test, y_test

# Example usage
