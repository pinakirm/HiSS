import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    # Load data from filde
    wine_red = pd.read_csv('./data/winequality-red.csv', delimiter=';')
    wine_white = pd.read_csv('./data/winequality-white.csv', delimiter=';')

    wine_red['type'] = 0  # 0 for red wine
    wine_white['type'] = 1  # 1 for white wine

    # Combine datasets
    wine_combined = pd.concat([wine_red, wine_white], ignore_index=True)

    # Separate features and target variable
    X = wine_combined.drop(columns=['type'])
    y = wine_combined['type']


    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    # Check the processed data shapes
    return X_train, y_train, X_test, y_test