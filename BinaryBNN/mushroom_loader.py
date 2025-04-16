import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def one_hot_encode_features(X):
    """
    One-hot encodes categorical features in the dataset.
    """
    X_encoded = pd.get_dummies(X, columns=X.columns, drop_first=False)
    return X_encoded.values, X_encoded.columns

def load_data(get_categorical_info=False):
    # Load data from file
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    mushroom_data = pd.read_csv('./data/mushrooms.data', header=None, names=columns)

    # Separate features and labels
    y = mushroom_data['class']
    X = mushroom_data.drop(columns=['class'])

    # Encode labels as binary (edible=0, poisonous=1)
    y = np.where(y == 'e', 0, 1)

    # One-hot encode categorical features
    X_encoded, encoded_columns = one_hot_encode_features(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


    if get_categorical_info:
        start_index = []
        cat_length = []
        start = 0
        for col in encoded_columns:
            start_index.append(start)
            cat_length.append(1)
            start += 1
        return X_train, y_train, X_test, y_test, start_index, cat_length
    else:
        return X_train, y_train, X_test, y_test