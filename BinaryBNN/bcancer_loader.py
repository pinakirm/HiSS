import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    # Load data from file
    columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
    breast_cancer_data = pd.read_csv('./data/wdbc.data', header=None, names=columns)

    # Drop the ID column
    breast_cancer_data = breast_cancer_data.drop(columns=['ID'])

    # Encode Diagnosis as binary (Malignant=1, Benign=0)
    y = np.where(breast_cancer_data['Diagnosis'] == 'M', 1, 0)
    X = breast_cancer_data.drop(columns=['Diagnosis']).values  # Convert DataFrame to NumPy array

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test