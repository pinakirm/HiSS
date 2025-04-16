import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def sequence_to_numeric(seq, amino_acids):
    return np.array([amino_acids.index(aa) for aa in seq])


def one_hot_encode_fast(sequences, max_length):
    num_sequences = len(sequences)
    encoded = np.zeros((num_sequences, max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            if aa in AA_TO_IDX:
                encoded[i, j, AA_TO_IDX[aa]] = 1
    return encoded



def load_data(get_categorical_info=False):
    # Load data from file
    df = pd.read_csv("hf://datasets/SaProtHub/Dataset-Beta_Lactamase-PEER/dataset.csv")
    df.drop(columns=['stage'])

    sequences = df['protein'].tolist()
    X = one_hot_encode_fast(sequences, max_length=286)
    X = X.reshape(X.shape[0], -1)


    # Separate features and labels
    y  = df['label'].values


    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test