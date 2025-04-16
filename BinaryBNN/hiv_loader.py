import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def one_hot_encode_sequences(sequences):
    """
    One-hot encodes amino acid sequences.
    Each sequence (string of 8 amino acids) is split into characters,
    and each character is one-hot encoded.
    """
    amino_acids = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]  # List of standard amino acids
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Initialize one-hot encoded array
    one_hot = np.zeros((len(sequences), len(sequences[0]), len(amino_acids)))

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in aa_to_index:  # Skip invalid amino acids
                one_hot[i, j, aa_to_index[aa]] = 1

    # Flatten each sequence into a single vector
    return one_hot.reshape(len(sequences), -1)

def load_data( get_categorical_info=False):
    """
    Load and preprocess the 1625Data.txt file.
    """
    amino_acids = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]  # List of standard amino acids
    # Load data from file
    data = pd.read_csv('./data/1625.txt', header=None, names=['sequence', 'label'])

    # Separate features (octamers) and labels
    sequences = data['sequence']
    labels = data['label']

    # One-hot encode the octamer sequences
    X_encoded = one_hot_encode_sequences(sequences)

    # Encode labels as binary (cleaved=-1 -> 0, not cleaved=+1 -> 1)
    y_encoded = np.where(labels == -1, 0, 1)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    if get_categorical_info:
        # Return categorical information for encoded sequences
        start_index = []
        cat_length = []
        for i in range(len(sequences[0])):  # Each position in the sequence
            start_index.append(i * len(amino_acids))
            cat_length.append(len(amino_acids))
        return X_train, y_train, X_test, y_test, start_index, cat_length
    else:
        return X_train, y_train, X_test, y_test

#