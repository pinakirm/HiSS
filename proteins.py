import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform



# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Function to one-hot encode protein sequences
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


# Load and preprocess dataset
df = pd.read_csv("hf://datasets/SaProtHub/Dataset-Beta_Lactamase-PEER/dataset.csv")

numeric_sequences = np.array([sequence_to_numeric(seq, AMINO_ACIDS) for seq in df['protein']])

# Compute pairwise Hamming distances
pairwise_distances = squareform(pdist(numeric_sequences, metric="hamming")) * numeric_sequences.shape[1]


energies = df['label'].values

# Compute pairwise absolute energy differences
pairwise_energy_differences = np.abs(energies[:, None] - energies[None, :])


import matplotlib.pyplot as plt

# Flatten distances and energy differences
distances_flat = pairwise_distances.flatten()
energy_diff_flat = pairwise_energy_differences.flatten()

# Plot
plt.scatter(distances_flat, energy_diff_flat, alpha=0.5)
plt.xlabel("Hamming Distance")
plt.ylabel("Absolute Energy Difference")
plt.title("Energy Difference vs. Sequence Similarity")
plt.show()


import seaborn as sns

# Create a heatmap for pairwise energy differences
sns.heatmap(pairwise_energy_differences, cmap="viridis")
plt.title("Pairwise Energy Differences")
plt.show()


# Select a single sequence and find its k-nearest neighbors
sequence_index = 0
k = 10
neighbors = np.argsort(pairwise_distances[sequence_index])[:k+1]

# Plot the energy of the sequence and its neighbors
neighbor_energies = energies[neighbors]
plt.plot(range(len(neighbor_energies)), neighbor_energies, marker='o')
plt.xlabel("Neighbor Index")
plt.ylabel("Energy (U(x))")
plt.title("Local Smoothness of Energy Landscape")
plt.show()

sequences = df['protein'].tolist()
X = one_hot_encode_fast(sequences, max_length=286)

train_idx = df['stage'] == 'train'
valid_idx = df['stage'] == 'valid'
test_idx = df['stage'] == 'test'

X_train, y_train = X[train_idx], df['label'].values[train_idx]
X_valid, y_valid = X[valid_idx], df['label'].values[valid_idx]
X_test, y_test = X[test_idx], df['label'].values[test_idx]

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Validation shape: {X_valid.shape}, {y_valid.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")


# Define the neural network model
def create_model_with_dropout(input_shape, dropout_rate=0.2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


# Train multiple independent models (chains)
def train_independent_chains(X_train, y_train, X_valid, y_valid, X_test, y_test, num_chains=5, seed=42):
    train_losses = []
    valid_losses = []
    train_mse_list = []
    test_mse_list = []
    test_predictions = []

    for chain in range(num_chains):
        print(f"Training independent chain {chain + 1}/{num_chains}...")
        set_seed(seed + chain)  # Set a unique seed for each chain

        # Create and train the model
        model = create_model_with_dropout(input_shape=(286, 20), dropout_rate=0.02)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=40,
            batch_size=32,
            verbose=1
        )

        # Store training and validation losses
        train_losses.append(history.history['loss'])
        valid_losses.append(history.history['val_loss'])

        # Evaluate on train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mse_list.append(mean_squared_error(y_train, y_train_pred))
        test_mse_list.append(mean_squared_error(y_test, y_test_pred))
        test_predictions.append(y_test_pred)

    # Compute mean and std error for predictions and MSE
    avg_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0) / np.sqrt(num_chains)

    avg_valid_loss = np.mean(valid_losses, axis=0)
    std_valid_loss = np.std(valid_losses, axis=0) / np.sqrt(num_chains)

    avg_train_mse = np.mean(train_mse_list)
    std_train_mse = np.std(train_mse_list) / np.sqrt(num_chains)

    avg_test_mse = np.mean(test_mse_list)
    std_test_mse = np.std(test_mse_list) / np.sqrt(num_chains)

    avg_test_pred = np.mean(test_predictions, axis=0)

    return avg_train_loss, std_train_loss, avg_valid_loss, std_valid_loss, avg_train_mse, std_train_mse, avg_test_mse, std_test_mse, avg_test_pred


# Train 5 independent chains
(
    avg_train_loss, std_train_loss,
    avg_valid_loss, std_valid_loss,
    avg_train_mse, std_train_mse,
    avg_test_mse, std_test_mse,
    avg_test_pred
) = train_independent_chains(X_train, y_train, X_valid, y_valid, X_test, y_test, num_chains=5, seed=42)

print(f"Average Train MSE: {avg_train_mse:.4f} ± {std_train_mse:.4f}")
print(f"Average Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")

from scipy.stats import pearsonr

# Compute the correlation coefficient for predictions vs. actual values
""""
correlation, _ = pearsonr(y_test, avg_test_pred)
print(f"Correlation Coefficient: {correlation:.4f}")
"""""

# Plot averaged training and validation loss with std error
epochs = range(1, len(avg_train_loss) + 1)
plt.plot(epochs, avg_train_loss, label='Avg Train Loss', color='blue')
plt.fill_between(epochs, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, color='blue', alpha=0.2)

plt.plot(epochs, avg_valid_loss, label='Avg Validation Loss', color='orange')
plt.fill_between(epochs, avg_valid_loss - std_valid_loss, avg_valid_loss + std_valid_loss, color='orange', alpha=0.2)




plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Averaged Training and Validation Loss with Std Error ')
plt.show()

# Plot averaged predictions vs actuals for the test set
plt.scatter(y_test, avg_test_pred, alpha=0.5)
plt.xlabel('Actual U(x)')
plt.ylabel('Averaged Predicted U(x)')
plt.title('Averaged Predictions vs. Actual Values ')
plt.show()