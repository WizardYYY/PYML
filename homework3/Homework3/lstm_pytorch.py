# -*- coding: utf-8 -*-
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Function to preprocess data
def preprocess_data(df, scaler=None, is_training=True):
    """Preprocess data for LSTM model

    Args:
        df: DataFrame to preprocess
        scaler: Fitted scaler if processing test data, None otherwise
        is_training: Whether this is training data

    Returns:
        Processed data and scaler if training
    """
    # Create a copy to avoid modifying original data
    df_scaled = df.copy()

    # Map wind direction to numerical values
    mapping = {'NE': 0, 'SE': 1, 'NW': 2, 'cv': 3}
    df_scaled['wnd_dir'] = df_scaled['wnd_dir'].map(mapping)

    # Convert and set date as index if it exists
    if 'date' in df_scaled.columns:
        df_scaled['date'] = pd.to_datetime(df_scaled['date'])
        df_scaled.set_index('date', inplace=True)

    # Select relevant columns
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    df_scaled = df_scaled[columns]

    # Scale data
    if is_training:
        scaler = MinMaxScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, scaler
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for test data")
        df_scaled[columns] = scaler.transform(df_scaled[columns])
        return df_scaled


# Function to create sequences for LSTM
def create_sequences(data, n_past, n_future):
    """Create sequences for LSTM model

    Args:
        data: Numpy array of data
        n_past: Number of past time steps
        n_future: Number of future time steps

    Returns:
        X and y sequences
    """
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, 1:])
        y.append(data[i + n_future - 1:i + n_future, 0])
    return np.array(X), np.array(y)


# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)

        # Select device (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # LSTM and fully connected layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # h0= nn.init.orthogonal_(h0)
        # c0 = nn.init.orthogonal_(c0)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Apply dropout to prevent overfitting
        out = self.dropout(out)

        # Fully connected layer to get final output
        out = self.fc(out)
        return out


# Function to train one epoch
def train_epoch(net, train_iter, optimizer, loss_fn):
    """Train model for one epoch

    Args:
        net: Neural network model
        train_iter: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function

    Returns:
        Average loss and evaluation metrics
    """
    net.train()
    train_loss = []
    predictions = []
    targets = []

    loop = tqdm(train_iter, desc='Train')
    device = next(net.parameters()).device

    for X, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        train_loss.append(loss.item())

        # Store predictions and targets for metrics
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Calculate metrics
    rmse = sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    return sum(train_loss) / len(train_loss), rmse, mae


# Function to evaluate model
@torch.no_grad()
def eval_model(net, test_iter, loss_fn):
    """Evaluate model on test data

    Args:
        net: Neural network model
        test_iter: Test data loader
        loss_fn: Loss function

    Returns:
        Average loss and evaluation metrics
    """
    net.eval()
    test_loss = []
    predictions = []
    targets = []

    loop = tqdm(test_iter, desc='Test')
    device = next(net.parameters()).device

    for X, y in loop:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        test_loss.append(loss.item())

        # Store predictions and targets for metrics
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())

    # Calculate metrics
    rmse = sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    return sum(test_loss) / len(test_loss), rmse, mae


# Function to train the model
def train_model(model, train_loader, test_loader, epochs=20, patience=3, lr=0.001):
    """Train the model

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate

    Returns:
        Training history and best model
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_losses, train_rmses, train_maes = [], [], []
    test_losses, test_rmses, test_maes = [], [], []

    best_rmse = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train and evaluate
        train_loss, train_rmse, train_mae = train_epoch(model, train_loader, optimizer, loss_fn)
        test_loss, test_rmse, test_mae = eval_model(model, test_loader, loss_fn)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Record training time
        train_time = time.time() - start_time

        # Store metrics
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        train_maes.append(train_mae)
        test_losses.append(test_loss)
        test_rmses.append(test_rmse)
        test_maes.append(test_mae)

        # Print progress
        print(f"Epoch: {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | "
              f"Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f} | "
              f"Train MAE: {train_mae:.6f} | Test MAE: {test_mae:.6f} | "
              f"Time: {train_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping based on RMSE
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Test RMSE: {best_rmse:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Save best model
    torch.save(best_model_state, 'best_lstm_model.pth')
    print("Best model saved as 'best_lstm_model.pth'")

    return {
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'train_maes': train_maes,
        'test_losses': test_losses,
        'test_rmses': test_rmses,
        'test_maes': test_maes
    }


# Function to plot learning curves
def plot_learning_curves(history):
    """Plot learning curves

    Args:
        history: Training history dictionary
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    # Plot losses
    axs[0].plot(history['train_losses'], label='Train Loss', color='blue')
    axs[0].plot(history['test_losses'], label='Test Loss', color='green')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss (MSE)')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    # Plot RMSEs
    axs[1].plot(history['train_rmses'], label='Train RMSE', color='blue')
    axs[1].plot(history['test_rmses'], label='Test RMSE', color='green')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('RMSE Curves')
    axs[1].legend()

    # Plot MAEs
    axs[2].plot(history['train_maes'], label='Train MAE', color='blue')
    axs[2].plot(history['test_maes'], label='Test MAE', color='green')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MAE')
    axs[2].set_title('MAE Curves')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


# Main function
def main():
    """Main function to run the entire pipeline"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    print("Loading data...")
    df_train = pd.read_csv('LSTM-Multivariate_pollution.csv')
    df_test = pd.read_csv('pollution_test_data1.csv.xls')

    # Check for missing values
    print("\nMissing values in training data:")
    print(df_train.isnull().sum())
    print("\nMissing values in test data:")
    print(df_test.isnull().sum())

    # Data summary
    print("\nTraining data summary:")
    print(df_train.describe())

    # Preprocess data
    print("\nPreprocessing data...")
    df_train_scaled, scaler = preprocess_data(df_train, is_training=True)
    df_test_scaled = preprocess_data(df_test, scaler=scaler, is_training=False)

    # Convert to numpy arrays
    df_train_scaled_np = np.array(df_train_scaled)
    df_test_scaled_np = np.array(df_test_scaled)

    # Create sequences
    print("\nCreating sequences...")
    n_past = 10
    n_future = 1
    X_train, y_train = create_sequences(df_train_scaled_np, n_past, n_future)
    X_test, y_test = create_sequences(df_test_scaled_np, n_past, n_future)

    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create data loaders
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Define model parameters
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64  # Increased from 32
    output_size = 1
    num_layers = 2
    dropout_rate = 0.3  # Reduced from 0.5

    # Create model
    print("\nCreating LSTM model...")
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    # Print model summary
    print("\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=30,  # Increased from 20
        patience=5,  # Increased from 3
        lr=0.001
    )

    # Plot learning curves
    print("\nPlotting learning curves...")
    plot_learning_curves(history)


if __name__ == '__main__':
    main()





















