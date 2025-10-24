# predictor.py
"""
Core prediction and training utilities for the GRU Stock Forecasting project.

Provides:
- Data download & caching (yfinance)
- Technical feature creation
- Sequence building for GRU input
- GRU model (with optional attention)
- Train / save / load model (model + scaler)
- Predict and evaluation helpers
"""

from typing import Tuple, List, Optional
import os
import math
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Config / paths / device
# ---------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RND_SEED = 42
torch.manual_seed(RND_SEED)
np.random.seed(RND_SEED)


# ---------------------------
# Data utilities
# ---------------------------
def download_stock(symbol: str, start: str = "2008-01-01", end: Optional[str] = None,
                   cache: bool = True) -> pd.DataFrame:
    """Download (or load cached) OHLCV data for symbol."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if cache and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(symbol, start=start, end=end, progress=False)
        df.to_csv(csv_path)
    if "Close" not in df.columns:
        raise ValueError("Downloaded data has no 'Close' column.")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few simple technical indicators used as features."""
    df = df.copy()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["STD_10"] = df["Close"].rolling(window=10).std()
    df["RET_1"] = df["Close"].pct_change().fillna(0)
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df


def build_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Turn feature DataFrame into sequences (X) and next-day target (y)."""
    arr = df[feature_cols].values
    sequences, targets = [], []
    for i in range(len(arr) - seq_len):
        sequences.append(arr[i:i + seq_len])
        targets.append(arr[i + seq_len][feature_cols.index("Close")])  # predict Close
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32).reshape(-1, 1)


# ---------------------------
# Model definitions
# ---------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: (batch, seq_len, hidden)
        score = torch.tanh(self.W(H))                   # (batch, seq_len, hidden)
        e = self.v(score).squeeze(-1)                   # (batch, seq_len)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (H * alpha).sum(dim=1)                # (batch, hidden)
        return context, alpha


class GRUWithOptionalAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2,
                 dropout: float = 0.0, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.att = Attention(hidden_size) if use_attention else None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden)
        if self.use_attention:
            context, _ = self.att(out)
            return self.fc(context)
        else:
            return self.fc(out[:, -1, :])  # last timestep


# ---------------------------
# Save / Load helpers
# ---------------------------
def save_model_and_scaler(symbol: str, model: nn.Module, scaler: MinMaxScaler):
    path = os.path.join(MODELS_DIR, f"{symbol}_gru.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
    torch.save(model.state_dict(), path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return path, scaler_path


def load_model_and_scaler(symbol: str, input_size: int, use_attention: bool = False, map_location: str = "cpu"):
    path = os.path.join(MODELS_DIR, f"{symbol}_gru.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
    if not os.path.exists(path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file not found for symbol: " + symbol)
    # build model (must match training config)
    model = GRUWithOptionalAttention(input_size=input_size, use_attention=use_attention).to(device)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    model.eval()
    return model, scaler


# ---------------------------
# Training & prediction
# ---------------------------
def train(symbol: str,
          start: str = "2008-01-01",
          end: Optional[str] = None,
          seq_len: int = 60,
          epochs: int = 10,
          batch_size: int = 32,
          lr: float = 1e-3,
          use_attention: bool = False,
          save: bool = True) -> Tuple[str, str]:
    """
    Train a GRU for a given symbol. Saves model and scaler to models/ by default.
    Returns paths (model_path, scaler_path).
    """
    df = download_stock(symbol, start=start, end=end, cache=True)
    df = add_technical_indicators(df)
    feature_cols = ["Close", "MA_5", "MA_10", "EMA_10", "STD_10", "RET_1"]
    X, y = build_sequences(df, feature_cols, seq_len=seq_len)
    if len(X) < 10:
        raise ValueError("Not enough sequence data to train.")

    # split train/val (simple)
    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    scaler = MinMaxScaler()
    # fit scaler on training features (reshape)
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat_train)
    # transform
    X_train_scaled = scaler.transform(flat_train).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    # convert to tensors
    X_tr = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # model
    input_size = X_tr.shape[-1]
    model = GRUWithOptionalAttention(input_size=input_size, use_attention=use_attention).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # training loop
    model.train()
    num_batches = max(1, int(np.ceil(len(X_tr) / batch_size)))
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_tr))
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb = X_tr[idx]; yb = y_tr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # validation performance
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t).cpu().numpy()
            val_rmse = math.sqrt(((y_val - pred_val) ** 2).mean())
            val_mape = (np.abs((y_val - pred_val) / (y_val + 1e-8)).mean()) * 100.0
        model.train()
        print(f"[{symbol}] Epoch {epoch+1}/{epochs} train_loss={(epoch_loss / num_batches):.6f} val_rmse={val_rmse:.4f} val_mape={val_mape:.2f}%")

    # save model + scaler
    model_path, scaler_path = "", ""
    if save:
        model_path, scaler_path = save_model_and_scaler(symbol, model, scaler)
        print(f"Saved model: {model_path} scaler: {scaler_path}")

    return model_path, scaler_path


def predict(symbol: str,
            start: str = "2008-01-01",
            end: Optional[str] = None,
            seq_len: int = 60,
            use_attention: bool = False,
            return_series: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load model and scaler for symbol and return (actual_close_array, predicted_array).
    The shapes are (n,) arrays aligned so predicted[i] corresponds to actual[i + seq_len] in original data.
    """
    df = download_stock(symbol, start=start, end=end, cache=True)
    df = add_technical_indicators(df)
    feature_cols = ["Close", "MA_5", "MA_10", "EMA_10", "STD_10", "RET_1"]
    X, y = build_sequences(df, feature_cols, seq_len=seq_len)
    if len(X) == 0:
        raise ValueError("Not enough data to produce sequences for prediction.")

    # load model + scaler
    model, scaler = load_model_and_scaler(symbol, input_size=X.shape[-1], use_attention=use_attention, map_location=device)
    # scale all sequences
    flat = X.reshape(-1, X.shape[-1])
    flat_scaled = scaler.transform(flat)
    X_scaled = flat_scaled.reshape(X.shape)

    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy().reshape(-1, 1)

    # preds correspond to scaled inverse transform only for close; since scaler used multivariate transform,
    # simplest approach here: we already predicted close in the original scale, because target was not scaled.
    # (We trained the model to predict the real close because scaler was applied to features;
    # the target used the original Close value, so preds are in original close scale.)
    preds = preds.flatten()
    actual = y.flatten()
    return actual, preds


# ---------------------------
# Metrics helpers
# ---------------------------
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(((y_true - y_pred) ** 2).mean())


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100.0)


# ---------------------------
# Quick demo (if run directly)
# ---------------------------
if __name__ == "__main__":
    # simple demo: train a tiny model for MSFT (short epochs)
    sym = "MSFT"
    print("Training demo (short):", sym)
    train(sym, start="2018-01-01", end=None, epochs=3, batch_size=64, save=False)
    print("Demo predict (will fail if model not saved):")
    try:
        actual, pred = predict(sym, start="2018-01-01", end=None)
        print("Actual len:", len(actual), "Pred len:", len(pred))
        print("RMSE:", compute_rmse(actual, pred), "MAPE:", compute_mape(actual, pred))
    except Exception as e:
        print("Predict error (expected if model not saved):", e)























# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import yfinance as yf

# # Load stock data (Fetch from Yahoo Finance)
# def load_stock_data(stock_symbol):
#     stock = yf.download(stock_symbol, start="2020-01-01", end="2024-04-01")
#     stock.to_csv(f"{stock_symbol}.csv")  # Save for future use
#     return stock[['Close']]

# # Preprocess Data
# def preprocess_data(df):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df)
#     return scaled_data, scaler

# # Create sequences for GRU
# def create_sequences(data, seq_length=60):
#     sequences, targets = [], []
#     for i in range(len(data) - seq_length):
#         sequences.append(data[i:i+seq_length])
#         targets.append(data[i+seq_length])
#     return np.array(sequences), np.array(targets)

# # Define GRU Model
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1, :])

# # Train model function
# def train_model(model, train_loader, criterion, optimizer, epochs=20):
#     for epoch in range(epochs):
#         for sequences, targets in train_loader:
#             sequences, targets = sequences.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(sequences)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # Prediction function
# def predict(model, data, scaler, seq_length=60):
#     model.eval()
#     inputs = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
#         predicted = model(inputs).cpu().numpy()
#     return scaler.inverse_transform(predicted)

# # Combined visualization function for all stocks
# def plot_all_predictions(stock_data, predictions, stock_symbols):
#     plt.figure(figsize=(12, 6))

#     colors = ['blue', 'green', 'red']
    
#     for i, stock in enumerate(stock_symbols):
#         actual_prices = stock_data[stock]['Close'].values
#         predicted_prices = predictions[stock]

#         # Plot actual prices
#         plt.plot(actual_prices, label=f'Actual {stock}', color=colors[i], linestyle='dotted')
        
#         # Plot predicted prices (aligned at the end)
#         plt.plot(range(len(actual_prices) - len(predicted_prices), len(actual_prices)), 
#                  predicted_prices, label=f'Predicted {stock}', color=colors[i])

#     plt.title('Stock Price Prediction (AMZN, IBM, MSFT)')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.show()

# # Load and process data
# stock_symbols = ['AMZN', 'IBM', 'MSFT']
# seq_length = 60
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# stock_data = {}
# predictions = {}

# for stock in stock_symbols:
#     df = load_stock_data(stock)
#     stock_data[stock] = df  # Store actual stock data

#     data, scaler = preprocess_data(df)
#     sequences, targets = create_sequences(data, seq_length)
    
#     # Convert to PyTorch tensors
#     sequences = torch.tensor(sequences, dtype=torch.float32)
#     targets = torch.tensor(targets, dtype=torch.float32)
    
#     # Create DataLoader
#     train_dataset = torch.utils.data.TensorDataset(sequences, targets)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
#     # Model initialization
#     model = GRUModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Train model
#     train_model(model, train_loader, criterion, optimizer)
    
#     # Save model
#     torch.save(model.state_dict(), f'{stock}_gru_model.pth')
#     print(f"Trained model saved for {stock}")
    
#     # Predict next day's stock price
#     predicted_price = predict(model, data, scaler)
#     predictions[stock] = predicted_price.flatten()  # Store predicted prices

# # Plot all stocks together
# plot_all_predictions(stock_data, predictions, stock_symbols)



