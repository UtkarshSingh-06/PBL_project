import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

# Load stock data (Fetch from Yahoo Finance)
def load_stock_data(stock_symbol):
    stock = yf.download(stock_symbol, start="2020-01-01", end="2024-04-01")
    stock.to_csv(f"{stock_symbol}.csv")  # Save for future use
    return stock[['Close']]

# Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Create sequences for GRU
def create_sequences(data, seq_length=60):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Train model function
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Prediction function
def predict(model, data, scaler, seq_length=60):
    model.eval()
    inputs = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(inputs).cpu().numpy()
    return scaler.inverse_transform(predicted)

# Combined visualization function for all stocks
def plot_all_predictions(stock_data, predictions, stock_symbols):
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'green', 'red']
    
    for i, stock in enumerate(stock_symbols):
        actual_prices = stock_data[stock]['Close'].values
        predicted_prices = predictions[stock]

        # Plot actual prices
        plt.plot(actual_prices, label=f'Actual {stock}', color=colors[i], linestyle='dotted')
        
        # Plot predicted prices (aligned at the end)
        plt.plot(range(len(actual_prices) - len(predicted_prices), len(actual_prices)), 
                 predicted_prices, label=f'Predicted {stock}', color=colors[i])

    plt.title('Stock Price Prediction (AMZN, IBM, MSFT)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Load and process data
stock_symbols = ['AMZN', 'IBM', 'MSFT']
seq_length = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stock_data = {}
predictions = {}

for stock in stock_symbols:
    df = load_stock_data(stock)
    stock_data[stock] = df  # Store actual stock data

    data, scaler = preprocess_data(df)
    sequences, targets = create_sequences(data, seq_length)
    
    # Convert to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(sequences, targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model initialization
    model = GRUModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, criterion, optimizer)
    
    # Save model
    torch.save(model.state_dict(), f'{stock}_gru_model.pth')
    print(f"Trained model saved for {stock}")
    
    # Predict next day's stock price
    predicted_price = predict(model, data, scaler)
    predictions[stock] = predicted_price.flatten()  # Store predicted prices

# Plot all stocks together
plot_all_predictions(stock_data, predictions, stock_symbols)



