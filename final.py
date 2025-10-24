import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GRU model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Prediction and visualization logic
def predict_and_plot(symbol):
    df = yf.download(symbol, start="2020-01-01", end="2024-04-01")
    close_prices = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    seq_length = 60
    sequences = []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i+seq_length])

    sequences = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

    model = GRUModel(1, 50, 2, 1).to(device)
    try:
        model.load_state_dict(torch.load(f"{symbol}_gru_model.pth"))
    except:
        messagebox.showerror("Model Error", f"Model for {symbol} not found.")
        return None, None

    model.eval()
    with torch.no_grad():
        predictions = model(sequences.unsqueeze(-1)).cpu().numpy()

    predictions = scaler.inverse_transform(predictions)
    actual_prices = close_prices.values[seq_length:]

    return actual_prices, predictions

# GUI App
def run_app():
    def on_predict():
        symbol = combo.get()
        if not symbol:
            messagebox.showwarning("Input Error", "Please select a stock symbol")
            return

        actual, predicted = predict_and_plot(symbol)
        if actual is None:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(actual, label="Actual Prices", color="blue")
        ax.plot(predicted, label="Predicted Prices", color="red")
        ax.set_title(f"{symbol} Stock Price Prediction")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    root = tk.Tk()
    root.title("Stock Price Prediction App (GRU)")
    root.geometry("700x600")

    tk.Label(root, text="Select Stock Symbol", font=("Arial", 12)).pack(pady=10)
    combo = ttk.Combobox(root, values=["AMZN", "IBM", "MSFT"], font=("Arial", 12))
    combo.pack(pady=5)

    predict_button = tk.Button(root, text="Predict & Show Graph", command=on_predict,
                               font=("Arial", 12), bg="#4CAF50", fg="white")
    predict_button.pack(pady=10)

    global frame
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    root.mainloop()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the app
if __name__ == "__main__":
    run_app()
