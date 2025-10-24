import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import time

# ---------- GRU Model ----------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ---------- Safe Download with Retry ----------
def safe_download(symbol, start, end, retries=3, delay=2):
    """Download stock data with retry mechanism."""
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, threads=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(delay)
    raise ValueError(f"No data found for {symbol}. Check date range or internet connection.")

# ---------- Train Model ----------
def train_model(symbol, start, end):
    df = safe_download(symbol, start, end)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)

    seq_len = 60
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    X = torch.tensor(np.array(X), dtype=torch.float32).view(-1, seq_len, 1).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)

    model = GRUModel(1, 50, 2, 1).to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"{symbol} Train Epoch {epoch+1}: {loss.item():.6f}")

    torch.save(model.state_dict(), f"{symbol}_gru_model.pth")
    messagebox.showinfo("Training Complete", f"Model trained and saved for {symbol}")

# ---------- Predict Data ----------
def get_predictions(symbol, start, end):
    df = safe_download(symbol, start, end)
    df = df[['Close']].dropna()

    if len(df) < 100:
        raise ValueError("Not enough data available.")

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)

    seq_len = 60
    seqs = [data[i:i+seq_len] for i in range(len(data)-seq_len)]
    X = torch.tensor(np.array(seqs), dtype=torch.float32).view(-1, seq_len, 1).to(device)

    model = GRUModel(1, 50, 2, 1).to(device)
    model.load_state_dict(torch.load(f"{symbol}_gru_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(X).cpu().numpy()

    preds = scaler.inverse_transform(preds)
    actual = df.values[seq_len:]
    return actual.flatten(), preds.flatten()

# ---------- GUI with Dark Mode & Animation ----------
def run_app():
    root = tk.Tk()
    root.title("📈 GRU Stock Predictor")
    root.geometry("980x800")
    dark = tk.BooleanVar(value=False)

    # ---------- Style ----------
    style = ttk.Style()
    style.theme_use('default')

    def apply_theme():
        bg = '#263238' if dark.get() else '#e1f5fe'
        fg = 'white' if dark.get() else 'black'
        root.configure(bg=bg)
        title.config(bg=bg, fg=fg)
        for lbl in labels:
            lbl.config(bg=bg, fg=fg)
        stock_label.config(bg=bg, fg='#80CBC4' if dark.get() else '#1B5E20')
        graph_frame.config(bg=bg)

    # ---------- UI Components ----------
    toggle = ttk.Checkbutton(root, text="Dark Mode", variable=dark, command=apply_theme)
    toggle.pack(anchor='ne', padx=10, pady=5)

    title = tk.Label(root, text="Stock Price Forecasting with GRU", font=("Segoe UI", 18, 'bold'))
    title.pack(pady=10)

    labels = []
    lbl1 = tk.Label(root, text="Symbol (e.g., AAPL):"); lbl1.pack(); labels.append(lbl1)
    combo = ttk.Combobox(root, values=["AMZN","IBM","MSFT","GOOGL","AAPL","TSLA","META","NFLX","NVDA","ADBE"])
    combo.pack(pady=5)

    stock_label = tk.Label(root, text="", font=("Segoe UI", 12)); stock_label.pack(pady=5)

    lbl2 = tk.Label(root, text="Start Date (YYYY-MM-DD):"); lbl2.pack(); labels.append(lbl2)
    start_entry = tk.Entry(root); start_entry.pack(pady=5)

    lbl3 = tk.Label(root, text="End Date (YYYY-MM-DD):"); lbl3.pack(); labels.append(lbl3)
    end_entry = tk.Entry(root); end_entry.pack(pady=5)

    ttk.Button(root, text="Train Model",
               command=lambda: train_model(combo.get(), start_entry.get(), end_entry.get())).pack(pady=5)
    ttk.Button(root, text="Predict & Animate", command=lambda: on_predict()).pack(pady=5)

    graph_frame = tk.Frame(root); graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
    apply_theme()

    # ---------- Prediction Logic ----------
    def on_predict():
        symbol = combo.get().strip()
        s, e = start_entry.get(), end_entry.get()
        if not symbol or not s or not e:
            messagebox.showwarning("Missing Input", "Please fill all fields.")
            return

        try:
            info = yf.Ticker(symbol).info
            price = info.get('regularMarketPrice', None)
            change = info.get('regularMarketChangePercent', None)
            if price:
                stock_label.config(text=f"Price: ${price} | Change: {change:.2f}%" if change else f"Price: ${price}")
        except:
            stock_label.config(text="Unable to fetch stock info.")

        try:
            actual, pred = get_predictions(symbol, s, e)
        except Exception as ex:
            messagebox.showerror("Data Error", str(ex))
            return

        for w in graph_frame.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(8,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(f"{symbol} Prediction", color='white' if dark.get() else 'black')
        ax.set_facecolor('#37474F' if dark.get() else 'white')
        line1, = ax.plot([], [], color='#80CBC4' if dark.get() else '#1976D2', label='Actual')
        line2, = ax.plot([], [], color='#FF8A65' if dark.get() else '#E53935', label='Predicted')
        ax.legend(facecolor=('#263238' if dark.get() else 'white'),
                  labelcolor=('white' if dark.get() else 'black'))
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        n = len(actual)

        # Smooth animation that stops at the end
        def animate(i=0):
            if i <= n:
                line1.set_data(range(i), actual[:i])
                line2.set_data(range(i), pred[:i])
                ax.relim(); ax.autoscale_view()
                canvas.draw()
                root.after(15, lambda: animate(i+1))

        animate()

    root.mainloop()

# ---------- Entry Point ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    run_app()
