from flask import Flask, jsonify, send_file
import torch
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load GRU Model
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Load stock data
def load_stock_data(stock_symbol):
    stock = yf.download(stock_symbol, start="2020-01-01", end="2024-04-01")
    return stock[['Close']]

# Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Predict stock price
def predict_stock(model, data, scaler, seq_length=60):
    model.eval()
    inputs = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predicted = model(inputs).numpy()
    return scaler.inverse_transform(predicted)

# Flask route to serve predictions
@app.route('/predict', methods=['GET'])
def get_predictions():
    stock_symbols = ['AMZN', 'IBM', 'MSFT']
    predictions = {}

    for stock in stock_symbols:
        df = load_stock_data(stock)
        data, scaler = preprocess_data(df)

        model = GRUModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(f"{stock}_gru_model.pth"))
        
        predicted_price = predict_stock(model, data, scaler)
        predictions[stock] = {
            "actual": df['Close'].tolist(),  
            "predicted": predicted_price.flatten().tolist()
        }

    return jsonify(predictions)

# Flask route to serve graph image
@app.route('/plot', methods=['GET'])
def plot_stock_graph():
    stock_symbols = ['AMZN', 'IBM', 'MSFT']
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(12, 6))

    for i, stock in enumerate(stock_symbols):
        df = load_stock_data(stock)
        data, scaler = preprocess_data(df)

        model = GRUModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(f"{stock}_gru_model.pth"))

        predicted_price = predict_stock(model, data, scaler).flatten()

        plt.plot(df['Close'].values, label=f'Actual {stock}', color=colors[i], linestyle='dotted')
        plt.plot(range(len(df['Close']) - len(predicted_price), len(df['Close'])), 
                 predicted_price, label=f'Predicted {stock}', color=colors[i])

    plt.title('Stock Price Prediction (AMZN, IBM, MSFT)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.savefig("static/stock_graph.png")  # Save the plot
    return send_file("static/stock_graph.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

