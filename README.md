# 📈 Stock Market Forecasting & Visualization Web App

This project predicts stock prices using a GRU-based deep learning model built with PyTorch, visualizes the predictions with Chart.js, and serves them on a dynamic webpage using Flask.

---

## 🚀 Features
- Real-time stock data fetching from Yahoo Finance (using `yfinance`)
- Data preprocessing and scaling (with `MinMaxScaler`)
- Deep learning model using GRU (Gated Recurrent Units)
- Model training and prediction for multiple stocks (Amazon, IBM, Microsoft)
- Prediction graph generation using `matplotlib`
- REST API using Flask
- Interactive front-end displaying prediction charts with `Chart.js`

---

## 📊 Tech Stack

### Backend (Python)
- **PyTorch**: GRU model definition, training, prediction
- **Pandas & NumPy**: Data loading and manipulation
- **yFinance**: Downloading historical stock data
- **Scikit-learn**: MinMaxScaler for normalization
- **Matplotlib**: Generating stock prediction plots
- **Flask**: Serving the trained model and predictions via API endpoints

### Frontend (Web)
- **HTML5**: Webpage structure
- **CSS**: Basic styling
- **JavaScript (Vanilla)**: Fetching prediction data from Flask API
- **Chart.js**: Displaying interactive charts for each stock

---

## 📂 File Structure
```
├── app.py                    # Flask app API
├── templates
│   └── index.html            # Main webpage
├── static
│   └── script.js             # JavaScript logic
├── AMZN_gru_model.pth       # Saved PyTorch model (Amazon)
├── IBM_gru_model.pth        # Saved PyTorch model (IBM)
├── MSFT_gru_model.pth       # Saved PyTorch model (Microsoft)
├── AMZN_predictions.json     # Predictions JSON file
├── IBM_predictions.json      # Predictions JSON file
├── MSFT_predictions.json     # Predictions JSON file
├── plot_AMZN.png            # Predicted graph (optional)
├── train_predict.py         # ML script to train and predict
└── README.md                # Project documentation
```

---

## 🔧 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train_predict.py
```

### 3. Run Flask Server
```bash
python app.py
```

### 4. Open Webpage
Visit `http://127.0.0.1:5000` in your browser to view stock prediction charts.

---

## 🌐 Optional Future Upgrades
- Dockerize the app for containerized deployment
- Use a database (e.g., PostgreSQL) to store historical data
- Deploy on AWS/GCP with Nginx reverse proxy
- Add user login & dashboard for stock subscriptions

---


