import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def import_data(stocks, start_date, end_date):
    stock_data = yf.download(stocks, start=start_date, end=end_date)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)

    return scaled_data, stock_data, scaler

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(X_train, y_train, input_size=1, hidden_size=50, num_layers=2, dropout=0.2, epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    y_train_tensor = y_train_tensor.view(-1, 1)
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    return model

def test_model(X, y, scaler, stock_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    X_tensor = X_tensor.unsqueeze(-1)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    test_dates = stock_data.index[-len(X):]
    y_test_rescaled = scaler.inverse_transform(y.reshape(-1, 1))
    return test_dates, y_test_rescaled, predictions

def plot_results(test_dates, y_test_rescaled, predictions):
    plt.figure(figsize=(12, 6), facecolor='black')
    plt.plot(test_dates, y_test_rescaled, color='blue', label='Real Stock Price')
    plt.plot(test_dates, predictions, color='red', label='Predicted Stock Price')
    plt.title('PANW stock price prediction', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Stock Price', color='white')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    stocks = 'PANW'
    start_date = '2000-01-01'
    end_date = '2025-02-05'
    print("Importing data...", end='\n\n')
    scaled_data, stock_data, scaler = import_data(stocks, start_date, end_date)
    print("Creating sequences...", end='\n\n')
    X, y = create_sequences(scaled_data)
    print("Splitting data...", end='\n\n')
    X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print("Training model...", end='\n\n')
    model = train_model(X_train, y_train)
    print("Testing model...", end='\n\n')
    test_dates, y_test_rescaled, predictions = test_model(X_test, y_test, scaler, stock_data, model)
    print("Plotting results...", end='\n\n')
    plot_results(test_dates, y_test_rescaled, predictions)

if __name__ == "__main__":
    main()