import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
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

def create_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X, y, epochs=50, batch_size=32)
    return model


def test_model(X, y, scaler, stock_data, model):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    test_dates = stock_data.index[-100:]
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
    scaled_data, stock_data, scaler = import_data(stocks, start_date, end_date)
    X, y = create_sequences(scaled_data)
    X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = create_model(X_train, y_train)
    test_dates, y_test_rescaled, predictions = test_model(X_test, y_test, scaler, stock_data, model)
    plot_results(test_dates, y_test_rescaled, predictions)

if __name__ == "__main__":
    main()