import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Define the tickers to fetch data for
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'ADBE', 'INTC', 'CSCO', 'IBM']

# Fetch data for each ticker using Yahoo Finance API
data = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    data[ticker] = stock.history(period='5 years')['Close']

# Combine data for all stocks into a single dataset
df = pd.concat(data.values(), axis=1, keys=data.keys())

# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values.reshape(-1,1))

# split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# create input/output arrays for training and testing data
def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

lookback = 60
x_train, y_train = create_dataset(train_data, lookback)
x_test, y_test = create_dataset(test_data, lookback)

# Build Model
model = Sequential() 
model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# make predictions and evaluate performance
predictions = model.__call__(x_test)
predictions = scaler.inverse_transform(predictions)
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# pickle data
model.save("lstm_model")

last_60_days = scaled_data[-60:]
X_test = []
X_test.append(last_60_days)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.__call__(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print('Next day\'s predicted stock price:', predicted_price)

plt.plot(y_test, label=True)
plt.plot(predictions, label="LSTM Value")
plt.plot(predicted_price, label="Predicted Price Value")
plt.title("Prediction by LSTM")
plt.xlabel("Time Scale")
plt.ylabel("USD")
plt.legend()
plt.show()
