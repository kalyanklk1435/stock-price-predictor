import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download stock data (Example: Apple)
data = yf.download("AAPL", start="2015-01-01", end="2023-01-01")

# Use only Close prices
data = data[['Close']]

# Create prediction column
data['Prediction'] = data[['Close']].shift(-30)

# Prepare dataset
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future prices
x_future = np.array(data.drop(['Prediction'], axis=1))[-30:]
forecast = model.predict(x_future)

print("Next 30 days prediction:")
print(forecast)

# Plot
plt.figure(figsize=(10,6))
plt.plot(data['Close'])
plt.title("Stock Price History")
plt.show()