import datetime
import math
import pickle
import requests
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pandas.tseries.offsets import BDay  # For business days

# Set your Alpha Vantage API key
API_KEY = '89SN94A059MA1JLL'  # Replace with your API key

# Function to fetch stock data from Alpha Vantage
def fetch_alpha_vantage_data(symbol, start_date, end_date):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full&datatype=csv"
    response = requests.get(url)

    if response.status_code == 200:
        data = pd.read_csv(io.StringIO(response.text))
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)  # Ensure dates are sorted

        # Debug available date range
        print(f"Available Data Range: {data.index.min()} to {data.index.max()}")

        # Filter data by date range
        data = data.loc[start_date:end_date]

        return data
    else:
        raise Exception(f"Failed to fetch data from Alpha Vantage. Status code: {response.status_code}")

# Fetch Google stock data using Alpha Vantage
df = fetch_alpha_vantage_data('GOOGL', '2022-07-16', '2025-02-07')

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Selecting required columns
df['HL_PCT'] = (df['high'] - df['low']) / df['low'] * 100
df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]

# Forecasting setup
forecast_col = 'close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))  # Predicting 1% of the dataset
df['label'] = df[forecast_col].shift(-forecast_out)

# Preparing data for training
X = np.array(df.drop(['label'], axis=1))
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale the features
X_lately = X[-forecast_out:]
X = X[:-forecast_out]  # Aligning the dataset
df.dropna(inplace=True)
y = np.array(df['label'])

# Train-Test Split (time-based split)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train a new Linear Regression model
clf = LinearRegression()
clf.fit(X_train, y_train)

# Model Accuracy and Predictions
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

# Display predictions and accuracy
print(f"Forecasted Prices: {forecast_set}")
print(f"Model Accuracy (R² Score): {accuracy}")
print(f"Forecast Out: {forecast_out}")

# Add Forecasted Data to DataFrame
df['Forecast'] = np.nan

# Get last available date and increment by one business day
last_date = df.index[-1]
next_date = last_date + BDay(1)

# Add predictions to the dataframe
for i in forecast_set:
    while next_date in df.index:  # Ensure we don't overwrite actual stock prices
        next_date += BDay(1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_date += BDay(1)  # Move to the next future business date

# Create a new DataFrame for plotting
plot_df = df[['close', 'Forecast']].copy()

# Plotting the Results
plt.figure(figsize=(12, 6))
plt.plot(plot_df.index, plot_df['close'], color='blue', label='Actual Price')
plt.plot(plot_df.index, plot_df['Forecast'], color='red', linestyle='dashed', label='Predicted Price')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Google Stock Price Prediction (Alpha Vantage)')
plt.show()