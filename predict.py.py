import math
import io
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import BDay

API_KEY = '7Q3UX1ZBNKCC4NKP'


def fetch_alpha_vantage_data(symbol, start_date, end_date):
    # NOTE: outputsize=full is a PREMIUM feature on free API keys.
    # Using compact (last 100 data points) which is free.
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}"
        f"&apikey={API_KEY}"
        f"&datatype=csv"
        # No outputsize=full — free tier only supports compact (last 100 days)
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"HTTP error: {response.status_code}")

    raw_text = response.text

    # CHECK: If Alpha Vantage returns a JSON error message instead of CSV data
    if '"Information"' in raw_text or '"Error Message"' in raw_text or '"Note"' in raw_text:
        raise Exception(
            f"Alpha Vantage API returned an error instead of data:\n{raw_text}\n\n"
            f"This usually means:\n"
            f"  1. Your API key hit the rate limit (5 calls/min, 500/day on free tier)\n"
            f"  2. The requested feature requires a premium plan\n"
            f"  3. The symbol is invalid\n"
            f"Wait 1 minute and try again, or upgrade your plan at https://www.alphavantage.co/premium/"
        )

    data = pd.read_csv(io.StringIO(raw_text))
    print("Columns returned by API:", data.columns.tolist())
    print(data.head(3))

    # Auto-detect the date column
    date_col = None
    for candidate in ['timestamp', 'date', 'Date', 'Timestamp']:
        if candidate in data.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = data.columns[0]
        print(f"Warning: Using first column as date: '{date_col}'")

    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data.sort_index(inplace=True)

    print(f"Available Data Range: {data.index.min()} to {data.index.max()}")

    filtered = data.loc[start_date:end_date]
    if filtered.empty:
        print(
            f"Warning: No data in range {start_date} to {end_date}. "
            f"Using all available data instead (free tier returns last ~100 days)."
        )
        return data  # Return all available data
    return filtered


# Fetch Google stock data
df = fetch_alpha_vantage_data('GOOGL', '2024-01-01', '2025-04-27')

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Auto-detect OHLCV column names
col_map = {}
for col in df.columns:
    cl = col.lower()
    if 'open' in cl:    col_map['open']   = col
    elif 'high' in cl:  col_map['high']   = col
    elif 'low' in cl:   col_map['low']    = col
    elif 'close' in cl: col_map['close']  = col
    elif 'volume' in cl:col_map['volume'] = col

print("Detected column mapping:", col_map)

missing = [k for k in ['open', 'high', 'low', 'close', 'volume'] if k not in col_map]
if missing:
    raise Exception(f"Could not find columns for: {missing}. Available: {df.columns.tolist()}")

df.rename(columns={v: k for k, v in col_map.items()}, inplace=True)

# Feature Engineering
df['HL_PCT']    = (df['high'] - df['low'])   / df['low']  * 100
df['PCT_change']= (df['close'] - df['open']) / df['open'] * 100
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]

# Forecasting setup
df.fillna(-99999, inplace=True)
forecast_out = max(1, int(math.ceil(0.01 * len(df))))  # at least 1 day
print(f"Total rows: {len(df)} | Forecasting {forecast_out} day(s) ahead")

df['label'] = df['close'].shift(-forecast_out)
df.dropna(inplace=True)

if len(df) < 10:
    raise Exception(
        f"Not enough data to train ({len(df)} rows after cleaning). "
        f"Free API returns ~100 rows; try a broader date range."
    )

X = np.array(df.drop(['label'], axis=1))
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_lately = X[-forecast_out:]
X        = X[:-forecast_out]
y        = np.array(df['label'])[:-forecast_out]

# Time-based train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test, y_test)
forecast_set= clf.predict(X_lately)

print(f"Forecasted Prices: {forecast_set}")
print(f"Model Accuracy (R² Score): {accuracy:.4f}")
print(f"Forecast Out (days): {forecast_out}")

# Add forecast to DataFrame
df['Forecast'] = np.nan
last_date  = df.index[-1]
next_date  = last_date + BDay(1)

for price in forecast_set:
    while next_date in df.index:
        next_date += BDay(1)
    df.loc[next_date] = [np.nan] * (len(df.columns) - 1) + [price]
    next_date += BDay(1)

# Plot
plot_df = df[['close', 'Forecast']].copy()

plt.figure(figsize=(12, 6))
plt.plot(plot_df.index, plot_df['close'],    color='blue', label='Actual Price')
plt.plot(plot_df.index, plot_df['Forecast'], color='red',  linestyle='dashed', label='Predicted Price')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Google (GOOGL) Stock Price Prediction — Linear Regression')
plt.tight_layout()
plt.show()
