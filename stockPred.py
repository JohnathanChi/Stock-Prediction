# Import necessary libraries
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Step 1: List of S&P 500 companies
sp500_companies = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.'
    # Add more companies as needed
}

# Step 2: Allow user to select a company
print("Select a company from the S&P 500:")
for ticker, name in sp500_companies.items():
    print(f"{ticker}: {name}")

selected_ticker = input("Enter the ticker symbol of the company: ").upper()

if selected_ticker not in sp500_companies:
    print("Invalid ticker symbol. Please try again.")
else:
    # Step 3: Data Collection and Visualization
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=5000)).strftime("%Y-%m-%d")

    data = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)

    # Plot stock data
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                            open=data["Open"], 
                                            high=data["High"],
                                            low=data["Low"], 
                                            close=data["Close"])])
    figure.update_layout(title=f"{sp500_companies[selected_ticker]} Stock Price Analysis", 
                         xaxis_rangeslider_visible=False)
    figure.show()

    # Section 4: ARIMA Model
    train_size = int(len(data) * 0.8)
    train, test = data['Close'][:train_size], data['Close'][train_size:]

    arima_model = pm.auto_arima(train, seasonal=False, trace=False)
    arima_pred = arima_model.predict(n_periods=len(test))

    arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
    arima_mae = mean_absolute_error(test, arima_pred)
    arima_r2 = r2_score(test, arima_pred)

    # Section 5: Prophet Model
    prophet_df = data[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']

    prophet_model = Prophet()
    prophet_model.fit(prophet_df[:train_size])

    future = prophet_model.make_future_dataframe(periods=len(test))
    forecast = prophet_model.predict(future)
    prophet_pred = forecast['yhat'][train_size:].values

    prophet_rmse = np.sqrt(mean_squared_error(test, prophet_pred))
    prophet_mae = mean_absolute_error(test, prophet_pred)
    prophet_r2 = r2_score(test, prophet_pred)

    # Section 6: Random Forest Model
    data['PrevClose'] = data['Close'].shift(1)
    data.dropna(inplace=True)

    features = ['PrevClose']
    target = 'Close'

    train_features = data[features][:train_size]
    test_features = data[features][train_size:]
    train_target = data[target][:train_size]
    test_target = data[target][train_size:]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_features, train_target)
    rf_pred = rf_model.predict(test_features)

    rf_rmse = np.sqrt(mean_squared_error(test_target, rf_pred))
    rf_mae = mean_absolute_error(test_target, rf_pred)
    rf_r2 = r2_score(test_target, rf_pred)

    # Section 7: LSTM Model
    x = data[["Open", "High", "Low", "Volume"]].values
    y = data["Close"].values
    y = y.reshape(-1, 1)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    lstm_model.add(LSTM(64, return_sequences=False))
    lstm_model.add(Dense(25))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(xtrain, ytrain, batch_size=1, epochs=10)

    lstm_pred = lstm_model.predict(xtest)

    lstm_rmse = np.sqrt(mean_squared_error(ytest, lstm_pred))
    lstm_mae = mean_absolute_error(ytest, lstm_pred)
    lstm_r2 = r2_score(ytest, lstm_pred)

    # Section 8: Compare Model Performance
    print(f"ARIMA: RMSE={arima_rmse}, MAE={arima_mae}, R2={arima_r2}")
    print(f"Prophet: RMSE={prophet_rmse}, MAE={prophet_mae}, R2={prophet_r2}")
    print(f"Random Forest: RMSE={rf_rmse}, MAE={rf_mae}, R2={rf_r2}")
    print(f"LSTM: RMSE={lstm_rmse}, MAE={lstm_mae}, R2={lstm_r2}")

    # Section 9: Visualize the Results
    plt.figure(figsize=(14,7))
    plt.plot(test.index, test, color='blue', label='Actual Price')
    plt.plot(test.index, arima_pred, color='green', label='ARIMA Prediction')
    plt.plot(test.index, prophet_pred, color='red', label='Prophet Prediction')
    plt.plot(test.index[:len(rf_pred)], rf_pred, color='purple', label='Random Forest Prediction')
    plt.plot(test.index, lstm_pred, color='orange', label='LSTM Prediction')
    plt.title(f'{sp500_companies[selected_ticker]} Stock Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
