import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Fetch historical data for selected stock
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Add additional columns (returns, moving averages, volatility)
def preprocess_data(stock_data):
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    stock_data['20-Day MA'] = stock_data['Adj Close'].rolling(window=20).mean()
    stock_data['50-Day MA'] = stock_data['Adj Close'].rolling(window=50).mean()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=30).std()
    return stock_data

# Visualizing the moving averages and adjusted close price
def visualize_data(stock_data, predicted_data):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close Price', color='blue')
    plt.plot(stock_data.index, stock_data['20-Day MA'], label='20-Day Moving Average', color='red')
    plt.plot(stock_data.index, stock_data['50-Day MA'], label='50-Day Moving Average', color='green')

    # Plot future predicted prices as a dotted line
    plt.plot(predicted_data.index, predicted_data['Predicted Price'], label='Predicted Prices (1 Year Ahead)', linestyle='--', color='orange')

    plt.title('Apple Stock Price, Moving Averages, and Predictions (Next Year)')
    plt.legend()
    plt.show()

# Plot the volatility
def plot_volatility(stock_data):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Volatility'], label='Volatility', color='purple')
    plt.title('Stock Price Volatility')
    plt.show()

# Train a Random Forest model and predict stock prices
def train_random_forest(stock_data):
    # Remove NaN values
    stock_data = stock_data.dropna()

    # Features (moving averages, volatility) and target (Adj Close)
    X = stock_data[['20-Day MA', '50-Day MA', 'Volatility']]
    y = stock_data['Adj Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions and evaluation on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse}")

    return model

# Predict future stock prices for one year ahead
def predict_future_prices(model, stock_data, days_ahead=365):
    # Create new dates for one year ahead
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    # Predict using last available moving averages and volatility for future days
    last_20ma = stock_data['20-Day MA'].iloc[-1]
    last_50ma = stock_data['50-Day MA'].iloc[-1]
    last_volatility = stock_data['Volatility'].iloc[-1]

    # Generate future features using the last available data
    future_features = pd.DataFrame({
        '20-Day MA': [last_20ma] * days_ahead,
        '50-Day MA': [last_50ma] * days_ahead,
        'Volatility': [last_volatility] * days_ahead
    }, index=future_dates)

    # Predict future stock prices
    future_predictions = model.predict(future_features)

    # Create a DataFrame to hold future predictions
    predicted_data = pd.DataFrame({'Predicted Price': future_predictions}, index=future_dates)

    return predicted_data

if __name__ == "__main__":
    # Get today's date for up-to-date stock data
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch stock data for Apple (AAPL)
    stock_data = get_stock_data('AAPL', '2018-01-01', end_date)
    
    # Preprocess the data
    stock_data = preprocess_data(stock_data)
    
    # Train Random Forest and predict stock prices
    model = train_random_forest(stock_data)
    
    # Predict future stock prices for one year ahead
    predicted_data = predict_future_prices(model, stock_data)

    # Visualize historical and predicted data
    visualize_data(stock_data, predicted_data)
    
    # Plot volatility
    plot_volatility(stock_data)
    
    # Save preprocessed data and predictions to CSV for Tableau
    stock_data.to_csv('AAPL_stock_data.csv')
    predicted_data.to_csv('AAPL_predicted_prices.csv')

    print("Data fetched, processed, visualized, predicted, and saved successfully.")
