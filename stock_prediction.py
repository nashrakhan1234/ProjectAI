import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

st.title("Stock Price Prediction App")
st.write("Predict future stock prices using the ARIMA model.")

# User input for stock ticker selection
ticker = st.selectbox("Select Stock Ticker:", ['GOOG', 'AAPL', 'MSFT', 'GME'])

# User input for prediction period slider
years_to_predict = st.slider("Select number of years to predict:", min_value=1, max_value=5, value=1)

# User input for date range
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

# Fetch historical data from Yahoo Finance with caching
@st.cache_data
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("No data found for the given ticker symbol.")
        return data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to plot raw stock data



import matplotlib.pyplot as plt

# Function to plot raw stock data using Matplotlib
def plot_raw_data_matplotlib(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Open'], label="Open Price", color='blue')
    plt.plot(data.index, data['Close'], label="Close Price", color='red')
    plt.title("Time Series Data with Open and Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())  # Display the Matplotlib figure in Streamlit

# Replace the call to `plot_raw_data` with `plot_raw_data_matplotlib` where necessary

# Store fetched data in session state to make it accessible across interactions
if st.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        st.session_state['data'] = fetch_stock_data(ticker, start_date, end_date)
        if 'data' in st.session_state and not st.session_state['data'].empty:
            st.write("### Historical Stock Price Data")
            st.write(st.session_state['data'].tail())  # Display last few rows of data
            st.write("### Raw Data")
            st.write(st.session_state['data'])  # Display the full raw data
            plot_raw_data_matplotlib(st.session_state['data'])  # Call the function to plot raw data

# Function to plot stock data if it exists
def plot_stock_data():
    if 'data' in st.session_state:
        data = st.session_state['data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
        fig.update_layout(title="Historical Close Price", xaxis_title="Date", yaxis_title="Close Price (USD)")
        st.plotly_chart(fig)
    else:
        st.warning("Please fetch data first.")

# if st.button("Plot Data"):
#     plot_stock_data()

# Train ARIMA model using the closing prices
def train_arima_model():
    if 'data' in st.session_state:
        data = st.session_state['data']
        close_prices = data['Close'].dropna()
        model = pm.auto_arima(close_prices, seasonal=False, trace=True, stepwise=True)
        return model
    else:
        st.warning("Please fetch data first.")
        return None

if st.button("Train Model"):
    with st.spinner("Training ARIMA model..."):
        model = train_arima_model()
        if model:
            st.write("### ARIMA Model Summary")
            st.write(model.summary())
            st.session_state['model'] = model

# Convert selected years into trading days for prediction
periods = years_to_predict * 252  # Assuming 252 trading days in a year

# Predict future prices

# Import Matplotlib if not already imported
import matplotlib.pyplot as plt

# Modify the `predict_future` function to return both data and forecast
def predict_future(periods=10):
    if 'model' in st.session_state and 'data' in st.session_state:
        model = st.session_state['model']
        data = st.session_state['data']
        forecast = model.predict(n_periods=periods)
        future_dates = pd.date_range(start=data.index[-1], periods=periods + 1, freq='B')[1:]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
        return data, forecast_df
    else:
        st.warning("Please train the model first.")
        return None, None

# Button to trigger future prediction plotting with Matplotlib
if st.button("Predict Future Prices"):
    with st.spinner("Predicting future prices..."):
        data, forecast_df = predict_future(periods=periods)
        if forecast_df is not None:
            st.write("### Predicted Prices")
            st.write(forecast_df)

            # Plot both historical and predicted prices using Matplotlib
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'], label="Historical Close Price", color='red')
            plt.plot(forecast_df['Date'], forecast_df['Forecast'], label="Predicted Close Price", color='blue')
            plt.title("Stock Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.grid(True)
            
            # Display the Matplotlib plot in Streamlit
            st.pyplot(plt.gcf())

# def predict_future(periods=10):
#     if 'model' in st.session_state and 'data' in st.session_state:
#         model = st.session_state['model']
#         data = st.session_state['data']
#         forecast = model.predict(n_periods=periods)
#         future_dates = pd.date_range(start=data.index[-1], periods=periods + 1, freq='B')[1:]
#         forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
#         return forecast_df
#     else:
#         st.warning("Please train the model first.")
#         return None

# if st.button("Predict Future Prices"):
#     with st.spinner("Predicting future prices..."):
#         forecast_df = predict_future(periods=periods)
#         if forecast_df is not None:
#             st.write("### Predicted Prices")
#             st.write(forecast_df)

#             # Plot both historical and predicted prices
#             data = st.session_state['data']
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close Price', line=dict(color='red')))
#             fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Predicted Close Price', line=dict(color='blue')))
#             fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
#             st.plotly_chart(fig)
