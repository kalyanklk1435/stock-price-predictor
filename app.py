import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import date

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Advanced Stock Price Predictor")

# Sidebar
st.sidebar.header("Stock Selection")

stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "TCS (India)": "TCS.NS",
    "Reliance (India)": "RELIANCE.NS",
    "Infosys (India)": "INFY.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
custom_stock = st.sidebar.text_input("Or Enter Custom Symbol")

start_date = st.sidebar.date_input("Start Date", date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

stock = stocks[selected_stock] if custom_stock == "" else custom_stock.upper()

if st.button("Predict"):

    with st.spinner("Fetching Data and Training Model..."):

        data = yf.download(stock, start=start_date, end=end_date)

        if data.empty:
            st.error("Invalid Stock Symbol!")
        else:
            # Fix for Current Price Error
            current_price = data['Close'].iloc[-1]

            if isinstance(current_price, pd.Series):
                current_price = current_price.values[0]

            st.subheader(f"📌 Current Price: {float(current_price):.2f}")

            # Prepare Data
            data = data[['Close']]
            data['Prediction'] = data['Close'].shift(-30)

            X = np.array(data.drop(['Prediction'], axis=1))[:-30]
            y = np.array(data['Prediction'])[:-30]

            # Train Model
            model = LinearRegression()
            model.fit(X, y)

            predictions = model.predict(X)
            score = r2_score(y, predictions)

            # Future Prediction
            x_future = np.array(data.drop(['Prediction'], axis=1))[-30:]
            forecast = model.predict(x_future)

            st.subheader("📊 Model Accuracy")
            st.write(f"R² Score: {score:.4f}")

            st.subheader("🔮 Next 30 Days Prediction")
            st.write(forecast)

            st.subheader("📈 Stock Price History")
            st.line_chart(data['Close'])