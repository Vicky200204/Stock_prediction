import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# 1. App Title
st.title("📈 Real-Time Stock Price Prediction System")

# 2. User Input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, INFY.NS):", value="AAPL")
predict_days = st.slider("Days to Predict:", 1, 30, 5)

# 3. Load Stock Data
@st.cache_data
def load_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period="3y")
    return df

if stock_symbol:
    df = load_data(stock_symbol)
    
    st.subheader(f"Recent Data for {stock_symbol}")
    st.dataframe(df.tail())

    # 4. Plot
    st.line_chart(df["Close"], use_container_width=True)

    # 5. Preprocessing
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 6. Training Data
    train_data = scaled_data[0:int(len(scaled_data)*0.8)]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 7. LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # 8. Predict Future
    last_60_days = data[-60:].values
    last_60_scaled = scaler.transform(last_60_days)
    pred_input = last_60_scaled.reshape(1, -1)
    temp_input = list(pred_input[0])

    future_output = []
    for i in range(predict_days):
        if len(temp_input) > 60:
            temp_input = temp_input[1:]
        x_input = np.array(temp_input).reshape(1, 60, 1)
        pred = model.predict(x_input, verbose=0)
        temp_input.append(pred[0][0])
        future_output.append(pred[0][0])

    pred_prices = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))

    # 9. Output Plot
    st.subheader(f"{predict_days}-Day Prediction for {stock_symbol}")
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
    pred_df = pd.DataFrame(pred_prices, index=future_dates, columns=["Predicted Price"])
    
    st.line_chart(pred_df, use_container_width=True)
    st.dataframe(pred_df)

