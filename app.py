import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📈 AI Stock Market Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL","TSLA","MSFT","GOOGL","AMZN",
     "TCS.NS","INFY.NS","RELIANCE.NS"]
)

# ---------------- LOAD DATA ----------------
with st.spinner("Fetching stock data..."):
    data = yf.download(stock, start="2023-01-01")

data.columns = data.columns.get_level_values(0)
data = data.dropna()

# ---------------- MOVING AVERAGES ----------------
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# ---------------- SIGNAL ----------------
data["Signal"] = 0
data.loc[data["MA20"] > data["MA50"], "Signal"] = 1
data.loc[data["MA20"] < data["MA50"], "Signal"] = -1

latest_signal = data["Signal"].iloc[-1]

# ================= ML MODEL =================
X = np.array(range(len(data))).reshape(-1,1)
y = data["Close"].values

model = LinearRegression()
model.fit(X, y)

# accuracy
pred_train = model.predict(X)
accuracy = r2_score(y, pred_train)

# next-day prediction
next_day = np.array([[len(data)]])
predicted_price = model.predict(next_day)[0]

# future 7-day prediction
future_days = np.array(range(len(data), len(data)+7)).reshape(-1,1)
future_prices = model.predict(future_days)

# ---------------- SUMMARY ----------------
st.subheader("📊 Stock Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current Price", round(data["Close"].iloc[-1],2))
col2.metric("Highest", round(data["High"].max(),2))
col3.metric("Lowest", round(data["Low"].min(),2))
col4.metric("🤖 Next Day Prediction", round(predicted_price,2))
col5.metric("Model Accuracy (R²)", round(accuracy,3))

# ---------------- SIGNAL DISPLAY ----------------
if latest_signal == 1:
    st.success("📈 BUY Signal (MA20 above MA50)")
elif latest_signal == -1:
    st.error("📉 SELL Signal (MA20 below MA50)")
else:
    st.warning("⚠️ Neutral Signal")

# ---------------- MAIN PRICE CHART ----------------
fig = go.Figure()

# candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

# moving averages
fig.add_trace(go.Scatter(
    x=data.index,
    y=data["MA20"],
    mode="lines",
    name="MA20"
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["MA50"],
    mode="lines",
    name="MA50"
))

# prediction line (future)
future_dates = pd.date_range(
    start=data.index[-1],
    periods=8,
    freq="D"
)[1:]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_prices,
    mode="lines+markers",
    name="AI Prediction"
))

fig.update_layout(
    title=f"{stock} Price + AI Prediction",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FUTURE GRAPH ----------------
st.subheader("🤖 Future 7-Day Prediction")

future_fig = go.Figure()

future_fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_prices,
    mode="lines+markers",
    name="Future Prediction"
))

future_fig.update_layout(
    title="Predicted Next 7 Days",
    height=350
)

st.plotly_chart(future_fig, use_container_width=True)

# ---------------- VOLUME ----------------
st.subheader("📊 Trading Volume")

volume_fig = go.Figure()

volume_fig.add_trace(go.Bar(
    x=data.index,
    y=data["Volume"],
    name="Volume"
))

volume_fig.update_layout(height=300)

st.plotly_chart(volume_fig, use_container_width=True)

# ---------------- DOWNLOAD REPORT ----------------
st.subheader("⬇️ Download Prediction Report")

report_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_prices
})

csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="prediction_report.csv",
    mime="text/csv"
)
