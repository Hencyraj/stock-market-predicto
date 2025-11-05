import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import feedparser
from nsepython import equity_history

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# ---------------- UI THEME ----------------
st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0d0d0d 0%, #141E30 50%, #243B55 100%);
    color: white;
}
.big-title {text-align:center; font-size:36px; color:#58a6ff; text-shadow:0px 0px 12px #0084ff;}
.card {padding:18px; border-radius:12px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.18); text-align:center;}
.metric-value{font-size:22px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=3600)
def fetch_data(symbol, years):
    start = date.today() - relativedelta(years=years)
    df = equity_history(symbol=symbol, series="EQ",
                        start_date=start.strftime("%d-%m-%Y"),
                        end_date=date.today().strftime("%d-%m-%Y"))
    if df is None or len(df)==0: return pd.DataFrame()
    df = df.rename(columns={
        "CH_TIMESTAMP":"date","CH_OPENING_PRICE":"open","CH_TRADE_HIGH_PRICE":"high",
        "CH_TRADE_LOW_PRICE":"low","CH_CLOSING_PRICE":"close","CH_TOT_TRADED_QTY":"volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    return df[["date","open","high","low","close","volume"]].sort_values("date")

# ---------------- FEATURES ----------------
def add_features(df):
    out = df.copy()
    for lag in [1,2,3,5,10]:
        out[f"lag_{lag}"] = out["close"].shift(lag)
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["target"] = out["close"].shift(-1)
    return out.dropna()

# ---------------- KNN MODEL ----------------
def knn_predict(feat, horizon, k):
    X = feat.drop(columns=["date","open","high","low","close","volume","target"]).values
    y = feat["target"].values

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)

    fc = []
    last = feat.iloc[-1].copy()
    for _ in range(horizon):
        pred = model.predict([last.drop(labels=["date","open","high","low","close","volume","target"]).values])[0]
        next_date = last["date"] + pd.tseries.offsets.BDay(1)
        last["date"], last["close"] = next_date, pred
        fc.append({"date":next_date,"pred_close":pred})

    mae = mean_absolute_error(y[-20:], model.predict(X)[-20:])
    return pd.DataFrame(fc), mae

# ---------------- FIXED LSTM MODEL ----------------
def lstm_predict(feat, horizon, steps, epochs):
    feature_cols = [c for c in feat.columns if c not in ["date","target","open","high","low","volume","close"]]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = feat[feature_cols].values
    y_raw = feat["target"].values.reshape(-1,1)

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X, y = [], []
    for i in range(len(X_scaled) - steps):
        X.append(X_scaled[i:i+steps])
        y.append(y_scaled[i+steps])
    X, y = np.array(X), np.array(y)

    if len(X) < 50:
        return pd.DataFrame(), None

    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(steps,X.shape[-1])),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    fc = []
    seq = X_scaled[-steps:]
    last_date = feat["date"].iloc[-1]

    for _ in range(horizon):
        yhat_scaled = model.predict(seq.reshape(1,steps,-1), verbose=0)[0][0]
        yhat = scaler_y.inverse_transform([[yhat_scaled]])[0][0]
        next_date = last_date + pd.tseries.offsets.BDay(1)
        fc.append({"date":next_date, "pred_close":yhat})
        last_date = next_date
        seq = np.vstack([seq[1:], [yhat_scaled]*seq.shape[1]])

    pred = scaler_y.inverse_transform(model.predict(X[-50:], verbose=0))
    true = scaler_y.inverse_transform(y[-50:])
    mae = mean_absolute_error(true, pred)

    return pd.DataFrame(fc), mae

# ---------------- SENTIMENT ----------------
def sentiment(symbol):
    url = f"https://www.moneycontrol.com/rss/company-news/{symbol.lower()}.xml"
    news = feedparser.parse(url).entries[:5]
    if not news: return "No News"
    scores = [sid.polarity_scores(n.title)["compound"] for n in news]
    avg = np.mean(scores)
    return "📈 Positive" if avg>0 else "📉 Negative"

# ---------------- UI ----------------
st.markdown("<h1 class='big-title'>📈 NSE Stock Predictor</h1>", unsafe_allow_html=True)

with st.sidebar:
    symbol = st.text_input("Enter NSE Symbol (UPPERCASE)", "TCS").upper()
    years = st.slider("Years of Data",1,10,3)
    horizon = st.slider("Forecast Days",5,30,10)
    k = st.slider("KNN Neighbors",3,15,5)
    steps = st.slider("LSTM Lookback",15,60,30)
    epochs = st.slider("LSTM Epochs",5,50,20)
    run = st.button("🔍 Search / Predict")

if not run:
    st.stop()

data = fetch_data(symbol, years)
if data.empty:
    st.error("⚠️ No data found. Check stock symbol (must be UPPERCASE).")
    st.stop()

feat = add_features(data)



feat = add_features(data)
fc_knn, mae_knn = knn_predict(feat,horizon,k)
fc_lstm, mae_lstm = lstm_predict(feat,horizon,steps,epochs)
news = sentiment(symbol)

# ---------------- GRAPH ----------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["date"],open=data["open"],high=data["high"],low=data["low"],close=data["close"],
    increasing_line_color='#00ff99', decreasing_line_color='#ff4d4d', name="Price"
))

fig.add_trace(go.Scatter(
    x=fc_knn["date"], y=fc_knn["pred_close"], mode="lines+markers",
    line=dict(color="#00eaff", width=3), name="KNN Prediction"
))

fig.add_trace(go.Scatter(
    x=fc_lstm["date"], y=fc_lstm["pred_close"], mode="lines+markers",
    line=dict(color="#ff00ff", width=3), name="LSTM Prediction"
))

fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")

c1,c2,c3 = st.columns(3)
c1.markdown(f"<div class='card'><div class='metric-value'>KNN MAE: {mae_knn:.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='metric-value'>LSTM MAE: {mae_lstm:.2f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='metric-value'>Sentiment: {news}</div></div>", unsafe_allow_html=True)

st.plotly_chart(fig, use_container_width=True)
