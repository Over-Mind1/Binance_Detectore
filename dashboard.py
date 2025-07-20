import requests
import pandas as pd
import streamlit as st
import time
import datetime
import altair as alt

st.set_page_config(page_title="Binance Pump Detector", layout="wide")
st.title("🚨 Binance Pump Detector (BTC Pairs)")
st.markdown("مراقبة العملات اللي على وشك Pump 🚀 بناءً على حركة الفوليوم والسعر")

# إعدادات
REFRESH_INTERVAL = 60  # ثواني
VOLUME_THRESHOLD = 3  # 300%
LOOKBACK_HOURS = 6

@st.cache_data(ttl=300)
def get_binance_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    return [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'BTC' and s['status'] == 'TRADING']

def get_klines(symbol, interval='1h', limit=LOOKBACK_HOURS + 1):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df[['Open time', 'Close', 'Volume']]

def analyze_symbol(symbol):
    df = get_klines(symbol)
    if len(df) < LOOKBACK_HOURS + 1:
        return None
    last_vol = df.iloc[-1]['Volume']
    avg_vol = df.iloc[:-1]['Volume'].mean()
    last_close = df.iloc[-1]['Close']
    prev_close = df.iloc[-2]['Close']
    price_change = (last_close - prev_close) / prev_close * 100
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

    if vol_ratio > VOLUME_THRESHOLD:
        return {
            'symbol': symbol,
            'last_close': last_close,
            'prev_close': prev_close,
            'price_change': price_change,
            'vol_ratio': vol_ratio,
            'df': df
        }
    return None

symbols = get_binance_symbols()
pump_candidates = []

progress = st.progress(0)
status = st.empty()

for i, symbol in enumerate(symbols):
    status.text(f"بيتم تحليل: {symbol} ({i+1}/{len(symbols)})")
    result = analyze_symbol(symbol)
    if result:
        pump_candidates.append(result)
    progress.progress((i+1)/len(symbols))

status.text("✅ التحليل انتهى")
progress.empty()

if pump_candidates:
    st.success(f"تم رصد {len(pump_candidates)} عملة فيها نشاط غير طبيعي 🚀")

    for coin in pump_candidates:
        st.subheader(f"🪙 {coin['symbol']}")
        st.markdown(f"""
        - **السعر الحالي:** {coin['last_close']:.8f} BTC  
        - **التغير السعري:** {coin['price_change']:.2f}%  
        - **زيادة الفوليوم:** {coin['vol_ratio']:.2f}x
        """)
        chart = alt.Chart(coin['df']).mark_line(point=True).encode(
            x='Open time:T',
            y=alt.Y('Close', title='السعر (BTC)'),
            tooltip=['Open time', 'Close', 'Volume']
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)
else:
    st.warning("لا يوجد حالياً عملات فيها Pump واضح على أزواج BTC. جرب بعد شوية ⏳")

st.caption("⏱️ يتم التحديث كل دقيقة.")
