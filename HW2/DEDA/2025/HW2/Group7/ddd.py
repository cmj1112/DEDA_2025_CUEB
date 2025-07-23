import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# 下载BTC数据
def getData(crypto, currency='EUR'):
    now = datetime.now()
    last_year = now - timedelta(days=365)
    data = yf.download(f'{crypto}-{currency}', start=last_year.strftime('%Y-%m-%d'), end=now.strftime('%Y-%m-%d'))
    return data

btc_data = getData('BTC')

# 确保列是平的
if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = btc_data.columns.get_level_values(0)

# 绘制蜡烛图
fig = go.Figure(data=[go.Candlestick(
    x=btc_data.index,
    open=btc_data['Open'],
    high=btc_data['High'],
    low=btc_data['Low'],
    close=btc_data['Close']
)])

fig.update_layout(
    title='Bitcoin Price with Range Slider',
    xaxis_title='Date',
    yaxis_title='Price (EUR)',
    xaxis_rangeslider_visible=True
)

fig.show()