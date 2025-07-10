import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


def get_btc_data_gecko(currency='eur', days=365):
    """Use CoinGecko API to get BTC price data (I can't access yahoo finance through API in China mainland)"""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': currency.lower(),
        'days': days,
        'interval': '1d'
    }

    response = requests.get(url, params=params)
    data = response.json()

    # 处理价格数据
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    # 处理OHLC数据（需要额外API调用）
    ohlc_url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    ohlc_params = {
        'vs_currency': currency.lower(),
        'days': days
    }
    ohlc_response = requests.get(ohlc_url, params=ohlc_params)
    ohlc_data = ohlc_response.json()

    # 解析OHLC数据
    ohlc_df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    ohlc_df['date'] = pd.to_datetime(ohlc_df['timestamp'], unit='ms')
    ohlc_df.set_index('date', inplace=True)
    ohlc_df.drop(columns=['timestamp'], inplace=True)

    return ohlc_df


# 获取数据
btc_data = get_btc_data_gecko(currency='EUR', days=365)

# 绘制图表
fig = go.Figure(data=[go.Candlestick(
    x=btc_data.index,
    open=btc_data['open'],
    high=btc_data['high'],
    low=btc_data['low'],
    close=btc_data['close']
)])

fig.update_layout(
    title=f'Time Series with Range slider for BTC',
    xaxis_title='Date',
    yaxis_title='Price (EUR)',
    xaxis_rangeslider_visible=True
)
fig.write_html('BTC_price.html')
import webbrowser
webbrowser.open('BTC_price.html')