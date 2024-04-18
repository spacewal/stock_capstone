import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import streamlit as st

# Streamlit page setup
st.title('S&P 500 Stock Analysis')

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = dt.datetime.now()

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

# Use Streamlit's selectbox widget to let the user pick a stock symbol
selected_symbol = st.selectbox('Select a stock symbol:', symbols_list)

df = yf.download(tickers= selected_symbol,
                 period='1y',
                 start=start_date,
                 end=end_date).stack()

df.index.names = ['date', 'ticker']

df_one_symbol = df[df.index.get_level_values('ticker') == selected_symbol]

