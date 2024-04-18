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

# Fetching S&P 500 data from Wikipedia
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')  # Replacing . with - to match yfinance format
symbols_list = sp500['Symbol'].unique().tolist()

# Setting up date range for fetching historical data
end_date = dt.datetime.now()
start_date = end_date - pd.DateOffset(days=365*8)

# Use Streamlit's selectbox widget to let the user pick a stock symbol
selected_symbol = st.selectbox('Select a stock symbol:', symbols_list)

# Display a loading message while fetching the data
with st.spinner(f'Loading data for {selected_symbol}...'):
    # Fetching the historical stock data from yfinance for the selected symbol
    df = yf.download(tickers=selected_symbol, start=start_date, end=end_date)

# Renaming the index for clarity
df.index.name = 'date'

# Display successful loading message
st.success('Data loading complete!')

# Display the DataFrame in the Streamlit app
st.write(df.head())  # Show the first few rows of the DataFrame
