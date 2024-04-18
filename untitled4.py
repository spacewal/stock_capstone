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

# Fetching S&P 500 company symbols from Wikipedia
sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-')  # Adjust symbols for yfinance compatibility
unique_symbols = sp500_table['Symbol'].unique().tolist()

# Setting up the historical data range
current_date = dt.datetime.now()
eight_years_ago = current_date - pd.DateOffset(days=365*8)

# Allow the user to pick a single stock symbol using a searchable dropdown
selected_symbol = st.selectbox('Select a stock symbol:', unique_symbols)

# Fetch and display stock data for the selected symbol
if selected_symbol:
    with st.spinner(f'Fetching data for {selected_symbol}...'):
        stock_data = yf.download(selected_symbol, start=eight_years_ago, end=current_date)
        
        # Remove the time component from the index and rename it to 'Date'
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'index': 'Date'}, inplace=True)
        
        st.success('Data fetched successfully!')
        st.write(stock_data.head())  # Display the first few rows of the stock data
