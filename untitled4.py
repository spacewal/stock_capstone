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
        df = yf.download(tickers=selected_symbol, start=start_date, end=end_date)

        # Format the date to remove the time part and reset the index without adding a new column
        df.index = df.index.date  # Convert the index to a date-only format
        df.reset_index(inplace=True)  # Reset the index to get 'Date' as a column
        df.rename(columns={'index': 'Date'}, inplace=True)  # Rename the column to 'Date'

        # Display the DataFrame without the index column
        st.dataframe(df.set_index('Date'), width=None, height=None)  # Set 'Date' as the index for display

        st.success('Data fetched successfully!')
