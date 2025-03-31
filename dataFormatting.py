import pandas_datareader as pdr
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
import json
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm

# This gets the main economic data. Returning it in a dataframe
def econ_data(start, end):
    econ = pdr.data.DataReader(['GDP', 'UNRATE', 'CPIAUCSL', 'PCE', 'DGS10', 'DGS2', 'FEDFUNDS'], 'fred', start, end)
    all_business_days = pd.date_range(start, end, freq='B')
    econ = econ.reindex(all_business_days).ffill()
    econ.index.rename("Date")
    econ.columns = ['GDP', 'Unemployment-Rate', 'CPI', 'Consum-Spend', '10Year-Yield', '2Year-Yield', 'IRate']

    # Getting S&P 500 market cap
    spy = yf.Ticker("SPY")
    spy_history = spy.history(start=start, end=end)
    shares_outstanding = spy.info["sharesOutstanding"]
    spy_history["Market Cap"] = spy_history["Close"] * shares_outstanding
    spy_history["S&P-Market-Cap"] = spy_history["Market Cap"] / 1e7
    spy_history = spy_history[["S&P-Market-Cap"]]
    spy_history.index = spy_history.index.tz_localize(None)
    spy_history = spy_history.bfill()

    # Merging this data
    econ = pd.concat([econ, spy_history], axis=1)
    econ = econ.dropna(subset=["S&P-Market-Cap"])

    # Buffet Ratio
    econ["Buffet-Ratio"] = econ["S&P-Market-Cap"] / econ["GDP"]

    return econ

# Getting the stock sentiment data
def sentiment_data(start, end, ticker="AAPL", limit=100):
    # Get the config.json info
    with open('secrets.json', 'r') as config_file:
        config = json.load(config_file)
        
    api_key = config.get("alpha_vantage_key")

    start = start.strftime("%Y%m%dT%H%M")
    end = end.strftime("%Y%m%dT%H%M")

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&topics=technology,earnings&time_from={start}&time_to={end}&apikey={api_key}&limit={limit}"

    response = requests.get(url)
    data = response.json()

    sum = 0
    count = 0

    # Extract sentiment score
    for article in data.get("feed", []):
        count += 1
        sum += article['overall_sentiment_score']

    average_sentiment = 0

    if count != 0:
        average_sentiment = sum / count

    return average_sentiment

# Get all of the fundamental data on the stock
def stock_data(start, end, ticker="AAPL"):
    stock = yf.Ticker(ticker)
    price_history = stock.history(start=start, end=end)

    # All request information
    fundamentals = {
        "Beta": stock.info.get("beta"),
        "Leverage-Ratio": stock.info.get("debtToEquity"),
        "P/E-Ratio": stock.info.get("trailingPE"),
        "Earnings": stock.info.get("trailingEps"),
        "Market-Cap": stock.info.get("marketCap"),
        "Days-to-Earnings": stock.info.get("earningsQuarterlyGrowth"),
        "3M-Average-Volume": stock.info.get("averageVolume"),
    }
    fundamentals_df = pd.DataFrame([fundamentals] * len(price_history), index=price_history.index)

    # Merge
    stock_df = pd.concat([price_history[['Close']], fundamentals_df], axis=1)

    # Formatting
    stock_df['Market-Cap'] /= 1e9
    stock_df.index = stock_df.index.tz_localize(None)

    return stock_df

# Getting trading information for options
def options_data():
    # Read dataset
    options_df = pd.read_csv("2021-2023.csv")
    options_df['expiration'] = pd.to_datetime(options_df['expiration'], format="%Y-%m-%d %H:%M:%S %z UTC")
    options_df['date'] = pd.to_datetime(options_df['date'], format="%Y-%m-%d %H:%M:%S %z UTC")

    # Format dates
    options_df['date'] = options_df['date'].dt.tz_localize(None)
    options_df['expiration'] = options_df['expiration'].dt.tz_localize(None)

    # Adjust data
    options_df['expiration'] = (options_df['expiration'] - options_df['date']).dt.days
    options_df["call"] = options_df['call_put'].astype(str).str.contains("Call").astype(int)
    options_df.drop(columns=["call_put"], inplace=True)

    return options_df

# Getting the realized vol of the stock
def realized_volatility(row, stock_df, max_end):
    start_date = row["date"]
    end_date = row["target_date"]

    # Get the prices over the option life
    price_data = stock_df.loc[(stock_df.index >= start_date) & (stock_df.index <= end_date) & (end_date <= max_end), "Close"]

    if len(price_data) < 2:
        return np.nan

    # Log returns
    log_returns = np.log(price_data.shift(1) / price_data).dropna()

    # Realized vol. Anualizing vol to make it consistent across any time to expiry
    return log_returns.std() * np.sqrt(252)

def main():
    start = datetime(2021, 1, 1)
    end = datetime(2023, 12, 31)

    # Get economic data
    econ = econ_data(start, end)

    # Get fundamental data
    stock_df = stock_data(start, end, ticker="AAPL")

    options_df = options_data()
    #combined_df = options_df

    # Combining per derivative
    combined_df = options_df.assign(**econ.iloc[0])
    combined_df = combined_df.assign(**stock_df.iloc[0])

    # Getting real vol
    combined_df["target_date"] = combined_df["date"] + pd.to_timedelta(combined_df["expiration"], unit="D")
    combined_df["realized_vol"] = combined_df.apply(lambda row: realized_volatility(row, stock_df, end), axis=1)

    # Dropping things
    combined_df.drop(columns=["target_date", "act_symbol"], inplace=True)
    combined_df.dropna(subset=["realized_vol"], inplace=True)

    print(combined_df)
    combined_df.to_csv("trainingData.csv", index=False)
    
    # For sentiment data (UNUSED)
    #sentiment_data(start, end, ticker="AAPL")

main()