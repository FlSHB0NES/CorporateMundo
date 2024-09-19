
import pandas as pd
import matplotlib.pyplot as plt
import math
from portfolio_env import PortfolioEnv  # Import the custom environment


# Constants for RSI thresholds and portfolio management
RSI_BUY = 30  # RSI value below which we consider buying
RSI_SELL = 70  # RSI value above which we consider selling
PORTFOLIO_TRADE_PERCENTAGE = 0.01  # Percentage of portfolio to trade per transaction
MINIMUM_TRADE = 100  # Minimum number of shares to trade

def calculate_rsi(data, window=14):
    # Calculate the daily price changes
    delta = data['Close/Last'].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)  # Gains only where price increased
    loss = (-delta.where(delta < 0, 0)).fillna(0)  # Losses only where price decreased

    # Calculate the Exponential Moving Average (EMA) of gains and losses
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()

    # Calculate the RS (Relative Strength) and RSI (Relative Strength Index)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))  # Standard RSI formula

    return rsi

def should_buy(row, buy_threshold=RSI_BUY):
    # Determines whether to buy based on RSI value being below the threshold
    return row['RSI'] < buy_threshold

def should_sell(row, sell_threshold=RSI_SELL):
    # Determines whether to sell based on RSI value being above the threshold
    return row['RSI'] > sell_threshold

def execute_trade(date, action, price, shares_to_trade):
    # Executes a buy or sell trade, adjusting cash and shares accordingly
    global current_cash, current_shares
    trade_cost = price * shares_to_trade
    if action == 'buy' and current_cash >= trade_cost:
        # If buying, update shares and subtract the cost from cash
        current_shares += shares_to_trade
        current_cash -= trade_cost
        log_trade(date, 'Buy', price, shares_to_trade, current_cash, current_shares)
    elif action == 'sell' and current_shares >= shares_to_trade:
        # If selling, update shares and add the sale proceeds to cash
        current_shares -= shares_to_trade
        current_cash += trade_cost
        log_trade(date, 'Sell', price, shares_to_trade, current_cash, current_shares)

def log_trade(date, ticker, action, price, shares, cash, stock_holdings):
    # Logs the details of a trade into a global trade log DataFrame
    global trade_log
    new_entry = {
        'Date': date,
        'Ticker': ticker,
        'Action': action,
        'Price': price,
        'Shares': shares,
        'Cash': cash,
        'Total Value': cash + price * stock_holdings,  # Total value includes cash and stock value
        'Stock Holdings': stock_holdings,
        'All Time ROI': round(((cash + price * stock_holdings) - initial_cash) / initial_cash, 3)
    }
    trade_log = pd.concat([trade_log, pd.DataFrame([new_entry])], ignore_index=True)

def calculate_trade_amount(price, current_shares):
    # Calculates the number of shares to trade based on portfolio trade percentage
    if price * current_shares > 0:
        return math.ceil((price * current_shares * PORTFOLIO_TRADE_PERCENTAGE) / price)
    return MINIMUM_TRADE  # If no current shares, default to minimum trade amount

def visualize(data):

    trade_log = pd.DataFrame(data)

    # Plotting the stock holdings and stock price over time
    fig, ax1 = plt.subplots()

    # Plot stock holdings
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Holdings', color='tab:blue')
    ax1.plot(trade_log['Date'], trade_log['Stock Holdings'], color='tab:blue', label='Stock Holdings')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis to plot stock price
    ax2 = ax1.twinx()
    ax2.set_ylabel('Stock Price', color='tab:red')
    ax2.plot(trade_log['Date'], trade_log['Price'], color='tab:red', label='Stock Price')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and show plot
    plt.title('Stock Holdings and Stock Price Over Time')
    plt.show()

# Constants for RSI thresholds and portfolio management
RSI_BUY = 30  # RSI value below which we consider buying
RSI_SELL = 70  # RSI value above which we consider selling
PORTFOLIO_TRADE_PERCENTAGE = 0.01  # Percentage of portfolio to trade per transaction
MINIMUM_TRADE = 100  # Minimum number of shares to trade

# Stock tickers to consider
stock_tickers = ['AAPL', 'TSLA', 'GOOGL', 'NVDA']

# Dictionary to hold stock data for each ticker
stock_data = {}

# Load the stock data for each ticker (assuming CSV files exist for each ticker)
for ticker in stock_tickers:
    data_path = f'./data/HistoricalData_{ticker}.csv'
    df = pd.read_csv(data_path)

    # Clean the data: Convert 'Date' to datetime, remove '$' from financial columns, and convert to float
    df['Date'] = pd.to_datetime(df['Date'])
    financial_columns = ['Close/Last', 'Open', 'High', 'Low']
    for col in financial_columns:
        df[col] = df[col].replace('[\\$,]', '', regex=True).astype(float)

    # Sort the data by date in ascending order
    df = df.sort_values(by='Date', ascending=True)

    # Add the RSI (Relative Strength Index) to the stock data
    df['RSI'] = calculate_rsi(df)

    # Store the data in the dictionary
    stock_data[ticker] = df

# Get the full range of dates covered by the stock data
all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
# Initialize portfolio tracking variables
initial_cash = 10000  # Example starting cash
current_cash = initial_cash
current_shares = {ticker: 0 for ticker in stock_tickers}  # Dictionary to track shares for each stock

# Initialize the logging DataFrame to store trade history
columns = ['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Cash', 'Total Value', 'Stock Holdings']
trade_log = pd.DataFrame(columns=columns)
trade_log['Date'] = pd.to_datetime(trade_log['Date'])

# Trade execution function to handle multiple stocks
def execute_trade(date, ticker, action, price, shares_to_trade):
    global current_cash, current_shares
    trade_cost = price * shares_to_trade
    if action == 'buy' and current_cash >= trade_cost:
        # If buying, update shares and subtract the cost from cash
        current_shares[ticker] += shares_to_trade
        current_cash -= trade_cost
        log_trade(date, ticker, 'Buy', price, shares_to_trade, current_cash, current_shares[ticker])
    elif action == 'sell' and current_shares[ticker] >= shares_to_trade:
        # If selling, update shares and add the proceeds to cash
        current_shares[ticker] -= shares_to_trade
        current_cash += trade_cost
        log_trade(date, ticker, 'Sell', price, shares_to_trade, current_cash, current_shares[ticker])
    else:
        log_trade(date, ticker, 'Hold', price, 0, current_cash, current_shares[ticker])

# Loop through each date, and for each date, evaluate all stocks
for date in all_dates:
    for ticker, df in stock_data.items():
        if date in df.index:
            row = df.loc[date]
            stock_trading_price = row['Close/Last']
            if should_buy(row):
                shares_to_trade = calculate_trade_amount(stock_trading_price, current_shares[ticker])
                execute_trade(date, ticker, 'buy', stock_trading_price, shares_to_trade)
            elif should_sell(row):
                shares_to_trade = calculate_trade_amount(stock_trading_price, current_shares[ticker])
                execute_trade(date, ticker, 'sell', stock_trading_price, shares_to_trade)
            else:
                log_trade(date, ticker, 'Hold', stock_trading_price, 0, current_cash, current_shares[ticker])

# Print the first 300 entries of the trade log to check the trades
print(trade_log)
#visualize(trade_log)
