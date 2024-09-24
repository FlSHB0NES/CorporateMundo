
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants for RSI thresholds and portfolio management
RSI_BUY = 30  # RSI value below which we consider buying
RSI_SELL = 70  # RSI value above which we consider selling
PORTFOLIO_TRADE_PERCENTAGE = 0.01  # Percentage of portfolio to trade per transaction
MINIMUM_TRADE = 100  # Minimum number of shares to trade

def visualize():
    for ticker in portfolio:
        aapl_data = trade_log[trade_log['Ticker'] == ticker]
        # Create the figure and the axis
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot stock price on the first axis (ax1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price', color='tab:blue')
        ax1.plot(aapl_data['Date'], aapl_data['Price'], color='tab:blue', label='Stock Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis for the shares owned of AAPL
        ax2 = ax1.twinx()
        ax2.set_ylabel('Shares Owned', color='tab:orange')
        ax2.plot(aapl_data['Date'], aapl_data['Stock Holdings'], color='tab:orange', label='Shares Owned')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Add a title and show the plot
        plt.title(f'{ticker} Price vs Shares Owned Over Time')
        fig.tight_layout()
        plt.show()
    
# Trade execution function to handle multiple stocks
def execute_trade(date, ticker, action, price, shares_to_trade):
    print(f'{date}: {action} {shares_to_trade} shares of {ticker} @${price}')
    global cash_portion, equity_portion, portfolio
    trade_cost = price * shares_to_trade
    if action == 'buy' and cash_portion >= trade_cost:
        # If buying, update shares and subtract the cost from cash
        portfolio[ticker] += shares_to_trade
        cash_portion -= trade_cost
        log_trade(date, ticker, 'Buy', price, shares_to_trade)
    elif action == 'sell' and portfolio[ticker] >= shares_to_trade:
        # If selling, update shares and add the proceeds to cash
        portfolio[ticker] -= shares_to_trade
        cash_portion += trade_cost
        log_trade(date, ticker, 'Sell', price, shares_to_trade)
    else:
        log_trade(date, ticker, 'Hold', price, 0)

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

def log_trade(date, ticker, action, price, shares_traded):
    # Logs the details of a trade into a global trade log DataFrame
    global cash_portion
    global trade_log
    global equity_portion

    stock_holdings = portfolio[ticker]

    # Update equity portion with new prices
    equity_portion = 0
    for stock in portfolio:
        stock_price = stock_data[stock].loc[stock_data[stock]['Date'] == date, 'Close/Last'].iloc[0]
        equity_portion += portfolio[stock] * stock_price

    new_entry = {
        'Date': date,
        'Ticker': ticker,
        'Action': action,
        'Price': price,
        'Shares Traded': shares_traded,
        'Total Shares': portfolio[ticker],
        'Cash': cash_portion,
        'Portfolio($)': cash_portion + equity_portion,  # Total value includes cash and stock value
        'Stock Holdings': stock_holdings,
        'All Time ROI': round(((cash_portion + equity_portion) - initial_cash) / initial_cash, 3)
    }
    trade_log = pd.concat([trade_log, pd.DataFrame([new_entry])], ignore_index=True)

def calculate_trade_amount(price, portfolio):
    # Calculates the number of shares to trade based on portfolio trade percentage
    if price * portfolio > 0:
        return math.ceil((price * portfolio * PORTFOLIO_TRADE_PERCENTAGE) / price)
    return MINIMUM_TRADE  # If no current shares, default to minimum trade amount

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
cash_portion = initial_cash
equity_portion = 0
portfolio = {ticker: 0 for ticker in stock_tickers}  # Dictionary to track shares for each stock

# Initialize the logging DataFrame to store trade history
columns = ['Date', 'Ticker', 'Action', 'Price', 'Shares Traded', 'Total Shares', 'Cash', 'Portfolio($)', 'Stock Holdings', 'All Time ROI']
trade_log = pd.DataFrame(columns=columns)
trade_log['Date'] = pd.to_datetime(trade_log['Date'])

# Loop through each date, and for each date, evaluate all stocks
for date in all_dates:
    for ticker, df in stock_data.items():
        if date in df.index:
            row = df.iloc[date]
            stock_trading_price = row['Close/Last']
            if should_buy(row):
                shares_to_trade = calculate_trade_amount(stock_trading_price, portfolio[ticker])
                execute_trade(row['Date'], ticker, 'buy', stock_trading_price, shares_to_trade)
            elif should_sell(row):
                shares_to_trade = calculate_trade_amount(stock_trading_price, portfolio[ticker])
                execute_trade(row['Date'], ticker, 'sell', stock_trading_price, shares_to_trade)
            else:
                log_trade(row['Date'], ticker, 'Hold', stock_trading_price, 0)

# Print the first 300 entries of the trade log to check the trades
print(trade_log)
trade_log.to_excel('trade_log.xlsx', index=False, engine='openpyxl')
visualize()