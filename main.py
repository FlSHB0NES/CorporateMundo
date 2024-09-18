import pandas as pd
import math

# Constants for RSI thresholds and portfolio management
RSI_BUY = 30  # RSI value below which we consider buying
RSI_SELL = 70  # RSI value above which we consider selling
PORTFOLIO_TRADE_PERCENTAGE = 0.4  # Percentage of portfolio to trade per transaction
MINIMUM_TRADE = 5  # Minimum number of shares to trade

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

def log_trade(date, action, price, shares, cash, stock_holdings):
    # Logs the details of a trade into a global trade log DataFrame
    global trade_log
    new_entry = {
        'Date': date,
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

# Load the stock data from the CSV file
data_path = './data/HistoricalData_1726632594353.csv'
apple_stock_data = pd.read_csv(data_path)

# Clean the data: Convert 'Date' to datetime, remove '$' from financial columns, and convert to float
apple_stock_data['Date'] = pd.to_datetime(apple_stock_data['Date'])
financial_columns = ['Close/Last', 'Open', 'High', 'Low']
for col in financial_columns:
    apple_stock_data[col] = apple_stock_data[col].replace('[\\$,]', '', regex=True).astype(float)

# Sort the data by date in ascending order
apple_stock_data = apple_stock_data.sort_values(by='Date', ascending=True)

# Add the RSI (Relative Strength Index) to the stock data
apple_stock_data['RSI'] = calculate_rsi(apple_stock_data)

# Engage in trading based on the RSI values

# Initialize the logging DataFrame to store trade history
columns = ['Date', 'Action', 'Price', 'Shares', 'Cash', 'Total Value', 'Stock Holdings']
trade_log = pd.DataFrame(columns=columns)
trade_log['Date'] = pd.to_datetime(trade_log['Date'])

# Assuming you start with some initial cash and no stocks
initial_cash = 10000  # Example starting cash
current_cash = initial_cash
current_shares = 0

# Loop through each row of stock data to evaluate and execute trades
for index, row in apple_stock_data.iterrows():
    if should_buy(row):
        stock_trading_price = row['Close/Last']
        shares_to_trade = calculate_trade_amount(stock_trading_price, current_shares)
        execute_trade(row['Date'], 'buy', stock_trading_price, shares_to_trade)

    elif should_sell(row):
        stock_trading_price = row['Close/Last']
        shares_to_trade = calculate_trade_amount(stock_trading_price, current_shares)
        execute_trade(row['Date'], 'sell', stock_trading_price, shares_to_trade)

# Print the first 300 entries of the trade log to check the trades
print(trade_log.head(300))
