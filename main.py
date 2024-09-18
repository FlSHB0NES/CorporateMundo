import pandas as pd

RSI_BUY = 30
RSI_SELL = 70
PORTFOLIO_TRADE_PERCENTAGE = 0.1
MINIMUM_TRADE = 100

def calculate_rsi(data, window=14):
    # Calculate the daily price changes
    delta = data['Close/Last'].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the Exponential Moving Average (EMA) of gains and losses
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()

    # Calculate the RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def should_buy(row, buy_threshold=RSI_BUY):
    return row['RSI'] < buy_threshold

def should_sell(row, sell_threshold=RSI_SELL):
    return row['RSI'] > sell_threshold

def execute_trade(date, action, price, shares_to_trade):
    global current_cash, current_shares
    trade_cost = price * shares_to_trade
    if action == 'buy' and current_cash >= trade_cost:
        current_shares += shares_to_trade
        current_cash -= trade_cost
        log_trade(date, 'Buy', price, shares_to_trade, current_cash, current_shares)
    elif action == 'sell' and current_shares >= shares_to_trade:
        current_shares -= shares_to_trade
        current_cash += trade_cost
        log_trade(date, 'Sell', price, shares_to_trade, current_cash, current_shares)

def log_trade(date, action, price, shares, cash, stock_holdings):
    global trade_log
    new_entry = {
        'Date': date,
        'Action': action,
        'Price': price,
        'Shares': shares,
        'Cash': cash,
        'Total Value': cash + price * stock_holdings,
        'Stock Holdings': stock_holdings
    }
    trade_log = trade_log.append(new_entry, ignore_index=True)

def calculate_trade_amount(date):
    # Assuming today's date is the current day
    current_date = pd.to_datetime(date)
    # Retrieve the current stock price and holdings from the log
    current_day_data = trade_log.loc[trade_log['Date'] == current_date]

    if not current_day_data.empty:
        return current_day_data['Close/Last'].values[0] * current_day_data['Holdings'].values[0]
    
    return 5

# Load the data from the CSV file
data_path = './data/HistoricalData_1726606875880.csv'
apple_stock_data = pd.read_csv(data_path)

# Clean the data: Convert date to datetime and remove '$' sign from financial columns and convert to float
apple_stock_data['Date'] = pd.to_datetime(apple_stock_data['Date'])
financial_columns = ['Close/Last', 'Open', 'High', 'Low']
for col in financial_columns:
    apple_stock_data[col] = apple_stock_data[col].replace('[\\$,]', '', regex=True).astype(float)

# Add RSI Field
apple_stock_data['RSI'] = calculate_rsi(apple_stock_data)

# Engage in trading

# Initialize the logging DataFrame
columns = ['Date', 'Action', 'Price', 'Shares', 'Cash', 'Total Value', 'Stock Holdings']
trade_log = pd.DataFrame(columns=columns)
trade_log['Date'] = pd.to_datetime(trade_log['Date'])


# Assuming you start with some initial cash and no stocks
initial_cash = 10000  # Example starting cash
current_cash = initial_cash
current_shares = 0

for index, row in apple_stock_data.iterrows():
    print(row['RSI'])
    if should_buy(row):
        amount_to_trade = calculate_trade_amount(row['Date'])
        execute_trade(row['Date'], 'buy', row['Close/Last'], amount_to_trade)

print(trade_log.head(10))