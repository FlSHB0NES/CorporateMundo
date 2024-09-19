import pandas as pd
import matplotlib.pyplot as plt

# Mocking up the trade_log data for plotting purposes as the original data isn't available
# Normally, this would come from actual trading data

# Generating sample data to represent trade log
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Stock Holdings': [i * 10 for i in range(100)],  # Simulating stock holdings
    'Price': [150 + i*0.5 for i in range(100)]  # Simulating stock price increase
}

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
