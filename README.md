# CorporateMundo

## Overview
CorporateMundo is a stock trading simulation platform that implements various trading algorithms, such as reinforcement learning (RL) and technical analysis, to evaluate and execute trades. The platform supports multiple stocks and uses indicators like RSI to make buy, sell, and hold decisions. The project is designed to allow experimentation with different strategies and their effects on a simulated portfolio.

## Features
- **Multiple Stock Support**: Trade across multiple stocks simultaneously.
- **Reinforcement Learning Algorithms**: RL-based decision-making for optimizing trade strategies.
- **Technical Indicators**: Uses RSI and other indicators for trade decisions.
- **Visualization**: Visual representation of trading history and stock performance.
  
## Repository Structure
- `main.py`: Core trading logic and RSI-based trading strategy.
- `RL_algo.py`: Implementation of reinforcement learning algorithms for trading.
- `visualizer.py`: Visualization tools to plot trading results.
- `data/`: Directory containing historical stock data in CSV format.
- `trade_log.xlsx`: Log of trades executed during simulation.

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FlSHB0NES/CorporateMundo.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Usage
Modify the `main.py` file to select different stocks, change trading strategies, or adjust portfolio settings. The trading algorithms are configured to iterate through each stock on a daily basis, making buy/sell decisions based on the chosen strategy.

## Contributions
Feel free to contribute by creating pull requests or opening issues for bugs and feature requests.

## License
This project is licensed under the MIT License.

--- 

You can adjust this template based on more specific details or requirements you want to include in your README.
