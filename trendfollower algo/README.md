# Systematic Multi-Asset Trend-Following & Portfolio Optimization Framework

## Overview
This project implements a semi-automated algorithmic trading system designed for multi-asset trend-following. The framework integrates technical indicator optimization with a quantitative portfolio "Rank Manager" to select high-momentum assets, bridging the gap between individual strategy research and active portfolio management.

## System Architecture

### 1. Strategy Layer & Optimization
Each asset in the universe is modeled using dedicated trend-following logic, optimized via **GridSearch** to identify the most robust parameters. 
* **Signal Generation:** Strategies utilize either a **Triple Exponential Moving Average (3EMA) combined with MACD** or a **Dual Kaufman Adaptive Moving Average (KAMA)** approach.
* **Validation:** Parameters are tuned using a **60/40 Train-Test split** to ensure out-of-sample stability and mitigate overfitting.

### 2. Quantitative Rank Manager
The system employs a proprietary ranking logic to construct the final portfolio:
* **Asset Selection:** The "Rank Manager" filters and ranks assets based on the **Relative Strength Index (RSI)** calculated on returns.
* **Concentration:** The algorithm selects only the **top 4 assets** for trading.
* **Momentum Filter:** A strict threshold is applied, requiring an **RSI > 50** to ensure the system only enters positions with strong positive momentum.

### 3. Execution Model & Workflow
The system is designed for a **semi-automatic operational workflow**:
* **Pre-Market Signal Generation:** The bot is executed prior to the market open to process historical data and generate specific order instructions.
* **Manual Order Placement:** To ensure oversight and risk control, orders are manually placed based on the bot's quantitative output.

## Technical Stack
**Language:** Python 
* **Core Libraries:** Pandas, NumPy (Data Processing), TA-Lib/Pandas_TA (Technical Analysis).
*Skills Demonstrated:** Trend Following, Momentum Strategies, Portfolio Optimization, Hyperparameter Tuning.
