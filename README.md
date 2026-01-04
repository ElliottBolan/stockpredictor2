# Stock and Crypto Predictor

A comprehensive stock and cryptocurrency price predictor using real-time price data APIs and LSTM (Long Short-Term Memory) neural networks. The application provides real-time price information, AI-powered price predictions, and the latest news for selected stocks or cryptocurrencies.

## Features

- 📊 **Real-time Price Data**: Fetches current and historical price data for stocks and cryptocurrencies using Yahoo Finance API
- 🤖 **LSTM Price Prediction**: Uses deep learning (LSTM neural network) to predict future prices based on historical patterns
- 📰 **Latest News**: Displays real-time news articles for selected stocks/cryptocurrencies, or general market news when no symbol is selected
- 💹 **Support for Stocks and Crypto**: Works with any stock ticker (e.g., AAPL, TSLA) or cryptocurrency (e.g., BTC-USD, ETH-USD)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ElliottBolan/stockpredictor2.git
   cd stockpredictor2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys (Optional but recommended)**
   
   For real-time news functionality, you need a NewsAPI key:
   
   - Sign up for a free API key at [https://newsapi.org/](https://newsapi.org/)
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API key:
     ```
     NEWS_API_KEY=your_actual_api_key_here
     ```

## Usage

### Analyze a Specific Stock or Cryptocurrency

```bash
python main.py SYMBOL
```

**Examples:**

```bash
# Analyze Apple stock
python main.py AAPL

# Analyze Bitcoin
python main.py BTC-USD

# Analyze Tesla stock
python main.py TSLA

# Analyze Ethereum
python main.py ETH-USD
```

### View General Market News

Run without any arguments to see general market news:

```bash
python main.py
```

## Output

When analyzing a symbol, the application provides:

1. **Real-time Price Data**
   - Current price
   - Previous close
   - Open price
   - Day high/low
   - Trading volume

2. **LSTM Price Predictions**
   - Model trained on 1 year of historical data
   - Predictions for the next 5 days
   - Model performance metrics

3. **Latest News Articles**
   - Recent news related to the selected stock/crypto
   - Source, publication date, and description
   - Direct links to full articles

## Project Structure

```
stockpredictor2/
├── main.py              # Main application entry point
├── price_fetcher.py     # Real-time and historical price data fetching
├── lstm_predictor.py    # LSTM neural network for price prediction
├── news_fetcher.py      # News article fetching and display
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment configuration
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technical Details

### Price Data
- Uses `yfinance` library to fetch real-time and historical data
- Supports any ticker available on Yahoo Finance
- Configurable time periods and intervals

### LSTM Model
- Built with TensorFlow/Keras
- 2-layer LSTM architecture with dropout for regularization
- Trained on 1 year of historical closing prices
- Uses 60-day sequences to predict future prices
- Early stopping to prevent overfitting

### News Integration
- Uses NewsAPI for real-time news articles
- Falls back to mock news when API key is not configured
- Displays up to 5 articles for specific symbols
- Shows general business news when no symbol is selected

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **tensorflow**: Deep learning framework for LSTM
- **scikit-learn**: Data preprocessing and scaling
- **yfinance**: Yahoo Finance API wrapper for price data
- **requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **newsapi-python**: NewsAPI wrapper for news articles
- **matplotlib**: Data visualization (future use)

## Limitations

- Predictions are based on historical patterns and should not be used as financial advice
- News functionality requires a valid NewsAPI key (free tier available)
- LSTM model requires at least 60 days of historical data
- Prediction accuracy depends on market conditions and data quality

## Future Enhancements

- Add data visualization (price charts, prediction graphs)
- Support for multiple technical indicators
- Model persistence (save/load trained models)
- Web interface for easier interaction
- Support for portfolio tracking
- Sentiment analysis on news articles

## License

MIT License - Feel free to use and modify as needed.

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Always do your own research and consult with financial professionals before making investment decisions.