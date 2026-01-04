"""
Stock and Crypto Predictor
Main application integrating price fetching, LSTM prediction, and news display
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from price_fetcher import PriceDataFetcher
from lstm_predictor import LSTMPredictor
from news_fetcher import NewsFetcher


class StockCryptoPredictor:
    """Main application class for stock and crypto prediction"""
    
    def __init__(self):
        self.price_fetcher = PriceDataFetcher()
        self.lstm_predictor = LSTMPredictor(sequence_length=60)
        self.news_fetcher = NewsFetcher()
    
    def run(self, symbol=None):
        """
        Run the predictor application
        
        Args:
            symbol (str): Stock/crypto ticker symbol (None for general news only)
        """
        print("\n" + "="*80)
        print("STOCK AND CRYPTO PREDICTOR")
        print("Real-time Price Data, LSTM Predictions, and News")
        print("="*80 + "\n")
        
        if symbol:
            self._analyze_symbol(symbol)
        else:
            self._show_general_news()
    
    def _analyze_symbol(self, symbol):
        """
        Analyze a specific stock or cryptocurrency
        
        Args:
            symbol (str): Ticker symbol
        """
        print(f"Analyzing: {symbol}")
        print("-"*80 + "\n")
        
        # 1. Fetch real-time price data
        print("📊 REAL-TIME PRICE DATA")
        print("-"*80)
        current_price_data = self.price_fetcher.get_current_price(symbol)
        
        if current_price_data:
            print(f"Name: {current_price_data['name']}")
            print(f"Symbol: {current_price_data['symbol']}")
            print(f"Current Price: ${current_price_data['current_price']}")
            print(f"Previous Close: ${current_price_data['previous_close']}")
            print(f"Open: ${current_price_data['open']}")
            print(f"Day High: ${current_price_data['day_high']}")
            print(f"Day Low: ${current_price_data['day_low']}")
            print(f"Volume: {current_price_data['volume']:,}" if isinstance(current_price_data['volume'], (int, float)) else f"Volume: {current_price_data['volume']}")
        else:
            print(f"Could not fetch current price data for {symbol}")
            print("Please check the symbol and try again.")
            return
        
        # 2. Fetch historical data and make predictions
        print("\n🤖 LSTM PRICE PREDICTION")
        print("-"*80)
        print("Fetching historical data...")
        
        historical_data = self.price_fetcher.fetch_data(symbol, period="1y", interval="1d")
        
        if historical_data is not None and len(historical_data) > 60:
            price_data = self.price_fetcher.prepare_data_for_lstm()
            
            print(f"Training LSTM model on {len(historical_data)} days of data...")
            print("This may take a moment...")
            
            try:
                # Prepare and train model
                self.lstm_predictor.prepare_data(price_data, train_size=0.8)
                self.lstm_predictor.build_model(units=50, dropout=0.2, layers=2)
                self.lstm_predictor.train(epochs=30, batch_size=32, verbose=0)
                
                # Make predictions
                test_predictions = self.lstm_predictor.predict()
                
                # Predict next day
                future_predictions = self.lstm_predictor.predict_next(price_data, steps=5)
                
                print("✓ Model trained successfully!")
                print(f"\nPredicted prices for next 5 days:")
                for i, pred in enumerate(future_predictions, 1):
                    print(f"  Day {i}: ${pred:.2f}")
                
                # Evaluate model
                test_loss = self.lstm_predictor.evaluate()
                print(f"\nModel Test Loss (MSE): {test_loss:.6f}")
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                print("Prediction requires sufficient historical data.")
        else:
            print(f"Insufficient historical data for {symbol}. Need at least 60 days.")
        
        # 3. Fetch and display news
        print("\n📰 LATEST NEWS")
        print("-"*80)
        company_name = current_price_data['name'] if current_price_data else None
        news_articles = self.news_fetcher.fetch_symbol_news(symbol, company_name, max_results=5)
        
        if news_articles:
            for i, article in enumerate(news_articles, 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Source: {article['source']} | Published: {article['published_at'][:10]}")
                if article['description']:
                    desc = article['description'][:150] + "..." if len(article['description']) > 150 else article['description']
                    print(f"   {desc}")
                print(f"   URL: {article['url']}")
        else:
            print("No news articles found.")
        
        print("\n" + "="*80 + "\n")
    
    def _show_general_news(self):
        """Show general market news when no symbol is selected"""
        print("No symbol selected - showing general market news")
        print("-"*80 + "\n")
        
        print("📰 GENERAL MARKET NEWS")
        print("-"*80)
        
        news_articles = self.news_fetcher.fetch_general_market_news(max_results=10)
        
        if news_articles:
            for i, article in enumerate(news_articles, 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Source: {article['source']} | Published: {article['published_at'][:10]}")
                if article['description']:
                    desc = article['description'][:150] + "..." if len(article['description']) > 150 else article['description']
                    print(f"   {desc}")
                print(f"   URL: {article['url']}")
        else:
            print("No news articles found.")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    predictor = StockCryptoPredictor()
    
    if len(sys.argv) > 1:
        # Symbol provided as command line argument
        symbol = sys.argv[1].upper()
        predictor.run(symbol)
    else:
        # No symbol - show general news
        print("\nUsage: python main.py [SYMBOL]")
        print("Examples:")
        print("  python main.py AAPL        # Analyze Apple stock")
        print("  python main.py BTC-USD     # Analyze Bitcoin")
        print("  python main.py TSLA        # Analyze Tesla stock")
        print("  python main.py             # Show general market news\n")
        
        predictor.run()


if __name__ == "__main__":
    main()
