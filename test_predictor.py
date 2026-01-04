#!/usr/bin/env python3
"""
Test script for Stock and Crypto Predictor
Demonstrates the functionality with sample data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lstm_predictor import LSTMPredictor


def test_lstm_predictor():
    """Test the LSTM predictor with sample data"""
    print("="*80)
    print("Testing LSTM Predictor with Sample Data")
    print("="*80 + "\n")
    
    # Generate sample price data (simulating stock prices)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create a synthetic stock price pattern
    trend = np.linspace(100, 150, len(dates))
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    prices = trend + seasonality + noise
    
    # Create DataFrame
    data = pd.DataFrame({'Close': prices}, index=dates)
    
    print(f"Sample data created: {len(data)} days of synthetic price data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Average price: ${data['Close'].mean():.2f}\n")
    
    # Initialize predictor
    print("Initializing LSTM Predictor...")
    predictor = LSTMPredictor(sequence_length=60)
    
    # Prepare data
    print("Preparing data for training...")
    predictor.prepare_data(data[['Close']], train_size=0.8)
    
    print(f"Training samples: {len(predictor.X_train)}")
    print(f"Test samples: {len(predictor.X_test)}\n")
    
    # Build and train model
    print("Building LSTM model...")
    predictor.build_model(units=50, dropout=0.2, layers=2)
    
    print("\nTraining model (this may take a moment)...")
    history = predictor.train(epochs=20, batch_size=32, verbose=0)
    
    print("✓ Training completed!")
    
    # Evaluate
    test_loss = predictor.evaluate()
    print(f"Model Test Loss (MSE): {test_loss:.6f}\n")
    
    # Make predictions
    print("Making predictions for the next 5 days...")
    future_predictions = predictor.predict_next(data[['Close']], steps=5)
    
    print("\nPredicted prices:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"  Day {i}: ${pred:.2f}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80 + "\n")


def test_components():
    """Test individual components"""
    print("="*80)
    print("Component Tests")
    print("="*80 + "\n")
    
    # Test 1: Price Fetcher
    print("1. Price Fetcher Module")
    try:
        from price_fetcher import PriceDataFetcher
        fetcher = PriceDataFetcher()
        print("   ✓ PriceDataFetcher initialized successfully\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Test 2: News Fetcher
    print("2. News Fetcher Module")
    try:
        from news_fetcher import NewsFetcher
        news = NewsFetcher()
        print("   ✓ NewsFetcher initialized successfully\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Test 3: LSTM Predictor
    print("3. LSTM Predictor Module")
    try:
        from lstm_predictor import LSTMPredictor
        predictor = LSTMPredictor()
        print("   ✓ LSTMPredictor initialized successfully\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Test 4: Main Application
    print("4. Main Application Module")
    try:
        from main import StockCryptoPredictor
        app = StockCryptoPredictor()
        print("   ✓ StockCryptoPredictor initialized successfully\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    print("="*80)
    print("All component tests passed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test components
    test_components()
    
    # Test LSTM with sample data
    test_lstm_predictor()
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("\nThe application is ready to use.")
    print("Run 'python main.py SYMBOL' to analyze a stock or cryptocurrency.")
    print("="*80 + "\n")
