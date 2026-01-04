"""
LSTM Model for Stock/Crypto Price Prediction
Implements an LSTM neural network for time series prediction
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class LSTMPredictor:
    """LSTM-based price predictor for stocks and cryptocurrencies"""
    
    def __init__(self, sequence_length=60):
        """
        Initialize LSTM Predictor
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaled_data = None
    
    def prepare_data(self, data, train_size=0.8):
        """
        Prepare data for LSTM training
        
        Args:
            data (pandas.DataFrame): Price data
            train_size (float): Proportion of data to use for training
        
        Returns:
            tuple: Training and testing data
        """
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * train_size)
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        # Reshape for LSTM [samples, time steps, features]
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_model(self, units=50, dropout=0.2, layers=2):
        """
        Build LSTM model
        
        Args:
            units (int): Number of LSTM units
            dropout (float): Dropout rate
            layers (int): Number of LSTM layers
        """
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=units, return_sequences=(layers > 1), 
                           input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, layers):
            return_seq = (i < layers - 1)
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(dropout))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model
    
    def train(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the LSTM model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation data proportion
            verbose (int): Verbosity mode
        
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return history
    
    def predict(self, data=None):
        """
        Make predictions
        
        Args:
            data (numpy.array): Data to predict on (uses test data if None)
        
        Returns:
            numpy.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        if data is None:
            data = self.X_test
        
        predictions = self.model.predict(data)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def predict_next(self, recent_data, steps=1):
        """
        Predict future prices
        
        Args:
            recent_data (pandas.DataFrame): Recent price data
            steps (int): Number of future steps to predict
        
        Returns:
            list: Future price predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Scale the recent data
        scaled_recent = self.scaler.transform(recent_data)
        
        predictions = []
        current_sequence = scaled_recent[-self.sequence_length:].copy()
        
        for _ in range(steps):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(current_batch, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred, axis=0)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten().tolist()
    
    def evaluate(self):
        """
        Evaluate model performance on test data
        
        Returns:
            float: Test loss
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return test_loss
    
    def save_model(self, filepath='lstm_model.keras'):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_model.keras'):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to model file
        """
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")
