import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def create_features(data, ticker, seq_length):
    """Create features with technical indicators using ta library"""
    features = pd.DataFrame(index=data.index)
    t = ticker
    
    # Price data
    close = data[f'{t}_Close']
    high = data[f'{t}_High']
    low = data[f'{t}_Low']
    
    # Technical indicators using ta library
    rsi = RSIIndicator(close, window=14)
    macd = MACD(close)
    atr = AverageTrueRange(high, low, close, window=14)
    
    features[f'{t}_RSI'] = rsi.rsi()
    features[f'{t}_MACD'] = macd.macd_diff()
    features[f'{t}_ATR'] = atr.average_true_range()
    
    # Volatility calculation
    features[f'{t}_VOL'] = close.rolling(21).std()
    
    # Lag features
    for lag in [1, 5, 21]:
        features[f'{t}_LAG_{lag}'] = close.shift(lag)
        
    return features.dropna()

class VolatilityAwareHybrid:
    def __init__(self, tickers, seq_length=21):
        self.tickers = tickers
        self.seq_length = seq_length
        self.scaler = RobustScaler()
        self.models = {}
        
    def prepare_data(self, data, ticker):
        """Create sequences with volatility awareness"""
        t = ticker
        features = create_features(data, t, self.seq_length)
        prices = data[f'{t}_Close'].values.reshape(-1, 1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X_seq, X_xgb, y = [], [], []
        for i in range(self.seq_length, len(features)-1):
            X_seq.append(scaled_features[i-self.seq_length:i])
            X_xgb.append(scaled_features[i])
            price_change = (prices[i+1] - prices[i]) / prices[i]
            y.append(price_change[0])
            
        return np.array(X_seq), np.array(X_xgb), np.array(y)
    
    def build_model(self, input_shape, xgb_input_dim, volatility_level):
        """Dynamically weighted hybrid model"""
        lstm_input = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(lstm_input)
        x = LSTM(64)(x)
        
        xgb_input = Input(shape=(xgb_input_dim,))
        d = Dense(64, activation='relu')(xgb_input)
        
        combined = Concatenate()([x, d])
        
        if volatility_level == 'high':
            outputs = Dense(1, activation='tanh')(combined)
        else:
            outputs = Dense(1)(combined)
            
        model = Model(inputs=[lstm_input, xgb_input], outputs=outputs)
        
        if volatility_level == 'high':
            loss = 'binary_crossentropy'
            lr = 0.001
        else:
            loss = 'mse'
            lr = 0.0005
            
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
        return model
    
    def train(self, data):
        """Train with volatility-based adaptation"""
        for t in self.tickers:
            X_seq, X_xgb, y = self.prepare_data(data, t)
            
            # Calculate volatility
            atr_values = data[f'{t}_ATR'].values[self.seq_length:-1]
            volatility_ratio = np.mean(atr_values) / np.median(atr_values)
            volatility_level = 'high' if volatility_ratio > 1.2 else 'low'
            
            # Build appropriate model
            self.models[t] = self.build_model(
                input_shape=(X_seq.shape[1], X_seq.shape[2]),
                xgb_input_dim=X_xgb.shape[1],
                volatility_level=volatility_level
            )
            
            # Train
            print(f"Training {t} with {volatility_level} volatility setting")
            self.models[t].fit(
                [X_seq, X_xgb], y,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                verbose=1
            )
    
    def predict(self, data, ticker):
        """Make volatility-aware predictions"""
        X_seq, X_xgb, _ = self.prepare_data(data, ticker)
        return self.models[ticker].predict([X_seq[-1:], X_xgb[-1:]])
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import yfinance as yf

# def fetch_stock_data(tickers, start_date, end_date):
#     """Fetch stock data with NSE fallback"""
#     data = {}
#     for ticker in tickers:
#         # Try global ticker first
#         df = yf.download(ticker, start=start_date, end=end_date, progress=False)
#         if df.empty:
#             # Try NSE fallback
#             df = yf.download(f"{ticker}.NS", start=start_date, end=end_date, progress=False)
#         if not df.empty:
#             data[ticker] = df['Adj Close']
#         else:
#             print(f"Warning: No data for {ticker}")
#     return pd.DataFrame(data).dropna()

# def time_series_preprocessor(data, seq_length=60, test_size=0.2):
#     """Enhanced time-series preprocessing with proper train-test split"""
#     # Split data first to prevent leakage
#     train_size = int(len(data) * (1 - test_size))
#     train_data = data.iloc[:train_size]
#     test_data = data.iloc[train_size - seq_length:]  # Maintain sequence
    
#     # Scale datasets
#     scalers = {}
#     train_scaled = pd.DataFrame()
#     test_scaled = pd.DataFrame()
    
#     for ticker in data.columns:
#         scaler = MinMaxScaler()
#         train_scaled[ticker] = scaler.fit_transform(train_data[[ticker]]).flatten()
#         test_scaled[ticker] = scaler.transform(test_data[[ticker]]).flatten()
#         scalers[ticker] = scaler

#     # Create sequences
#     def create_sequences(dataset):
#         X, y = [], []
#         for i in range(seq_length, len(dataset)):
#             X.append(dataset.iloc[i-seq_length:i].values)
#             y.append(dataset.iloc[i].values)
#         return np.array(X), np.array(y)
    
#     X_train, y_train = create_sequences(train_scaled)
#     X_test, y_test = create_sequences(test_scaled)
    
#     return (X_train, y_train), (X_test, y_test), scalers

# def build_lstm_model(input_shape, output_units):
#     """Enhanced LSTM architecture with regularization"""
#     model = Sequential([
#         LSTM(128, return_sequences=True, input_shape=input_shape,
#              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#         Dropout(0.4),
#         LSTM(64, return_sequences=True),
#         Dropout(0.3),
#         LSTM(32),
#         Dense(64, activation='relu'),
#         Dropout(0.2),
#         Dense(output_units)
#     ])
    
#     optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
#     model.compile(optimizer=optimizer,
#                  loss='mse',
#                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
#     return model

# def evaluate_predictions(model, X_test, y_test, scalers):
#     """Comprehensive model evaluation with financial metrics"""
#     # Generate predictions
#     y_pred_scaled = model.predict(X_test)
    
#     # Inverse transform predictions
#     tickers = list(scalers.keys())
#     y_pred = np.zeros_like(y_pred_scaled)
#     y_test_actual = np.zeros_like(y_test)
    
#     for i, ticker in enumerate(tickers):
#         y_pred[:, i] = scalers[ticker].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()
#         y_test_actual[:, i] = scalers[ticker].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()
    
#     # Calculate returns
#     pred_returns = pd.DataFrame(y_pred).pct_change().dropna()
#     actual_returns = pd.DataFrame(y_test_actual).pct_change().dropna()
    
#     # Calculate metrics
#     metrics = {
#         'MSE': mean_squared_error(y_test_actual, y_pred),
#         'RMSE': np.sqrt(mean_squared_error(y_test_actual, y_pred)),
#         'MAE': mean_absolute_error(y_test_actual, y_pred),
#         'R2': r2_score(y_test_actual, y_pred),
#         'Direction_Accuracy': np.mean(np.sign(actual_returns) == np.sign(pred_returns)),
#         'Sharpe_Ratio': (pred_returns.mean() / pred_returns.std()).mean()
#     }
    
#     return metrics, y_pred

# def train_and_evaluate_model(X_train, y_train, X_test, y_test, scalers, epochs=100, batch_size=64):
#     """Train model with early stopping and return evaluation metrics"""
#     model = build_lstm_model((X_train.shape[1], X_train.shape[2]), X_train.shape[2])
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
#     history = model.fit(
#         X_train, y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_split=0.2,
#         callbacks=[early_stop],
#         verbose=1
#     )
    
#     metrics, predictions = evaluate_predictions(model, X_test, y_test, scalers)
#     return model, metrics, predictions, history

# def predict_next_day_price(model, data, scalers, seq_length=60):
#     """Predict next day's price for all assets"""
#     last_sequence = data[-seq_length:].copy()
    
#     # Scale the last sequence
#     scaled_sequence = pd.DataFrame()
#     for ticker in data.columns:
#         scaled_values = scalers[ticker].transform(last_sequence[[ticker]])
#         scaled_sequence[ticker] = scaled_values.flatten()
    
#     # Reshape for prediction
#     X_pred = scaled_sequence.values.reshape(1, seq_length, len(data.columns))
    
#     # Make prediction and inverse transform
#     pred_scaled = model.predict(X_pred)
#     predictions = {}
    
#     for i, ticker in enumerate(data.columns):
#         pred_value = scalers[ticker].inverse_transform(pred_scaled[0, i].reshape(-1, 1))[0, 0]
#         predictions[ticker] = pred_value
    
#     return predictions

# # Save and load functions remain the same
# def save_model_and_scalers(model, scalers, base_filename):
#     """Save model and scalers"""
#     model.save(f"{base_filename}_model.h5")
#     np.save(f"{base_filename}_scalers.npy", scalers)

# def load_model_and_scalers(base_filename):
#     """Load model and scalers"""
#     model = load_model(f"{base_filename}_model.h5")
#     scalers = np.load(f"{base_filename}_scalers.npy", allow_pickle=True).item()
#     return model, scalers

# import numpy as np
# import pandas as pd 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler

# def preprocess_multi_asset(data, sequence_length=60):
#     """
#     Preprocess data for LSTM model with better error handling
#     """
#     scalers = {}
#     scaled_data = pd.DataFrame()
    
#     # Remove any NaN values first
#     data = data.dropna()
    
#     for ticker in data.columns:
#         scaler = MinMaxScaler()
#         values = data[ticker].values.reshape(-1, 1)
#         # Check for any infinite values
#         values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
#         scaled_values = scaler.fit_transform(values)
#         scaled_data[ticker] = scaled_values.flatten()
#         scalers[ticker] = scaler
    
#     X, y = [], []
#     for i in range(sequence_length, len(scaled_data)):
#         X.append(scaled_data.iloc[i-sequence_length:i].values)
#         y.append(scaled_data.iloc[i].values)
    
#     return np.array(X), np.array(y), scalers

# def build_multi_lstm(input_shape, output_units):
#     model = Sequential([
#         LSTM(64, return_sequences=True, input_shape=input_shape),
#         Dropout(0.3),
#         LSTM(32),
#         Dense(output_units)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model



# # utils/lstm_model.py
# import numpy as np
# import pandas as pd
# import os
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # ----- Data Fetching with NSE Fallback -----
# def fetch_stock_data(tickers, start_date, end_date):
#     """
#     Fetch stock data for multiple tickers from Yahoo Finance.
#     Tries global ticker first; if data is empty, then appends '.NS'.
#     Returns a DataFrame of Close prices.
#     """
#     data = {}
#     for ticker in tickers:
#         # Try global ticker
#         df = yf.download(ticker, start=start_date, end=end_date, progress=False)
#         if df.empty:
#             # Try NSE fallback
#             df = yf.download(f"{ticker}.NS", start=start_date, end=end_date, progress=False)
#         if not df.empty:
#             data[ticker] = df['Close']
#         else:
#             print(f"Warning: No data for {ticker}")
#     # Convert dictionary to DataFrame and drop rows with missing data
#     if data:
#         return pd.DataFrame(data).dropna()
#     else:
#         return pd.DataFrame()

# # ----- Preprocessing -----
# def preprocess_data(data, sequence_length=60):
#     """
#     Scale the data and create sequences for LSTM training.
#     Returns X, y arrays and the scaler used.
#     """
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data)
    
#     X, y = [], []
#     for i in range(sequence_length, len(scaled_data)):
#         X.append(scaled_data[i-sequence_length:i])
#         y.append(scaled_data[i, 0])  # Predicting the first asset's price
#     return np.array(X), np.array(y), scaler

# # ----- Build LSTM Model -----
# def build_lstm_model(input_shape):
#     """
#     Build and compile the LSTM model.
#     """
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     return model

# # ----- Train and Evaluate Model -----
# def train_and_evaluate_model(X, y, tickers, epochs=10, batch_size=32):
#     """
#     Split the data into training and testing sets,
#     train the LSTM model, evaluate it, and save the trained model.
#     Returns the model filename and evaluation metrics.
#     """
#     split_ratio = 0.8
#     train_size = int(len(X) * split_ratio)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
    
#     model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
#     # Evaluate the model on test data
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     # Save the trained model
#     model_filename = f"lstm_model_{'_'.join(sorted(tickers))}.h5"
#     model.save(model_filename)
#     return model_filename, (mse, rmse, mae, r2)

# # ----- Save and Load Scaler -----
# def save_scaler(scaler, filename):
#     """
#     Save scaler parameters (min and scale) to a .npy file.
#     """
#     np.save(filename, [scaler.min_, scaler.scale_])

# def load_scaler(filename):
#     """
#     Load scaler parameters and reconstruct a MinMaxScaler.
#     """
#     scaler = MinMaxScaler()
#     scaler.min_, scaler.scale_ = np.load(filename, allow_pickle=True)
#     return scaler

# # ----- Prediction -----
# def predict_next_day_price(model_filename, scaler_filename, data, tickers, sequence_length=60):
#     """
#     Load the saved model and scaler, then predict the next day's price
#     for the first ticker.
#     """
#     model = load_model(model_filename)
#     scaler = load_scaler(scaler_filename)
    
#     last_sequence = data[-sequence_length:]
#     last_sequence_scaled = scaler.transform(last_sequence)
#     last_sequence_reshaped = np.reshape(last_sequence_scaled, (1, sequence_length, len(tickers)))
    
#     predicted_price_scaled = model.predict(last_sequence_reshaped)
#     dummy_array = np.zeros((1, len(tickers)))
#     dummy_array[0, 0] = predicted_price_scaled[0][0]
#     predicted_price = scaler.inverse_transform(dummy_array)[0][0]
#     return predicted_price
