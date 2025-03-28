import pandas as pd
import numpy as np
import logging

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            # Moving Averages
            self.indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
            self.indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
            self.indicators['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.indicators['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            self.indicators['MACD'] = macd
            self.indicators['MACD_Signal'] = signal

            # Bollinger Bands
            middle = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            self.indicators['BB_Upper'] = middle + (std * 2)
            self.indicators['BB_Middle'] = middle
            self.indicators['BB_Lower'] = middle - (std * 2)

            # Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            self.indicators['Stoch_K'] = k
            self.indicators['Stoch_D'] = k.rolling(window=3).mean()

            return self.indicators

        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 'HOLD'

        try:
            # RSI signals
            signals.loc[self.indicators['RSI'] < 30, 'Signal'] = 'BUY'
            signals.loc[self.indicators['RSI'] > 70, 'Signal'] = 'SELL'

            # MACD signals
            signals.loc[self.indicators['MACD'] > self.indicators['MACD_Signal'], 'Signal'] = 'BUY'
            signals.loc[self.indicators['MACD'] < self.indicators['MACD_Signal'], 'Signal'] = 'SELL'

            # Moving Average signals
            signals.loc[self.indicators['SMA_20'] > self.indicators['SMA_50'], 'Signal'] = 'BUY'
            signals.loc[self.indicators['SMA_20'] < self.indicators['SMA_50'], 'Signal'] = 'SELL'

            # Bollinger Bands signals
            signals.loc[df['Close'] < self.indicators['BB_Lower'], 'Signal'] = 'BUY'
            signals.loc[df['Close'] > self.indicators['BB_Upper'], 'Signal'] = 'SELL'

            return signals['Signal']

        except Exception as e:
            logging.error(f"Error generating technical signals: {str(e)}")
            raise