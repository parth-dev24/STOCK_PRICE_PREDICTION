import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataProcessor:
    def __init__(self):
        self.data = None
        self.indian_stocks = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'HDFCBANK.NS': 'HDFC Bank',
            'INFY.NS': 'Infosys',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ICICIBANK.NS': 'ICICI Bank',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'WIPRO.NS': 'Wipro',
            'AXISBANK.NS': 'Axis Bank',
            'SBIN.NS': 'State Bank of India',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'MARUTI.NS': 'Maruti Suzuki',
            'LT.NS': 'Larsen & Toubro',
            'ASIANPAINT.NS': 'Asian Paints',
            'SUNPHARMA.NS': 'Sun Pharma',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'HCLTECH.NS': 'HCL Technologies',
            'NESTLEIND.NS': 'Nestle India',
            'TITAN.NS': 'Titan Company'
        }

    def get_stock_list(self):
        """Return list of available Indian stocks"""
        return [(symbol, name) for symbol, name in self.indian_stocks.items()]

    def fetch_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        try:
            # Append .NS if not already present for Indian stocks
            if not symbol.endswith('.NS') and symbol in self.indian_stocks:
                symbol = f"{symbol}.NS"

            stock = yf.Ticker(symbol)
            self.data = stock.history(period=period, interval=interval)

            if self.data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return self.data

        except Exception as e:
            logging.error(f"Error fetching stock data: {str(e)}")
            raise

    def get_fundamental_data(self, symbol):
        """Fetch fundamental data for a stock"""
        try:
            # Append .NS if not already present for Indian stocks
            if not symbol.endswith('.NS') and symbol in self.indian_stocks:
                symbol = f"{symbol}.NS"

            stock = yf.Ticker(symbol)
            info = stock.info

            fundamentals = {
                'Market Cap': info.get('marketCap'),
                'P/E Ratio': info.get('trailingPE'),
                'EPS': info.get('trailingEps'),
                'Revenue': info.get('totalRevenue'),
                'Profit Margin': info.get('profitMargins'),
                'Debt to Equity': info.get('debtToEquity'),
                '52 Week High': info.get('fiftyTwoWeekHigh'),
                '52 Week Low': info.get('fiftyTwoWeekLow'),
                'Beta': info.get('beta'),
                'Dividend Yield': info.get('dividendYield')
            }

            return fundamentals

        except Exception as e:
            logging.error(f"Error fetching fundamental data: {str(e)}")
            raise

    def prepare_features(self, df):
        """Prepare features for model training"""
        try:
            features = pd.DataFrame()

            # Price-based features (4 features)
            features['Close'] = df['Close']
            features['High'] = df['High']
            features['Low'] = df['Low']
            features['Open'] = df['Open']

            # Volume feature (1 feature)
            features['Volume'] = df['Volume']

            # Technical indicator (1 feature)
            features['SMA_20'] = df['Close'].rolling(window=20).mean()

            # Drop NaN values
            features = features.dropna()

            logging.info(f"Prepared features shape: {features.shape}")
            return features

        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            raise