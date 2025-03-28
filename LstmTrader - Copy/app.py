import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging

from models.lstm_model import LSTMModel
from utils.technical_indicators import TechnicalAnalyzer
from utils.data_processor import DataProcessor
from utils.stock_recommender import StockRecommender # Add to imports

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize components
data_processor = DataProcessor()
technical_analyzer = TechnicalAnalyzer()
lstm_model = LSTMModel()
stock_recommender = StockRecommender(data_processor, technical_analyzer) # Add after initializing other components

def plot_stock_data(df, indicators, predictions=None):
    """Create interactive stock chart with indicators"""
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))

    # Add technical indicators
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators['SMA_20'],
        name='SMA 20', line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=indicators['SMA_50'],
        name='SMA 50', line=dict(color='orange')
    ))

    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=df.index[-len(predictions):],
            y=predictions,
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title='Stock Price Analysis',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    return fig

def format_price_prediction(current_price, predicted_price):
    """Format price prediction with change percentage"""
    change = ((predicted_price - current_price) / current_price) * 100
    return f"₹{predicted_price:.2f} ({change:+.2f}%)"

def main():
    st.set_page_config(page_title="Algorion Trading Platform", layout="wide")

    st.title("🚀 Algorion: AI-Driven Indian Stock Trading Platform")

    # Sidebar inputs
    st.sidebar.header("Configuration")

    # Get list of Indian stocks
    stock_list = data_processor.get_stock_list()
    stock_options = {f"{name} ({symbol})": symbol for symbol, name in stock_list}

    # Stock selection dropdown
    selected_stock = st.sidebar.selectbox(
        "Select Indian Stock",
        options=list(stock_options.keys()),
        index=0
    )
    symbol = stock_options[selected_stock]

    period = st.sidebar.selectbox(
        "Time Period", 
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    try:
        # Fetch and process data
        with st.spinner("Fetching stock data..."):
            df = data_processor.fetch_stock_data(symbol, period)
            features = data_processor.prepare_features(df)
            indicators = technical_analyzer.calculate_all_indicators(df)

        # Display stock information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}", 
                     f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f}")
        with col2:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        with col3:
            st.metric("RSI", f"{indicators['RSI'].iloc[-1]:.2f}")
        with col4:
            day_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            st.metric("Day Change %", f"{day_change:.2f}%")

        # Plot stock chart
        st.plotly_chart(plot_stock_data(df, indicators), use_container_width=True)

        # Technical Analysis Section
        st.header("Technical Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Technical Indicators")
            tech_df = pd.DataFrame({
                'RSI': indicators['RSI'].iloc[-1],
                'MACD': indicators['MACD'].iloc[-1],
                'Stochastic K': indicators['Stoch_K'].iloc[-1],
                'Stochastic D': indicators['Stoch_D'].iloc[-1]
            }, index=[0])
            st.dataframe(tech_df)

        with col2:
            st.subheader("Trading Signals")
            signals = technical_analyzer.generate_signals(df)
            current_signal = signals.iloc[-1]
            signal_color = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'yellow'
            }
            st.markdown(f"Current Signal: "
                       f":<span style='color: {signal_color[current_signal]}'>"
                       f"{current_signal}</span>",
                       unsafe_allow_html=True)

        # LSTM Predictions
        st.header("AI Price Predictions")
        if st.button("Generate Predictions"):
            with st.spinner("Training LSTM model and generating predictions..."):
                X, y = lstm_model.prepare_data(features)
                split = int(0.8 * len(X))
                X_train, y_train = X[:split], y[:split]

                # Train model
                lstm_model.train(X_train, y_train)

                # Display model accuracy metrics
                metrics = lstm_model.get_metrics()
                st.subheader("Model Performance Metrics")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Directional Accuracy", f"{metrics['Directional Accuracy']:.2f}%")
                    st.metric("R² Score", f"{metrics['R²']:.4f}")

                with col2:
                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
                    st.metric("Root Mean Square Error", f"{metrics['RMSE']:.4f}")

                # Generate predictions for different time periods
                current_price = df['Close'].iloc[-1]
                predictions = lstm_model.get_price_predictions(X, current_price)

                # Display predictions in a clean format
                st.subheader("Projected Prices")
                cols = st.columns(3)

                for idx, (period, price) in enumerate(predictions.items()):
                    with cols[idx % 3]:
                        st.metric(
                            period,
                            format_price_prediction(current_price, price),
                            delta=f"{((price - current_price) / current_price) * 100:.2f}%"
                        )

                # Update chart with short-term predictions
                next_month_pred = lstm_model.predict_future(X, X[-1], 21)  # 21 trading days
                st.plotly_chart(plot_stock_data(df, indicators, next_month_pred),
                             use_container_width=True)

        # Fundamental Analysis
        st.header("Fundamental Analysis")
        fundamentals = data_processor.get_fundamental_data(symbol)
        st.dataframe(pd.DataFrame([fundamentals]))

        # AI Stock Recommendations # Add to main() function, after the "Fundamental Analysis" section
        st.header("🤖 AI Stock Recommendations")
        if st.button("Generate Stock Recommendations"):
            with st.spinner("Analyzing Indian stocks..."):
                recommendations = stock_recommender.get_top_recommendations(n=5)

                st.subheader("Top 5 Recommended Stocks")
                for rec in recommendations:
                    with st.expander(f"{rec['name']} ({rec['symbol']}) - {rec['recommendation']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Current Price", f"₹{rec['current_price']:.2f}", 
                                    f"{rec['price_change_5d']:.1f}%")
                            st.metric("AI Score", f"{rec['score']:.2f}", 
                                    f"Technical: {rec['technical_score']:.2f}")

                        with col2:
                            st.metric("RSI", f"{rec['rsi']:.1f}")
                            st.metric("Fundamental Score", f"{rec['fundamental_score']:.2f}")

                        # Display insights
                        insights = stock_recommender.get_recommendation_insights(rec)
                        if insights:
                            st.write("💡 Key Insights:")
                            for insight in insights:
                                st.write(f"• {insight}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()