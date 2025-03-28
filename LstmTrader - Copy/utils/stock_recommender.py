import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

class StockRecommender:
    def __init__(self, data_processor, technical_analyzer):
        self.data_processor = data_processor
        self.technical_analyzer = technical_analyzer
        
    def calculate_technical_score(self, indicators: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0
            
            # RSI Score (0-30: Oversold, 70-100: Overbought)
            rsi = indicators['RSI'].iloc[-1]
            if rsi < 30:
                score += 1  # Buying opportunity
            elif rsi > 70:
                score -= 1  # Selling pressure
                
            # MACD Signal
            if indicators['MACD'].iloc[-1] > indicators['MACD_Signal'].iloc[-1]:
                score += 1  # Bullish
            else:
                score -= 1  # Bearish
                
            # Moving Average Signal
            if indicators['SMA_20'].iloc[-1] > indicators['SMA_50'].iloc[-1]:
                score += 1  # Uptrend
            else:
                score -= 1  # Downtrend
                
            # Normalize score to 0-1 range
            return (score + 3) / 6  # Normalizing from [-3, 3] to [0, 1]
            
        except Exception as e:
            logging.error(f"Error calculating technical score: {str(e)}")
            raise

    def calculate_fundamental_score(self, fundamentals: Dict) -> float:
        """Calculate fundamental analysis score"""
        try:
            score = 0
            
            # P/E Ratio Analysis
            pe_ratio = fundamentals.get('P/E Ratio')
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    score += 1  # Potentially undervalued
                elif pe_ratio > 30:
                    score -= 1  # Potentially overvalued
                    
            # Profit Margin
            profit_margin = fundamentals.get('Profit Margin')
            if profit_margin and profit_margin > 0.2:  # 20% or higher
                score += 1
                
            # Debt to Equity
            debt_equity = fundamentals.get('Debt to Equity')
            if debt_equity and debt_equity < 1:  # Less than 1 is generally good
                score += 1
                
            # Normalize score to 0-1 range
            return (score + 3) / 6  # Normalizing from [-3, 3] to [0, 1]
            
        except Exception as e:
            logging.error(f"Error calculating fundamental score: {str(e)}")
            raise

    def get_top_recommendations(self, n: int = 5) -> List[Dict]:
        """Get top stock recommendations"""
        try:
            recommendations = []
            stocks = self.data_processor.get_stock_list()
            
            for symbol, name in stocks:
                try:
                    # Fetch data
                    df = self.data_processor.fetch_stock_data(symbol, period='6mo')
                    indicators = self.technical_analyzer.calculate_all_indicators(df)
                    fundamentals = self.data_processor.get_fundamental_data(symbol)
                    
                    # Calculate scores
                    technical_score = self.calculate_technical_score(indicators)
                    fundamental_score = self.calculate_fundamental_score(fundamentals)
                    
                    # Combined score (equal weight)
                    total_score = (technical_score + fundamental_score) / 2
                    
                    # Current price and recent performance
                    current_price = df['Close'].iloc[-1]
                    price_change = ((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
                    
                    recommendations.append({
                        'symbol': symbol,
                        'name': name,
                        'score': total_score,
                        'technical_score': technical_score,
                        'fundamental_score': fundamental_score,
                        'current_price': current_price,
                        'price_change_5d': price_change,
                        'rsi': indicators['RSI'].iloc[-1],
                        'recommendation': 'Strong Buy' if total_score > 0.7 
                                  else 'Buy' if total_score > 0.5 
                                  else 'Hold' if total_score > 0.3 
                                  else 'Sell'
                    })
                    
                except Exception as e:
                    logging.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Sort by score and return top N
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n]
            
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            raise
            
    def get_recommendation_insights(self, stock_data: Dict) -> List[str]:
        """Generate insights for a stock recommendation"""
        insights = []
        
        # Technical Analysis Insights
        if stock_data['technical_score'] > 0.7:
            insights.append("Strong technical indicators suggest positive momentum")
        elif stock_data['technical_score'] < 0.3:
            insights.append("Technical indicators show weakness")
            
        # RSI Analysis
        rsi = stock_data['rsi']
        if rsi < 30:
            insights.append("Stock may be oversold (RSI < 30)")
        elif rsi > 70:
            insights.append("Stock may be overbought (RSI > 70)")
            
        # Recent Performance
        if stock_data['price_change_5d'] > 5:
            insights.append(f"Strong recent performance: +{stock_data['price_change_5d']:.1f}% in 5 days")
        elif stock_data['price_change_5d'] < -5:
            insights.append(f"Recent weakness: {stock_data['price_change_5d']:.1f}% in 5 days")
            
        return insights
