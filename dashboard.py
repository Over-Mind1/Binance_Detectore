import requests
import pandas as pd
import streamlit as st
import time
import datetime
import numpy as np
import altair as alt
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Binance Pump Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .pump-alert {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-alert {
        background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PumpCandidate:
    symbol: str
    current_price: float
    price_change_1h: float
    price_change_24h: float
    volume_ratio_1h: float
    volume_ratio_24h: float
    pump_score: float
    confidence_level: str
    detection_strategy: str
    market_cap_rank: int = 0
    liquidity_score: float = 0.0
    order_book_imbalance: float = 0.0
    volatility_squeeze: bool = False
    timestamp: datetime.datetime = None

class AdvancedPumpDetector:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Binance-Pump-Detector/1.0'
        })
        
        # Enhanced Detection Parameters
        self.volume_thresholds = {
            'conservative': 2.5,
            'moderate': 4.0,
            'aggressive': 6.0
        }
        
        self.price_thresholds = {
            'conservative': 5.0,   # 5% price increase
            'moderate': 10.0,      # 10% price increase
            'aggressive': 15.0     # 15% price increase
        }
        
        self.lookback_periods = {
            '1h': 24,   # 24 hours of 1h candles
            '15m': 96,  # 24 hours of 15m candles
            '5m': 288   # 24 hours of 5m candles
        }

    @st.cache_data(ttl=300)
    def get_trading_pairs(_self, quote_assets=['USDT', 'BTC', 'ETH', 'BNB']) -> List[str]:
        """Get all active trading pairs for specified quote assets"""
        try:
            url = f"{_self.base_url}/api/v3/exchangeInfo"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            pairs = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] in quote_assets):
                    # Additional filtering for active pairs
                    if ('SPOT' in symbol_info.get('permissions', []) or 
                        len(symbol_info.get('permissions', [])) == 0):
                        pairs.append(symbol_info['symbol'])
            
            # If no pairs found, try without permission filtering
            if not pairs:
                for symbol_info in data['symbols']:
                    if (symbol_info['status'] == 'TRADING' and 
                        symbol_info['quoteAsset'] in quote_assets):
                        pairs.append(symbol_info['symbol'])
            
            return sorted(pairs)
        except Exception as e:
            st.error(f"Error fetching trading pairs: {e}")
            # Fallback: return some common pairs for testing
            fallback_pairs = []
            for quote in quote_assets:
                if quote == 'USDT':
                    fallback_pairs.extend(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT'])
                elif quote == 'BTC':
                    fallback_pairs.extend(['ETHBTC', 'BNBBTC', 'ADABTC', 'XRPBTC'])
            return fallback_pairs[:50]

    def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 24) -> Optional[pd.DataFrame]:
        """Fetch kline data with enhanced error handling"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert to appropriate data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_volume', 'trades_count', 'taker_buy_base_volume', 
                             'taker_buy_quote_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
            
        except Exception as e:
            # Add debug information
            st.write(f"Error fetching {symbol}: {str(e)}")
            return None

    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker statistics"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            # Try alternative endpoint
            try:
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                price_data = response.json()
                # Return basic structure
                return {
                    'symbol': symbol,
                    'price': price_data['price'],
                    'priceChangePercent': '0'  # Fallback
                }
            except Exception:
                return None

    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get order book data for imbalance analysis"""
        try:
            url = f"{self.base_url}/api/v3/depth"
            params = {'symbol': symbol, 'limit': min(limit, 100)}  # Binance free API limit
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception:
            return None

    def calculate_volume_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Advanced volume analysis with multiple timeframes"""
        if len(df) < 3:
            return {'volume_ratio_1h': 0, 'volume_ratio_6h': 0, 'volume_ratio_24h': 0}
        
        current_volume = df.iloc[-1]['volume']
        
        # Different volume ratios
        avg_volume_6h = df.iloc[-6:]['volume'].mean() if len(df) >= 6 else df['volume'].mean()
        avg_volume_24h = df['volume'].mean()
        
        return {
            'volume_ratio_1h': current_volume / df.iloc[-2]['volume'] if df.iloc[-2]['volume'] > 0 else 0,
            'volume_ratio_6h': current_volume / avg_volume_6h if avg_volume_6h > 0 else 0,
            'volume_ratio_24h': current_volume / avg_volume_24h if avg_volume_24h > 0 else 0
        }

    def calculate_price_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Comprehensive price movement analysis"""
        if len(df) < 2:
            return {'price_change_1h': 0, 'price_change_6h': 0, 'price_change_24h': 0}
        
        current_price = df.iloc[-1]['close']
        
        changes = {}
        if len(df) >= 2:
            changes['price_change_1h'] = ((current_price - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100
        if len(df) >= 6:
            changes['price_change_6h'] = ((current_price - df.iloc[-6]['close']) / df.iloc[-6]['close']) * 100
        if len(df) >= 24:
            changes['price_change_24h'] = ((current_price - df.iloc[-24]['close']) / df.iloc[-24]['close']) * 100
        else:
            changes['price_change_24h'] = ((current_price - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
        
        return changes

    def detect_volatility_squeeze(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect volatility squeeze (Bollinger Bands compression)"""
        if len(df) < lookback:
            return False
        
        df_recent = df.iloc[-lookback:].copy()
        df_recent['sma'] = df_recent['close'].rolling(window=20).mean()
        df_recent['std'] = df_recent['close'].rolling(window=20).std()
        df_recent['bb_upper'] = df_recent['sma'] + (df_recent['std'] * 2)
        df_recent['bb_lower'] = df_recent['sma'] - (df_recent['std'] * 2)
        df_recent['bb_width'] = (df_recent['bb_upper'] - df_recent['bb_lower']) / df_recent['sma']
        
        current_width = df_recent['bb_width'].iloc[-1]
        avg_width = df_recent['bb_width'].mean()
        
        return current_width < (avg_width * 0.7)  # Squeeze detected

    def calculate_order_book_imbalance(self, order_book: Dict) -> float:
        """Calculate order book imbalance (buy vs sell pressure)"""
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.0
        
        total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:10])
        total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:10])
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
        
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        return imbalance

    def calculate_pump_score(self, volume_analysis: Dict, price_analysis: Dict, 
                           order_book_imbalance: float, volatility_squeeze: bool,
                           detection_mode: str = 'moderate') -> Tuple[float, str]:
        """Advanced pump scoring algorithm"""
        
        score = 0.0
        confidence_factors = []
        
        # Volume Score (40% weight)
        volume_score = 0
        if volume_analysis['volume_ratio_1h'] > self.volume_thresholds[detection_mode]:
            volume_score += 40 * min(volume_analysis['volume_ratio_1h'] / 10, 1.0)
            confidence_factors.append("High 1h volume")
        
        if volume_analysis['volume_ratio_6h'] > 2.0:
            volume_score += 20 * min(volume_analysis['volume_ratio_6h'] / 5, 1.0)
            confidence_factors.append("Sustained volume increase")
        
        score += min(volume_score, 40)
        
        # Price Score (30% weight)
        price_score = 0
        if price_analysis['price_change_1h'] > 0:
            price_score += 15 * min(price_analysis['price_change_1h'] / self.price_thresholds[detection_mode], 1.0)
        
        if price_analysis['price_change_1h'] > self.price_thresholds[detection_mode]:
            price_score += 15
            confidence_factors.append(f"Strong 1h price pump ({price_analysis['price_change_1h']:.1f}%)")
        
        score += min(price_score, 30)
        
        # Order Book Imbalance (20% weight)
        if abs(order_book_imbalance) > 0.3:
            imbalance_score = 20 * min(abs(order_book_imbalance), 1.0)
            score += imbalance_score
            if order_book_imbalance > 0:
                confidence_factors.append("Strong buy pressure")
            else:
                confidence_factors.append("Strong sell pressure")
        
        # Volatility Squeeze Bonus (10% weight)
        if volatility_squeeze:
            score += 10
            confidence_factors.append("Volatility squeeze breakout")
        
        # Determine confidence level
        if score >= 70:
            confidence = "Very High"
        elif score >= 50:
            confidence = "High"
        elif score >= 30:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return min(score, 100), confidence

    def analyze_symbol(self, symbol: str, detection_mode: str = 'moderate') -> Optional[PumpCandidate]:
        """Comprehensive symbol analysis"""
        try:
            # Get kline data
            df = self.get_kline_data(symbol, '1h', 24)
            if df is None or len(df) < 5:
                return None
            
            # Get 24h ticker
            ticker_24h = self.get_24h_ticker(symbol)
            if not ticker_24h:
                return None
            
            # Get order book
            order_book = self.get_order_book(symbol)
            
            # Perform analyses
            volume_analysis = self.calculate_volume_analysis(df)
            price_analysis = self.calculate_price_analysis(df)
            order_book_imbalance = self.calculate_order_book_imbalance(order_book) if order_book else 0.0
            volatility_squeeze = self.detect_volatility_squeeze(df)
            
            # Calculate pump score
            pump_score, confidence = self.calculate_pump_score(
                volume_analysis, price_analysis, order_book_imbalance, 
                volatility_squeeze, detection_mode
            )
            
            # Filter based on minimum criteria
            min_score = {'conservative': 50, 'moderate': 35, 'aggressive': 25}
            if pump_score < min_score[detection_mode]:
                return None
            
            # Create pump candidate
            return PumpCandidate(
                symbol=symbol,
                current_price=df.iloc[-1]['close'],
                price_change_1h=price_analysis.get('price_change_1h', 0),
                price_change_24h=float(ticker_24h.get('priceChangePercent', 0)),
                volume_ratio_1h=volume_analysis['volume_ratio_1h'],
                volume_ratio_24h=volume_analysis['volume_ratio_24h'],
                pump_score=pump_score,
                confidence_level=confidence,
                detection_strategy=detection_mode,
                order_book_imbalance=order_book_imbalance,
                volatility_squeeze=volatility_squeeze,
                timestamp=datetime.datetime.now()
            )
            
        except Exception as e:
            return None

def run_pump_detection():
    """Main pump detection function"""
    detector = AdvancedPumpDetector()
    
    # Sidebar Configuration
    st.sidebar.title("🔧 Detection Settings")
    
    quote_assets = st.sidebar.multiselect(
        "Quote Assets", 
        ['USDT', 'BTC', 'ETH', 'BNB'], 
        default=['USDT', 'BTC']
    )
    
    detection_mode = st.sidebar.selectbox(
        "Detection Mode", 
        ['conservative', 'moderate', 'aggressive'],
        index=1
    )
    
    max_pairs = st.sidebar.slider("Max Pairs to Analyze", 50, 500, 200, 50)
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
    
    # Main Dashboard
    st.title("🚨 Advanced Binance Pump Detection System")
    st.markdown(f"**Detection Mode:** {detection_mode.title()} | **Quote Assets:** {', '.join(quote_assets)}")
    
    # Get trading pairs
    with st.spinner("Loading trading pairs..."):
        all_pairs = detector.get_trading_pairs(quote_assets)
        
        # Debug information
        if not all_pairs:
            st.error("❌ No trading pairs retrieved from API")
            st.info("🔧 Trying manual connection test...")
            
            # Test basic connection
            try:
                test_url = "https://api.binance.com/api/v3/ping"
                test_response = requests.get(test_url, timeout=10)
                if test_response.status_code == 200:
                    st.success("✅ Binance API connection successful")
                    
                    # Test exchange info
                    exchange_url = "https://api.binance.com/api/v3/exchangeInfo"
                    exchange_response = requests.get(exchange_url, timeout=15)
                    if exchange_response.status_code == 200:
                        data = exchange_response.json()
                        total_symbols = len(data.get('symbols', []))
                        st.info(f"📊 Exchange info loaded: {total_symbols} total symbols available")
                        
                        # Show sample symbols for debugging
                        sample_symbols = [s['symbol'] for s in data['symbols'][:10] if s['quoteAsset'] in quote_assets]
                        if sample_symbols:
                            st.info(f"🔍 Sample matching pairs: {', '.join(sample_symbols)}")
                        else:
                            st.warning("⚠️ No symbols match your quote asset criteria")
                    else:
                        st.error(f"❌ Exchange info failed: {exchange_response.status_code}")
                else:
                    st.error(f"❌ Binance API ping failed: {test_response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection test failed: {e}")
            
            return
        
        pairs_to_analyze = all_pairs[:max_pairs]
        st.info(f"📈 Found {len(all_pairs)} pairs, analyzing top {len(pairs_to_analyze)}")
    
    if not pairs_to_analyze:
        st.error("No trading pairs found!")
        return
    
    # Analysis Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pump_candidates = []
    
    # Use ThreadPoolExecutor for parallel processing (reduced workers for free API)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(detector.analyze_symbol, symbol, detection_mode): symbol 
            for symbol in pairs_to_analyze
        }
        
        completed = 0
        for future in as_completed(future_to_symbol):
            completed += 1
            progress = completed / len(pairs_to_analyze)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing: {completed}/{len(pairs_to_analyze)} pairs...")
            
            try:
                result = future.result()
                if result:
                    pump_candidates.append(result)
            except Exception as e:
                # Silently continue on individual symbol errors
                continue
            
            # Add small delay to respect rate limits
            time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by pump score
    pump_candidates.sort(key=lambda x: x.pump_score, reverse=True)
    
    # Display Results
    if pump_candidates:
        st.success(f"🎯 Found {len(pump_candidates)} potential pump candidates!")
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidates", len(pump_candidates))
        with col2:
            high_confidence = len([c for c in pump_candidates if c.confidence_level in ['High', 'Very High']])
            st.metric("High Confidence", high_confidence)
        with col3:
            avg_score = np.mean([c.pump_score for c in pump_candidates])
            st.metric("Avg Pump Score", f"{avg_score:.1f}")
        with col4:
            max_change = max([c.price_change_1h for c in pump_candidates])
            st.metric("Max 1H Change", f"{max_change:.2f}%")
        
        # Detailed Results Table
        st.subheader("🔍 Detailed Analysis")
        
        results_data = []
        for candidate in pump_candidates:
            results_data.append({
                'Symbol': candidate.symbol,
                'Current Price': f"{candidate.current_price:.8f}",
                '1H Change %': f"{candidate.price_change_1h:.2f}%",
                '24H Change %': f"{candidate.price_change_24h:.2f}%",
                'Volume Ratio 1H': f"{candidate.volume_ratio_1h:.2f}x",
                'Volume Ratio 24H': f"{candidate.volume_ratio_24h:.2f}x",
                'Pump Score': f"{candidate.pump_score:.1f}",
                'Confidence': candidate.confidence_level,
                'Order Book Imbalance': f"{candidate.order_book_imbalance:.3f}",
                'Volatility Squeeze': "✅" if candidate.volatility_squeeze else "❌"
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Top 5 Detailed Analysis
        st.subheader("🏆 Top 5 Pump Candidates")
        
        for i, candidate in enumerate(pump_candidates[:5], 1):
            with st.expander(f"#{i} {candidate.symbol} - Score: {candidate.pump_score:.1f} ({candidate.confidence_level})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Price Analysis:**
                    - Current Price: {candidate.current_price:.8f}
                    - 1H Change: {candidate.price_change_1h:.2f}%
                    - 24H Change: {candidate.price_change_24h:.2f}%
                    
                    **Volume Analysis:**
                    - 1H Volume Ratio: {candidate.volume_ratio_1h:.2f}x
                    - 24H Volume Ratio: {candidate.volume_ratio_24h:.2f}x
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Advanced Indicators:**
                    - Order Book Imbalance: {candidate.order_book_imbalance:.3f}
                    - Volatility Squeeze: {'Yes' if candidate.volatility_squeeze else 'No'}
                    - Detection Strategy: {candidate.detection_strategy.title()}
                    - Timestamp: {candidate.timestamp.strftime('%H:%M:%S')}
                    """)
                
                # Price chart
                df_chart = detector.get_kline_data(candidate.symbol, '1h', 24)
                if df_chart is not None:
                    chart = alt.Chart(df_chart).mark_line(point=True, color='#ff6b6b').encode(
                        x=alt.X('open_time:T', title='Time'),
                        y=alt.Y('close:Q', title='Price'),
                        tooltip=['open_time:T', 'close:Q', 'volume:Q']
                    ).properties(
                        width=600,
                        height=200,
                        title=f"{candidate.symbol} - Last 24 Hours"
                    )
                    st.altair_chart(chart, use_container_width=True)
    
    else:
        st.warning(f"No pump candidates detected with {detection_mode} settings. Try adjusting the detection mode or parameters.")
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    run_pump_detection()