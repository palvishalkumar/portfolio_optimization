import yfinance as yf
import pandas as pd
import streamlit as st

# ---- Asset Configurations ---- #
ASSETS = {
    'crypto': ['BTC-USD', 'ETH-USD'],
    'metals': ['GC=F', 'SI=F']
}

# ---- General Data Fetcher for Stocks/Assets ---- #
def fetch_data(symbol, start, end):
    try:
        if symbol in ASSETS['crypto'] + ASSETS['metals']:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if not data.empty:
                return data
        else:
            for exchange in ['.NS', '.BO']:
                data = yf.download(f"{symbol}{exchange}", start=start, end=end, progress=False)
                if not data.empty:
                    return data
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
    return None

# ---- Fetch Market Indices (NIFTY, SENSEX, VIX) ---- #
def fetch_market_data():
    symbols = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN', 'India VIX': '^INDIAVIX'}
    results = {}
    
    for name, symbol in symbols.items():
        try:
            data = yf.download(symbol, period='2d', progress=False, auto_adjust=True)
            if len(data) >= 2:  # Make sure we have at least 2 days of data
                last_close = float(data['Close'].iloc[-1])
                prev_close = float(data['Close'].iloc[-2])
                pct_change = ((last_close - prev_close) / prev_close) * 100
            else:
                last_close = float(data['Close'].iloc[-1]) if not data.empty else 0
                pct_change = 0
                
            results[name] = {
                'value': f"₹{last_close:,.2f}",
                'change': f"{pct_change:.2f}%"
            }
        except Exception as e:
            st.error(f"Error fetching {name}: {str(e)}")
            results[name] = {'value': 'N/A', 'change': 'N/A'}
    
    return results['NIFTY 50'], results['SENSEX'], results['India VIX']
# ---- Fetch Global & Indian Indices ---- #
def fetch_global_data():
    global_symbols = {'Dow Jones': '^DJI', 'NASDAQ': '^IXIC', 'FTSE 100': '^FTSE', 'Nikkei 225': '^N225'}
    indian_symbols = {'Bank Nifty': '^NSEBANK', 'Nifty IT': '^CNXIT', 'Nifty FMCG': '^CNXFMCG'}

    global_data = []
    indian_data = []

    for name, symbol in global_symbols.items():
        try:
            data = yf.download(symbol, period='2d', progress=False, auto_adjust=True)
            if len(data) >= 2:
                close = float(data['Close'].iloc[-1])
                prev_close = float(data['Close'].iloc[-2])
                pct_change = ((close - prev_close) / prev_close) * 100
            else:
                close = float(data['Close'].iloc[-1]) if not data.empty else 0
                pct_change = 0
            
            global_data.append({
                'Index': name,
                'Last': f"{close:,.2f}",
                'Change %': f"{pct_change:.2f}%"
            })
        except Exception as e:
            st.error(f"Error fetching {name}: {str(e)}")
            global_data.append({'Index': name, 'Last': 'N/A', 'Change %': 'N/A'})

    for name, symbol in indian_symbols.items():
        try:
            data = yf.download(symbol, period='2d', progress=False, auto_adjust=True)
            if len(data) >= 2:
                close = float(data['Close'].iloc[-1])
                prev_close = float(data['Close'].iloc[-2])
                pct_change = ((close - prev_close) / prev_close) * 100
            else:
                close = float(data['Close'].iloc[-1]) if not data.empty else 0
                pct_change = 0
            
            indian_data.append({
                'Index': name,
                'Last': f"{close:,.2f}",
                'Change %': f"{pct_change:.2f}%"
            })
        except Exception as e:
            st.error(f"Error fetching {name}: {str(e)}")
            indian_data.append({'Index': name, 'Last': 'N/A', 'Change %': 'N/A'})

    return pd.DataFrame(global_data), pd.DataFrame(indian_data)

# ---- Fetch Most Active Stocks ---- #
def fetch_active_stocks():
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    active_list = []
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, period='1d', progress=False)
            if not data.empty:
                # Convert Series values to float
                close = float(data['Close'].iloc[-1])
                volume = float(data['Volume'].iloc[-1])
                active_list.append({
                    'Stock': symbol.replace('.NS', ''),  # Remove .NS suffix for display
                    'Price': f"₹{close:,.2f}",
                    'Volume': f"{volume:,.0f}"
                })
        except Exception as e:
            st.error(f"Error fetching {symbol}: {str(e)}")
            active_list.append({
                'Stock': symbol.replace('.NS', ''),
                'Price': 'N/A',
                'Volume': 'N/A'
            })
    
    return pd.DataFrame(active_list)


# ---- Gainers & Losers (Sample for now) ---- #
def fetch_gainers_losers():
    gainers = pd.DataFrame({
        'Stock': ['TCS.NS', 'INFY.NS'],
        'Price': ['₹3,400', '₹1,450'],
        'Change %': ['+3.5%', '+2.1%']
    })
    losers = pd.DataFrame({
        'Stock': ['RELIANCE.NS', 'HDFC.NS'],
        'Price': ['₹2,500', '₹2,800'],
        'Change %': ['-1.5%', '-2.2%']
    })
    return gainers, losers

# ---- Portfolio Summary & Allocation ---- #
def fetch_portfolio_data():
    # Define your portfolio holdings with all 6 asset classes
    portfolio = {
        'Stocks': [
            {'symbol': 'RELIANCE.NS', 'quantity': 50},
            {'symbol': 'TCS.NS', 'quantity': 25},
            {'symbol': 'INFY.NS', 'quantity': 30}
        ],
        'ETFs': [
            {'symbol': 'NIFTYBEES.NS', 'quantity': 100},  # Nifty ETF
            {'symbol': 'BANKBEES.NS', 'quantity': 50},    # Banking ETF
            {'symbol': 'GOLDSHARE.NS', 'quantity': 40}    # Gold ETF
        ],
        'Crypto': [
            {'symbol': 'BTC-INR', 'quantity': 0.01},
            {'symbol': 'ETH-INR', 'quantity': 0.15}
        ],
        'Bonds': [
            {'symbol': 'IN10Y.NS', 'amount': 200000},  # 10-year Gov bonds
            {'symbol': 'IN05Y.NS', 'amount': 100000}   # 5-year Gov bonds
        ],
        'Commodities': [
            {'symbol': 'GLD', 'quantity': 10},     # Gold ETF
            {'symbol': 'USO', 'quantity': 15}      # Oil ETF
        ],
        'Metals': [
            {'symbol': 'SLV', 'quantity': 20},     # Silver ETF
            {'symbol': 'PPLT', 'quantity': 5}      # Platinum ETF
        ]
    }
    
    total_value = 0
    asset_values = {
        'Stocks': 0, 
        'ETFs': 0, 
        'Crypto': 0, 
        'Bonds': 0, 
        'Commodities': 0, 
        'Metals': 0
    }
    previous_total = 0
    
    try:
        # Calculate real-time values for each asset class
        for asset_class, holdings in portfolio.items():
            for holding in holdings:
                ticker = yf.Ticker(holding['symbol'])
                current_data = ticker.history(period='2d')
                
                if not current_data.empty:
                    current_price = float(current_data['Close'].iloc[-1])
                    prev_price = float(current_data['Close'].iloc[-2])
                    
                    # Calculate current and previous values
                    if 'quantity' in holding:
                        current_value = current_price * holding['quantity']
                        prev_value = prev_price * holding['quantity']
                    else:  # For bonds using amount
                        current_value = holding['amount']
                        prev_value = holding['amount']
                    
                    asset_values[asset_class] += current_value
                    total_value += current_value
                    previous_total += prev_value

        # Calculate total return (assuming initial investment was 90% of total_value)
        initial_investment = total_value * 0.9
        total_return = total_value - initial_investment
        
        # Calculate daily change percentage
        daily_change_pct = ((total_value - previous_total) / previous_total) * 100 if previous_total > 0 else 0
        
        # Calculate allocation percentages
        total_allocation = sum(asset_values.values())
        allocation_data = [
            {
                'Asset': asset,
                'Allocation': (value/total_allocation)*100 if total_allocation > 0 else 0,
                'Value': f"₹{value:,.2f}"
            }
            for asset, value in asset_values.items()
        ]

        summary = {
            'total_value': f"₹{total_value:,.2f}",
            'total_change': f"{daily_change_pct:+.2f}%",
            'total_return': f"₹{total_return:,.2f}",
            'return_change': f"{(total_return/initial_investment)*100:+.2f}%" if initial_investment > 0 else "0.00%"
        }

        return summary, pd.DataFrame(allocation_data)

    except Exception as e:
        st.error(f"Error calculating portfolio data: {str(e)}")
        # Fallback with all six asset classes
        summary = {
            'total_value': "₹0.00",
            'total_change': "0.00%",
            'total_return': "₹0.00",
            'return_change': "0.00%"
        }
        allocation_data = [
            {'Asset': asset, 'Allocation': 0, 'Value': "₹0.00"} 
            for asset in ['Stocks', 'ETFs', 'Crypto', 'Bonds', 'Commodities', 'Metals']
        ]
        return summary, pd.DataFrame(allocation_data)

# ---- Transactions ---- #
def fetch_transactions():
    return pd.DataFrame({
        'Date': ['2024-01-10', '2024-02-15'],
        'Asset': ['INFY.NS', 'BTC-USD'],
        'Action': ['Buy', 'Sell'],
        'Quantity': [10, 0.05],
        'Price': [1500, 42000],
        'Total': [15000, 2100]
    })

def fetch_commodities():
    # Using more reliable commodity symbols
    commodities = {
        'Gold': 'GLD',           # SPDR Gold Shares ETF
        'Silver': 'SLV',         # iShares Silver Trust
        'Crude Oil': 'USO',      # United States Oil Fund
        'Natural Gas': 'UNG'     # United States Natural Gas Fund
    }
    
    data = []
    for name, symbol in commodities.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info and 'regularMarketPrice' in info:
                current_price = info['regularMarketPrice']
                prev_price = info.get('previousClose', current_price)
                
                # Calculate change percentage
                change = ((current_price - prev_price) / prev_price) * 100 if prev_price else 0
                
                data.append({
                    'Commodity': name,
                    'Price': f"${current_price:,.2f}",
                    'Change %': f"{change:+.2f}%"
                })
            else:
                # Fallback to historical data if info is not available
                hist = ticker.history(period='2d')
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    data.append({
                        'Commodity': name,
                        'Price': f"${current_price:,.2f}",
                        'Change %': f"{change:+.2f}%"
                    })
                else:
                    # If both methods fail, add with placeholder values
                    data.append({
                        'Commodity': name,
                        'Price': 'Updating...',
                        'Change %': '--'
                    })
        except Exception as e:
            # Add the commodity with placeholder values instead of skipping
            data.append({
                'Commodity': name,
                'Price': 'Updating...',
                'Change %': '--'
            })
    
    return pd.DataFrame(data)

# ---- Currencies ---- #
def fetch_currencies():
    # Major currency pairs with INR
    currencies = {
        'USD/INR': 'INR=X',
        'EUR/INR': 'EURINR=X',
        'GBP/INR': 'GBPINR=X',
        'JPY/INR': 'JPYINR=X'
    }
    
    data = []
    for pair, symbol in currencies.items():
        try:
            ticker = yf.Ticker(symbol)
            current_rate = ticker.info.get('regularMarketPrice', 0)
            prev_rate = ticker.info.get('previousClose', 0)
            change = ((current_rate - prev_rate) / prev_rate) * 100 if prev_rate else 0
            
            data.append({
                'Currency': pair,
                'Rate': f"₹{current_rate:.2f}",
                'Change %': f"{change:+.2f}%"
            })
        except Exception as e:
            st.error(f"Error fetching {pair}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

# ---- Bonds ---- #
def fetch_bonds():
    # Indian Government Bond yields
    bonds = {
        '10Y India': 'IN10Y.NS',  # 10-year Indian Government Bond
        '5Y India': 'IN05Y.NS',   # 5-year Indian Government Bond
        '2Y India': 'IN02Y.NS'    # 2-year Indian Government Bond
    }
    
    data = []
    for name, symbol in bonds.items():
        try:
            ticker = yf.Ticker(symbol)
            current_yield = ticker.info.get('regularMarketPrice', 0)
            prev_yield = ticker.info.get('previousClose', 0)
            change = current_yield - prev_yield if prev_yield else 0
            
            data.append({
                'Bond': name,
                'Yield': f"{current_yield:.2f}%",
                'Change': f"{change:+.2f}%"
            })
        except Exception as e:
            # For Indian bonds, use fixed values if real-time data is unavailable
            if '10Y' in name:
                data.append({'Bond': name, 'Yield': '7.10%', 'Change': '+0.02%'})
            elif '5Y' in name:
                data.append({'Bond': name, 'Yield': '6.50%', 'Change': '+0.01%'})
            elif '2Y' in name:
                data.append({'Bond': name, 'Yield': '6.20%', 'Change': '+0.01%'})
            continue
    
    return pd.DataFrame(data)
# import yfinance as yf
# import pandas as pd
# import streamlit as st

# # Assets configuration at the top
# ASSETS = {'crypto': ['BTC-USD', 'ETH-USD'], 'metals': ['GC=F', 'SI=F']}

# def fetch_data(symbol, start, end):
#     """
#     Fetch data for given symbol, trying Indian exchanges for stocks
#     and direct download for crypto/metals
#     """
#     try:
#         # For crypto and metals, try direct download
#         if symbol in ASSETS['crypto'] + ASSETS['metals']:
#             data = yf.download(symbol, start=start, end=end, progress=False)
#             if not data.empty:
#                 return data
#         else:
#             # For stocks, try Indian exchanges
#             for exchange in ['.NS', '.BO']:
#                 data = yf.download(f"{symbol}{exchange}", start=start, end=end, progress=False)
#                 if not data.empty:
#                     return data
#     except Exception as e:
#         st.error(f"Error fetching {symbol}: {str(e)}")
#     return None

# def fetch_stock_info(symbol):
#     """
#     Fetch stock information with better handling for different asset types
#     """
#     try:
#         if symbol in ASSETS['crypto']:
#             stock = yf.Ticker(symbol)
#             info = stock.info
#             market_cap = info.get('marketCap', 0)
#             volume = info.get('volume24h', info.get('volume', 0))
            
#             return {
#                 'trailingPE': info.get('trailingPE', 'N/A (Crypto)'),
#                 'marketCap': market_cap if market_cap else 'N/A',
#                 'volume': volume if volume else 'N/A',
#                 'beta': info.get('beta', 'N/A (Crypto)'),
#                 'dividendYield': info.get('dividendYield', 'N/A (Crypto)'),
#                 'returnOnEquity': info.get('returnOnEquity', 'N/A (Crypto)'),
#                 'debtToEquity': info.get('debtToEquity', 'N/A (Crypto)'),
#                 'sector': 'Cryptocurrency'
#             }
            
#         elif symbol in ASSETS['metals']:
#             stock = yf.Ticker(symbol)
#             info = stock.info
#             volume = info.get('volume', 0)
            
#             return {
#                 'trailingPE': 'N/A (Commodity)',
#                 'marketCap': 'N/A (Commodity)',
#                 'volume': volume if volume else 'N/A',
#                 'beta': info.get('beta', 'N/A (Commodity)'),
#                 'dividendYield': info.get('dividendYield', 'N/A (Commodity)'),
#                 'returnOnEquity': info.get('returnOnEquity', 'N/A (Commodity)'),
#                 'debtToEquity': info.get('debtToEquity', 'N/A (Commodity)'),
#                 'sector': 'Commodity'
#             }
            
#         else:
#             # Try Indian exchanges for stocks
#             for exchange in ['.NS', '.BO']:
#                 try:
#                     stock = yf.Ticker(f"{symbol}{exchange}")
#                     info = stock.info
#                     if info:
#                         market_cap = info.get('marketCap', 0)
#                         volume = info.get('averageVolume', info.get('volume', 0))
#                         pe_ratio = info.get('trailingPE', 0)
#                         beta = info.get('beta', 0)
#                         div_yield = info.get('dividendYield', 0)
#                         roe = info.get('returnOnEquity', 0)
#                         debt_equity = info.get('debtToEquity', 0)
                        
#                         return {
#                             'trailingPE': pe_ratio if pe_ratio else 'N/A',
#                             'marketCap': market_cap if market_cap else 'N/A',
#                             'volume': volume if volume else 'N/A',
#                             'beta': beta if beta else 'N/A',
#                             'dividendYield': div_yield if div_yield is not None else 'N/A',
#                             'returnOnEquity': roe if roe is not None else 'N/A',
#                             'debtToEquity': debt_equity if debt_equity else 'N/A',
#                             'sector': info.get('sector', 'Unknown')
#                         }
#                 except:
#                     continue

#         # If no data found from any source
#         return {
#             'trailingPE': 'N/A',
#             'marketCap': 'N/A',
#             'volume': 'N/A',
#             'beta': 'N/A',
#             'dividendYield': 'N/A',
#             'returnOnEquity': 'N/A',
#             'debtToEquity': 'N/A',
#             'sector': 'Unknown'
#         }
            
#     except Exception as e:
#         st.error(f"Error fetching info for {symbol}: {str(e)}")
#         return {
#             'trailingPE': 'N/A',
#             'marketCap': 'N/A',
#             'volume': 'N/A',
#             'beta': 'N/A',
#             'dividendYield': 'N/A',
#             'returnOnEquity': 'N/A',
#             'debtToEquity': 'N/A',
#             'sector': 'Unknown'
#         }