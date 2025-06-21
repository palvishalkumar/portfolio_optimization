import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import seaborn as sns
from utils.portfolio_opt import optimize_portfolio

def calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.065):
    """Calculate portfolio metrics using MPT"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return portfolio_return, portfolio_risk, sharpe_ratio

def optimize_portfolio(returns, cov_matrix, risk_free_rate=0.065):
    """Optimize portfolio using Markowitz MPT"""
    num_assets = len(returns.columns)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0-1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Objective: Maximize Sharpe Ratio
    def objective(weights):
        portfolio_return, portfolio_risk, sharpe = calculate_portfolio_metrics(
            weights, returns, cov_matrix, risk_free_rate
        )
        return -sharpe  # Negative because we're minimizing
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def get_asset_fundamentals(symbol):
    """Fetch fundamental data for each asset"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get current price
        current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
        
        # Calculate market cap
        market_cap = 0
        if 'marketCap' in info and info['marketCap']:
            market_cap = info['marketCap']
        elif 'sharesOutstanding' in info and info['sharesOutstanding']:
            market_cap = info['sharesOutstanding'] * current_price
        elif 'totalAssets' in info:
            market_cap = info['totalAssets']
            
        # Get other metrics
        pe_ratio = info.get('trailingPE', info.get('priceToBook', 0))
        beta = info.get('beta3Year', info.get('beta', 1.0))
        
        # Get historical data for 52-week high/low
        hist = stock.history(period="1y")
        week_52_high = hist['High'].max() if not hist.empty else current_price
        week_52_low = hist['Low'].min() if not hist.empty else current_price
        
        # Calculate dividend yield using trailing annual dividend yield
        div_yield = info.get('trailingAnnualDividendYield', 0)
        if div_yield:
            div_yield = div_yield * 100  # Convert to percentage
        else:
            # Try alternative dividend yield calculation
            annual_dividend = info.get('trailingAnnualDividendRate', 0)
            div_yield = (annual_dividend / current_price * 100) if current_price > 0 else 0

        return {
            'P/E Ratio': pe_ratio if pe_ratio and pe_ratio > 0 else 0,
            'Market Cap': market_cap,
            '52W High': float(week_52_high),
            '52W Low': float(week_52_low),
            'Dividend Yield': float(div_yield),
            'Beta': float(beta) if beta else 1.0
        }
            
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def format_value(val, column):
    """Format values based on column type"""
    if pd.isna(val) or val == 'N/A':
        return "Not Available"
    
    try:
        if column == 'Market Cap':
            val = float(val)
            if val >= 1e9:
                return f"₹{val/1e9:.2f}B"
            elif val >= 1e6:
                return f"₹{val/1e6:.2f}M"
            else:
                return f"₹{val:.2f}"
        elif column in ['52W High', '52W Low']:
            return f"₹{float(val):.2f}"
        elif column == 'Dividend Yield':
            return f"{float(val):.2f}%"
        elif column == 'P/E Ratio' or column == 'Beta':
            return f"{float(val):.2f}"
        else:
            return str(val)
    except:
        return "Not Available"

def show_asset_analysis(predictions):
    st.subheader("Asset Analysis")
    
    analysis_data = []
    for pred in predictions:
        data = get_asset_fundamentals(pred['Asset'])
        if data:
            analysis_data.append({
                'Asset': pred['Name'],
                **data
            })
    
    df = pd.DataFrame(analysis_data)
    
    # Format the values properly
    df['Market Cap'] = df['Market Cap'].apply(lambda x: f"₹{x/1e9:.2f}B")
    df['52W High'] = df['52W High'].apply(lambda x: f"₹{x:.2f}")
    df['52W Low'] = df['52W Low'].apply(lambda x: f"₹{x:.2f}")
    df['Dividend Yield'] = df['Dividend Yield'].apply(lambda x: f"{x:.2f}%")
    df['P/E Ratio'] = df['P/E Ratio'].apply(lambda x: f"{x:.2f}")
    df['Beta'] = df['Beta'].apply(lambda x: f"{x:.2f}")

    # Display the styled table
    st.dataframe(
        df.style.set_properties(**{
            'background-color': 'rgba(46, 15, 90, 0.2)',
            'color': 'white',
            'font-size': '14px',
            'padding': '12px 15px'
        }),
        use_container_width=True
    )

    # Add legend/explanation
    with st.expander("📚 Understanding the Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Key Metrics Explained:**
            - **P/E Ratio**: Price to Earnings ratio - lower is generally better
            - **Market Cap**: Total market value in billions
            - **52W High/Low**: Price range in last 52 weeks
            """)
        with col2:
            st.markdown("""
            - **Dividend Yield**: Annual dividend as % of price
            - **Beta**: Market volatility comparison (1 = market average)
            - **N/A**: Data not available
            """)

def show_portfolio():
    st.title("📊 Portfolio Optimization")

    if 'predictions' not in st.session_state or 'historical_data' not in st.session_state:
        st.warning("⚠️ Please generate predictions first!")
        return

    predictions = st.session_state.predictions
    historical_data = st.session_state.historical_data

    # 1. Asset Analysis Section
    st.header("🎯 Asset Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🔄 Correlation Analysis", "⚖️ Optimization Results"])
    
    with tab1:
        # Create metrics grid
        show_asset_analysis(predictions)

    with tab2:
        # Process returns data - Fix for the DataFrame creation
        try:
            # Initialize empty DataFrame with proper index
            first_data = list(historical_data.values())[0]
            returns_data = pd.DataFrame(index=first_data.index)
            
            # Add each asset's returns as a column
            for symbol, data in historical_data.items():
                # Extract just the 'Close' price changes
                returns_data[symbol] = data['Close'].pct_change()
            
            # Remove any rows with missing values
            returns_data = returns_data.dropna()
            
            # Correlation heatmap
            st.subheader("Asset Correlation Matrix")
            corr_matrix = returns_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                labels=dict(color="Correlation")
            )
            fig.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title={
                    'text': "Interactive Correlation Heatmap",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation interpretation
            st.info("""
            🔍 **Interpretation:**
            - Dark Blue: Strong positive correlation (1.0)
            - White: No correlation (0)
            - Dark Red: Strong negative correlation (-1.0)
            
            Diversification benefits increase with lower correlations between assets.
            """)

        except Exception as e:
            st.error(f"Error processing returns data: {str(e)}")
            st.info("Please ensure you have selected valid assets with historical data.")
            return

    with tab3:
        # Calculate optimal portfolio
        cov_matrix = returns_data.cov()
        num_assets = len(predictions)
        
        def portfolio_metrics(weights):
            returns = returns_data.mean() * 252
            port_return = np.sum(returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = port_return / port_risk
            return port_return, port_risk, sharpe

        # Optimization
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        def objective(weights):
            return -portfolio_metrics(weights)[2]  # Maximize Sharpe
        
        initial_weights = np.array([1/num_assets] * num_assets)
        optimal_weights = minimize(objective, initial_weights, 
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints).x

        # Display optimization results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Optimal weights visualization
            fig_weights = go.Figure(data=[
                go.Bar(
                    x=[pred['Name'] for pred in predictions],
                    y=optimal_weights * 100,
                    marker_color='#3D1C66'
                )
            ])
            fig_weights.update_layout(
                title="Optimal Portfolio Weights",
                yaxis_title="Allocation (%)",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        with col2:
            # Portfolio metrics
            opt_return, opt_risk, opt_sharpe = portfolio_metrics(optimal_weights)
            
            st.markdown("""
            <style>
            .big-metric {
                font-size: 24px;
                font-weight: bold;
                color: #FFFFFF;
            }
            .metric-label {
                color: #B8B8B8;
                font-size: 16px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Portfolio Metrics")
            st.markdown(f"""
            <div class="metric-label">Expected Annual Return</div>
            <div class="big-metric">{opt_return*100:.2f}%</div>
            <br>
            <div class="metric-label">Portfolio Risk</div>
            <div class="big-metric">{opt_risk*100:.2f}%</div>
            <br>
            <div class="metric-label">Sharpe Ratio</div>
            <div class="big-metric">{opt_sharpe:.2f}</div>
            """, unsafe_allow_html=True)

    # Add disclaimer at bottom
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer:** This optimization uses Modern Portfolio Theory (MPT) and assumes normally distributed returns.
    Past performance does not guarantee future results. Consider consulting with a financial advisor for personalized advice.
    """)

# import streamlit as st
# from utils.portfolio_opt import fetch_stock_data, optimize_portfolio
# import plotly.express as px
# import pandas as pd

# def show_portfolio():
#     st.title("💰 Portfolio Optimization")

#     tickers = st.text_input("Enter comma-separated tickers", "RELIANCE.NS, TCS.NS, INFY.NS")
#     start_date = st.date_input("Start Date")
#     end_date = st.date_input("End Date")

#     if st.button("Optimize Portfolio"):
#         with st.spinner("Fetching data and optimizing..."):
#             ticker_list = [ticker.strip() for ticker in tickers.split(",")]
#             data = fetch_stock_data(ticker_list, start_date, end_date)
#             weights = optimize_portfolio(data)

#             st.subheader("Optimal Portfolio Allocation")
#             allocation = pd.DataFrame({
#                 'Stock': ticker_list,
#                 'Allocation %': [round(w * 100, 2) for w in weights]
#             })

#             st.dataframe(allocation)

#             fig = px.pie(allocation, names='Stock', values='Allocation %', title='Portfolio Allocation')
#             st.plotly_chart(fig)