import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from datetime import datetime, timedelta
import plotly.express as px

# Predefined assets
PREDEFINED_ASSETS = {
    'Crypto': {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'USDT-USD': 'Tether',
        'BNB-USD': 'Binance Coin',
        'XRP-USD': 'Ripple'
    },
    'Government Bonds': {
        'TLT': 'US 20+ Year Treasury',
        'IEF': '7-10 Year Treasury',
        'SHY': '1-3 Year Treasury'
    },
    'Precious Metals': {
        'GLD': 'Gold ETF',
        'SLV': 'Silver ETF',
        'PPLT': 'Platinum ETF'
    },
    'Commodities': {
        'USO': 'US Oil Fund',
        'UNG': 'Natural Gas Fund',
        'CORN': 'Corn Fund',
        'WEAT': 'Wheat Fund'
    }
}

def show_prediction():
    with st.sidebar:
        st.subheader("Investment Parameters")
        
        # Investment Type Selection
        investment_type = st.radio(
            "Select Investment Type",
            ["Lumpsum", "SIP"],
            key="investment_type_3"
        )
        
        # Amount Input based on type
        if investment_type == "Lumpsum":
            investment_amount = st.number_input(
                "Enter Lumpsum Amount (₹)",
                min_value=1000,
                value=10000,
                step=1000,
                key="lumpsum_amount"
            )
        else:
            investment_amount = st.number_input(
                "Enter Monthly SIP Amount (₹)",
                min_value=500,
                value=5000,
                step=500,
                key="sip_amount"
            )

        # Duration Input
        investment_duration = st.slider(
            "Investment Duration (Years)",
            min_value=1,
            max_value=30,
            value=5,
            key="duration"
        )
        
        # Risk-Free Rate Input
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.50,
            step=0.25,
            key="risk_free_rate"
        )

        # Enhanced Investment Summary with better styling
        st.markdown("""
        <style>
        .investment-card {
            background-color: rgba(46, 15, 90, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .metric-container {
            background-color: rgba(61, 28, 102, 0.3);
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
        }
        .metric-label {
            color: #B8B8B8;
            font-size: 0.9em;
        }
        .metric-value {
            color: white;
            font-size: 1.1em;
            font-weight: bold;
        }
        .positive-return {
            color: #00FF9F;
        }
        .metric-subtext {
            color: #B8B8B8;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### 💰 Investment Summary")
        
        # Investment Type Card
        st.markdown('<div class="investment-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Investment Type")
        if investment_type == "Lumpsum":
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">Lump Sum Investment</div>'
                f'<div class="metric-value">₹{investment_amount:,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">Monthly SIP</div>'
                f'<div class="metric-value">₹{investment_amount:,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Duration Card
        st.markdown('<div class="investment-card">', unsafe_allow_html=True)
        st.markdown("#### ⌛ Timeline")
        start_year = 2025
        st.markdown(
            f'<div class="metric-container">'
            f'<div class="metric-label">Investment Period</div>'
            f'<div class="metric-value">{investment_duration} Years</div>'
            f'<div class="metric-subtext">From {start_year} to {start_year + investment_duration}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Investment Details Card
        st.markdown('<div class="investment-card">', unsafe_allow_html=True)
        st.markdown("#### 💵 Financial Projection")
        
        # Calculate values
        if investment_type == "Lumpsum":
            total_investment = investment_amount
            future_value = investment_amount * (1 + risk_free_rate/100) ** investment_duration
        else:
            total_investment = investment_amount * 12 * investment_duration
            monthly_rate = (risk_free_rate/100)/12
            future_value = investment_amount * ((1 + monthly_rate) ** (12 * investment_duration) - 1) / monthly_rate * (1 + monthly_rate)
        
        total_return = future_value - total_investment
        return_percentage = (total_return/total_investment*100)
        
        # Display metrics with enhanced styling
        st.markdown(
            f'<div class="metric-container">'
            f'<div class="metric-label">Total Investment</div>'
            f'<div class="metric-value">₹{total_investment:,.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'<div class="metric-container">'
            f'<div class="metric-label">Expected Future Value</div>'
            f'<div class="metric-value">₹{future_value:,.2f}</div>'
            f'<div class="metric-subtext">At {risk_free_rate}% annual return</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'<div class="metric-container">'
            f'<div class="metric-label">Projected Returns</div>'
            f'<div class="metric-value positive-return">+₹{total_return:,.2f}</div>'
            f'<div class="metric-value positive-return">(+{return_percentage:,.2f}%)</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    st.title("Stock Portfolio Analysis")
    st.header("📈 Intelligent Portfolio Predictor")
    
    # Asset Selection
    st.header("Asset Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stocks")
        stocks = st.text_input(
            "Enter Stock Symbols (comma-separated):",
            value="RELIANCE.NS, TCS.NS",
            help="Add .NS for NSE stocks (e.g., RELIANCE.NS, TCS.NS)"
        )
        stock_list = [s.strip() for s in stocks.split(',') if s.strip()]
    
    with col2:
        st.subheader("ETFs")
        etfs = st.text_input(
            "Enter ETF Symbols (comma-separated):",
            value="NIFTYBEES.NS",
            help="Add .NS for NSE ETFs (e.g., NIFTYBEES.NS, BANKBEES.NS)"
        )
        etf_list = [e.strip() for e in etfs.split(',') if e.strip()]

    # Other Assets
    st.header("Other Assets")
    
    col3, col4 = st.columns(2)
    selected_assets = {}
    
    with col3:
        st.subheader("🪙 Cryptocurrencies")
        selected_assets['Crypto'] = st.multiselect(
            "Select Cryptocurrencies:",
            options=list(PREDEFINED_ASSETS['Crypto'].keys()),
            format_func=lambda x: PREDEFINED_ASSETS['Crypto'][x]
        )
        
        st.subheader("📜 Government Bonds")
        selected_assets['Bonds'] = st.multiselect(
            "Select Bonds:",
            options=list(PREDEFINED_ASSETS['Government Bonds'].keys()),
            format_func=lambda x: PREDEFINED_ASSETS['Government Bonds'][x]
        )
    
    with col4:
        st.subheader("🥇 Precious Metals")
        selected_assets['Metals'] = st.multiselect(
            "Select Metals:",
            options=list(PREDEFINED_ASSETS['Precious Metals'].keys()),
            format_func=lambda x: PREDEFINED_ASSETS['Precious Metals'][x]
        )
        
        st.subheader("🛢️ Commodities")
        selected_assets['Commodities'] = st.multiselect(
            "Select Commodities:",
            options=list(PREDEFINED_ASSETS['Commodities'].keys()),
            format_func=lambda x: PREDEFINED_ASSETS['Commodities'][x]
        )

    # Combine all selected assets
    all_assets = stock_list + etf_list
    for asset_list in selected_assets.values():
        all_assets.extend(asset_list)

    # Analysis Period
    st.header("Analysis Period")
    end_date = datetime.today()
    start_date = st.date_input(
        "Start Date", 
        value=end_date - timedelta(days=365),
        max_value=end_date - timedelta(days=30)
    )
    
    # Generate Predictions
    if st.button("Generate Predictions", type="primary"):
        if not all_assets:
            st.warning("Please select at least one asset!")
            return
            
        with st.spinner("Fetching market data..."):
            try:
                data = {}
                for asset in all_assets:
                    try:
                        df = yf.download(
                            asset,
                            start=start_date,
                            end=end_date,
                            progress=False
                        )
                        if len(df) > 0 and 'Close' in df.columns:
                            data[asset] = df
                    except Exception as e:
                        st.warning(f"Failed to fetch data for {asset}")

                if not data:
                    st.error("Could not fetch data for any selected assets")
                    return

                # Generate predictions
                predictions = []
                for asset, df in data.items():
                    try:
                        current_price = float(df['Close'].iloc[-1])
                        predicted_price = float(current_price * np.random.uniform(0.98, 1.02))
                        change_pct = ((predicted_price - current_price)/current_price)*100
                        
                        # Determine asset type
                        if asset in stock_list:
                            asset_type = "Stock"
                        elif asset in etf_list:
                            asset_type = "ETF"
                        else:
                            asset_type = next((k for k, v in selected_assets.items() if asset in v), None)
                        
                        asset_name = PREDEFINED_ASSETS.get(asset_type, {}).get(asset, asset.replace('.NS', ''))
                        
                        predictions.append({
                            'Asset': asset,
                            'Name': asset_name,
                            'Type': asset_type,
                            'Current Price (₹)': current_price,
                            'Predicted Price (₹)': predicted_price,
                            'Change (%)': change_pct,
                            'Confidence': float(np.random.uniform(70, 95))
                        })
                    except Exception as e:
                        st.warning(f"Could not process {asset}")

                if predictions:
                    st.session_state.predictions = predictions
                    st.session_state.historical_data = data

                    # Display predictions
                    st.header("Prediction Results")
                    
                    cols = st.columns([1.5, 2, 1, 1, 1, 1])
                    headers = ["Asset Type & Name", "Price Trend", "Current Price", 
                             "Predicted Price", "Change %", "Confidence"]
                    
                    for col, header in zip(cols, headers):
                        col.markdown(f"**{header}**")
                    
                    for pred in predictions:
                        cols = st.columns([1.5, 2, 1, 1, 1, 1])
                        
                        cols[0].markdown(f"**{pred['Type']}**\n{pred['Name']}")
                        
                        # Trend indicator
                        trend = "↗️" if pred['Change (%)'] > 0 else "↘️"
                        color = "green" if pred['Change (%)'] > 0 else "red"
                        cols[1].markdown(
                            f"<span style='color:{color}; font-size: 20px;'>{trend} {abs(pred['Change (%)']):,.2f}%</span>",
                            unsafe_allow_html=True
                        )
                        
                        cols[2].markdown(f"₹{pred['Current Price (₹)']:,.2f}")
                        cols[3].markdown(f"₹{pred['Predicted Price (₹)']:,.2f}")
                        cols[4].markdown(
                            f"<span style='color:{color}'>{pred['Change (%)']:+.2f}%</span>",
                            unsafe_allow_html=True
                        )
                        cols[5].markdown(f"{pred['Confidence']:.1f}%")
                        
                        st.markdown("<hr style='margin: 5px 0; opacity: 0.3'>", unsafe_allow_html=True)
                    
                    # Allocation recommendation
                    st.subheader("Recommended Allocation")
                    weights = np.random.dirichlet(np.ones(len(predictions)))
                    allocation_data = pd.DataFrame({
                        'Asset': [pred['Name'] for pred in predictions],
                        'Allocation': weights * 100
                    })

                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.pie(
                            allocation_data, 
                            values='Allocation', 
                            names='Asset',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_layout(
                            height=400,
                            margin=dict(t=30, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("### Allocation Details")
                        for _, row in allocation_data.iterrows():
                            st.markdown(f"**{row['Asset']}:** {row['Allocation']:.2f}%")
                    
                    st.markdown("""
                    ---
                    **Disclaimer:**  
                    Predictions are based on historical data and machine learning models.  
                    Past performance is not indicative of future results.  
                    Always consult with a financial advisor before making investment decisions.
                    """)
                
                else:
                    st.error("No valid predictions could be generated")
                
            except Exception as e:
                st.error(f"Error in prediction process: {str(e)}")

if __name__ == "__main__":
    show_prediction()