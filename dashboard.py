import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_fetcher import fetch_market_data, fetch_global_data, fetch_portfolio_data, fetch_transactions, fetch_active_stocks, fetch_gainers_losers, fetch_commodities, fetch_currencies, fetch_bonds

# -----------------------------------------------
# Market Overview Cards (API Integrated)
# -----------------------------------------------
def show_market_overview():
    st.subheader("📊 Market Overview")
    nifty, sensex, vix = fetch_market_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NIFTY 50", nifty['value'], nifty['change'])
    with col2:
        st.metric("SENSEX", sensex['value'], sensex['change'])
    with col3:
        st.metric("India VIX", vix['value'], vix['change'], delta_color="inverse")

# -----------------------------------------------
# Global Markets & Indian Sector Indices (API Integrated)
# -----------------------------------------------
def show_global_indian_markets():
    st.subheader("🌎 Global & 🇮🇳 Indian Markets")
    global_indices, indian_indices = fetch_global_data()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🌍 Global Markets")
        st.dataframe(
            global_indices.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %']),
            hide_index=True,
            use_container_width=True
        )
    with col2:
        st.markdown("##### 🇮🇳 Indian Sector Indices")
        st.dataframe(
            indian_indices.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %']),
            hide_index=True,
            use_container_width=True
        )
# -----------------------------------------------
# Portfolio Summary & Asset Allocation (API Integrated)
# -----------------------------------------------
def show_portfolio_section():
    st.subheader("📈 Portfolio Snapshot")
    portfolio_summary, asset_allocation = fetch_portfolio_data()

    col1, col2 = st.columns(2)
    with col1:
        # Portfolio metrics
        st.metric("Total Value", portfolio_summary['total_value'], portfolio_summary['total_change'])
        st.metric("Total Return", portfolio_summary['total_return'], portfolio_summary['return_change'])

    with col2:
        # Asset allocation pie chart
        fig = px.pie(
            asset_allocation, 
            values='Allocation', 
            names='Asset', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            showlegend=True,
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0)'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------
# Portfolio Summary & Asset Allocation (API Integrated)
# -----------------------------------------------
def show_commodities_currencies():
    st.subheader("💰 Market Instruments")
    commodities, currencies, bonds = fetch_commodities(), fetch_currencies(), fetch_bonds()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Commodities")
        st.dataframe(
            commodities.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %'])
            .format({
                'Price': '₹{:,.2f}',
                'Change %': '{:,.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("##### Currencies")
        st.dataframe(
            currencies.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %'])
            .format({
                'Rate': '₹{:,.2f}',
                'Change %': '{:,.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col3:
        st.markdown("##### Bonds")
        st.dataframe(
            bonds.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %'])
            .format({
                'Yield': '{:,.2f}%',
                'Change %': '{:,.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )   
# -----------------------------------------------
# Recent Transactions (API Integrated)
# -----------------------------------------------
def show_recent_transactions():
    st.subheader("📝 Recent Transactions")
    transactions = fetch_transactions()

    with st.expander("View Transactions"):
        st.dataframe(
            transactions.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            })
            .format({
                'Price': '₹{:,.2f}',
                'Total': '₹{:,.2f}'
            })
            .applymap(lambda x: 'color: #00FF00' if isinstance(x, str) and x == 'Buy' else 
                    ('color: #FF4444' if isinstance(x, str) and x == 'Sell' else 'color: white')),
            hide_index=True,
            use_container_width=True
        )
# -----------------------------------------------
# Most Active Stocks & Gainers/Losers (API Integrated)
# -----------------------------------------------
def show_stock_activity():
    st.subheader("🔥 Active Stocks & Movers")
    most_active, gainers, losers = fetch_active_stocks(), *fetch_gainers_losers()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Most Active")
        # Updated styling for most active stocks
        st.dataframe(
            most_active.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("##### Top Gainers / Losers")
        st.markdown("**Gainers**")
        # Updated styling for gainers with green background
        st.dataframe(
            gainers.style
            .set_properties(**{
                'background-color': 'rgba(0, 255, 0, 0.1)',
                'color': 'white',
                'border': '1px solid #4A4A4A'
            })
            .apply(lambda x: ['color: #00FF00' if '+' in str(v) else 'color: white' for v in x], 
                  subset=['Change %']),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("**Losers**")
        # Updated styling for losers with red background
        st.dataframe(
            losers.style
            .set_properties(**{
                'background-color': 'rgba(255, 0, 0, 0.1)',
                'color': 'white',
                'border': '1px solid #4A4A4A'
            })
            .apply(lambda x: ['color: #FF4444' if '-' in str(v) else 'color: white' for v in x], 
                  subset=['Change %']),
            hide_index=True,
            use_container_width=True
        )
# -----------------------------------------------
# Commodities, Currencies, Bonds (API Integrated)
# -----------------------------------------------
# -----------------------------------------------
# Market Instruments (API Integrated)
# -----------------------------------------------
def show_commodities_currencies():
    st.subheader("💰 Market Instruments")
    commodities, currencies, bonds = fetch_commodities(), fetch_currencies(), fetch_bonds()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Commodities")
        st.dataframe(
            commodities.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("##### Currencies")
        st.dataframe(
            currencies.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col3:
        st.markdown("##### Bonds")
        st.dataframe(
            bonds.style
            .set_properties(**{
                'background-color': '#2D2D2D',
                'color': 'white',
                'border': '1px solid #4A4A4A',
                'padding': '8px'
            }),
            hide_index=True,
            use_container_width=True
        )
# -----------------------------------------------
# Main Dashboard Layout
# -----------------------------------------------
def show_dashboard():
    st.title(f"📊 Welcome, {st.session_state.get('name', 'Investor')}! 👋")
    st.markdown("Stay updated with live market data and portfolio insights.")

    show_market_overview()
    st.markdown("---")
    show_global_indian_markets()
    st.markdown("---")
    show_portfolio_section()
    st.markdown("---")
    show_recent_transactions()
    st.markdown("---")
    show_stock_activity()
    st.markdown("---")
    show_commodities_currencies()
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This dashboard displays real-time market data fetched from various financial APIs. While we strive for accuracy, please verify critical information from official sources before making investment decisions.")


# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # -----------------------------------------------
# # Main Dashboard Function
# # -----------------------------------------------
# def show_dashboard():
#     st.title(f"👋 Welcome, {st.session_state.get('name', 'Investor')}!")
#     st.caption("Your personalized financial dashboard")

#     # Market Overview
#     show_market_overview()
#     st.divider()

#     # Global & Indian Markets
#     show_global_indian_markets()
#     st.divider()

#     # Portfolio Snapshot
#     show_portfolio_section()
#     st.divider()

#     # Recent Transactions
#     show_recent_transactions()
#     st.divider()

#     # Stock Activity (Active Stocks, Gainers, Losers)
#     show_stock_activity()
#     st.divider()

#     # Commodities, Currencies, Bonds
#     show_commodities_currencies()
#     st.divider()

#     # Disclaimer
#     st.caption("⚠️ **Disclaimer:** This is a demo dashboard using placeholder data for UI visualization purposes only.")


# # -----------------------------------------------
# # Market Overview Section
# # -----------------------------------------------
# def show_market_overview():
#     st.subheader("📊 Market Overview")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("NIFTY 50", "22,212.70", "+0.15%")
#     col2.metric("SENSEX", "73,327.94", "+0.12%")
#     col3.metric("India VIX", "12.45", "-3.2%", delta_color="inverse")


# # -----------------------------------------------
# # Global & Indian Market Indices
# # -----------------------------------------------
# def show_global_indian_markets():
#     st.subheader("🌎 Global & 🇮🇳 Indian Markets")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("##### 🌎 Global Markets")
#         global_df = pd.DataFrame({
#             'Index': ['Nasdaq', 'Dow Jones', 'FTSE 100'],
#             'Value': ['16,250.25', '38,790.12', '7,923.60'],
#             'Change': ['+0.32%', '-0.12%', '+0.28%']
#         })
#         st.dataframe(global_df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("##### 🇮🇳 Indian Sector Indices")
#         india_df = pd.DataFrame({
#             'Index': ['Bank Nifty', 'Midcap', 'IT Index'],
#             'Value': ['46,790.30', '42,200.40', '33,120.12'],
#             'Change': ['+0.50%', '+0.18%', '-0.30%']
#         })
#         st.dataframe(india_df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Portfolio Summary & Asset Allocation
# # -----------------------------------------------
# def show_portfolio_section():
#     st.subheader("📈 Portfolio Snapshot")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric("Total Value", "₹1,00,000", "+5%")
#         st.metric("Total Return", "12.5%", "+2.5%")

#     with col2:
#         allocation_data = {
#             'Asset': ['Stocks', 'Crypto', 'Gold', 'Cash'],
#             'Allocation': [60, 20, 15, 5]
#         }
#         fig = px.pie(allocation_data, values='Allocation', names='Asset', hole=0.4)
#         fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
#         st.plotly_chart(fig, use_container_width=True)


# # -----------------------------------------------
# # Recent Transactions (Collapsible)
# # -----------------------------------------------
# def show_recent_transactions():
#     st.subheader("📝 Recent Transactions")
#     with st.expander("View Transaction History"):
#         transactions_df = pd.DataFrame({
#             'Date': ['2024-03-11', '2024-03-10', '2024-03-09'],
#             'Type': ['Buy', 'Sell', 'Buy'],
#             'Asset': ['RELIANCE', 'TCS', 'NIFTYBEES'],
#             'Quantity': [10, 5, 20],
#             'Price': [2500, 3800, 200],
#             'Total': [25000, 19000, 4000]
#         })
#         st.dataframe(
#             transactions_df.style.format({'Price': '₹{:,.2f}', 'Total': '₹{:,.2f}'}),
#             hide_index=True,
#             use_container_width=True
#         )


# # -----------------------------------------------
# # Most Active, Top Gainers & Losers
# # -----------------------------------------------
# def show_stock_activity():
#     st.subheader("🔥 Stock Market Highlights")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("##### Most Active Stocks")
#         active_df = pd.DataFrame({
#             'Stock': ['RELIANCE', 'TCS', 'INFY'],
#             'Price': ['₹2,500', '₹3,800', '₹1,500'],
#             'Change': ['+1.5%', '-0.5%', '+0.8%']
#         })
#         st.dataframe(active_df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("##### Top Gainers & Losers")
#         gainers_df = pd.DataFrame({'Stock': ['ADANIGREEN', 'BAJFINANCE'], 'Change': ['+4.2%', '+3.8%']})
#         losers_df = pd.DataFrame({'Stock': ['ONGC', 'COALINDIA'], 'Change': ['-2.1%', '-1.8%']})
#         st.markdown("**Top Gainers**")
#         st.dataframe(gainers_df, hide_index=True, use_container_width=True)
#         st.markdown("**Top Losers**")
#         st.dataframe(losers_df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Commodities, Currencies & Bonds
# # -----------------------------------------------
# def show_commodities_currencies():
#     st.subheader("💰 Commodities, Currencies & Bonds")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown("##### Commodities")
#         commodities_df = pd.DataFrame({'Asset': ['Gold', 'Silver'], 'Price': ['₹65,500', '₹72,000']})
#         st.dataframe(commodities_df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("##### Currencies")
#         currency_df = pd.DataFrame({'Pair': ['USD/INR', 'EUR/INR'], 'Rate': ['₹83.12', '₹89.50']})
#         st.dataframe(currency_df, hide_index=True, use_container_width=True)

#     with col3:
#         st.markdown("##### Bonds")
#         bond_df = pd.DataFrame({'Bond': ['India 10Y'], 'Yield': ['7.05%']})
#         st.dataframe(bond_df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Entry Point for Running Standalone (Optional)
# # -----------------------------------------------
# if __name__ == "__main__":
#     show_dashboard()



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # -----------------------------------------------
# # Market Overview Section (Indian Indices)
# # -----------------------------------------------
# def show_market_overview():
#     st.subheader("📊 Market Overview")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("NIFTY 50", "22,212.70", "+0.15%")
#     with col2:
#         st.metric("SENSEX", "73,327.94", "+0.12%")
#     with col3:
#         st.metric("India VIX", "12.45", "-3.2%", delta_color="inverse")


# # -----------------------------------------------
# # Global Markets & Indian Indices Section
# # -----------------------------------------------
# def show_global_markets():
#     st.subheader("🌎 Global & Indian Markets")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("#### 🌎 Global Markets")
#         data = {
#             'Index': ['Nasdaq', 'Dow Jones', 'FTSE 100'],
#             'Value': ['16,250.25', '38,790.12', '7,923.60'],
#             'Change': ['+0.32%', '-0.12%', '+0.28%']
#         }
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("#### 🇮🇳 Indian Sector Indices")
#         data = {
#             'Index': ['Bank Nifty', 'Midcap', 'IT Index'],
#             'Value': ['46,790.30', '42,200.40', '33,120.12'],
#             'Change': ['+0.50%', '+0.18%', '-0.30%']
#         }
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Stock Action Section (Most Active)
# # -----------------------------------------------
# def show_stock_action():
#     st.subheader("🔥 Most Active Stocks")
#     data = {
#         'Stock': ['RELIANCE', 'TCS', 'INFY'],
#         'Price': ['₹2,500', '₹3,800', '₹1,500'],
#         'Change': ['+1.5%', '-0.5%', '+0.8%'],
#         'Volume (Cr)': [15.2, 8.5, 6.1]
#     }
#     df = pd.DataFrame(data)
#     st.dataframe(df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Top Gainers and Losers Section
# # -----------------------------------------------
# def show_top_gainers_losers():
#     st.subheader("🏆 Top Gainers & 📉 Top Losers")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("##### 🏆 Top Gainers")
#         data = {
#             'Stock': ['ADANIGREEN', 'BAJFINANCE', 'HINDALCO'],
#             'Price': ['₹1,400', '₹7,200', '₹525'],
#             'Change': ['+4.2%', '+3.8%', '+3.5%']
#         }
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("##### 📉 Top Losers")
#         data = {
#             'Stock': ['ONGC', 'COALINDIA', 'TATAMOTORS'],
#             'Price': ['₹225', '₹330', '₹940'],
#             'Change': ['-2.1%', '-1.8%', '-1.2%']
#         }
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Commodities, Currencies, Bonds Section
# # -----------------------------------------------
# def show_commodities_currencies_bonds():
#     st.subheader("💰 Commodities, 💱 Currencies & 🏦 Bonds")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown("##### 💰 Commodities")
#         data = {'Asset': ['Gold', 'Silver', 'Crude Oil'], 'Price': ['₹65,500', '₹72,000', '$82.15'], 'Change': ['+0.3%', '-0.1%', '+0.6%']}
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)

#     with col2:
#         st.markdown("##### 💱 Currencies")
#         data = {'Pair': ['USD/INR', 'EUR/INR', 'JPY/INR'], 'Rate': ['₹83.12', '₹89.50', '₹0.57'], 'Change': ['+0.1%', '-0.2%', '+0.05%']}
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)

#     with col3:
#         st.markdown("##### 🏦 Bonds/Yields")
#         data = {'Bond': ['India 10Y', 'US 10Y'], 'Yield': ['7.05%', '4.15%'], 'Change': ['+0.02%', '-0.01%']}
#         df = pd.DataFrame(data)
#         st.dataframe(df, hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Portfolio Summary (Your existing section retained)
# # -----------------------------------------------
# def show_portfolio_summary():
#     st.subheader("📈 Portfolio Summary")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric("Total Portfolio Value", "₹1,00,000", "+5%")
#     with col2:
#         st.metric("Total Return (%)", "12.5%", "+2.5%")


# # -----------------------------------------------
# # Asset Allocation Pie Chart (Enhanced)
# # -----------------------------------------------
# def show_asset_allocation():
#     st.subheader("💼 Asset Allocation")
#     allocation_data = {'Asset': ['Stocks', 'Crypto', 'Gold', 'Cash'], 'Allocation': [60, 20, 15, 5]}
#     fig = px.pie(allocation_data, values='Allocation', names='Asset', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
#     fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'})
#     st.plotly_chart(fig, use_container_width=True)


# # -----------------------------------------------
# # Recent Transactions Section (Your existing section retained)
# # -----------------------------------------------
# def show_recent_transactions():
#     st.subheader("📝 Recent Transactions")
#     transactions = pd.DataFrame({
#         'Date': ['2024-03-11', '2024-03-10', '2024-03-09'],
#         'Type': ['Buy', 'Sell', 'Buy'],
#         'Asset': ['RELIANCE', 'TCS', 'NIFTYBEES'],
#         'Quantity': [10, 5, 20],
#         'Price': [2500, 3800, 200],
#         'Total': [25000, 19000, 4000]
#     })
#     st.dataframe(transactions.style.format({'Price': '₹{:,.2f}', 'Total': '₹{:,.2f}'}), hide_index=True, use_container_width=True)


# # -----------------------------------------------
# # Main Dashboard Layout
# # -----------------------------------------------
# def show_dashboard():
#     st.title(f"Welcome, {st.session_state.get('name', 'Investor')}! 👋")
#     st.markdown("Stay updated with live market insights and manage your portfolio effectively.")

#     show_market_overview()
#     st.markdown("---")
#     show_global_markets()
#     st.markdown("---")
#     show_stock_action()
#     st.markdown("---")
#     show_top_gainers_losers()
#     st.markdown("---")
#     show_commodities_currencies_bonds()
#     st.markdown("---")

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         show_portfolio_summary()
#     with col2:
#         show_asset_allocation()

#     st.markdown("---")
#     show_recent_transactions()

#     st.markdown("---")
#     st.caption("**Disclaimer:** This is a demo dashboard. Values are placeholders. Verify data independently.")

# # -----------------------------------------------
# # Run the Dashboard
# # -----------------------------------------------
# if __name__ == "__main__":
#     show_dashboard()

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from utils.data_fetcher import fetch_data, ASSETS

# def show_market_overview():
#     """Display market overview section with key indices"""
#     st.subheader("📊 Market Overview")
    
#     # Create columns for different market metrics
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric(
#             label="NIFTY 50",
#             value="22,212.70",
#             delta="0.15%",
#             help="Current NIFTY 50 Index Value"
#         )
    
#     with col2:
#         st.metric(
#             label="SENSEX",
#             value="73,327.94",
#             delta="0.12%",
#             help="Current SENSEX Index Value"
#         )
    
#     with col3:
#         st.metric(
#             label="India VIX",
#             value="12.45",
#             delta="-3.2%",
#             delta_color="inverse",
#             help="Market Volatility Index"
#         )

# def show_portfolio_summary():
#     """Display portfolio summary with key metrics"""
#     st.subheader("📈 Portfolio Summary")
    
#     # Create two columns for portfolio metrics
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Portfolio Value Chart
#         fig = go.Figure(go.Indicator(
#             mode = "number+delta",
#             value = float(st.session_state.get('investment_amount', 100000)),
#             delta = {"reference": 95000, "relative": True, "position": "right"},
#             title = {"text": "Portfolio Value"},
#             domain = {'x': [0, 1], 'y': [0, 1]}
#         ))
#         fig.update_layout(
#             height=200,
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font={'color': '#ffffff'}
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         # Returns Chart
#         fig = go.Figure(go.Indicator(
#             mode = "gauge+number",
#             value = 12.5,
#             title = {'text': "Total Return (%)"},
#             gauge = {
#                 'axis': {'range': [-20, 20], 'tickcolor': "#ffffff"},
#                 'bar': {'color': "#1f77b4"},
#                 'steps': [
#                     {'range': [-20, 0], 'color': "rgba(255, 99, 71, 0.3)"},
#                     {'range': [0, 20], 'color': "rgba(144, 238, 144, 0.3)"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 0
#                 }
#             }
#         ))
#         fig.update_layout(
#             height=200,
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font={'color': '#ffffff'}
#         )
#         st.plotly_chart(fig, use_container_width=True)

# def show_asset_allocation():
#     """Display asset allocation breakdown"""
#     st.subheader("💼 Asset Allocation")
    
#     # Sample data for asset allocation
#     allocation_data = {
#         'Asset': ['Stocks', 'Crypto', 'Gold', 'Cash'],
#         'Allocation': [60, 20, 15, 5]
#     }
    
#     # Create pie chart
#     fig = px.pie(
#         allocation_data, 
#         values='Allocation', 
#         names='Asset',
#         hole=0.4,
#         color_discrete_sequence=px.colors.qualitative.Set3
#     )
#     fig.update_layout(
#         height=300,
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': '#ffffff'}
#     )
#     st.plotly_chart(fig, use_container_width=True)

# def show_recent_transactions():
#     """Display recent transactions table"""
#     st.subheader("📝 Recent Transactions")
    
#     # Sample transaction data
#     transactions = pd.DataFrame({
#         'Date': ['2024-03-11', '2024-03-10', '2024-03-09'],
#         'Type': ['Buy', 'Sell', 'Buy'],
#         'Asset': ['RELIANCE', 'TCS', 'NIFTYBEES'],
#         'Quantity': [10, 5, 20],
#         'Price': [2500, 3800, 200],
#         'Total': [25000, 19000, 4000]
#     })
    
#     # Add styling
#     st.markdown("""
#     <style>
#     .stDataFrame {
#         background-color: rgba(255, 255, 255, 0.1);
#         border-radius: 10px;
#         padding: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Display styled dataframe
#     st.dataframe(
#         transactions.style
#         .format({
#             'Price': '₹{:,.2f}',
#             'Total': '₹{:,.2f}'
#         })
#         .applymap(lambda x: 'color: #90EE90' if isinstance(x, str) and x == 'Buy' else 
#                           ('color: #FFB6C1' if isinstance(x, str) and x == 'Sell' else '')),
#         hide_index=True,
#         use_container_width=True
#     )

# def show_dashboard():
#     """Main dashboard display function"""
#     st.title(f"Welcome, {st.session_state['name']}! 👋")
    
#     # Add a brief description
#     st.markdown("""
#     Monitor your investments, track performance, and analyze your asset allocation all in one place.
#     Let's check how your portfolio is performing today.
#     """)
    
#     # Display all dashboard components
#     show_market_overview()
    
#     st.markdown("---")
    
#     show_portfolio_summary()
    
#     st.markdown("---")
    
#     # Create two columns for allocation and transactions
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         show_asset_allocation()
    
#     with col2:
#         show_recent_transactions()
    
#     # Add disclaimer at the bottom
#     st.markdown("---")
#     st.caption("""
#     **Disclaimer:** All values shown are for demonstration purposes. Past performance 
#     is not indicative of future results. Please verify all data independently.
#     """)

# if __name__ == "__main__":
#     show_dashboard()