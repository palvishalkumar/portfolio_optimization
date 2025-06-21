import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.065):
    """Calculate portfolio metrics using MPT"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return portfolio_return, portfolio_risk, sharpe_ratio

def optimize_portfolio(returns_data, risk_free_rate=0.065):
    """Core optimization function"""
    # Calculate mean returns and covariance matrix
    cov_matrix = returns_data.cov()
    num_assets = len(returns_data.columns)
    
    # Optimization setup
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    def objective(weights):
        return -calculate_portfolio_metrics(weights, returns_data, cov_matrix, risk_free_rate)[2]
    
    # Run optimization
    initial_weights = np.array([1/num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # Get optimal metrics
    opt_weights = result.x
    opt_metrics = calculate_portfolio_metrics(opt_weights, returns_data, cov_matrix, risk_free_rate)
    
    return {
        'weights': opt_weights,
        'return': opt_metrics[0],
        'risk': opt_metrics[1],
        'sharpe': opt_metrics[2],
        'cov_matrix': cov_matrix
    }

def optimize_portfolio_with_predictions(historical_data, predictions, risk_free_rate=0.065):
    """
    Optimize portfolio using both historical data and predicted returns
    """
    try:
        # Ensure we have DataFrame
        if not isinstance(historical_data, pd.DataFrame):
            raise ValueError("Historical data must be a DataFrame")
            
        # Ensure we have valid predictions
        if not predictions:
            raise ValueError("No predictions provided")
            
        # Create arrays for optimization
        assets = [p['Asset'] for p in predictions]
        exp_returns = np.array([p['Change (%)'] / 100 for p in predictions])
        
        # Validate we have all required data
        missing_assets = [asset for asset in assets if asset not in historical_data.columns]
        if missing_assets:
            raise ValueError(f"Missing historical data for assets: {missing_assets}")
            
        # Calculate covariance matrix
        returns = historical_data[assets].pct_change().dropna()
        cov_matrix = returns.cov() * 252  # Annualize covariance
        
        # Simple equal weight portfolio for testing
        num_assets = len(assets)
        weights = np.array([1/num_assets] * num_assets)
        
        # Calculate basic metrics
        portfolio_return = np.sum(exp_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk != 0 else 0
        
        return {
            'success': True,
            'weights': dict(zip(assets, weights)),
            'metrics': {
                'expected_return': portfolio_return * 100,
                'risk': portfolio_risk * 100,
                'sharpe_ratio': sharpe_ratio,
                'beta': 1.0,
                'diversification_ratio': num_assets
            }
        }
        
    except Exception as e:
        print(f"Optimization error details: {str(e)}")  # For debugging
        return {
            'success': False,
            'error': str(e)
        }

def calculate_portfolio_beta(returns, weights):
    """Calculate portfolio beta relative to market"""
    try:
        market_returns = returns.mean(axis=1)  # Using equal-weighted market proxy
        portfolio_returns = returns.dot(weights)
        covariance = np.cov(portfolio_returns, market_returns)[0,1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 1
    except:
        return 1

def calculate_diversification_ratio(weights, cov_matrix):
    """Calculate portfolio diversification ratio"""
    try:
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        weighted_volatilities = np.sqrt(np.diag(cov_matrix)) * weights
        sum_weighted_vols = np.sum(weighted_volatilities)
        return sum_weighted_vols / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1
    except:
        return 1

def generate_efficient_frontier(historical_data, predictions, risk_free_rate=0.065, num_portfolios=100):
    """Generate efficient frontier points"""
    try:
        returns = historical_data.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        
        # Get expected returns from predictions
        exp_returns = np.array([pred['Change (%)'] / 100 for pred in predictions])
        
        # Generate random portfolios
        num_assets = len(predictions)
        all_weights = np.random.dirichlet(np.ones(num_assets), num_portfolios)
        
        # Calculate returns and risks for each portfolio
        all_returns = []
        all_risks = []
        all_sharpe_ratios = []
        
        for weights in all_weights:
            portfolio_return, portfolio_risk, sharpe = calculate_portfolio_metrics(
                weights, returns, cov_matrix, risk_free_rate
            )
            all_returns.append(portfolio_return)
            all_risks.append(portfolio_risk)
            all_sharpe_ratios.append(sharpe)
        
        return {
            'returns': np.array(all_returns) * 100,  # Convert to percentage
            'risks': np.array(all_risks) * 100,  # Convert to percentage
            'sharpe_ratios': np.array(all_sharpe_ratios),
            'weights': all_weights
        }
    except Exception as e:
        return None

def simple_optimize(predictions, historical_data):
    """
    Simple equal-weight portfolio optimization for testing
    """
    try:
        num_assets = len(predictions)
        equal_weight = 1.0 / num_assets
        
        weights = {pred['Asset']: equal_weight for pred in predictions}
        
        return {
            'success': True,
            'weights': weights,
            'metrics': {
                'expected_return': sum(pred['Change (%)'] for pred in predictions) / num_assets,
                'risk': 0.0,
                'sharpe_ratio': 0.0
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def ensure_all_assets_included(selected_assets, predictions):
    # Ensure all selected assets have predictions
    missing_assets = [asset for asset in selected_assets if asset not in [p['Asset'] for p in predictions]]
    if missing_assets:
        print(f"Missing predictions for assets: {missing_assets}")
        # Add default predictions for missing assets
        for asset in missing_assets:
            predictions.append({'Asset': asset, 'Change (%)': 0})

# Example usage
selected_assets = ['NIFTYBEES', 'RELIANCE', 'TCS']  # Example selected assets
predictions = [{'Asset': 'NIFTYBEES', 'Change (%)': 2.5}]  # Example predictions
ensure_all_assets_included(selected_assets, predictions)

# Now all selected assets should be included in predictions
print(predictions)