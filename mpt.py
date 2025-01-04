import yfinance as yf 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############ Calculate the required annualized return rate ###########
def required_annualized_return_calculation(initial, duration, target):
    required_annualized_return = ((target / initial) ** (1/duration)) - 1
    required_annualized_return = format(required_annualized_return, '.2f')
    required_annualized_return = float(required_annualized_return)
    return required_annualized_return


initial = 100000
duration = 4
target = 150000

required_return = required_annualized_return_calculation(initial, duration, target)
print(f"Required Annualized Return: {required_return}")

########### Calculate the annual return of each stock in S&P 500 ##############
def annual_return_calculation():
    df = pd.read_csv('s&p500.csv')
    annual_return_dict = {}

    for ticker in df['Symbol']:
        try:
            data = yf.download(ticker, start='2020-12-31', end='2024-12-31')['Close']

            # If data is empty (which happens sometimes for yfinance)
            if data.empty:
                print(f"No data available for {ticker}")
                continue

            returns = data.pct_change().dropna()
            mean_daily_returns = returns.mean()
            annual_return = mean_daily_returns * 252

            # Add to dict
            annual_return_dict[ticker] = annual_return
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return annual_return_dict

annual_return_dict = annual_return_calculation()

########## Calculate the annual risks (standard deviations of returns) of stocks ##########
def stock_risk_calculation():
    # Load S&P 500 tickers from the CSV file
    df = pd.read_csv('s&p500.csv')
    risk_dict = {}

    for ticker in df['Symbol']:
        try:
            data = yf.download(ticker, start='2020-12-31', end='2024-12-31')['Close']

            if data.empty:
                print(f"No data available for {ticker}")
                continue

            returns = data.pct_change().dropna()
            std_dev = returns.std()
            annualized_risk = std_dev * np.sqrt(252)
            risk_dict[ticker] = annualized_risk
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    return risk_dict

stock_risk_calculation()

############### Calculate covariance matrix ###############
def covariance_calculation():
    df = pd.read_csv('s&p500.csv')
    ticker_list = df['Symbol']

    returns_data = {}

    for ticker in ticker_list:
        data = yf.download(ticker, start='2020-12-31', end='2024-12-31')['Close']
        
        data = data.squeeze()

        daily_returns = data.pct_change().dropna()

        returns_data[ticker] = daily_returns

    returns_df = pd.DataFrame(returns_data)

    returns_df = returns_df.dropna(how='any')

    covariance_matrix = returns_df.cov()

    return covariance_matrix

try:
    cov_matrix = covariance_calculation()
    print(cov_matrix)
except Exception as e:
    print(f"An error occurred: {e}")

########### Define portfolio performance function #############
def portfolio_performance(weights, annual_returns, cov_matrix):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk

# Generate random portfolios
def generate_portfolios(annual_return_dict, cov_matrix, num_portfolios=10000):
    results = []
    weights_record = []
    tickers = list(annual_return_dict.keys())
    annual_returns = np.array(list(annual_return_dict.values()))

    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        portfolio_return, portfolio_risk = portfolio_performance(weights, annual_returns, cov_matrix)
        sharpe_ratio = portfolio_return / portfolio_risk

        results.append((portfolio_return, portfolio_risk, sharpe_ratio, weights))
        weights_record.append(weights)

    results_df = pd.DataFrame(results, columns=['Return', 'Risk', 'Sharpe Ratio', 'Weights'])
    return results_df, weights_record

portfolios_df, weights_list = generate_portfolios(annual_return_dict, cov_matrix)

########## Get most recent stock prices ##########
def get_stock_prices():
    prices = {}
    # Load the CSV file
    df = pd.read_csv('s&p500.csv')
    
    for ticker in df['Symbol']:
        try:
            stock = yf.Ticker(ticker)
            prices[ticker] = stock.history(period='1d')['Close'][-1]
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return prices

stock_prices = get_stock_prices()

print(stock_prices)

######### Choose suitable portfolios based on conditions provided by users #########

def filter_portfolios(portfolios_df, required_return, stock_prices, initial):
    valid_portfolios = []

    for _, row in portfolios_df.iterrows():
        weights = row['Weights']  # Portfolio weights
        
        # Calculate the total cost of the portfolio
        total_cost = sum(
            w * initial / stock_prices[ticker] 
            for ticker, w in zip(stock_prices.keys(), weights)
        )
        
        # Ensure portfolio satisfies both return and cost constraints
        if row['Return'] >= required_return and total_cost <= initial:
            portfolio_data = dict(row)
            portfolio_data['Total Cost'] = total_cost
            valid_portfolios.append(portfolio_data)

    filtered_df = pd.DataFrame(valid_portfolios)
    
    if not filtered_df.empty:
        return filtered_df.sort_values(by='Sharpe Ratio', ascending=False).head(5)
    else:
        print("No portfolios satisfy the constraints.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid portfolios

top_5_portfolios = filter_portfolios(portfolios_df, required_return, stock_prices, initial)

print(top_5_portfolios)

######## Print top 5 portfolios ########
for i, row in top_5_portfolios.iterrows():
    print(f"Portfolio {i + 1}")
    
    portfolio_return = row['Return'][0] if isinstance(row['Return'], np.ndarray) else row['Return']
    portfolio_risk = row['Risk'][0] if isinstance(row['Risk'], np.ndarray) else row['Risk']
    sharpe_ratio = row['Sharpe Ratio'][0] if isinstance(row['Sharpe Ratio'], np.ndarray) else row['Sharpe Ratio']
    
    print(f"Return: {portfolio_return:.2%}")
    print(f"Risk: {portfolio_risk:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    formatted_weights = ', '.join([f"{weight:.2%}" for weight in row['Weights']])
    print(f"Weights: {formatted_weights}\n")

########### Final results ###########
def calculate_quantities(top_5_portfolios, stock_prices, initial):
    portfolio_details = []
    
    for i, row in top_5_portfolios.iterrows():
        portfolio_weights = row['Weights']
        portfolio_costs = []
        stock_quantities = {}
        
        for stock, weight in zip(stock_prices.keys(), portfolio_weights):
            stock_allocation = weight * initial
            stock_price = stock_prices[stock]
            quantity = stock_allocation // stock_price  
            
            if quantity > 0:  # Include only stocks with non-zero quantities
                portfolio_costs.append(quantity * stock_price)
                stock_quantities[stock] = quantity
        
        total_cost = sum(portfolio_costs)
        
        if total_cost <= initial:
            portfolio_details.append({
                'Portfolio': i + 1,
                'Stocks': stock_quantities,
                'Total Cost': total_cost,
                'Remaining Budget': initial - total_cost,
                'Return': float(row['Return']),
                'Risk': float(row['Risk']),
                'Sharpe Ratio': float(row['Sharpe Ratio'])
            })
    
    return portfolio_details

portfolio_details = calculate_quantities(top_5_portfolios, stock_prices, initial)

# Display portfolios
if portfolio_details:
    for portfolio in portfolio_details:
        print(f"Portfolio {portfolio['Portfolio']}:")
        print(f"  Stocks and Quantities: {portfolio['Stocks']}")
        print(f"  Total Cost: ${portfolio['Total Cost']:.2f}")
        print(f"  Remaining Budget: ${portfolio['Remaining Budget']:.2f}")
        print(f"  Return: {portfolio['Return']:.2%}")
        print(f"  Risk: {portfolio['Risk']:.2%}")
        print(f"  Sharpe Ratio: {portfolio['Sharpe Ratio']:.2f}\n")
else:
    print("No valid portfolios found within the given constraints.")

