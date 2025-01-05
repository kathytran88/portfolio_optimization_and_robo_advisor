from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import yfinance as yf
from transformers import pipeline
import spacy
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# Load NPL model. My laptop has GPU so I specify to run with CUDA
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
sentiment_analyzer = pipeline("sentiment-analysis", device=0)

nlp = spacy.load("en_core_web_sm")

def ensure_complete_sentences(text):
    doc = nlp(text)
    final_sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text.endswith(('.', '!', '?')):
            final_sentences.append(sent_text)
        else:
            pass

    return " ".join(final_sentences)


def analyze_article(text):
    # Summarize the article
    try:
        raw_summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        summary = ensure_complete_sentences(raw_summary)
    except Exception as e:
        summary = "Error summarizing the article."
        print(f"Summarization error: {e}")

    # Extract entities
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        entities = []
        print(f"Entity extraction error: {e}")

    # Organizations
    org = []
    for entity, label in entities:
        if label == 'ORG' and entity not in org: 
            org.append(entity)
    if len(org) == 0:
        org = 'No organizations affected'

    # Analyze sentiment
    try:
        sentiment = sentiment_analyzer(text)[0]
    except Exception as e:
        sentiment = {"label": "Neutral", "score": 0}
        print(f"Sentiment analysis error: {e}")

    # Find stock
    try:
        df = pd.read_csv('company_tickers.csv')
    except Exception as e:
        print(f"Error loading ticker CSV: {e}")
        return {
            "summary": summary,
            "org": org,
            "sentiment": sentiment,
            "ticker": "Error loading ticker data",
            "current price": "N/A",
            "new price": "N/A"
        }

    ticker = []
    for entity, label in entities:
        for index, comp in df['company'].items():
            if entity.lower() == comp.lower():
                if df['ticker'][index] not in ticker:
                    ticker.append(df['ticker'][index]) 

    ### Stock price ###
    scaling_factor = 0.05
    percentage_change = scaling_factor * sentiment['score']

    if len(ticker) == 0:
        ticker = 'No stocks affected'
        current_price = ['N/A']
        new_price = ['N/A']  

    elif len(ticker) == 1:
        stock = yf.Ticker(ticker[0])  
        try:
            current_price = [stock.history(period="1d")["Close"].iloc[-1]]
            # Predicted price
            if sentiment['label'] == 'POSITIVE':
                new_price = [current_price[0] + (percentage_change * current_price[0])]
            elif sentiment['label'] == 'NEGATIVE':
                new_price = [current_price[0] - (percentage_change * current_price[0])]
            else:
                new_price = [current_price[0]]
        except Exception:
            current_price = ['N/A']
            new_price = ['N/A']

    elif len(ticker) > 1:
        current_price = []
        new_price = []
        for each_ticker in ticker:
            stock = yf.Ticker(each_ticker)
            try:
                each_current_price = stock.history(period="1d")["Close"].iloc[-1]
                current_price.append(each_current_price)
                if sentiment['label'] == 'POSITIVE':
                    new_price.append(each_current_price + (percentage_change * each_current_price))
                elif sentiment['label'] == 'NEGATIVE':
                    new_price.append(each_current_price - (percentage_change * each_current_price))
                else:
                    new_price.append(each_current_price)
            except Exception:
                current_price.append('N/A')
                new_price.append('N/A')

    # results
    return {
        "summary": summary,
        "org": org,
        "sentiment": sentiment,
        "ticker": ticker,
        "current price": current_price,
        "new price": new_price
    }

####################### Ro-bo advisors #######################
#### I have pre-calculated the following data and export as csv files ####
# annual returns
annual_df = pd.read_csv('annual_returns.csv')
annual_return_dict = pd.Series(annual_df['Annual Return'].values, index=annual_df['Ticker']).to_dict()

# covariance matrix
cov_matrix = pd.read_csv('cov_matrix.csv', index_col=0)

# stock prices
stock_prices_df = pd.read_csv('stock_prices.csv')
stock_prices = pd.Series(stock_prices_df['Stock Price'].values, index=stock_prices_df['Ticker']).to_dict()

##############################################
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article_text = request.form['article_text']
        results = analyze_article(article_text)
        return render_template('index.html', results=results, submitted=True)
    return render_template('index.html', submitted=False)

@app.route('/robo', methods=['GET', 'POST'])
def robo_advisor():
    if request.method == 'POST':
        try:
            # Get user inputs
            initial = float(request.form['initial_investment'])
            duration = int(request.form['duration'])
            target = float(request.form['target_return'])

            # Calculate required annualized return from the inputs
            required_return = ((target / initial) ** (1 / duration)) - 1

            # Cost
            def integer_shares_cost(weights, stock_prices, initial):
                total_cost = 0
                for ticker, w in zip(stock_prices.keys(), weights):
                    dollars_allocated = w * initial
                    price = stock_prices[ticker]
                    quantity = int(dollars_allocated // price) 
                    total_cost += (quantity * price)
                return total_cost

            def portfolio_performance(weights, annual_returns, cov_matrix):
                portfolio_return = np.dot(weights, annual_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_return, portfolio_risk

            def generate_portfolios(annual_return_dict, cov_matrix, num_portfolios=10000):
                results = []
                tickers = list(annual_return_dict.keys())
                annual_returns = np.array(list(annual_return_dict.values()))

                for _ in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)

                    portfolio_return, portfolio_risk = portfolio_performance(
                        weights, annual_returns, cov_matrix
                    )
                    sharpe_ratio = (
                        portfolio_return / portfolio_risk
                        if portfolio_risk != 0
                        else 0
                    )

                    results.append((portfolio_return, portfolio_risk, sharpe_ratio, weights))

                results_df = pd.DataFrame(
                    results, columns=['Return', 'Risk', 'Sharpe Ratio', 'Weights']
                )
                return results_df

            # Generate random portfolios
            portfolios_df = generate_portfolios(annual_return_dict, cov_matrix)
            
            def filter_portfolios(portfolios_df, required_return, stock_prices, initial):
                valid_portfolios = []

                for _, row in portfolios_df.iterrows():
                    weights = row['Weights'] 

                    total_cost = integer_shares_cost(weights, stock_prices, initial)

                    # Ensure portfolio satisfies both return and cost constraints
                    if row['Return'] >= required_return and total_cost <= initial:
                        portfolio_data = dict(row)
                        portfolio_data['Total Cost'] = total_cost
                        valid_portfolios.append(portfolio_data)

                filtered_df = pd.DataFrame(valid_portfolios)
                if filtered_df.empty:
                    print("No portfolios satisfy the constraints.")
                    return pd.DataFrame()
                return filtered_df.sort_values(by='Sharpe Ratio', ascending=False).head(5)

            top_5_portfolios = filter_portfolios(
                portfolios_df, required_return, stock_prices, initial
            )

            def calculate_quantities(top_5_portfolios, stock_prices, initial):
                portfolio_details = []

                for i, row in top_5_portfolios.iterrows():
                    portfolio_weights = row['Weights']
                    
                    weighted_stocks = sorted(
                        zip(stock_prices.keys(), portfolio_weights),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    leftover = initial
                    stock_quantities = {}

                    for stock, weight in weighted_stocks:
                        if weight > 0:
                            stock_price = stock_prices[stock]
                            allocation = leftover * weight
                            shares = int(allocation // stock_price)
                            cost = shares * stock_price
                            leftover -= cost
                            if shares > 0:
                                stock_quantities[stock] = shares

                    purchase_possible = True
                    while purchase_possible:
                        purchase_possible = False
                        for stock, weight in weighted_stocks:
                            price = stock_prices[stock]
                            if leftover >= price > 0:  
                                leftover -= price
                                stock_quantities[stock] = stock_quantities.get(stock, 0) + 1
                                purchase_possible = True

                    total_cost = initial - leftover

                    portfolio_details.append({
                        'Portfolio': len(portfolio_details) + 1,
                        'Stocks': stock_quantities,
                        'Total Cost': total_cost,
                        'Remaining Budget': leftover,
                        'Return': float(row['Return']),
                        'Risk': float(row['Risk']),
                        'Sharpe Ratio': float(row['Sharpe Ratio'])
                    })

                return portfolio_details

            portfolio_details = calculate_quantities(top_5_portfolios, stock_prices, initial)

            # ouput results
            return render_template(
                'robo.html',
                portfolios=portfolio_details,
                initial=initial,
                duration=duration,
                target=target
            )

        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('robo_advisor'))

    return render_template('robo.html', portfolios=None)

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
