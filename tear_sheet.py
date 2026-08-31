import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas_datareader.data as web
import statsmodels.api as sm
import base64


tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
benchmark = "^GSPC"

start_date = "2019-01-01"
end_date = "2024-12-31"

data = yf.download(
    tickers + [benchmark],
    start=start_date,
    end=end_date,
    auto_adjust=True  
)["Close"]

print(data.head())
print(data.shape)

returns = data.pct_change().dropna()
# just for viewing — doesn't change `returns` itself
print(returns.head().map(lambda x: f"{x:.2%}"))

TRADING_DAYS = 252
risk_free_rate = 0.02  

def cagr(prices):
    n_years = len(prices) / TRADING_DAYS
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1

def annual_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(daily_returns, rf=risk_free_rate):
    ann_return = daily_returns.mean() * TRADING_DAYS
    ann_vol = annual_volatility(daily_returns)
    return (ann_return - rf) / ann_vol

def sortino_ratio(daily_returns, rf=risk_free_rate):
    ann_return = daily_returns.mean() * TRADING_DAYS
    downside_returns = daily_returns[daily_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    return (ann_return - rf) / downside_dev

# Run for every ticker + benchmark
for col in data.columns:
    print(f"\n--- {col} ---")
    print(f"CAGR: {cagr(data[col]):.2%}")
    print(f"Annual Volatility: {annual_volatility(returns[col]):.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio(returns[col]):.2f}")
    print(f"Sortino Ratio: {sortino_ratio(returns[col]):.2f}")

equity_curve = (1 + returns).cumprod()

plt.figure(figsize=(12, 6))
for col in equity_curve.columns:
    plt.plot(equity_curve.index, equity_curve[col], label=col)

plt.title("Equity Curve: Growth of $1 Invested (2019–2024)")
plt.xlabel("Date")
plt.ylabel("Growth Multiple")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.tight_layout()
plt.savefig("equity_curve.png")  
plt.show()

running_max = equity_curve.cummax()
drawdown = (equity_curve - running_max) / running_max

max_drawdown = drawdown.min()
print(max_drawdown)

plt.figure(figsize=(12, 6))
for col in drawdown.columns:
    plt.plot(drawdown.index, drawdown[col], label=col)

plt.title("Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.fill_between(drawdown.index, 0, drawdown["^GSPC"], alpha=0.1)  
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("drawdown.png")
plt.close()  

window = 126 

rolling_mean = returns.rolling(window).mean() * TRADING_DAYS
rolling_std = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std

plt.figure(figsize=(12, 6))
for col in rolling_sharpe.columns:
    plt.plot(rolling_sharpe.index, rolling_sharpe[col], label=col)

plt.axhline(0, color="black", linewidth=0.8)
plt.title(f"Rolling {window}-Day Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rolling_sharpe.png")
plt.close()

from scipy import stats
import numpy as np


log_returns = np.log(data / data.shift(1)).dropna()

daily_rf = risk_free_rate / TRADING_DAYS  

market_excess = log_returns["^GSPC"] - daily_rf

capm_results = {}

for col in tickers:
    asset_excess = log_returns[col] - daily_rf
    beta, alpha, r_value, p_value, std_err = stats.linregress(market_excess, asset_excess)
    capm_results[col] = {
        "alpha_daily": alpha,
        "alpha_annualized": alpha * TRADING_DAYS,
        "beta": beta,
        "r_squared": r_value**2
    }

for col, res in capm_results.items():
    print(f"\n--- {col} ---")
    print(f"Beta: {res['beta']:.2f}")
    print(f"Alpha (annualized): {res['alpha_annualized']:.2%}")
    print(f"R-squared: {res['r_squared']:.2f}")

ff3_factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start_date, end=end_date)[0]
print(ff3_factors.head())
print(ff3_factors.columns)

ff3_factors.index = ff3_factors.index.to_timestamp()
ff3_factors = ff3_factors / 100  # convert from percentage points to decimals

aligned = log_returns.join(ff3_factors, how="inner")
print(aligned.shape)
print(aligned.head())

ff3_results = {}

for col in tickers:
    y = aligned[col] - aligned["RF"]  # asset's excess return
    X = aligned[["Mkt-RF", "SMB", "HML"]]
    X = sm.add_constant(X)  # adds the alpha (intercept) term

    model = sm.OLS(y, X).fit()

    ff3_results[col] = {
        "alpha_daily": model.params["const"],
        "alpha_annualized": model.params["const"] * TRADING_DAYS,
        "beta_mkt": model.params["Mkt-RF"],
        "beta_smb": model.params["SMB"],
        "beta_hml": model.params["HML"],
        "r_squared": model.rsquared
    }

for col, res in ff3_results.items():
    print(f"\n--- {col} ---")
    print(f"Alpha (annualized): {res['alpha_annualized']:.2%}")
    print(f"Beta (Market): {res['beta_mkt']:.2f}")
    print(f"Beta (SMB): {res['beta_smb']:.2f}")
    print(f"Beta (HML): {res['beta_hml']:.2f}")
    print(f"R-squared: {res['r_squared']:.2f}")

ff5_factors = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start_date, end=end_date)[0]
ff5_factors.index = ff5_factors.index.to_timestamp()
ff5_factors = ff5_factors / 100  

aligned5 = log_returns.join(ff5_factors, how="inner")

ff5_results = {}

for col in tickers:
    y = aligned5[col] - aligned5["RF"]
    X = aligned5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    ff5_results[col] = {
        "alpha_annualized": model.params["const"] * TRADING_DAYS,
        "beta_mkt": model.params["Mkt-RF"],
        "beta_smb": model.params["SMB"],
        "beta_hml": model.params["HML"],
        "beta_rmw": model.params["RMW"],
        "beta_cma": model.params["CMA"],
        "r_squared": model.rsquared
    }

for col, res in ff5_results.items():
    print(f"\n--- {col} ---")
    print(f"Alpha (annualized): {res['alpha_annualized']:.2%}")
    print(f"Beta (Market): {res['beta_mkt']:.2f}")
    print(f"Beta (SMB): {res['beta_smb']:.2f}")
    print(f"Beta (HML): {res['beta_hml']:.2f}")
    print(f"Beta (RMW): {res['beta_rmw']:.2f}")
    print(f"Beta (CMA): {res['beta_cma']:.2f}")
    print(f"R-squared: {res['r_squared']:.2f}")

print("\n--- Risk Categorization (based on FF5 R²) ---")
for col, res in ff5_results.items():
    systematic = res["r_squared"] * 100
    idiosyncratic = (1 - res["r_squared"]) * 100
    print(f"{col}: Systematic {systematic:.1f}% | Idiosyncratic {idiosyncratic:.1f}%")
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

equity_b64 = img_to_base64("equity_curve.png")
drawdown_b64 = img_to_base64("drawdown.png")
rolling_sharpe_b64 = img_to_base64("rolling_sharpe.png")
def build_metrics_table():
    rows = ""
    for col in tickers + [benchmark]:
        rows += f"""
        <tr>
            <td>{col}</td>
            <td>{cagr(data[col]):.2%}</td>
            <td>{annual_volatility(returns[col]):.2%}</td>
            <td>{sharpe_ratio(returns[col]):.2f}</td>
            <td>{sortino_ratio(returns[col]):.2f}</td>
            <td>{max_drawdown[col]:.2%}</td>
        </tr>"""
    return rows

def build_factor_table():
    rows = ""
    for col in tickers:
        capm = capm_results[col]
        ff3 = ff3_results[col]
        ff5 = ff5_results[col]
        rows += f"""
        <tr>
            <td>{col}</td>
            <td>{capm['alpha_annualized']:.2%}</td>
            <td>{ff3['alpha_annualized']:.2%}</td>
            <td>{ff5['alpha_annualized']:.2%}</td>
            <td>{ff5['beta_mkt']:.2f}</td>
            <td>{ff5['r_squared']*100:.1f}%</td>
            <td>{(1-ff5['r_squared'])*100:.1f}%</td>
        </tr>"""
    return rows
html_report = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; color: #222; }}
    h1 {{ border-bottom: 3px solid #222; padding-bottom: 10px; }}
    h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
    th {{ background: #222; color: white; }}
    td:first-child, th:first-child {{ text-align: left; font-weight: bold; }}
    img {{ max-width: 100%; margin: 15px 0; }}
</style>
</head>
<body>

<h1>Quant Tear Sheet</h1>
<p>Portfolio: {", ".join(tickers)} | Benchmark: {benchmark} | Period: {start_date} to {end_date}</p>

<h2>Performance Metrics</h2>
<table>
<tr><th>Ticker</th><th>CAGR</th><th>Volatility</th><th>Sharpe</th><th>Sortino</th><th>Max Drawdown</th></tr>
{build_metrics_table()}
</table>

<h2>Equity Curve</h2>
<img src="data:image/png;base64,{equity_b64}">

<h2>Drawdown</h2>
<img src="data:image/png;base64,{drawdown_b64}">

<h2>Rolling {window}-Day Sharpe Ratio</h2>
<img src="data:image/png;base64,{rolling_sharpe_b64}">

<h2>Factor Regressions (Alpha Comparison &amp; Risk Categorization)</h2>
<table>
<tr><th>Ticker</th><th>CAPM Alpha</th><th>FF3 Alpha</th><th>FF5 Alpha</th><th>Market Beta</th><th>Systematic %</th><th>Idiosyncratic %</th></tr>
{build_factor_table()}
</table>

</body>
</html>
"""

with open("tear_sheet_report.html", "w") as f:
    f.write(html_report)

print("Report saved to tear_sheet_report.html")
