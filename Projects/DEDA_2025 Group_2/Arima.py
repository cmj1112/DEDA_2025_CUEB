import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('TkAgg')

# 1. Load data
file_path = "stable_time_series.csv"
df = pd.read_csv(file_path, encoding='UTF-8', parse_dates=['trade_date'])

# 2. Data preprocessing
df.sort_values('trade_date', inplace=True)  # Sort ascending (oldest first)

# Calculate daily statistics
daily_stats = df.groupby('trade_date')['price'].agg(
    ['count', 'min', 'max', 'mean', 'median', 'std']
).reset_index()

# Rename columns
daily_stats.columns = ['date', 'transaction_count', 'daily_min', 'daily_max',
                       'daily_mean', 'daily_median', 'daily_std']

# 3. Ensure date continuity
full_dates = pd.date_range(start=daily_stats['date'].min(),
                           end=daily_stats['date'].max(), freq='D')
full_df = pd.DataFrame({'date': full_dates})

# Merge data, fill missing values
ts = full_df.merge(daily_stats, on='date', how='left')
ts['transaction_count'] = ts['transaction_count'].fillna(0)


# 关键修复：检查序列是否恒定
def is_constant(series):
    """检查序列是否为常数"""
    return series.nunique() == 1


# 只填充非恒定序列
for col in ['daily_min', 'daily_max', 'daily_mean', 'daily_median', 'daily_std']:
    if not is_constant(ts[col]):
        ts[col] = ts[col].fillna(method='ffill')
    else:
        # 如果序列是常数，使用前向填充但跳过ADF检验
        ts[col] = ts[col].fillna(method='ffill')
        print(f"Warning: Column {col} is constant after filling")

ts.set_index('date', inplace=True)

# 4. Data visualization
plt.figure(figsize=(14, 12))

# Price trend
plt.subplot(3, 1, 1)
plt.plot(ts.index, ts.daily_mean, label='Daily Average Price', color='blue')
plt.fill_between(ts.index, ts.daily_min, ts.daily_max,
                 color='lightblue', alpha=0.5, label='Price Range')
plt.title('Daily Price Trend & Volatility')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Transaction volume
plt.subplot(3, 1, 2)
plt.bar(ts.index, ts.transaction_count, color='green', alpha=0.7)
plt.title('Daily Transaction Volume')
plt.ylabel('Transactions')
plt.grid(True)

# Price change rate
plt.subplot(3, 1, 3)
daily_returns = ts.daily_mean.pct_change().dropna() * 100
plt.plot(ts.index[1:], daily_returns, color='purple')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title('Daily Price Change Rate')
plt.ylabel('Change Rate (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('price_trend_analysis.jpg', dpi=300)
plt.show()

# 5. ARIMA modeling - using average price
target = ts['daily_mean'].copy()


# 关键修复：增强ADF检验函数
def safe_adf_test(series, title=''):
    """安全的ADF检验，处理恒定序列"""
    if series.nunique() == 1:
        print(f"Warning: Series '{title}' is constant. Cannot perform ADF test.")
        return 1.0  # 返回大于0.05的值强制差分

    result = adfuller(series.dropna(), autolag='AIC')
    output = {
        'ADF Test Statistic': result[0],
        'p-value': result[1],
        '# of Lags': result[2],
        'Critical Values': result[4]
    }
    print(f'ADF Test for {title}')
    for key, value in output.items():
        print(f'  {key}: {value}')
    return result[1]


print("\nStationarity Test - Original Series:")
p_value_original = safe_adf_test(target, 'Original Series')

# Differentiation
if p_value_original > 0.05:
    print("\nSeries is non-stationary - applying differencing...")
    diff_target = target.diff().dropna()
    print("\nStationarity Test - First Differences:")
    p_value_diff = safe_adf_test(diff_target, 'First Differences')

    # Visualize differentiation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    diff_target.plot(ax=axes[0], title='First Differences')
    plot_acf(diff_target, ax=axes[1], lags=20, title='ACF of First Differences')
    plt.tight_layout()
    plt.savefig('differencing_analysis.jpg', dpi=300)
    plt.show()

    d = 1  # Differencing order
else:
    print("\nSeries is stationary - no differencing needed")
    diff_target = target
    d = 0

# 6. Auto-ARIMA parameter search
print("\nSearching for optimal ARIMA parameters...")
try:
    auto_model = auto_arima(
        target,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        d=d,
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Get best parameters
    best_order = auto_model.order
    print(f"\nOptimal ARIMA parameters (p,d,q): {best_order}")
except Exception as e:
    print(f"Auto-ARIMA failed: {e}")
    print("Using default parameters (1,1,1)")
    best_order = (1, d, 1)

# 7. Model training
try:
    model = ARIMA(target, order=best_order)
    model_fit = model.fit()

    # Print model summary
    print("\nModel Summary:")
    print(model_fit.summary())
except Exception as e:
    print(f"Model training failed: {e}")
    print("Trying simpler model (0,1,0)")
    model = ARIMA(target, order=(0, d, 0))
    model_fit = model.fit()
    print("\nSimplified Model Summary:")
    print(model_fit.summary())

# 8. Residual diagnostics
residuals = model_fit.resid
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
residuals.plot(title='Model Residuals', ax=plt.gca())
plt.axhline(y=0, color='red', linestyle='--')

plt.subplot(2, 2, 2)
sns.histplot(residuals, kde=True, ax=plt.gca())
plt.title('Residual Distribution')

plt.subplot(2, 2, 3)
plot_acf(residuals, ax=plt.gca(), lags=20)
plt.title('Residual ACF')

plt.subplot(2, 2, 4)
from statsmodels.graphics.gofplots import qqplot

qqplot(residuals.dropna(), line='q', ax=plt.gca())
plt.title('Q-Q Plot')

plt.tight_layout()
plt.savefig('residual_diagnostics.jpg', dpi=300)
plt.show()

# 9. Forecasting and backtesting
# Split into train-test (last 5 days)
split_idx = -5
train_data = target.iloc[:split_idx]
test_data = target.iloc[split_idx:]

# Train model on training data
try:
    refit_model = ARIMA(train_data, order=best_order)
    refit_model_fit = refit_model.fit()

    # Forecast test set
    test_predictions = refit_model_fit.forecast(steps=len(test_data))

    # Calculate forecast errors
    mae = mean_absolute_error(test_data, test_predictions)
    mape = mean_absolute_percentage_error(test_data, test_predictions) * 100
except Exception as e:
    print(f"Forecasting failed: {e}")
    # Use simple mean as fallback
    test_predictions = [train_data.mean()] * len(test_data)
    mae = mean_absolute_error(test_data, test_predictions)
    mape = mean_absolute_percentage_error(test_data, test_predictions) * 100

# Format dates for better visualization
date_fmt = '%Y-%m-%d'
train_data.index = pd.to_datetime(train_data.index).strftime(date_fmt)
test_data.index = pd.to_datetime(test_data.index).strftime(date_fmt)

# 10. Visualize results
plt.figure(figsize=(14, 8))

# Plot full series
plt.plot(pd.to_datetime(target.index).strftime(date_fmt),
         target, 'b-', label='Actual Price')

# Plot test predictions
plt.plot(test_data.index, test_predictions, 'ro-',
         linewidth=2, markersize=8,
         label=f'Test Forecast (MAE={mae:.2f}, MAPE={mape:.1f}%)')

# Plot confidence interval
plt.fill_between(test_data.index,
                 test_predictions - mae,
                 test_predictions + mae,
                 color='r', alpha=0.1)

plt.title('ARIMA Model Forecast Performance')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_performance.jpg', dpi=300)
plt.show()

# 11. Future price forecast
future_days = 7  # Forecast 7 days into future
last_date = pd.to_datetime(target.index[-1])

# Generate future dates
future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, future_days + 1)]

# Forecast and confidence intervals
try:
    forecast = model_fit.forecast(steps=future_days)
    forecast_ci = model_fit.get_forecast(steps=future_days).conf_int()
except Exception as e:
    print(f"Future forecast failed: {e}")
    # Use last value as fallback
    forecast = [target.iloc[-1]] * future_days
    forecast_ci = pd.DataFrame({
        'lower': [target.iloc[-1] - target.std()] * future_days,
        'upper': [target.iloc[-1] + target.std()] * future_days
    })

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast': forecast,
    'lower_bound': forecast_ci.iloc[:, 0],
    'upper_bound': forecast_ci.iloc[:, 1]
}).set_index('date')

# Format dates
formatted_dates = [d.strftime(date_fmt) for d in forecast_df.index]

# 12. Visualize future forecast
plt.figure(figsize=(14, 7))

# Plot last 30 days
recent_data = target.tail(30)
plt.plot(pd.to_datetime(recent_data.index).strftime(date_fmt),
         recent_data, 'b-o', label='Historical Price')

# Plot forecast
plt.plot(formatted_dates, forecast, 'ro-', linewidth=2,
         markersize=8, label=f'{future_days}-Day Forecast')

# Confidence interval
plt.fill_between(formatted_dates,
                 forecast_df['lower_bound'],
                 forecast_df['upper_bound'],
                 color='r', alpha=0.1)

plt.title(f'Future Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('future_price_forecast.jpg', dpi=300)
plt.show()

# Print forecast results
print("\nFuture Price Forecast:")
print(forecast_df.round(2))