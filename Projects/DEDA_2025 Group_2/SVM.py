import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 1. 价格清洗函数
def clean_price(price_str):
    # 处理空值
    if pd.isna(price_str) or price_str in ['', 'NaN', 'nan', 'None']:
        return np.nan

    # 移除货币符号、逗号、空格等非数字字符
    cleaned = re.sub(r'[^\d.]', '', str(price_str))

    # 处理空字符串
    if cleaned == '':
        return np.nan

    # 转换数值并验证有效性
    try:
        value = float(cleaned)
        # 验证价格合理性
        if value <= 0 or value > 1000000:  # 假设合理价格范围
            return np.nan
        return value
    except ValueError:
        return np.nan

# 2. 读取数据
df = pd.read_csv(r'C:\Work\DEDA\pythonProject\Buff\data\buff_price_history.csv',
                 usecols=[1, 2, 3],
                 header=None,
                 skiprows=1,
                 names=['series_name', 'price', 'trade_date'])

# 3. 应用价格清洗
df['price'] = df['price'].apply(clean_price)

# 4. 检查清洗结果
invalid_prices = df[df['price'].isna()]
if not invalid_prices.empty:
    print(f"警告: {len(invalid_prices)} 行价格清洗失败")
    print("样本无效数据:")
    print(invalid_prices.head(3))

# 5. 处理同一天多价格记录
daily_stats = df.groupby(['series_name', 'trade_date'], sort=False).agg(
    daily_high=('price', 'max'),
    daily_low=('price', 'min'),
    daily_range=('price', lambda x: x.max() - x.min())
).reset_index()


# 在分组统计前添加数据验证
def validate_data(group):
    """验证数据完整性"""
    # 检查有效价格数量
    valid_prices = group['price'].notna().sum()
    if valid_prices < 2:  # 至少需要2个有效价格计算范围
        return None

    # 计算每日指标
    daily_high = group['price'].max()
    daily_low = group['price'].min()

    # 验证价格范围合理性
    if daily_high <= daily_low:
        return None

    return pd.Series({
        'daily_high': daily_high,
        'daily_low': daily_low,
        'daily_range': daily_high - daily_low
    })


# 应用验证
daily_stats = df.groupby(['series_name', 'trade_date'], sort=False).apply(validate_data).reset_index()
daily_stats = daily_stats.dropna()  # 移除无效数据


# 6. 按Xn分组
xn_groups = daily_stats.groupby('series_name', sort=False)

# 7. EWMA波动率计算
lambda_ = 0.94
volatility_results = {}
feature_data = []  # 存储特征数据

for name, group in xn_groups:
    # 直接使用原始降序数据
    log_range = np.log(group['daily_high'] / group['daily_low'])

    # 反向EWMA计算
    variance = pd.Series(np.zeros(len(log_range)), index=group.index)
    # 从最新日期向最旧日期计算
    for t in range(len(log_range) - 2, -1, -1):
        variance.iloc[t] = lambda_ * variance.iloc[t + 1] + (1 - lambda_) * log_range.iloc[t] ** 2

    # 年化波动率
    annual_vol = np.sqrt(variance.mean() * 252)
    volatility_results[name] = annual_vol

vol_df = pd.DataFrame.from_dict(volatility_results,
                                orient='index',
                                columns=['volatility'])
threshold = vol_df['volatility'].median()
vol_df['category'] = np.where(vol_df['volatility'] > threshold,
                             '变化剧烈',
                             '变化稳定')
print(vol_df)


stable_daily = daily_stats[daily_stats['series_name'].isin(
    vol_df[vol_df['category'] == '变化稳定'].index
)]

# 重命名变量避免冲突
stable_series_for_arima = stable_daily.groupby('series_name')


# 从原始数据中提取这些序列
stable_data = df[df['series_name'].isin(stable_series_for_arima)]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_arima_data(group):
    """为单个序列准备回归数据"""
    # 创建时间特征
    group = group.sort_values('trade_date')
    group['time_index'] = range(len(group))

    # 添加滞后特征
    group['price_lag1'] = group['price'].shift(1)
    group['price_lag2'] = group['price'].shift(2)

    # 移除NaN
    group = group.dropna()

    return group[['trade_date', 'daily_avg']].set_index('trade_date')


# 应用预处理
stable_data = stable_data.groupby('series_name').apply(prepare_arima_data)

# 选择特征和目标变量
X = stable_data[['time_index', 'price_lag1', 'price_lag2']]
y = stable_data['price']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(group):
    """为单个序列拟合ARIMA模型"""
    # 自动选择最优参数
    model = ARIMA(group['daily_avg'], order=(1, 0, 1))
    results = model.fit()

    # 预测
    forecast = results.get_forecast(steps=5)  # 预测未来5天
    return forecast.predicted_mean


# 应用ARIMA
arima_results = stable_data.groupby('series_name').apply(fit_arima)

# 选择示例序列
sample_series = stable_daily[stable_daily['series_name'] == stable_series_for_arima.first_valid_index()]

# 拟合ARIMA
model = fit_arima(prepare_arima_data(sample_series))
results = model.fit()

# 预测
forecast = results.get_forecast(steps=10)
forecast_index = pd.date_range(
    start=sample_series['trade_date'].iloc[-1],
    periods=11,  # 包含最后实际值
    freq='D'
)[1:]  # 从第二天开始

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(sample_series['trade_date'], sample_series['price'], 'b-', label='历史价格')
plt.plot(forecast_index, forecast.predicted_mean, 'r--', label='预测价格')
plt.fill_between(
    forecast_index,
    forecast.conf_int().iloc[:, 0],
    forecast.conf_int().iloc[:, 1],
    color='pink', alpha=0.3, label='95%置信区间'
)
plt.title(f"{stable_series_for_arima[0]} 价格预测")
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=300)


    # 修改特征提取
def extract_features(group):
        """增强的特征提取函数"""
        features = {
            'series_name': name,
            'annual_vol': volatility_results[name]
        }

        # 仅当有有效波动率时才计算其他特征
        if not np.isnan(features['annual_vol']):
            features.update({
                'avg_daily_range': group['daily_range'].mean(),
                'max_daily_range': group['daily_range'].max(),
                'volatility_std': group['daily_range'].std(),
                'skewness': group['daily_range'].skew() if len(group) > 2 else 0,
                'kurtosis': group['daily_range'].kurtosis() if len(group) > 3 else 0,
                'autocorr': group['daily_range'].autocorr() if len(group) > 1 else 0
            })
        else:
            # 设置为默认值
            features.update({
                'avg_daily_range': np.nan,
                'max_daily_range': np.nan,
                'volatility_std': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'autocorr': np.nan
            })

        return features
# 创建特征DataFrame
feature_data = []
for name, group in xn_groups:
    features = extract_features(group)
    feature_data.append(features)

# 创建DataFrame并直接设置索引
features_df = pd.DataFrame(feature_data).set_index('series_name')

# 9. 创建分类标签（基于EWMA波动率）
threshold = features_df['annual_vol'].median()
features_df['label'] = np.where(features_df['annual_vol'] > threshold, 1, 0)  # 1=变化剧烈, 0=变化稳定

# 10. 准备SVM数据
X = features_df.drop(['annual_vol', 'label'], axis=1)  # 特征矩阵
y = features_df['label']  # 标签

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 在SVM训练前添加数据清洗
def prepare_svm_data(features_df):
    """准备SVM数据并处理异常值"""
    # 移除无效行
    valid_df = features_df.dropna()

    # 移除方差为零的特征
    nonzero_var_cols = valid_df.columns[valid_df.var() > 1e-6]
    X = valid_df[nonzero_var_cols]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建标签
    threshold = valid_df['annual_vol'].median()
    y = (valid_df['annual_vol'] > threshold).astype(int)

    return X_scaled, y, valid_df.index


# 应用
X_scaled, y, valid_idx = prepare_svm_data(features_df)

# 11. SVM模型训练与评估
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

svm_model = SVC(
    kernel='rbf',  # 径向基函数核
    C=1.0,  # 正则化参数
    gamma='scale',  # 核函数系数
    probability=True  # 启用概率估计
)
svm_model.fit(X_train, y_train)

# 模型评估
y_pred = svm_model.predict(X_test)
print("\nSVM分类报告:")
print(classification_report(y_test, y_pred))

# 12. 可视化分析
# 12.1 波动率分布与分类结果
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 6))
colors = ['green' if vol <= threshold else 'red' for vol in features_df['annual_vol']]
plt.scatter(features_df.index, features_df['annual_vol'], c=colors, alpha=0.7)
plt.axhline(threshold, color='blue', linestyle='--', label=f'阈值: {threshold:.4f}')
plt.title('Xn波动率分类')
plt.ylabel('年化波动率')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('volatility_classification.png', dpi=300)

# 12.2 SVM决策边界可视化（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 训练简化SVM模型
svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_2d.fit(X_pca, y)

# 创建网格
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.title('SVM非线性决策边界 (PCA降维)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(label='波动类别 (0=稳定, 1=剧烈)')
plt.savefig('svm_decision_boundary.png', dpi=300)

# 13. 输出最终分类结果
final_classification = features_df[['annual_vol', 'label']].copy()
final_classification['svm_prediction'] = svm_model.predict(X_scaled)
final_classification['category'] = np.where(
    final_classification['svm_prediction'] == 1,
    '变化剧烈',
    '变化稳定'
)

print("\n最终分类结果:")
print(final_classification)

# 从之前分类结果中提取稳定序列
stable_series = final_classification[final_classification['category'] == '变化稳定'].index

# 从原始数据中提取这些序列
stable_data = daily_stats[daily_stats['series_name'].isin(stable_series)]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_regression_data(group):
    """为单个序列准备回归数据"""
    # 创建时间特征
    group = group.sort_values('trade_date')
    group['time_index'] = range(len(group))

    # 添加滞后特征
    group['price_lag1'] = group['price'].shift(1)
    group['price_lag2'] = group['price'].shift(2)

    # 移除NaN
    group = group.dropna()

    return group


# 应用预处理
stable_data = stable_data.groupby('series_name').apply(prepare_regression_data)

# 选择特征和目标变量
X = stable_data[['time_index', 'price_lag1', 'price_lag2']]
y = stable_data['price']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(group):
    """为单个序列拟合ARIMA模型"""
    # 自动选择最优参数
    model = ARIMA(group['price'], order=(1, 0, 1))  # ARIMA(1,0,1)
    results = model.fit()

    # 预测
    forecast = results.get_forecast(steps=5)  # 预测未来5天
    return forecast.predicted_mean


# 应用ARIMA
arima_results = stable_data.groupby('series_name').apply(fit_arima)

# 选择示例序列
sample_series = stable_data[stable_data['series_name'] == stable_series[0]]

# 拟合ARIMA
model = ARIMA(sample_series['price'], order=(1,0,1))
results = model.fit()

# 预测
forecast = results.get_forecast(steps=10)
forecast_index = pd.date_range(
    start=sample_series['trade_date'].iloc[-1],
    periods=11,  # 包含最后实际值
    freq='D'
)[1:]  # 从第二天开始

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(sample_series['trade_date'], sample_series['price'], 'b-', label='历史价格')
plt.plot(forecast_index, forecast.predicted_mean, 'r--', label='预测价格')
plt.fill_between(
    forecast_index,
    forecast.conf_int().iloc[:, 0],
    forecast.conf_int().iloc[:, 1],
    color='pink', alpha=0.3, label='95%置信区间'
)
plt.title(f"{stable_series[0]} 价格预测")
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=300)
