import pandas as pd
import numpy as np
import re


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

# 可视化波动性分布
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 6))
colors = vol_df['category'].map({'变化剧烈': 'red', '变化稳定': 'green'})
plt.bar(vol_df.index, vol_df['volatility'], color=colors)
plt.axhline(threshold, color='blue', linestyle='--', label=f'阈值: {threshold:.4f}')
plt.title('价格波动性分类')
plt.ylabel('年化波动率')
plt.xticks(rotation=45, ha="right",size=2)
plt.legend()
plt.tight_layout()
plt.savefig('volatility_classification.png', dpi=3000)

# 获取所有变化稳定的系列名称
stable_series = vol_df[vol_df['category'] == '变化稳定'].index.tolist()

# 从原始数据中提取这些系列的所有数据
stable_data = df[df['series_name'].isin(stable_series)]

# 或者从每日统计数据中提取（根据后续分析需要选择其一）
# stable_data = daily_stats[daily_stats['series_name'].isin(stable_series)]

# 检查结果
print(f"变化稳定的系列数量: {len(stable_series)}")
print(f"变化稳定的数据行数: {len(stable_data)}")
print("\n变化稳定的系列列表:")
print(stable_series)

# 保存结果到新表格
stable_data.to_csv('stable_time_series1.csv', index=False)

print(f"提取到 {len(stable_data)} 条稳定序列数据")
