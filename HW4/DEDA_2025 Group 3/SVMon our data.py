import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== 第一步：读取 Excel 文件 =====
file_path = r"D:\佳佳\暑期课\scrape.xlsx"  # 修改为你的本地路径，注意 r 表示原始字符串
df = pd.read_excel(file_path)

# ===== 第二步：数据清洗 =====
df_clean = df[['District', 'Year', 'Price (HKD/sq ft)']].dropna()

# ===== 第三步：构造目标变量“是否上涨” =====
df_clean['Price Next Year'] = df_clean.groupby('District')['Price (HKD/sq ft)'].shift(-1)
df_clean['Up'] = (df_clean['Price Next Year'] > df_clean['Price (HKD/sq ft)']).astype(int)
df_clean = df_clean.dropna(subset=['Price Next Year'])  # 去掉最后一年

# ===== 第四步：特征工程 =====
le = LabelEncoder()
df_clean['District Code'] = le.fit_transform(df_clean['District'])
X = df_clean[['District Code', 'Year']]
y = df_clean['Up']

# ===== 第五步：训练集与测试集划分 =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 第六步：SVM 模型训练 =====
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# ===== 第七步：预测与评估 =====
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# ===== 输出结果 =====
print("✅ SVM 模型训练完成！")
print(f"🎯 准确率 Accuracy: {accuracy:.2f}")
print("\n📋 分类报告 Classification Report:")
print(report)

# ===== 可视化混淆矩阵 =====
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()
